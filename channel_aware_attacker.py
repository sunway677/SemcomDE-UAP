"""
Channel-aware UAP generation and adversarial learning (Sections II-III).

Symbols follow the paper. ChannelAwareUAPAttacker.step() performs one
training iteration of Section III-E on a mini-batch (B_x, B_t) of B pairs:

  1. M SNR values gamma_1..gamma_M are drawn once by stratified sampling
     over the curriculum range [gamma_min^(e), gamma_max] (Eq. 15). The same
     gamma_m conditions both generators and parameterizes both channels.
  2. For every gamma_m:
       image branch  c_x,i = [e_txt_i ; f_snr(gamma_m)]                (Eq. 2)
                     delta_hat_x,i = G_x(z_x ; c_x,i)                  (Eq. 4)
                     delta_x = clip(mean_i delta_hat_x,i, -eps_x, eps_x)
       text branch   c_t = [e_img ; f_snr(gamma_m)]                    (Eq. 3)
                     delta_t = norm-clamp(G_t(z_t ; c_t))              (Eq. 5)
       adversarial inputs x_i + delta_x and T_i,1 (+) delta_t, encoded
       into v_adv_i and u_adv_i                                        (Eq. 6)
       R channel views of every adversarial embedding                  (Eq. 7)
       semantic targets mined on the first view                        (Eq. 8)
       L_CL = L_CL^(v) + L_CL^(u)                                      (Eqs. 9-11)
       L_SEP on the noiseless adversarial embeddings                   (Eq. 12)
       L_MVC, only when gamma_m >= gamma_mvc                           (Eq. 13)
  3. L = 1/M sum_m (L_CL,m + lambda_1 L_SEP,m + lambda_2 L_MVC,m)       (Eq. 16),
     followed by a single backward pass and one step of three Adam
     optimizers: (G_x, G_t, f_snr) at eta_g, z_x at eta_z = alpha * eta_g
     and z_t, with its own dedicated Adam, at eta_z.

snr_min is a mutable attribute so that the training script can apply the
curriculum schedule of Eq. 14 at every epoch.
"""

import contextlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Surrogate VLP models supported for UAP generation (Sec. IV-A).
SURROGATES = ('ALBEF', 'TCL', 'BLIP')

# d: dimension of the shared image-text embedding space of the surrogates.
EMBED_DIM = 256

# d_snr: dimension of the SNR embedding f_snr(gamma).
SNR_EMB_DIM = 64

# Dimension of one token embedding of the surrogates' BERT text encoders,
# i.e. of one row of the text UAP delta_t.
TOKEN_EMB_DIM = 768


# ==========================================
# Channel model and SNR sampling
# ==========================================

def stratified_snr_sample(snr_min, snr_max, m_strata):
    """Stratified SNR sampling over [snr_min, snr_max] (Eq. 15).

    The range is split into M equal-width intervals and one SNR value (dB)
    is drawn uniformly from each, giving gamma_1 < ... < gamma_M.
    """
    width = (snr_max - snr_min) / m_strata
    return [float(np.random.uniform(snr_min + m * width, snr_min + (m + 1) * width))
            for m in range(m_strata)]


def noise_std(snr_db, dim):
    """sigma(gamma), with sigma^2(gamma) = 1 / (d * 10^(gamma/10)) (Sec. II)."""
    return math.sqrt(1.0 / (dim * 10.0 ** (snr_db / 10.0)))


def channel_views(adv_feats, snr_db, r_views):
    """R independent AWGN realizations of the adversarial embeddings (Eq. 7).

    adv_feats: (B, d) L2-normalized adversarial embeddings.
    Returns a list of R tensors norm(adv_feats + n^(r)), n^(r) ~ N(0, sigma^2(gamma) I).
    """
    sigma = noise_std(snr_db, adv_feats.shape[-1])
    return [F.normalize(adv_feats + sigma * torch.randn_like(adv_feats), dim=-1)
            for _ in range(r_views)]


def build_condition(snr_encoder, semantic_feats, snr_db):
    """Semantic-channel condition [semantic ; f_snr(gamma)] (Eq. 2 / Eq. 3).

    semantic_feats: (n, d), one row per generator input; the SNR embedding
    is shared by all rows.
    """
    snr_emb = snr_encoder(snr_db, semantic_feats.device)
    return torch.cat([semantic_feats, snr_emb.expand(semantic_feats.shape[0], -1)], dim=-1)


# ==========================================
# Losses
# ==========================================

@torch.no_grad()
def mine_semantic_targets(first_view, clean_img_feats, clean_txt_feats):
    """Target indices y_i = argmin_{j != i} (a_i^T v_j + a_i^T u_j) (Eq. 8).

    first_view: (B, d) first channel view of the adversarial embeddings of
    one branch. Requires B >= 2.
    """
    sim = first_view @ clean_img_feats.T + first_view @ clean_txt_feats.T
    sim.fill_diagonal_(float('inf'))
    return sim.argmin(dim=1)


def contrastive_loss(views, targets, clean_self_feats, temperature, margin):
    """Branch contrastive loss L_CL^(v) (Eq. 10) or L_CL^(u) (Eq. 11).

    views:            the R channel views (B, d) of the branch.
    targets:          the target features of every sample, [v_{y_i}, u_{y_i}],
                      each (B, d): the target set P_i (or Q_i).
    clean_self_feats: (B, d) clean features of the branch's own modality
                      (v for the image branch, u for the text branch).

    For every view r, sample i and target p (Eq. 9):
        l = -log( sim_mu(a, p) / Omega ),
        Omega = sim(a, p) + sim(a, c_i) + sum_{j != i} sim(a, c_j),
    with a the view, c the clean features of the branch's modality,
    sim(a, b) = exp(a^T b / tau) and sim_mu(a, b) = exp((a^T b - mu) / tau).
    The result is the average over the R views, the B samples and the two
    targets, i.e. 1/(2RB) times the sum.
    """
    terms = []
    for view in views:
        # a_i^T c_j / tau for all j, including the self term j = i.
        logits_clean = view @ clean_self_feats.T / temperature
        for target in targets:
            logit_target = (view * target).sum(dim=-1) / temperature
            log_partition = torch.logsumexp(
                torch.cat([logit_target.unsqueeze(1), logits_clean], dim=1), dim=1)
            terms.append(log_partition - (logit_target - margin / temperature))
    return torch.stack(terms).mean()


def separation_loss(v_adv, v_clean, u_adv, u_clean):
    """L_SEP = 1/(2B) sum_i (v_adv_i^T v_i + u_adv_i^T u_i) (Eq. 12).

    Computed on the noiseless adversarial embeddings.
    """
    return ((v_adv * v_clean).sum() + (u_adv * u_clean).sum()) / (2 * v_adv.shape[0])


def consistency_term(views):
    """1 - 1/((R-1)B) sum_i sum_{r=2..R} (view_i^(r))^T view_i^(1) (Eq. 13).

    The first view, which is also used for target mining, is the reference.
    """
    reference = views[0]
    return 1.0 - torch.stack([(view * reference).sum(dim=-1) for view in views[1:]]).mean()


def consistency_loss(img_views, txt_views):
    """L_MVC: average of the image-branch and text-branch terms (Eq. 13)."""
    return 0.5 * (consistency_term(img_views) + consistency_term(txt_views))


# ==========================================
# Joint image/text UAP attacker
# ==========================================

class ChannelAwareUAPAttacker:
    """Joint optimizer of the image and text UAP generators (Sec. III-B to III-E).

    Args:
      img_generator / txt_generator: G_x and G_t (generator.Generator).
      snr_encoder:   the channel state encoder f_snr shared by both generators.
      z_x_shape / z_t_shape: shapes of the single latent codes z_x and z_t,
                     each initialized from N(0, I) and shared by all samples.
      eps_x:         l_inf budget of the image UAP (in [0, 1] pixel units).
      eps_t:         text budget in tokens, i.e. the number of rows of delta_t.
      min_norm / max_norm: L2-norm range of the word-embedding table, used to
                     norm-clamp every row of delta_t.
      lr:            eta_g, learning rate of G_x, G_t and f_snr.
      z_lr_scale:    alpha, so that z_x and z_t use eta_z = alpha * eta_g.
      temperature / margin: tau and mu of Eq. 9.
      m_strata / r_views:   M SNR strata per step and R channel views per stratum.
      lambda_sep / lambda_mvc: lambda_1 and lambda_2 of Eq. 16.
      mvc_snr_threshold: gamma_mvc (dB); lambda_2 = 0 when gamma_m < gamma_mvc.
      snr_min / snr_max: current SNR range [gamma_min^(e), gamma_max] (dB).
      betas:         Adam betas of all three optimizers.
    """

    def __init__(self, img_generator, txt_generator, snr_encoder,
                 z_x_shape, z_t_shape, eps_x, eps_t, min_norm, max_norm,
                 lr=1e-4, z_lr_scale=0.1, temperature=0.07, margin=0.02,
                 m_strata=3, r_views=3, lambda_sep=0.2, lambda_mvc=0.1,
                 mvc_snr_threshold=5.0, snr_min=0.0, snr_max=20.0,
                 betas=(0.5, 0.999), device='cuda'):
        self.device = torch.device(device)
        self.img_generator = img_generator.to(self.device)
        self.txt_generator = txt_generator.to(self.device)
        self.snr_encoder = snr_encoder.to(self.device)

        # One learnable latent code per generator (Eqs. 4-5), z ~ N(0, I).
        self.z_x = nn.Parameter(torch.randn(*z_x_shape, device=self.device))
        self.z_t = nn.Parameter(torch.randn(*z_t_shape, device=self.device))

        self.eps_x = eps_x
        self.eps_t = eps_t
        self.min_norm = float(min_norm)
        self.max_norm = float(max_norm)
        self.temperature = temperature
        self.margin = margin
        self.m_strata = m_strata
        self.r_views = r_views
        self.lambda_sep = lambda_sep
        self.lambda_mvc = lambda_mvc
        self.mvc_snr_threshold = mvc_snr_threshold
        self.snr_min = snr_min
        self.snr_max = snr_max

        # G_x, G_t and f_snr are optimized jointly by one Adam at eta_g; z_x
        # and z_t by separate Adam optimizers at eta_z = alpha * eta_g.
        generator_params = (list(self.img_generator.parameters())
                            + list(self.txt_generator.parameters())
                            + list(self.snr_encoder.parameters()))
        self.optim_g = optim.Adam(generator_params, lr=lr, betas=betas)
        self.optim_zx = optim.Adam([self.z_x], lr=lr * z_lr_scale, betas=betas)
        self.optim_zt = optim.Adam([self.z_t], lr=lr * z_lr_scale, betas=betas)

    def _autocast(self):
        """bf16 autocast for the generators and the surrogate on CUDA."""
        if self.device.type == 'cuda':
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def generate_image_uap(self, txt_cond_feats, snr_db):
        """Image UAP delta_x of shape (1, 3, H, W) (Eqs. 2 and 4).

        txt_cond_feats: (B, d) mean caption embeddings e_txt_i of the batch.
        Every sample i yields a candidate G_x(z_x ; c_x,i) from the shared
        latent code z_x; the candidates are averaged over the batch and the
        mean is clipped to [-eps_x, eps_x].
        """
        cond = build_condition(self.snr_encoder, txt_cond_feats, snr_db)
        with self._autocast():
            candidates = self.img_generator(
                self.z_x.expand(cond.shape[0], -1, -1, -1), cond)
        candidates = candidates.float()
        return torch.clamp(candidates.mean(dim=0, keepdim=True), -self.eps_x, self.eps_x)

    def generate_text_uap(self, img_prototype, snr_db):
        """Text UAP delta_t of shape (eps_t, D) (Eqs. 3 and 5).

        img_prototype: (1, d) batch-level image prototype e_img. Each of the
        eps_t rows of G_t(z_t ; c_t) is norm-clamped to [min_norm, max_norm].
        """
        cond = build_condition(self.snr_encoder, img_prototype, snr_db)
        with self._autocast():
            delta = self.txt_generator(self.z_t, cond)
        delta = delta.float().reshape(self.eps_t, -1)
        norm = delta.norm(dim=1, keepdim=True)
        return delta * (norm.clamp(self.min_norm, self.max_norm) / norm.clamp_min(1e-12))

    def step(self, model, normalization, images, text_inputs,
             positions, padding_zeros, txt_cond_feats, clean_img_feats, clean_txt_feats):
        """One optimization step on a mini-batch of B >= 2 image-text pairs.

        images:          (B, 3, H, W) clean images x_i in [0, 1].
        text_inputs:     tokenized first captions T_i,1.
        positions / padding_zeros: per caption, the word-piece offsets of the
                         eps_t most important words and their number of word
                         pieces minus one (where delta_t is substituted).
        txt_cond_feats:  (B, d) mean caption embeddings e_txt_i (Eq. 2).
        clean_img_feats / clean_txt_feats: (B, d) clean features v_i and u_i
                         (Eq. 1), L2-normalized.

        Returns (stats, delta_x, delta_t): the mean loss terms over the M
        strata, and the image and text UAPs generated at the last stratum
        gamma_M of this step (before the parameter update).
        """
        v_clean = clean_img_feats
        u_clean = clean_txt_feats

        self.img_generator.train()
        self.txt_generator.train()
        self.snr_encoder.train()
        self.optim_g.zero_grad(set_to_none=True)
        self.optim_zx.zero_grad(set_to_none=True)
        self.optim_zt.zero_grad(set_to_none=True)

        # Eq. 15: gamma_1..gamma_M are drawn once and shared by both branches.
        snr_list = stratified_snr_sample(self.snr_min, self.snr_max, self.m_strata)

        # Eq. 3: batch-level image prototype from the cached image embeddings.
        img_prototype = F.normalize(v_clean.mean(dim=0, keepdim=True), dim=-1)

        losses, cl_terms, sep_terms, mvc_terms = [], [], [], []
        delta_x = delta_t = None
        for snr_db in snr_list:
            delta_x = self.generate_image_uap(txt_cond_feats, snr_db)
            delta_t = self.generate_text_uap(img_prototype, snr_db)

            # Eq. 6: adversarial inputs and embeddings.
            adv_images = torch.clamp(images + delta_x, 0.0, 1.0)
            with self._autocast():
                img_out = model.inference_image(normalization(adv_images))
                txt_out = model.inference_text_replace(
                    text_inputs, delta_t, positions, padding_zeros)
            v_adv = F.normalize(img_out['image_feat'].float(), dim=-1)
            u_adv = F.normalize(txt_out['text_feat'].float(), dim=-1)

            # Eq. 7: R channel views per adversarial embedding.
            v_views = channel_views(v_adv, snr_db, self.r_views)
            u_views = channel_views(u_adv, snr_db, self.r_views)

            # Eq. 8: targets mined on the first view, shared by all views.
            y_v = mine_semantic_targets(v_views[0], v_clean, u_clean)
            y_u = mine_semantic_targets(u_views[0], v_clean, u_clean)

            # Eqs. 9-11: clean image features are the self term and the
            # negatives of the image branch, clean text features those of
            # the text branch.
            l_cl = (contrastive_loss(v_views, [v_clean[y_v], u_clean[y_v]], v_clean,
                                     self.temperature, self.margin)
                    + contrastive_loss(u_views, [v_clean[y_u], u_clean[y_u]], u_clean,
                                       self.temperature, self.margin))

            # Eq. 12.
            l_sep = separation_loss(v_adv, v_clean, u_adv, u_clean)
            loss = l_cl + self.lambda_sep * l_sep

            # Eq. 13; lambda_2 = 0 when gamma_m < gamma_mvc (Eq. 16). The term
            # needs at least two views, so it is also absent when R = 1.
            if snr_db >= self.mvc_snr_threshold and self.r_views >= 2:
                l_mvc = consistency_loss(v_views, u_views)
                loss = loss + self.lambda_mvc * l_mvc
                mvc_terms.append(l_mvc.detach())

            losses.append(loss)
            cl_terms.append(l_cl.detach())
            sep_terms.append(l_sep.detach())

        # Eq. 16: average over the M strata, one backward pass, one step.
        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        self.optim_g.step()
        self.optim_zx.step()
        self.optim_zt.step()

        stats = {
            'loss': total_loss.item(),
            'cl': torch.stack(cl_terms).mean().item(),
            'sep': torch.stack(sep_terms).mean().item(),
            'mvc': torch.stack(mvc_terms).mean().item() if mvc_terms else float('nan'),
            'n_mvc': len(mvc_terms),
            'snr': snr_list,
        }
        return stats, delta_x.detach(), delta_t.detach()

    def state_dict(self):
        return {'img_generator': self.img_generator.state_dict(),
                'txt_generator': self.txt_generator.state_dict(),
                'snr_encoder': self.snr_encoder.state_dict(),
                'z_x': self.z_x.detach().cpu(),
                'z_t': self.z_t.detach().cpu()}
