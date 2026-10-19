"""
Data-efficient SemCom UAP training (single surrogate VLP model).

Learns an image UAP delta_x and a text UAP delta_t from N image-text pairs
against one surrogate (ALBEF, TCL or BLIP), following Sections II-III of the
paper with the setup of Section IV-A:

  - Training data: N pairs (--few_shot_n, default 500) sampled uniformly at
    random (--seed) from the Flickr30K training split, i.e. the entries of
    the training annotation whose image is not in the test annotation. The
    subset is written to data_annotation/fewshot_<N>_seed<seed>.json.
  - Clean features (Eq. 1): v_i = norm(f_img(x_i)) and u_i = norm(f_txt(T_i,1))
    from the first caption, plus the mean caption embedding e_txt_i of Eq. 2,
    cached once in a feature bank.
  - Text substitution positions: the eps_t most important words of the
    first caption T_i,1 of every image, where delta_t is substituted (Eq. 6).
  - Curriculum (Eq. 14): the lower SNR bound gamma_min^(e) of epoch e widens
    from gamma_max - Delta/3 to gamma_max - 2 Delta/3 and then to gamma_min.
  - Every mini-batch runs one joint step of channel_aware_attacker.py
    (Eqs. 2-13, 15, 16).

Outputs, in <save_dir>/<MODEL>/flickr30k/ (only the latest epoch is kept):
  uap_noise-<epoch>.pth      delta_x, float32 tensor (1, 3, H, W) in [-eps_x, eps_x]
  uap_embedding-<epoch>.pth  delta_t, float32 tensor (eps_t, 768), norm-clamped
  generators-<epoch>.pth     G_x, G_t, f_snr, z_x and z_t
uap_noise / uap_embedding are the final generated image and text UAPs (end
of Sec. III-E): the delta_x and delta_t produced in the last optimization
step of the epoch, at the last sampled SNR stratum gamma_M of that step.

Usage (Sec. IV-A defaults):
python train_data_efficient_uap.py \
    --config ./configs/Retrieval_flickr_train.yaml \
    --primary_encoder ALBEF \
    --primary_ckpt ./checkpoint/ALBEF/flickr30k.pth \
    --save_dir output_semcom_de_uap
"""

import argparse
import json
import os
import random

import numpy as np
import ruamel.yaml as yaml
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from channel_aware_attacker import (
    ChannelAwareUAPAttacker, SURROGATES, EMBED_DIM, SNR_EMB_DIM, TOKEN_EMB_DIM)
from dataset import paired_dataset
from generator import Generator, SNREncoder

# Maximum token length of a caption, shared by all text encoder calls.
MAX_TEXT_LENGTH = 30

# Image normalization of the surrogate VLP models.
normalization = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073),
    (0.26862954, 0.26130258, 0.27577711))


# ==========================================
# Arguments
# ==========================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Data-efficient channel-aware UAP training (Sec. III, setup of Sec. IV-A).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default='./configs/Retrieval_flickr_train.yaml',
                        help='dataset config; annotation_file is the training annotation')
    parser.add_argument('--test_annotation', default='./data_annotation/flickr30k_test.json',
                        help='test annotation; its images are excluded from the training pool')
    parser.add_argument('--primary_encoder', type=str, required=True, choices=SURROGATES,
                        help='surrogate VLP model f')
    parser.add_argument('--primary_ckpt', type=str, required=True,
                        help='surrogate checkpoint, e.g. ./checkpoint/ALBEF/flickr30k.pth')
    parser.add_argument('--source_text_encoder', type=str, default='bert-base-uncased',
                        help='HuggingFace name of the BERT tokenizer')
    parser.add_argument('--save_dir', type=str, default='output_semcom_de_uap',
                        help='outputs go to <save_dir>/<MODEL>/flickr30k/')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (also selects the N training pairs)')
    parser.add_argument('--few_shot_n', type=int, default=500,
                        help='number N of training image-text pairs')
    parser.add_argument('--epochs', type=int, default=50, help='number E of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size B')
    parser.add_argument('--eps', type=int, default=12,
                        help='image budget eps_x, in units of 1/255')
    parser.add_argument('--word_num', type=int, default=1,
                        help='text budget eps_t, in tokens')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate eta_g of G_x, G_t and f_snr')
    parser.add_argument('--z_lr_scale', type=float, default=0.1,
                        help='alpha, so that z_x and z_t use eta_z = alpha * eta_g')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='temperature tau of Eq. 9')
    parser.add_argument('--margin', type=float, default=0.02,
                        help='margin mu of Eq. 9')
    parser.add_argument('--snr_min', type=float, default=0.0,
                        help='gamma_min (dB), lower end of the operating SNR range')
    parser.add_argument('--snr_max', type=float, default=20.0,
                        help='gamma_max (dB), upper end of the operating SNR range')
    parser.add_argument('--m_strata', type=int, default=3,
                        help='number M of SNR strata per training step (Eq. 15)')
    parser.add_argument('--r_views', type=int, default=3,
                        help='number R of channel views per SNR stratum (Eq. 7)')
    parser.add_argument('--lambda_sep', type=float, default=0.2,
                        help='lambda_1, weight of L_SEP (Eq. 16)')
    parser.add_argument('--lambda_mvc', type=float, default=0.1,
                        help='lambda_2, weight of L_MVC (Eq. 16)')
    parser.add_argument('--mvc_snr_threshold', type=float, default=5.0,
                        help='gamma_mvc (dB); lambda_2 = 0 for gamma_m < gamma_mvc')
    parser.add_argument('--wi_batch_size', type=int, default=256,
                        help='batch size of the word-importance computation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker processes')
    args = parser.parse_args(argv)

    if args.few_shot_n < 2:
        parser.error('--few_shot_n must be at least 2')
    if args.batch_size < 2:
        parser.error('--batch_size must be at least 2 (Eq. 8 selects a target j != i)')
    if args.few_shot_n % args.batch_size == 1:
        parser.error(f'--few_shot_n {args.few_shot_n} leaves a last mini-batch of one pair, '
                     f'for which Eq. 8 has no target j != i; choose another --batch_size')
    if args.m_strata < 1 or args.r_views < 1 or args.word_num < 1:
        parser.error('--m_strata, --r_views and --word_num must be positive')
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_transform(image_res):
    """Resize to image_res x image_res and convert to [0, 1]; no augmentation."""
    return transforms.Compose([
        transforms.Resize((image_res, image_res), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ])


# ==========================================
# Training subset (Sec. IV-A)
# ==========================================

def make_fewshot_annotation(train_file, test_file, n, seed):
    """Sample N training pairs uniformly at random and write them to a file.

    The pool is the training annotation without the entries whose image
    appears in the test annotation (the shipped flickr30k_train.json
    contains 955 of the 1,000 test images). Returns the path of
    <dir of train_file>/fewshot_<N>_seed<seed>.json.
    """
    train_data = json.load(open(train_file, 'r'))
    test_images = {entry['image'] for entry in json.load(open(test_file, 'r'))}
    pool = [entry for entry in train_data if entry['image'] not in test_images]
    if n > len(pool):
        raise ValueError(f'--few_shot_n {n} exceeds the {len(pool)} test-disjoint training pairs')
    subset = random.Random(seed).sample(pool, n)
    out_path = os.path.join(os.path.dirname(train_file), f'fewshot_{n}_seed{seed}.json')
    with open(out_path, 'w') as f:
        json.dump(subset, f)
    print(f'[Data] Sampled N={n} pairs from {len(pool)} test-disjoint training pairs '
          f'({len(train_data) - len(pool)} entries with test images excluded) -> {out_path}')
    return out_path


# ==========================================
# Feature bank (Eqs. 1-2)
# ==========================================

def _autocast(device):
    if device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return torch.autocast(device_type='cpu', enabled=False)


def _tokenize(tokenizer, texts, device):
    return tokenizer(texts, padding='max_length', truncation=True,
                     max_length=MAX_TEXT_LENGTH, return_tensors='pt').to(device)


class FeatureBank:
    """Clean features of the N training pairs, indexed by dataset index.

    img: v_i = norm(f_img(x_i))                     (Eq. 1)
    txt: u_i = norm(f_txt(T_i,1)), first caption    (Eq. 1)
    txt_mean: e_txt_i = norm(mean_k f_txt(T_i,k))   (Eq. 2, image-branch condition)
    """

    def __init__(self, n, dim):
        self.img = torch.zeros(n, dim)
        self.txt = torch.zeros(n, dim)
        self.txt_mean = torch.zeros(n, dim)

    @torch.no_grad()
    def build(self, model, tokenizer, data_loader, device):
        model.eval()
        for images, texts_group, img_ids, _ in tqdm(data_loader, desc='  Feature bank'):
            images = images.to(device)
            first_caps = [captions[0] for captions in texts_group]
            all_caps = [c for captions in texts_group for c in captions]
            with _autocast(device):
                img_f = model.inference_image(normalization(images))['image_feat']
                txt_f = model.inference_text(_tokenize(tokenizer, first_caps, device))['text_feat']
                all_f = model.inference_text(_tokenize(tokenizer, all_caps, device))['text_feat']
            img_f = F.normalize(img_f.float(), dim=-1).cpu()
            txt_f = F.normalize(txt_f.float(), dim=-1).cpu()
            all_f = F.normalize(all_f.float(), dim=-1).cpu()
            start = 0
            for b, idx in enumerate(img_ids):
                k = len(texts_group[b])
                self.img[idx] = img_f[b]
                self.txt[idx] = txt_f[b]
                self.txt_mean[idx] = F.normalize(all_f[start:start + k].mean(dim=0), dim=-1)
                start += k

    def to(self, device):
        self.img, self.txt, self.txt_mean = (
            self.img.to(device), self.txt.to(device), self.txt_mean.to(device))
        return self

    def state_dict(self):
        return {'img': self.img.cpu(), 'txt': self.txt.cpu(), 'txt_mean': self.txt_mean.cpu()}

    def load_state_dict(self, state):
        self.img, self.txt, self.txt_mean = state['img'], state['txt'], state['txt_mean']


# ==========================================
# Word importance (substitution positions of Eq. 6)
# ==========================================

class WordImportance:
    """Positions where delta_t is substituted into the first caption T_i,1.

    The words of T_i,1 are ranked by the KL divergence between the
    surrogate's text feature of the caption and that of the caption with
    the word replaced by [UNK]. For each of the eps_t highest-ranked words
    that start inside the encoded caption, positions[i] holds its word-piece
    offset (counted without [CLS]) and padding_zeros[i] its number of word
    pieces minus one, the format of model.inference_text_replace.
    """

    def __init__(self):
        self.positions = {}
        self.padding_zeros = {}

    @torch.no_grad()
    def build(self, model, tokenizer, data_loader, num_words, device, batch_size=256):
        model.eval()
        kl = torch.nn.KLDivLoss(reduction='none')
        # The vendored BertTokenizer encodes a caption as [CLS] + word pieces
        # (no trailing [SEP]), so word-piece offsets 0 .. MAX_TEXT_LENGTH - 2
        # (token indices 1 .. MAX_TEXT_LENGTH - 1) lie inside the encoded
        # caption; this is the same rule as the evaluation's TextAttacker.
        max_offset = MAX_TEXT_LENGTH - 2
        for _, texts_group, img_ids, _ in tqdm(data_loader, desc='  Word importance'):
            texts = [captions[0] for captions in texts_group]
            with _autocast(device):
                origin = model.inference_text(_tokenize(tokenizer, texts, device))['text_feat']
            origin = origin.flatten(1).float()

            masked, spans = [], []
            for text in texts:
                words = text.split(' ')
                spans.append((len(masked), len(masked) + len(words)))
                masked += [' '.join(words[:w] + ['[UNK]'] + words[w + 1:])
                           for w in range(len(words))]
            masked_feats = []
            for s in range(0, len(masked), batch_size):
                with _autocast(device):
                    feat = model.inference_text(
                        _tokenize(tokenizer, masked[s:s + batch_size], device))['text_feat']
                masked_feats.append(feat.flatten(1).float())
            masked_feats = torch.cat(masked_feats, dim=0)

            for b, (text, idx) in enumerate(zip(texts, img_ids)):
                s, e = spans[b]
                scores = kl(masked_feats[s:e].log_softmax(dim=-1),
                            origin[b].softmax(dim=-1).unsqueeze(0).expand(e - s, -1)).sum(dim=-1)
                ranking = sorted(range(e - s), key=lambda w: scores[w].item(), reverse=True)

                keys, offset = [], 0
                for word in text.split(' '):
                    n_pieces = len(tokenizer.tokenize(word))
                    keys.append((offset, n_pieces))
                    offset += n_pieces

                pos, pz = [], []
                for w in ranking:
                    if len(pos) == num_words:
                        break
                    start, n_pieces = keys[w]
                    if n_pieces > 0 and start <= max_offset:
                        pos.append(start)
                        pz.append(n_pieces - 1)
                self.positions[idx] = pos
                self.padding_zeros[idx] = pz

    def get_batch(self, img_ids):
        return ([self.positions[i] for i in img_ids],
                [self.padding_zeros[i] for i in img_ids])

    def state_dict(self):
        return {'positions': self.positions, 'padding_zeros': self.padding_zeros}

    def load_state_dict(self, state):
        self.positions, self.padding_zeros = state['positions'], state['padding_zeros']


def load_or_build(path, fingerprint, obj, build_fn):
    """Load a cached FeatureBank / WordImportance if it matches the training subset."""
    if os.path.exists(path):
        state = torch.load(path, map_location='cpu')
        if state.get('fingerprint') == fingerprint:
            obj.load_state_dict(state['data'])
            print(f'  Loaded cache {path}')
            return obj
        print(f'  Cache {path} does not match the training subset, rebuilding')
    build_fn()
    torch.save({'fingerprint': fingerprint, 'data': obj.state_dict()}, path)
    return obj


# ==========================================
# Curriculum (Eq. 14)
# ==========================================

def curriculum_snr_min(epoch, total_epochs, snr_min, snr_max):
    """Lower SNR bound gamma_min^(e) of epoch e (0-based) (Eq. 14).

    phi = e / E; gamma_max - Delta/3 for phi < 1/3, gamma_max - 2 Delta/3 for
    1/3 <= phi < 2/3, and gamma_min otherwise, with Delta = gamma_max - gamma_min.
    """
    phi = epoch / total_epochs
    delta = snr_max - snr_min
    if phi < 1 / 3:
        return snr_max - delta / 3
    if phi < 2 / 3:
        return snr_max - 2 * delta / 3
    return snr_min


def word_embedding_norm_range(model_name):
    """[min, max] L2-norm range of the word-embedding table (norm clamp of Eq. 5)."""
    path = f'word_embedding_{model_name}.pth'
    if not os.path.exists(path):
        path = 'word_embedding_ALBEF.pth'
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Word-embedding table 'word_embedding_{model_name}.pth' not found. "
            f"Generate it with: python scripts/generate_word_embeddings.py --all")
    norms = torch.load(path, map_location='cpu').float().norm(p=2, dim=1)
    print(f'  Word-embedding table {path}: L2 norm range [{norms.min():.4f}, {norms.max():.4f}]')
    return norms.min().item(), norms.max().item()


# ==========================================
# Training loop
# ==========================================

def train(model, tokenizer, data_loader, args, save_dir, device):
    """Train G_x, G_t, f_snr, z_x and z_t and export the UAPs after every epoch."""
    n_pairs = len(data_loader.dataset)

    print('=' * 60)
    print(f'SemCom-DE-UAP training: surrogate={args.primary_encoder}  N={n_pairs}  B={args.batch_size}'
          f'  E={args.epochs}')
    print(f'  eps_x={args.eps}/255  eps_t={args.word_num}  eta_g={args.lr}  alpha={args.z_lr_scale}')
    print(f'  M={args.m_strata}  R={args.r_views}  SNR=[{args.snr_min}, {args.snr_max}] dB'
          f'  tau={args.temperature}  mu={args.margin}')
    print(f'  lambda_1={args.lambda_sep}  lambda_2={args.lambda_mvc}  gamma_mvc={args.mvc_snr_threshold} dB')
    print('=' * 60)

    model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Clean features (Eqs. 1-2) and substitution positions, cached per subset.
    fingerprint = {'model': args.primary_encoder, 'ckpt': os.path.abspath(args.primary_ckpt),
                   'images': list(data_loader.dataset.image),
                   'texts': list(data_loader.dataset.text)}
    bank = FeatureBank(n_pairs, EMBED_DIM)
    load_or_build(os.path.join(save_dir, 'feature_bank.pth'), fingerprint, bank,
                  lambda: bank.build(model, tokenizer, data_loader, device))
    bank.to(device)
    importance = WordImportance()
    load_or_build(os.path.join(save_dir, 'word_importance.pth'),
                  dict(fingerprint, word_num=args.word_num), importance,
                  lambda: importance.build(model, tokenizer, data_loader, args.word_num,
                                           device, args.wi_batch_size))

    # G_x and G_t share the channel state encoder f_snr; both are conditioned
    # on [d-dim semantic feature ; d_snr-dim SNR embedding].
    ctx_dim = EMBED_DIM + SNR_EMB_DIM
    img_generator = Generator(100, [[1024, 512], [256, 128], [64, 32]], 3, args.batch_size,
                              first_kernel_size=4, num_heads=1, context_dim=ctx_dim)
    txt_generator = Generator(32, [[256, 128], [64, 32]], args.word_num, args.batch_size,
                              first_kernel_size=1, num_heads=1, context_dim=ctx_dim)
    snr_encoder = SNREncoder(snr_emb_dim=SNR_EMB_DIM)
    min_norm, max_norm = word_embedding_norm_range(args.primary_encoder)

    attacker = ChannelAwareUAPAttacker(
        img_generator, txt_generator, snr_encoder,
        z_x_shape=(1, 100, 3, 3), z_t_shape=(1, 32, 1, 3),
        eps_x=args.eps / 255, eps_t=args.word_num,
        min_norm=min_norm, max_norm=max_norm,
        lr=args.lr, z_lr_scale=args.z_lr_scale,
        temperature=args.temperature, margin=args.margin,
        m_strata=args.m_strata, r_views=args.r_views,
        lambda_sep=args.lambda_sep, lambda_mvc=args.lambda_mvc,
        mvc_snr_threshold=args.mvc_snr_threshold,
        snr_min=args.snr_min, snr_max=args.snr_max, device=device)

    for epoch in range(args.epochs):
        # Eq. 14: curriculum-adjusted lower SNR bound of this epoch.
        attacker.snr_min = curriculum_snr_min(epoch, args.epochs, args.snr_min, args.snr_max)
        print(f'Epoch {epoch}: SNR range [{attacker.snr_min:.2f}, {attacker.snr_max:.2f}] dB')

        delta_x = delta_t = None
        running, n_running = {'loss': 0.0, 'cl': 0.0, 'sep': 0.0}, 0
        for bi, (images, texts_group, img_ids, _) in enumerate(tqdm(data_loader)):
            images = images.to(device)
            # Eq. 6: delta_t is applied to the first caption T_i,1.
            texts = [captions[0] for captions in texts_group]
            text_inputs = _tokenize(tokenizer, texts, device)
            positions, padding_zeros = importance.get_batch(img_ids)

            stats, delta_x, delta_t = attacker.step(
                model, normalization, images, text_inputs, positions, padding_zeros,
                txt_cond_feats=bank.txt_mean[img_ids],
                clean_img_feats=bank.img[img_ids],
                clean_txt_feats=bank.txt[img_ids])

            for key in running:
                running[key] += stats[key]
            n_running += 1
            if (bi + 1) % 10 == 0 or bi + 1 == len(data_loader):
                print(f'E{epoch} B{bi + 1}/{len(data_loader)} | '
                      + ' '.join(f'{k.upper()}:{v / n_running:.4f}' for k, v in running.items())
                      + f" | MVC strata:{stats['n_mvc']}/{args.m_strata}")
                running, n_running = {key: 0.0 for key in running}, 0

        # Export the final generated UAPs of this epoch: delta_x and delta_t of
        # the last step, at its last SNR stratum gamma_M (end of Sec. III-E).
        assert delta_x.shape == (1, 3) + tuple(images.shape[-2:])
        assert delta_t.shape == (args.word_num, TOKEN_EMB_DIM)
        torch.save(delta_x.float().cpu(), os.path.join(save_dir, f'uap_noise-{epoch}.pth'))
        torch.save(delta_t.float().cpu(), os.path.join(save_dir, f'uap_embedding-{epoch}.pth'))
        torch.save(attacker.state_dict(), os.path.join(save_dir, f'generators-{epoch}.pth'))
        if epoch > 0:
            for name in ('uap_noise', 'uap_embedding', 'generators'):
                prev = os.path.join(save_dir, f'{name}-{epoch - 1}.pth')
                if os.path.exists(prev):
                    os.remove(prev)
        print(f'Epoch {epoch}: saved uap_noise-{epoch}.pth and uap_embedding-{epoch}.pth')

    print('Training complete.')


# ==========================================
# Entry point
# ==========================================

def main(argv=None):
    args = parse_args(argv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    cudnn.benchmark = True

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    save_dir = os.path.join(args.save_dir, args.primary_encoder, config['dataset'])
    os.makedirs(save_dir, exist_ok=True)

    ann_file = make_fewshot_annotation(config['annotation_file'], args.test_annotation,
                                       args.few_shot_n, args.seed)

    from utils import load_model
    print(f'Loading surrogate {args.primary_encoder} from {args.primary_ckpt}')
    model, _, tokenizer = load_model(args.primary_encoder, args.primary_ckpt,
                                     args.source_text_encoder, config, device)
    model.to(device)

    # All N pairs are used: no drop_last, so the last mini-batch may be smaller
    # than B. The order is fixed; samples are indexed by their dataset index.
    dataset = paired_dataset(ann_file, build_transform(config['image_res']), config['image_root'])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=args.num_workers, collate_fn=dataset.collate_fn)
    print(f'Training pairs: {len(dataset)} in {len(loader)} mini-batches')

    train(model, tokenizer, loader, args, save_dir, device)


if __name__ == '__main__':
    main()
