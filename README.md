# SemCom-DE-UAP

Official PyTorch implementation of **"Data-Efficient Universal Adversarial Perturbations
for Cross-Modal Retrieval in Semantic Communications"**, published at the
*IEEE Military Communications Conference (MILCOM), 2026*.

Ziyu Wang, Rui Ning, Shuai Hao, Mohammad Ghasemigol, Daniel Takabi, Lusi Li

This work attacks cross-modal (image-to-text / text-to-image) retrieval in
vision-language pre-training (VLP)-based semantic communication (SemCom) systems with
universal adversarial perturbations (UAPs) that are learned from a **limited set of
image-text pairs** (N = 500, less than 2% of the Flickr30K training data) and remain
effective under AWGN channel distortion. The framework combines:

- **semantic- and channel-guided UAP generation**: two conditional generators produce
  the image and text UAPs from cross-modal semantic features and an SNR embedding given
  by a channel state encoder;
- **channel-aware adversarial learning**: multiple AWGN realizations ("channel views")
  of each adversarial embedding are used as semantic-space training views, with
  semantic target mining, a multi-view adversarial contrastive loss, a semantic
  separation term and a multi-view consistency regularizer;
- **curriculum-guided multi-SNR training**: the training SNR range is expanded
  progressively and sampled by stratification.

The code builds on the [C-PGC](https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks)
codebase (*"One Perturbation is Enough"*, Fang et al., ICCV 2025) for the VLP model
wrappers and data loading.

## Repository layout

```
SemCom-DE-UAP/
├── generator.py                  # Conditional generators G_x, G_t and channel state
│                                   encoder f_snr (Sec. III-B)
├── channel_aware_attacker.py     # UAP generation (Eqs. 2-6), channel views (Eq. 7),
│                                   target mining (Eq. 8), losses (Eqs. 9-13) and the
│                                   joint objective / optimizers (Eq. 16)
├── train_data_efficient_uap.py   # Training: limited-data subset, feature bank,
│                                   curriculum (Eq. 14) and stratified SNR sampling (Eq. 15)
├── eval_data_efficient_uap.py    # Evaluation: Recall@1 and CR-ASR over SNRs
├── attacker.py, utils.py, dataset.py   # From C-PGC: caption word substitution, model
│                                   loading, dataset
├── models/                       # VLP backbones: ALBEF / TCL and BLIP
├── configs/                      # Dataset / model config files
├── data_annotation/              # Flickr30K train/test annotation JSONs
├── word_idx_ALBEF.npy            # Vocabulary for the adversarial-word lookup
├── scripts/
│   ├── generate_word_embeddings.py     # Builds the word-embedding caches
│   ├── train_{ALBEF,TCL,BLIP}.sh
│   └── eval_{ALBEF,TCL,BLIP}.sh
├── logs/                         # SLURM stdout/stderr
├── LICENSE                       # MIT (SemCom-DE-UAP code)
├── THIRD_PARTY_NOTICES.md        # Licenses of the vendored third-party code
└── licenses/
```

## Notation: paper symbols to command-line flags

All defaults of `train_data_efficient_uap.py` follow Sec. IV-A of the paper.

| Paper | Flag | Default | Where |
|---|---|---|---|
| training pairs `N` | `--few_shot_n` | `500` | Sec. IV-A |
| epochs `E`, batch size `B` | `--epochs`, `--batch_size` | `50`, `32` | Sec. IV-A |
| image budget `eps_x` | `--eps` (in 1/255) | `12` | Sec. III-A |
| text budget `eps_t` | `--word_num` | `1` token | Sec. III-A |
| generator learning rate `eta_g` | `--lr` | `1e-4` | Eq. (16) |
| latent learning-rate ratio `alpha` (`eta_z = alpha * eta_g`) | `--z_lr_scale` | `0.1` | Eq. (16) |
| SNR range `[gamma_min, gamma_max]` | `--snr_min`, `--snr_max` | `0`, `20` dB | Eqs. (14)-(15) |
| SNR strata `M` | `--m_strata` | `3` | Eq. (15) |
| channel views `R` | `--r_views` | `3` | Eq. (7) |
| separation weight `lambda_1` | `--lambda_sep` | `0.2` | Eq. (16) |
| consistency weight `lambda_2` | `--lambda_mvc` | `0.1` | Eq. (16) |
| consistency threshold `gamma_mvc` | `--mvc_snr_threshold` | `5` dB | Sec. III-D |
| temperature `tau`, margin `mu` | `--temperature`, `--margin` | `0.07`, `0.02` | Eq. (9) |

## Setup

```bash
conda env create -f environment.yml
conda activate semcom-de-uap
```

`environment.yml` installs `requirements.txt`. transformers >= 5 is not supported by the
vendored ALBEF/BLIP text encoders. Training and evaluation run on a CUDA GPU (the paper
used a single NVIDIA H100); `bert-base-uncased` (tokenizer/config) is downloaded from the
HuggingFace Hub on first use.

### Datasets and checkpoints

1. Download the [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/) images and
   set `image_root` in `configs/Retrieval_flickr_train.yaml` and
   `configs/Retrieval_flickr_test.yaml` to the directory that **contains**
   `flickr30k-images/` (the annotation entries are `flickr30k-images/<id>.jpg`; the
   default `''` means `./flickr30k-images/`). The annotation files are included under
   `data_annotation/`.
2. Download the Flickr30K-finetuned retrieval checkpoints of
   [ALBEF](https://github.com/salesforce/ALBEF), [TCL](https://github.com/uta-smile/TCL)
   and [BLIP](https://github.com/salesforce/BLIP) (ViT-B), and save them as:
   ```
   checkpoint/
   ├── ALBEF/flickr30k.pth
   ├── TCL/flickr30k.pth
   └── BLIP/flickr30k.pth
   ```
3. Build the word-embedding caches (requires `checkpoint/ALBEF/flickr30k.pth`):
   ```bash
   python scripts/generate_word_embeddings.py --all
   ```
   This writes `word_embedding_ALBEF.pth` (the fine-tuned ALBEF text-encoder
   word-embedding table) and the aliases `word_embedding_{TCL,BLIP}.pth` and
   `word_idx_{TCL,BLIP}.npy`. Training uses the embedding norms to clamp the text UAP;
   evaluation uses the table to map the text UAP to its nearest vocabulary word.

## Training

Each script trains the image and text UAP generators against one surrogate VLP model.
Run a script directly, or submit it with SLURM from the repository root (the
`#SBATCH -p` partition line is commented out; set it for your cluster):

```bash
bash scripts/train_ALBEF.sh      # or train_TCL.sh / train_BLIP.sh; or: sbatch scripts/train_ALBEF.sh
```

**Training data.** `N` image-text pairs are sampled uniformly at random (seed 42) from
the training split, i.e. the entries of `data_annotation/flickr30k_train.json` whose
image does not appear in the test split `data_annotation/flickr30k_test.json`. The
subset is written to `data_annotation/fewshot_<N>_seed<seed>.json`, and all `N` pairs
are used in every epoch.

**Each training iteration** on a mini-batch of `B` pairs:

1. draws `M` SNRs `gamma_1..gamma_M` by stratified sampling (Eq. 15) from the
   curriculum range `[gamma_min^(e), gamma_max]` (Eq. 14); the same `gamma_m` is used
   for both branches;
2. for each `gamma_m`:
   - builds the conditions `c_x,i = [e_txt_i ; f_snr(gamma_m)]` (mean caption embedding,
     Eq. 2) and `c_t = [e_img ; f_snr(gamma_m)]` (batch image prototype, Eq. 3);
   - generates the image UAP as the clipped batch mean of the per-sample candidates
     `G_x(z_x; c_x,i)` (Eq. 4) and the norm-clamped text UAP `G_t(z_t; c_t)` (Eq. 5);
   - applies them to the images and to the most important word of the first caption
     (Eq. 6), encodes the adversarial inputs, and draws `R` AWGN channel views of each
     adversarial embedding with `sigma^2(gamma) = 1 / (d * 10^(gamma/10))` (Eq. 7);
   - mines the semantic targets on the first view (Eq. 8) and computes the multi-view
     contrastive loss (Eqs. 9-11), the separation term on the noiseless adversarial
     embeddings (Eq. 12) and, when `gamma_m >= gamma_mvc`, the multi-view consistency
     term (Eq. 13);
3. averages the objective over the `M` strata (Eq. 16) and takes one optimizer step:
   one Adam for `G_x`, `G_t` and `f_snr` at `eta_g`, and dedicated Adam optimizers for
   `z_x` and `z_t` at `eta_z = alpha * eta_g`.

**Outputs** are written to `<save_dir>/<MODEL>/flickr30k/`. At the end of every epoch the
final generated UAPs (those of the epoch's last optimization step, at its last SNR
stratum) are saved as `uap_noise-<epoch>.pth` (image UAP, shape `(1, 3, H, W)`) and
`uap_embedding-<epoch>.pth` (text UAP, shape `(eps_t, 768)`), together with the
generator checkpoint `generators-<epoch>.pth` (`G_x`, `G_t`, `f_snr`, `z_x`, `z_t`); only
the latest epoch is kept, so the evaluation reads epoch 49 by default. The feature-bank
and word-importance caches are kept in the same directory.

## Evaluation

```bash
bash scripts/eval_ALBEF.sh       # or eval_TCL.sh / eval_BLIP.sh
```

The UAPs of one source (surrogate) model are evaluated on the target models ALBEF, TCL
and BLIP on the 1,000-image Flickr30K test split, at SNR in {inf, 20, 15, 10, 5, 0} dB:

- **Queries.** The image UAP is added to every test image (I2T queries). The text UAP is
  mapped to its nearest vocabulary word, which is substituted into every test caption
  with the source model's word-importance ranking (T2I queries); the adversarial
  captions are built once and used for all target models. The clean and adversarial
  query embeddings are transmitted over the AWGN channel of Sec. II
  (`norm(e + n)`, `n ~ N(0, sigma^2(gamma) I)`; no noise at `gamma = inf`).
- **Galleries.** I2T retrieves from the clean text gallery and T2I from the clean image
  gallery; the gallery embeddings are neither perturbed nor transmitted.
- **Metric.** Retrieval succeeds at Recall@1 (I2T: any ground-truth caption ranked
  first; T2I: the ground-truth image ranked first). The clean-retrieval attack success
  rate is

  `CR-ASR(gamma) = |S(gamma) ∩ F| / |S(gamma)|`,

  where `S(gamma)` is the set of clean queries retrieved successfully when transmitted at
  SNR `gamma`, and `F` is the set of adversarial queries whose retrieval fails after UAP
  injection (evaluated without channel noise).

The script prints the clean Recall@1, the adversarial Recall@1 and the CR-ASR for every
target, direction and SNR, and writes them to
`outputs/<output>/<SOURCE>_flickr30k_cr_asr.xlsx`.

Results vary between training runs (GPU nondeterminism and the stochastic channel
simulation) and may differ from the numbers reported in the paper.

## Citation

If you find this code useful, please cite:

```bibtex
@inproceedings{wang2026dataefficient,
  title     = {Data-Efficient Universal Adversarial Perturbations for Cross-Modal Retrieval in Semantic Communications},
  author    = {Wang, Ziyu and Ning, Rui and Hao, Shuai and Ghasemigol, Mohammad and Takabi, Daniel and Li, Lusi},
  booktitle = {2026 IEEE Military Communications Conference (MILCOM)},
  year      = {2026}
}
```

## Acknowledgements

The base pipeline, model wrappers and data loading are adapted from
[C-PGC](https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks):

```bibtex
@inproceedings{fang2025one,
  title     = {One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models},
  author    = {Fang, Hao and Kong, Jiawei and Yu, Wenbo and Chen, Bin and Li, Jiawei and Wu, Hao and Xia, Shu-Tao and Xu, Ke},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {4090--4100},
  year      = {2025},
  doi       = {10.1109/ICCV51701.2025.00390}
}
```

We also thank the authors of [ALBEF](https://github.com/salesforce/ALBEF),
[TCL](https://github.com/uta-smile/TCL) and [BLIP](https://github.com/salesforce/BLIP)
for releasing their code and models.

## License

The SemCom-DE-UAP code is released under the [MIT License](LICENSE). The vendored
third-party code keeps its original license; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
