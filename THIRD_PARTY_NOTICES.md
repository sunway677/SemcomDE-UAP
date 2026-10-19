# Third-party notices

The SemCom-DE-UAP-specific code (`train_data_efficient_uap.py`,
`channel_aware_attacker.py`, `eval_data_efficient_uap.py`, `scripts/`, and the
SNR conditioning in `generator.py`) is released under the MIT license in
[`LICENSE`](LICENSE). This repository also redistributes code and data from
the projects below, which remain under their original licenses. Full license
texts are in [`licenses/`](licenses/); the original copyright headers are kept
in the individual files.

Modified third-party files: the text-embedding substitution that inserts the
text UAP into a caption is fixed in this repository, in `BertEmbeddings` of
`models/xbert.py` (ALBEF/TCL) and `models/blip_model/med.py` (BLIP) and in
`inference_text_replace` of `models/model_retrieval.py` and
`models/blip_model/blip_retrieval.py`. The changes are marked
"modified for SemCom-DE-UAP" in the source; these files remain under their
upstream licenses.

| Component | Files in this repo | License | Copyright |
|---|---|---|---|
| [ALBEF](https://github.com/salesforce/ALBEF) | `models/model_retrieval.py` (modified), `models/vit.py`, `models/xbert.py` (modified), `models/tokenization_bert.py` | BSD-3-Clause ([text](licenses/ALBEF_LICENSE.txt)) | (c) 2021, Salesforce.com, Inc. |
| [BLIP](https://github.com/salesforce/BLIP) | `models/blip_model/{blip,blip_retrieval,med,vit}.py`, `models/blip_model/med_config.json` (`med.py` and `blip_retrieval.py` modified) | BSD-3-Clause ([text](licenses/BLIP_LICENSE.txt)) | (c) 2022, Salesforce.com, Inc. |
| [HuggingFace Transformers](https://github.com/huggingface/transformers) (BERT code adapted by ALBEF and BLIP) | `models/xbert.py` (modified), `models/tokenization_bert.py`, `models/blip_model/med.py` (modified) | Apache-2.0 ([text](licenses/Apache-2.0.txt)) | (c) The Google AI Language Team Authors, The HuggingFace Inc. team, NVIDIA Corporation, The OpenAI Team Authors |
| [timm](https://github.com/huggingface/pytorch-image-models) (ViT code adapted by ALBEF and BLIP) | `models/vit.py`, `models/blip_model/vit.py` | Apache-2.0 ([text](licenses/Apache-2.0.txt)) | (c) 2019 Ross Wightman |
| [latent-diffusion](https://github.com/CompVis/latent-diffusion) (via C-PGC's `attention.py`) | cross-attention blocks in `generator.py` | MIT ([text](licenses/latent-diffusion_LICENSE.txt)) | (c) 2022 Machine Vision and Learning Group, LMU Munich |
| [C-PGC](https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks) (Fang et al., ICCV 2025) | `attacker.py` (reduced to the caption word-substitution of `TextAttacker`), `utils.py`, `dataset.py`, `configs/`, the generator backbone in `generator.py`, and the base structure of the training/evaluation pipeline; `word_idx_ALBEF.npy` is the vocabulary index file read by the C-PGC evaluation code (not included in the C-PGC repository) | no license file in the upstream repository | (c) the C-PGC authors |
| Flickr30K annotations (as distributed with C-PGC) | `data_annotation/flickr30k_{train,test}.json` | [Flickr30K terms of use](https://shannon.cs.illinois.edu/DenotationGraph/) (non-commercial research) | (c) the Flickr30K authors |

The Flickr30K images and the fine-tuned VLP checkpoints are not redistributed;
download them from the original sources listed in the README.
