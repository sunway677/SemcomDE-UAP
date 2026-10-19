"""
Precompute the word-embedding / vocabulary files read by
train_data_efficient_uap.py and eval_data_efficient_uap.py.

- word_embedding_ALBEF.pth: the word-embedding table of the fine-tuned ALBEF
  text encoder (text_encoder.embeddings.word_embeddings.weight of
  ./checkpoint/ALBEF/flickr30k.pth). Training uses the min/max L2 norm of its
  rows as the magnitude range of the norm clamp on the text UAP delta_t
  (Sec. III-B.4b); evaluation uses it for the nearest-word lookup that maps
  delta_t to a valid token substitution.
- word_embedding_{TCL,BLIP}.pth and word_idx_{TCL,BLIP}.npy: aliases of
  word_embedding_ALBEF.pth and of the shipped word_idx_ALBEF.npy (the
  candidate token ids of the nearest-word lookup). ALBEF, TCL and BLIP all
  use the bert-base-uncased vocabulary, and the ALBEF table is used for all
  three surrogates. The aliases are relative symlinks, or copies where
  symlinks are not available; train and eval also fall back to the ALBEF
  files when a per-model file is missing.

All files are written to the repository root, where train and eval look for
them.

Usage (run once, after downloading the ALBEF checkpoint):
    python scripts/generate_word_embeddings.py --all
"""
import argparse
import os
import shutil
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Same import root as train/eval (sys.path[0] would otherwise be scripts/).
sys.path.insert(0, REPO_ROOT)

WORD_EMBEDDING_KEY = "text_encoder.embeddings.word_embeddings.weight"
BERT_ALIASES = ["TCL", "BLIP"]


def save_bert_word_embedding(albef_ckpt: str, out_path: str):
    if not os.path.isfile(albef_ckpt):
        raise SystemExit(f"ALBEF checkpoint not found: {albef_ckpt} "
                         f"(download the fine-tuned Flickr30K ALBEF checkpoint or pass --albef_ckpt)")
    checkpoint = torch.load(albef_ckpt, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    if WORD_EMBEDDING_KEY not in state_dict:
        raise SystemExit(f"{albef_ckpt} has no '{WORD_EMBEDDING_KEY}'; expected an ALBEF checkpoint")
    word_embedding = state_dict[WORD_EMBEDDING_KEY].detach().clone()
    torch.save(word_embedding, out_path)
    norms = word_embedding.float().norm(p=2, dim=1)
    print(f"[bert] saved {tuple(word_embedding.shape)} -> {out_path} "
          f"(L2 norm range [{norms.min():.4f}, {norms.max():.4f}])")


def link_or_copy(src_name: str, dst_name: str):
    """Make REPO_ROOT/dst_name an alias of REPO_ROOT/src_name."""
    src = os.path.join(REPO_ROOT, src_name)
    dst = os.path.join(REPO_ROOT, dst_name)
    if not os.path.isfile(src):
        raise SystemExit(f"missing {src}")
    if os.path.islink(dst) and os.readlink(dst) == src_name:
        return
    if os.path.lexists(dst):
        # A stale alias or copy from an earlier run.
        os.remove(dst)
    try:
        os.symlink(src_name, dst)
        kind = "symlink"
    except OSError:
        shutil.copyfile(src, dst)
        kind = "copy"
    print(f"[alias] {dst_name} -> {src_name} ({kind})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write word_embedding_{ALBEF,TCL,BLIP}.pth and word_idx_{TCL,BLIP}.npy to the repo root.")
    parser.add_argument("--albef_ckpt", default=os.path.join(REPO_ROOT, "checkpoint", "ALBEF", "flickr30k.pth"),
                        help="Fine-tuned ALBEF Flickr30K checkpoint whose text-encoder word embeddings "
                             "are shared by ALBEF/TCL/BLIP")
    parser.add_argument("--all", action="store_true",
                        help="Generate word_embedding_{ALBEF,TCL,BLIP}.pth and the word_idx aliases")
    args = parser.parse_args()

    if not args.all:
        raise SystemExit("pass --all (all surrogates share the ALBEF word-embedding table)")

    save_bert_word_embedding(args.albef_ckpt, os.path.join(REPO_ROOT, "word_embedding_ALBEF.pth"))
    for name in BERT_ALIASES:
        link_or_copy("word_embedding_ALBEF.pth", f"word_embedding_{name}.pth")
        link_or_copy("word_idx_ALBEF.npy", f"word_idx_{name}.npy")
