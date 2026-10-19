#!/bin/bash
#SBATCH -J de_uap_ALBEF
# #SBATCH -p <partition>   # uncomment and set your cluster GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

source ~/.bashrc
conda activate semcom-de-uap
# repo root: script location under bash, submission directory under sbatch
if [ -f "$(dirname "$0")/../train_data_efficient_uap.py" ]; then cd "$(dirname "$0")/.."; else cd "$SLURM_SUBMIT_DIR"; fi
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# All other hyperparameters (Sec. IV-A) are the argparse defaults of
# train_data_efficient_uap.py.
python train_data_efficient_uap.py \
  --config ./configs/Retrieval_flickr_train.yaml \
  --primary_encoder ALBEF \
  --primary_ckpt ./checkpoint/ALBEF/flickr30k.pth \
  --save_dir output_de_uap \
  --few_shot_n 500 \
  --epochs 50 \
  --batch_size 32
