#!/bin/bash
#SBATCH -J eval_de_uap_BLIP
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

# --epoch 49: the UAPs saved after the last of the 50 training epochs (0-indexed).
python eval_data_efficient_uap.py \
  --config ./configs/Retrieval_flickr_test.yaml \
  --source_model BLIP \
  --checkpoint ./checkpoint \
  --load_dir ./output_de_uap \
  --epoch 49 \
  --snr_list inf 20 15 10 5 0 \
  --output result_de_uap_BLIP
