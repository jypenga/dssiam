#!/bin/bash
#
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -C TitanX
#SBATCH --gres=gpu:1
#SBATCH --job-name="siamfc"
module load cuda90

export PATH="/var/scratch/jypenga/anaconda3/bin:$PATH"
python train.py --model="siamfc" --save="/var/scratch/jypenga/temp" --root="/var/scratch/jypenga/data/GOT-10k" --batch_size=64 --num_workers=16 --seq_len=2 --epoch_n=50 --subset='train' --ablation=1
