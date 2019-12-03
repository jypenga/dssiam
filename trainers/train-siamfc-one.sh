#!/bin/bash
#
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -C TitanX
#SBATCH --gres=gpu:1
#SBATCH --job-name="siamfc"
module load cuda90

export PATH="/var/scratch/jypenga/anaconda3/bin:$PATH"
python train.py --model="siamfc" --save="/var/scratch/jypenga/temp" --root="/var/scratch/jypenga/data" --batch_size=64 --num_workers=16 --seq_n=580 --epoch_n=10
