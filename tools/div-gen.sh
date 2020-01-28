#!/bin/bash
#
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -C TitanX
#SBATCH --gres=gpu:1
#SBATCH --job-name="div-gen"
module load cuda90

export PATH="/var/scratch/jypenga/anaconda3/bin:$PATH"
python div_gen.py --root='/var/scratch/jypenga/data' --report='/var/scratch/jypenga/figures' --models='/var/scratch/jypenga/models' --benchmark='OTB2015' --subset='val'
