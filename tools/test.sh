#!/bin/bash
#
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -C TitanX
#SBATCH --gres=gpu:1
#SBATCH --job-name="test"
module load cuda90

export PATH="/var/scratch/jypenga/anaconda3/bin:$PATH"
python test.py --model='dssiam' --weights='' --root='/var/scratch/jypenga/data' --results='/var/scratch/jypenga/results' --report='/var/scratch/jypenga/reports' --subset='val'
