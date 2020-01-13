#!/bin/bash
#
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -C TitanX
#SBATCH --gres=gpu:1
#SBATCH --job-name="test-fr"
module load cuda90

export PATH="/var/scratch/jypenga/anaconda3/bin:$PATH"
python test-fr.py --model='dssiam' --weights='/var/scratch/jypenga/models/dssiam_n2_e50.pth' --root='/var/scratch/jypenga/data' --report='/var/scratch/jypenga/reports' --subset='val'
