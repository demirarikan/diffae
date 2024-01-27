#!/bin/sh
#SBATCH --job-name=diffae
#SBATCH --output=checkpoint/diffae-%A.out # Standard output of the script (Can be absolute or
relative path). %A adds the job id to the file name so you can launch the same
script multiple times and get different logging files
#SBATCH --error=checkpoints/diffae-%A.err # Standard error of the script
#SBATCH --time=0-24:00:00 # Limit on the total run time (format: dayshours:
minutes:seconds)
#SBATCH --gres=gpu:1 # Number of GPUs if needed
#SBATCH --cpus-per-task=6 # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=32G # Memory in GB (Don't use more than 48/24 per GPU unless you
absolutely need it and know what you are doing)
# run the program

ml miniconda3 # load default miniconda and python module

conda deactivate
source activate env-diffae

python main.py --dataset cifar10 --dataroot data --cuda
