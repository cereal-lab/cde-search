#!/bin/bash 
#SBATCH --job-name=num-game-sp
#SBATCH --time=72:00:00
#SBATCH --output=num-game-sp.out
#SBATCH --mem=8G
#SBATCH --array=0-3

games=('GreaterThanGame'
    'IntransitiveRegionGame'
    'FocusingGame'
    'CompareOnOneGame')

game=${games[$SLURM_ARRAY_TASK_ID]}

module rm apps/python/3.8.5
module load apps/anaconda/5.3.1

source activate cde-search-env

python ~/cde-search/cli.py game-space -gid "$game" -out "$WORK/num-game-spaces"