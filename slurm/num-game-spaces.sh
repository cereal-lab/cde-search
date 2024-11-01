#!/bin/bash 
#SBATCH --job-name=num-game-sp
#SBATCH --time=72:00:00
#SBATCH --output=num-game-sp.out
#SBATCH --mem=8G
#SBATCH --array=0-3

games=('GreaterThanGame',
    'IntransitiveRegionGame',
    'FocusingGame',
    'CompareOnOneGame')

game=${games[$SLURM_ARRAY_TASK_ID]}

python3 cli.py game-space -gid "$game" -out "$WORK/num-game-spaces"