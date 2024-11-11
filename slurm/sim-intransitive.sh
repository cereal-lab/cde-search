#!/bin/bash 
#SBATCH --job-name=sim-space
#SBATCH --time=72:00:00
#SBATCH --output=out/sim-space.out
#SBATCH --mem=8G
#SBATCH --array=0-37

games=('rand:IntransitiveRegionGame'
    'hc-pmo-i:IntransitiveRegionGame'
    'hc-pmo-p:IntransitiveRegionGame'
    'hc-r-i:IntransitiveRegionGame'
    'hc-r-p:IntransitiveRegionGame'
    'de-l:IntransitiveRegionGame'
    'de-d-0:IntransitiveRegionGame'
    'de-d-1:IntransitiveRegionGame'
    'de-d-m:IntransitiveRegionGame'
    'de-d-g:IntransitiveRegionGame'
    'de-d-s:IntransitiveRegionGame'
    'de-d-d-0:IntransitiveRegionGame'
    'de-d-d-1:IntransitiveRegionGame'
    'de-d-d-2:IntransitiveRegionGame'
    'de-d-d-5:IntransitiveRegionGame'
    'de-d-d-100:IntransitiveRegionGame'
    'des-mea:IntransitiveRegionGame'
    'des-mea-0:IntransitiveRegionGame'
    'des-mea-1:IntransitiveRegionGame'
    'des-mea-2:IntransitiveRegionGame'
    'des-mea-5:IntransitiveRegionGame'
    'des-mea-100:IntransitiveRegionGame'
    'des-med:IntransitiveRegionGame'
    'des-med-0:IntransitiveRegionGame'
    'des-med-1:IntransitiveRegionGame'
    'des-med-2:IntransitiveRegionGame'
    'des-med-5:IntransitiveRegionGame'
    'des-med-100:IntransitiveRegionGame'
    'pl-l-0:IntransitiveRegionGame'
    'pl-l-1:IntransitiveRegionGame'
    'pl-l-2:IntransitiveRegionGame'
    'pl-l-5:IntransitiveRegionGame'
    'pl-l-100:IntransitiveRegionGame'
    'pl-d-0:IntransitiveRegionGame'
    'pl-d-1:IntransitiveRegionGame'
    'pl-d-2:IntransitiveRegionGame'
    'pl-d-5:IntransitiveRegionGame'
    'pl-d-100:IntransitiveRegionGame')

game=${games[$SLURM_ARRAY_TASK_ID]}

echo "Starting job $SLURM_ARRAY_TASK_ID with game $game"

module rm apps/python/3.8.5
module load apps/anaconda/5.3.1

source activate cde-search-env

python ~/cde-search/cli.py game -sim "$game" --times 30 --metrics "$WORK/cde-search/spaces.jsonlist"

echo "Done job $SLURM_ARRAY_TASK_ID with game $game"