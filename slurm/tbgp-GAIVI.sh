#!/bin/bash 
#SBATCH --job-name=tbgp
#SBATCH --time=72:00:00
#SBATCH --output=out/tbgp.out
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH --array=0-29

WORK=/data/dvitel/cde-search

games=('gp:cmp6'
        'gp:cmp8'
        'gp:maj6'
        'gp:mux6'
        'gp:par5'
        'gp:disc1'
        'gp:disc2'
        'gp:disc3'
        'gp:disc4'
        'gp:disc5'
        'gp:malcev1'
        'gp:malcev2'
        'gp:malcev3'
        'gp:malcev4'
        'gp:malcev5'
        'ifs:cmp6'
        'ifs:cmp8'
        'ifs:maj6'
        'ifs:mux6'
        'ifs:par5'
        'ifs:disc1'
        'ifs:disc2'
        'ifs:disc3'
        'ifs:disc4'
        'ifs:disc5'
        'ifs:malcev1'
        'ifs:malcev2'
        'ifs:malcev3'
        'ifs:malcev4'
        'ifs:malcev5')

game=${games[$SLURM_ARRAY_TASK_ID]}

echo "Starting job $SLURM_ARRAY_TASK_ID with setup $game"

conda activate cde-search-env

srun python ~/cde-search/cli.py objs -sid "$game" -n 30 -out "$WORK/cde-search/tbgp-2.jsonlist"

echo "Done job $SLURM_ARRAY_TASK_ID with game $game"