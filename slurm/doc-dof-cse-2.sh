#!/bin/bash 
#SBATCH --job-name=doc-dof-cse
#SBATCH --time=72:00:00
#SBATCH --output=out/doc-dof-cse.out
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH --array=0-29

games=('ifs:cmp6'
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
        'ifs:malcev5'
        'ifs0:cmp6'
        'ifs0:cmp8'
        'ifs0:maj6'
        'ifs0:mux6'
        'ifs0:par5'
        'ifs0:disc1'
        'ifs0:disc2'
        'ifs0:disc3'
        'ifs0:disc4'
        'ifs0:disc5'
        'ifs0:malcev1'
        'ifs0:malcev2'
        'ifs0:malcev3'
        'ifs0:malcev4'
        'ifs0:malcev5')

game=${games[$SLURM_ARRAY_TASK_ID]}

echo "Starting job $SLURM_ARRAY_TASK_ID with setup $game"

module rm apps/python/3.8.5
module load apps/anaconda/5.3.1

source activate cde-search-env

srun python ~/cde-search/cli.py objs -sid "$game" -out "$WORK/cde-search/gp-objs.jsonlist"

echo "Done job $SLURM_ARRAY_TASK_ID with game $game"