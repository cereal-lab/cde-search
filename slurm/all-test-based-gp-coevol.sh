#!/bin/bash 
#SBATCH --job-name=atbgpc
#SBATCH --time=72:00:00
#SBATCH --output=out/atbgpc.out
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH --array=0-305

# NOTE1: latest (after test-based-gp, coevol2-6)
# NOTE2: includes NSGA-3 and lexicase 
# NOTE3: only most important methods
# NOTE4: used for final data colletion for GECCO 25 article

games=('coevol_uo_40:cmp6'
        'coevol_uo_40:cmp8'
        'coevol_uo_40:maj6'
        'coevol_uo_40:mux6'
        'coevol_uo_40:par5'
        'coevol_uo_40:maj8'
        'coevol_uo_40:mux11'
        'coevol_uo_40:par7'
        'coevol_uo_40:disc1'
        'coevol_uo_40:disc2'
        'coevol_uo_40:disc3'
        'coevol_uo_40:disc4'
        'coevol_uo_40:disc5'
        'coevol_uo_40:malcev1'
        'coevol_uo_40:malcev2'
        'coevol_uo_40:malcev3'
        'coevol_uo_40:malcev4'
        'coevol_uo_40:malcev5'
        'coevol_uo2_50:cmp6'
        'coevol_uo2_50:cmp8'
        'coevol_uo2_50:maj6'
        'coevol_uo2_50:mux6'
        'coevol_uo2_50:par5'
        'coevol_uo2_50:maj8'
        'coevol_uo2_50:mux11'
        'coevol_uo2_50:par7'
        'coevol_uo2_50:disc1'
        'coevol_uo2_50:disc2'
        'coevol_uo2_50:disc3'
'coevol_uo2_50:disc4'
'coevol_uo2_50:disc5'
'coevol_uo2_50:malcev1'
'coevol_uo2_50:malcev2'
'coevol_uo2_50:malcev3'
'coevol_uo2_50:malcev4'
'coevol_uo2_50:malcev5'
'coevol_uo_d_30:cmp6'
'coevol_uo_d_30:cmp8'
'coevol_uo_d_30:maj6'
'coevol_uo_d_30:mux6'
'coevol_uo_d_30:par5'
'coevol_uo_d_30:maj8'
'coevol_uo_d_30:mux11'
'coevol_uo_d_30:par7'
'coevol_uo_d_30:disc1'
'coevol_uo_d_30:disc2'
'coevol_uo_d_30:disc3'
'coevol_uo_d_30:disc4'
'coevol_uo_d_30:disc5'
'coevol_uo_d_30:malcev1'
'coevol_uo_d_30:malcev2'
'coevol_uo_d_30:malcev3'
'coevol_uo_d_30:malcev4'
'coevol_uo_d_30:malcev5')

# NOTE: 255 games in total
game=${games[$SLURM_ARRAY_TASK_ID]}

echo "Starting job $SLURM_ARRAY_TASK_ID with setup $game"

module rm apps/python/3.8.5
module load apps/anaconda/5.3.1

source activate cde-search-env

srun python ~/cde-search/cli.py objs -sid "$game" -n 30 -out "$WORK/cde-search/atbgpc.jsonlist"

echo "Done job $SLURM_ARRAY_TASK_ID with game $game"