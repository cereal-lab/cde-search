#!/bin/bash 
#SBATCH --job-name=coevol
#SBATCH --time=72:00:00
#SBATCH --output=out/coevol.out
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH --array=0-119

games=('coevol_uo_5:cmp6'
        'coevol_uo_5:cmp8'
        'coevol_uo_5:maj6'
        'coevol_uo_5:mux6'
        'coevol_uo_5:par5'
        'coevol_uo_5:disc1'
        'coevol_uo_5:disc2'
        'coevol_uo_5:disc3'
        'coevol_uo_5:disc4'
        'coevol_uo_5:disc5'
        'coevol_uo_5:malcev1'
        'coevol_uo_5:malcev2'
        'coevol_uo_5:malcev3'
        'coevol_uo_5:malcev4'
        'coevol_uo_5:malcev5'
        'coevol_uo_10:cmp6'
        'coevol_uo_10:cmp8'
        'coevol_uo_10:maj6'
        'coevol_uo_10:mux6'
        'coevol_uo_10:par5'
        'coevol_uo_10:disc1'
        'coevol_uo_10:disc2'
        'coevol_uo_10:disc3'
        'coevol_uo_10:disc4'
        'coevol_uo_10:disc5'
        'coevol_uo_10:malcev1'
        'coevol_uo_10:malcev2'
        'coevol_uo_10:malcev3'
        'coevol_uo_10:malcev4'
        'coevol_uo_10:malcev5'
        'coevol_uo_15:cmp6'
        'coevol_uo_15:cmp8'
        'coevol_uo_15:maj6'
        'coevol_uo_15:mux6'
        'coevol_uo_15:par5'
        'coevol_uo_15:disc1'
        'coevol_uo_15:disc2'
        'coevol_uo_15:disc3'
        'coevol_uo_15:disc4'
        'coevol_uo_15:disc5'
        'coevol_uo_15:malcev1'
        'coevol_uo_15:malcev2'
        'coevol_uo_15:malcev3'
        'coevol_uo_15:malcev4'
        'coevol_uo_15:malcev5'
        'coevol_uo:cmp6'
        'coevol_uo:cmp8'
        'coevol_uo:maj6'
        'coevol_uo:mux6'
        'coevol_uo:par5'
        'coevol_uo:disc1'
        'coevol_uo:disc2'
        'coevol_uo:disc3'
        'coevol_uo:disc4'
        'coevol_uo:disc5'
        'coevol_uo:malcev1'
        'coevol_uo:malcev2'
        'coevol_uo:malcev3'
        'coevol_uo:malcev4'
        'coevol_uo:malcev5'
        'coevol_uo_25:cmp6'
        'coevol_uo_25:cmp8'
        'coevol_uo_25:maj6'
        'coevol_uo_25:mux6'
        'coevol_uo_25:par5'
        'coevol_uo_25:disc1'
        'coevol_uo_25:disc2'
        'coevol_uo_25:disc3'
        'coevol_uo_25:disc4'
        'coevol_uo_25:disc5'
        'coevol_uo_25:malcev1'
        'coevol_uo_25:malcev2'
        'coevol_uo_25:malcev3'
        'coevol_uo_25:malcev4'
        'coevol_uo_25:malcev5'
        'coevol_uo_30:cmp6'
        'coevol_uo_30:cmp8'
        'coevol_uo_30:maj6'
        'coevol_uo_30:mux6'
        'coevol_uo_30:par5'
        'coevol_uo_30:disc1'
        'coevol_uo_30:disc2'
        'coevol_uo_30:disc3'
        'coevol_uo_30:disc4'
        'coevol_uo_30:disc5'
        'coevol_uo_30:malcev1'
        'coevol_uo_30:malcev2'
        'coevol_uo_30:malcev3'
        'coevol_uo_30:malcev4'
        'coevol_uo_30:malcev5'
        'coevol_uo_35:cmp6'
        'coevol_uo_35:cmp8'
        'coevol_uo_35:maj6'
        'coevol_uo_35:mux6'
        'coevol_uo_35:par5'
        'coevol_uo_35:disc1'
        'coevol_uo_35:disc2'
        'coevol_uo_35:disc3'
        'coevol_uo_35:disc4'
        'coevol_uo_35:disc5'
        'coevol_uo_35:malcev1'
        'coevol_uo_35:malcev2'
        'coevol_uo_35:malcev3'
        'coevol_uo_35:malcev4'
        'coevol_uo_35:malcev5'
        'coevol_uo_40:cmp6'
        'coevol_uo_40:cmp8'
        'coevol_uo_40:maj6'
        'coevol_uo_40:mux6'
        'coevol_uo_40:par5'
        'coevol_uo_40:disc1'
        'coevol_uo_40:disc2'
        'coevol_uo_40:disc3'
        'coevol_uo_40:disc4'
        'coevol_uo_40:disc5'
        'coevol_uo_40:malcev1'
        'coevol_uo_40:malcev2'
        'coevol_uo_40:malcev3'
        'coevol_uo_40:malcev4'
        'coevol_uo_40:malcev5')

game=${games[$SLURM_ARRAY_TASK_ID]}

echo "Starting job $SLURM_ARRAY_TASK_ID with setup $game"

module rm apps/python/3.8.5
module load apps/anaconda/5.3.1

source activate cde-search-env

srun python ~/cde-search/cli.py objs -sid "$game" -n 30 -out "$WORK/cde-search/coevol.jsonlist"

echo "Done job $SLURM_ARRAY_TASK_ID with game $game"