#!/bin/bash 
#SBATCH --job-name=coevol2
#SBATCH --time=72:00:00
#SBATCH --output=out/coevol2.out
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH --array=0-149

games=('coevol_uo2_10:cmp6'
        'coevol_uo2_10:cmp8'
        'coevol_uo2_10:maj6'
        'coevol_uo2_10:mux6'
        'coevol_uo2_10:par5'
        'coevol_uo2_10:disc1'
        'coevol_uo2_10:disc2'
        'coevol_uo2_10:disc3'
        'coevol_uo2_10:disc4'
        'coevol_uo2_10:disc5'
        'coevol_uo2_10:malcev1'
        'coevol_uo2_10:malcev2'
        'coevol_uo2_10:malcev3'
        'coevol_uo2_10:malcev4'
        'coevol_uo2_10:malcev5'
        'coevol_uo2_20:cmp6'
        'coevol_uo2_20:cmp8'
        'coevol_uo2_20:maj6'
        'coevol_uo2_20:mux6'
        'coevol_uo2_20:par5'
        'coevol_uo2_20:disc1'
        'coevol_uo2_20:disc2'
        'coevol_uo2_20:disc3'
        'coevol_uo2_20:disc4'
        'coevol_uo2_20:disc5'
        'coevol_uo2_20:malcev1'
        'coevol_uo2_20:malcev2'
        'coevol_uo2_20:malcev3'
        'coevol_uo2_20:malcev4'
        'coevol_uo2_20:malcev5'
        'coevol_uo2_30:cmp6'
        'coevol_uo2_30:cmp8'
        'coevol_uo2_30:maj6'
        'coevol_uo2_30:mux6'
        'coevol_uo2_30:par5'
        'coevol_uo2_30:disc1'
        'coevol_uo2_30:disc2'
        'coevol_uo2_30:disc3'
        'coevol_uo2_30:disc4'
        'coevol_uo2_30:disc5'
        'coevol_uo2_30:malcev1'
        'coevol_uo2_30:malcev2'
        'coevol_uo2_30:malcev3'
        'coevol_uo2_30:malcev4'
        'coevol_uo2_30:malcev5'
        'coevol_uo2_40:cmp6'
        'coevol_uo2_40:cmp8'
        'coevol_uo2_40:maj6'
        'coevol_uo2_40:mux6'
        'coevol_uo2_40:par5'
        'coevol_uo2_40:disc1'
        'coevol_uo2_40:disc2'
        'coevol_uo2_40:disc3'
        'coevol_uo2_40:disc4'
        'coevol_uo2_40:disc5'
        'coevol_uo2_40:malcev1'
        'coevol_uo2_40:malcev2'
        'coevol_uo2_40:malcev3'
        'coevol_uo2_40:malcev4'
        'coevol_uo2_40:malcev5'
        'coevol_uo2_50:cmp6'
        'coevol_uo2_50:cmp8'
        'coevol_uo2_50:maj6'
        'coevol_uo2_50:mux6'
        'coevol_uo2_50:par5'
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
        'coevol_uo2_60:cmp6'
        'coevol_uo2_60:cmp8'
        'coevol_uo2_60:maj6'
        'coevol_uo2_60:mux6'
        'coevol_uo2_60:par5'
        'coevol_uo2_60:disc1'
        'coevol_uo2_60:disc2'
        'coevol_uo2_60:disc3'
        'coevol_uo2_60:disc4'
        'coevol_uo2_60:disc5'
        'coevol_uo2_60:malcev1'
        'coevol_uo2_60:malcev2'
        'coevol_uo2_60:malcev3'
        'coevol_uo2_60:malcev4'
        'coevol_uo2_60:malcev5'
        'coevol_uo2_70:cmp6'
        'coevol_uo2_70:cmp8'
        'coevol_uo2_70:maj6'
        'coevol_uo2_70:mux6'
        'coevol_uo2_70:par5'
        'coevol_uo2_70:disc1'
        'coevol_uo2_70:disc2'
        'coevol_uo2_70:disc3'
        'coevol_uo2_70:disc4'
        'coevol_uo2_70:disc5'
        'coevol_uo2_70:malcev1'
        'coevol_uo2_70:malcev2'
        'coevol_uo2_70:malcev3'
        'coevol_uo2_70:malcev4'
        'coevol_uo2_70:malcev5'
        'coevol_uo2_80:cmp6'
        'coevol_uo2_80:cmp8'
        'coevol_uo2_80:maj6'
        'coevol_uo2_80:mux6'
        'coevol_uo2_80:par5'
        'coevol_uo2_80:disc1'
        'coevol_uo2_80:disc2'
        'coevol_uo2_80:disc3'
        'coevol_uo2_80:disc4'
        'coevol_uo2_80:disc5'
        'coevol_uo2_80:malcev1'
        'coevol_uo2_80:malcev2'
        'coevol_uo2_80:malcev3'
        'coevol_uo2_80:malcev4'
        'coevol_uo2_80:malcev5'
        'coevol_uo2_90:cmp6'
        'coevol_uo2_90:cmp8'
        'coevol_uo2_90:maj6'
        'coevol_uo2_90:mux6'
        'coevol_uo2_90:par5'
        'coevol_uo2_90:disc1'
        'coevol_uo2_90:disc2'
        'coevol_uo2_90:disc3'
        'coevol_uo2_90:disc4'
        'coevol_uo2_90:disc5'
        'coevol_uo2_90:malcev1'
        'coevol_uo2_90:malcev2'
        'coevol_uo2_90:malcev3'
        'coevol_uo2_90:malcev4'
        'coevol_uo2_90:malcev5'
        'coevol_uo2_100:cmp6'
        'coevol_uo2_100:cmp8'
        'coevol_uo2_100:maj6'
        'coevol_uo2_100:mux6'
        'coevol_uo2_100:par5'
        'coevol_uo2_100:disc1'
        'coevol_uo2_100:disc2'
        'coevol_uo2_100:disc3'
        'coevol_uo2_100:disc4'
        'coevol_uo2_100:disc5'
        'coevol_uo2_100:malcev1'
        'coevol_uo2_100:malcev2'
        'coevol_uo2_100:malcev3'
        'coevol_uo2_100:malcev4'
        'coevol_uo2_100:malcev5')

game=${games[$SLURM_ARRAY_TASK_ID]}

echo "Starting job $SLURM_ARRAY_TASK_ID with setup $game"

module rm apps/python/3.8.5
module load apps/anaconda/5.3.1

source activate cde-search-env

srun python ~/cde-search/cli.py objs -sid "$game" -n 30 -out "$WORK/cde-search/coevol2.jsonlist"

echo "Done job $SLURM_ARRAY_TASK_ID with game $game"