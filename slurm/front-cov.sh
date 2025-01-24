#!/bin/bash 
#SBATCH --job-name=front-cov
#SBATCH --time=72:00:00
#SBATCH --output=out/front-cov.out
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH --array=0-89

games=('cov_ht_bp:cmp6'
        'cov_ht_bp:cmp8'
        'cov_ht_bp:maj6'
        'cov_ht_bp:mux6'
        'cov_ht_bp:par5'
        'cov_ht_bp:disc1'
        'cov_ht_bp:disc2'
        'cov_ht_bp:disc3'
        'cov_ht_bp:disc4'
        'cov_ht_bp:disc5'
        'cov_ht_bp:malcev1'
        'cov_ht_bp:malcev2'
        'cov_ht_bp:malcev3'
        'cov_ht_bp:malcev4'
        'cov_ht_bp:malcev5'
        'cov_et_bp:cmp6'
        'cov_et_bp:cmp8'
        'cov_et_bp:maj6'
        'cov_et_bp:mux6'
        'cov_et_bp:par5'
        'cov_et_bp:disc1'
        'cov_et_bp:disc2'
        'cov_et_bp:disc3'
        'cov_et_bp:disc4'
        'cov_et_bp:disc5'
        'cov_et_bp:malcev1'
        'cov_et_bp:malcev2'
        'cov_et_bp:malcev3'
        'cov_et_bp:malcev4'
        'cov_et_bp:malcev5'
        'cov_rt_bp:cmp6'
        'cov_rt_bp:cmp8'
        'cov_rt_bp:maj6'
        'cov_rt_bp:mux6'
        'cov_rt_bp:par5'
        'cov_rt_bp:disc1'
        'cov_rt_bp:disc2'
        'cov_rt_bp:disc3'
        'cov_rt_bp:disc4'
        'cov_rt_bp:disc5'
        'cov_rt_bp:malcev1'
        'cov_rt_bp:malcev2'
        'cov_rt_bp:malcev3'
        'cov_rt_bp:malcev4'
        'cov_rt_bp:malcev5'
        'cov_ht_rp:cmp6'
        'cov_ht_rp:cmp8'
        'cov_ht_rp:maj6'
        'cov_ht_rp:mux6'
        'cov_ht_rp:par5'
        'cov_ht_rp:disc1'
        'cov_ht_rp:disc2'
        'cov_ht_rp:disc3'
        'cov_ht_rp:disc4'
        'cov_ht_rp:disc5'
        'cov_ht_rp:malcev1'
        'cov_ht_rp:malcev2'
        'cov_ht_rp:malcev3'
        'cov_ht_rp:malcev4'
        'cov_ht_rp:malcev5'
        'cov_et_rp:cmp6'
        'cov_et_rp:cmp8'
        'cov_et_rp:maj6'
        'cov_et_rp:mux6'
        'cov_et_rp:par5'
        'cov_et_rp:disc1'
        'cov_et_rp:disc2'
        'cov_et_rp:disc3'
        'cov_et_rp:disc4'
        'cov_et_rp:disc5'
        'cov_et_rp:malcev1'
        'cov_et_rp:malcev2'
        'cov_et_rp:malcev3'
        'cov_et_rp:malcev4'
        'cov_et_rp:malcev5'
        'cov_rt_rp:cmp6'
        'cov_rt_rp:cmp8'
        'cov_rt_rp:maj6'
        'cov_rt_rp:mux6'
        'cov_rt_rp:par5'
        'cov_rt_rp:disc1'
        'cov_rt_rp:disc2'
        'cov_rt_rp:disc3'
        'cov_rt_rp:disc4'
        'cov_rt_rp:disc5'
        'cov_rt_rp:malcev1'
        'cov_rt_rp:malcev2'
        'cov_rt_rp:malcev3'
        'cov_rt_rp:malcev4'
        'cov_rt_rp:malcev5')

game=${games[$SLURM_ARRAY_TASK_ID]}

echo "Starting job $SLURM_ARRAY_TASK_ID with setup $game"

module rm apps/python/3.8.5
module load apps/anaconda/5.3.1

source activate cde-search-env

srun python ~/cde-search/cli.py objs -sid "$game" -n 30 -out "$WORK/cde-search/front-cov.jsonlist"

echo "Done job $SLURM_ARRAY_TASK_ID with game $game"