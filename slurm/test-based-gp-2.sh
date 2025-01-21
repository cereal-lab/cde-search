#!/bin/bash 
#SBATCH --job-name=test-based-gp
#SBATCH --time=72:00:00
#SBATCH --output=out/test-based-gp.out
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH --array=0-59

games=('do_rand:cmp6'
        'do_rand:cmp8'
        'do_rand:maj6'
        'do_rand:mux6'
        'do_rand:par5'
        'do_rand:disc1'
        'do_rand:disc2'
        'do_rand:disc3'
        'do_rand:disc4'
        'do_rand:disc5'
        'do_rand:malcev1'
        'do_rand:malcev2'
        'do_rand:malcev3'
        'do_rand:malcev4'
        'do_rand:malcev5'
        'do_fo:cmp6'
        'do_fo:cmp8'
        'do_fo:maj6'
        'do_fo:mux6'
        'do_fo:par5'
        'do_fo:disc1'
        'do_fo:disc2'
        'do_fo:disc3'
        'do_fo:disc4'
        'do_fo:disc5'
        'do_fo:malcev1'
        'do_fo:malcev2'
        'do_fo:malcev3'
        'do_fo:malcev4'
        'do_fo:malcev5'
        'doc_p:cmp6'
        'doc_p:cmp8'
        'doc_p:maj6'
        'doc_p:mux6'
        'doc_p:par5'
        'doc_p:disc1'
        'doc_p:disc2'
        'doc_p:disc3'
        'doc_p:disc4'
        'doc_p:disc5'
        'doc_p:malcev1'
        'doc_p:malcev2'
        'doc_p:malcev3'
        'doc_p:malcev4'
        'doc_p:malcev5'
        'doc_d:cmp6'
        'doc_d:cmp8'
        'doc_d:maj6'
        'doc_d:mux6'
        'doc_d:par5'
        'doc_d:disc1'
        'doc_d:disc2'
        'doc_d:disc3'
        'doc_d:disc4'
        'doc_d:disc5'
        'doc_d:malcev1'
        'doc_d:malcev2'
        'doc_d:malcev3'
        'doc_d:malcev4'
        'doc_d:malcev5')

game=${games[$SLURM_ARRAY_TASK_ID]}

echo "Starting job $SLURM_ARRAY_TASK_ID with setup $game"

module rm apps/python/3.8.5
module load apps/anaconda/5.3.1

source activate cde-search-env

srun python ~/cde-search/cli.py objs -sid "$game" -n 30 -out "$WORK/cde-search/gp-objs.jsonlist"

echo "Done job $SLURM_ARRAY_TASK_ID with game $game"