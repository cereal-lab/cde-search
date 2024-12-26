#!/bin/bash 
#SBATCH --job-name=doc-dof-cse
#SBATCH --time=72:00:00
#SBATCH --output=out/doc-dof-cse.out
#SBATCH --mem=8G
#SBATCH --array=0-344

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
    'ifs:malcev5'
    'do_rand:cmp6'
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
    'do_nsga:cmp6'
    'do_nsga:cmp8'
    'do_nsga:maj6'
    'do_nsga:mux6'
    'do_nsga:par5'
    'do_nsga:disc1'
    'do_nsga:disc2'
    'do_nsga:disc3'
    'do_nsga:disc4'
    'do_nsga:disc5'
    'do_nsga:malcev1'
    'do_nsga:malcev2'
    'do_nsga:malcev3'
    'do_nsga:malcev4'
    'do_nsga:malcev5'
    'doc:cmp6'
    'doc:cmp8'
    'doc:maj6'
    'doc:mux6'
    'doc:par5'
    'doc:disc1'
    'doc:disc2'
    'doc:disc3'
    'doc:disc4'
    'doc:disc5'
    'doc:malcev1'
    'doc:malcev2'
    'doc:malcev3'
    'doc:malcev4'
    'doc:malcev5'
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
    'doc_d:malcev5'
    'dof_w_2:cmp6'
    'dof_w_2:cmp8'
    'dof_w_2:maj6'
    'dof_w_2:mux6'
    'dof_w_2:par5'
    'dof_w_2:disc1'
    'dof_w_2:disc2'
    'dof_w_2:disc3'
    'dof_w_2:disc4'
    'dof_w_2:disc5'
    'dof_w_2:malcev1'
    'dof_w_2:malcev2'
    'dof_w_2:malcev3'
    'dof_w_2:malcev4'
    'dof_w_2:malcev5'
    'dof_w_3:cmp6'
    'dof_w_3:cmp8'
    'dof_w_3:maj6'
    'dof_w_3:mux6'
    'dof_w_3:par5'
    'dof_w_3:disc1'
    'dof_w_3:disc2'
    'dof_w_3:disc3'
    'dof_w_3:disc4'
    'dof_w_3:disc5'
    'dof_w_3:malcev1'
    'dof_w_3:malcev2'
    'dof_w_3:malcev3'
    'dof_w_3:malcev4'
    'dof_w_3:malcev5'
    'dof_wh_2:cmp6'
    'dof_wh_2:cmp8'
    'dof_wh_2:maj6'
    'dof_wh_2:mux6'
    'dof_wh_2:par5'
    'dof_wh_2:disc1'
    'dof_wh_2:disc2'
    'dof_wh_2:disc3'
    'dof_wh_2:disc4'
    'dof_wh_2:disc5'
    'dof_wh_2:malcev1'
    'dof_wh_2:malcev2'
    'dof_wh_2:malcev3'
    'dof_wh_2:malcev4'
    'dof_wh_2:malcev5'
    'dof_wh_3:cmp6'
    'dof_wh_3:cmp8'
    'dof_wh_3:maj6'
    'dof_wh_3:mux6'
    'dof_wh_3:par5'
    'dof_wh_3:disc1'
    'dof_wh_3:disc2'
    'dof_wh_3:disc3'
    'dof_wh_3:disc4'
    'dof_wh_3:disc5'
    'dof_wh_3:malcev1'
    'dof_wh_3:malcev2'
    'dof_wh_3:malcev3'
    'dof_wh_3:malcev4'
    'dof_wh_3:malcev5'
    'dof_w_2_80:cmp6'
    'dof_w_2_80:cmp8'
    'dof_w_2_80:maj6'
    'dof_w_2_80:mux6'
    'dof_w_2_80:par5'
    'dof_w_2_80:disc1'
    'dof_w_2_80:disc2'
    'dof_w_2_80:disc3'
    'dof_w_2_80:disc4'
    'dof_w_2_80:disc5'
    'dof_w_2_80:malcev1'
    'dof_w_2_80:malcev2'
    'dof_w_2_80:malcev3'
    'dof_w_2_80:malcev4'
    'dof_w_2_80:malcev5'
    'dof_w_3_80:cmp6'
    'dof_w_3_80:cmp8'
    'dof_w_3_80:maj6'
    'dof_w_3_80:mux6'
    'dof_w_3_80:par5'
    'dof_w_3_80:disc1'
    'dof_w_3_80:disc2'
    'dof_w_3_80:disc3'
    'dof_w_3_80:disc4'
    'dof_w_3_80:disc5'
    'dof_w_3_80:malcev1'
    'dof_w_3_80:malcev2'
    'dof_w_3_80:malcev3'
    'dof_w_3_80:malcev4'
    'dof_w_3_80:malcev5'
    'dof_wh_2_80:cmp6'
    'dof_wh_2_80:cmp8'
    'dof_wh_2_80:maj6'
    'dof_wh_2_80:mux6'
    'dof_wh_2_80:par5'
    'dof_wh_2_80:disc1'
    'dof_wh_2_80:disc2'
    'dof_wh_2_80:disc3'
    'dof_wh_2_80:disc4'
    'dof_wh_2_80:disc5'
    'dof_wh_2_80:malcev1'
    'dof_wh_2_80:malcev2'
    'dof_wh_2_80:malcev3'
    'dof_wh_2_80:malcev4'
    'dof_wh_2_80:malcev5'
    'dof_wh_3_80:cmp6'
    'dof_wh_3_80:cmp8'
    'dof_wh_3_80:maj6'
    'dof_wh_3_80:mux6'
    'dof_wh_3_80:par5'
    'dof_wh_3_80:disc1'
    'dof_wh_3_80:disc2'
    'dof_wh_3_80:disc3'
    'dof_wh_3_80:disc4'
    'dof_wh_3_80:disc5'
    'dof_wh_3_80:malcev1'
    'dof_wh_3_80:malcev2'
    'dof_wh_3_80:malcev3'
    'dof_wh_3_80:malcev4'
    'dof_wh_3_80:malcev5'
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
    'do_pca:cmp6'
    'do_pca:cmp8'
    'do_pca:maj6'
    'do_pca:mux6'
    'do_pca:par5'
    'do_pca:disc1'
    'do_pca:disc2'
    'do_pca:disc3'
    'do_pca:disc4'
    'do_pca:disc5'
    'do_pca:malcev1'
    'do_pca:malcev2'
    'do_pca:malcev3'
    'do_pca:malcev4'
    'do_pca:malcev5'
    'cov_ht_bp:cmp6'
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

srun python ~/cde-search/cli.py objs -sid "$game" --out "$WORK/cde-search/gp-objs.jsonlist"

echo "Done job $SLURM_ARRAY_TASK_ID with game $game"