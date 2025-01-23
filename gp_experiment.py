''' Running of DOC, DOF and CSE on benchmarks from bool and alg0 '''

import json
from tabulate import tabulate
from gp import gp_sim_names
import gp
from gp_benchmarks import all_benchmarks
from nsga2 import nsga2_sim_names
import nsga2
from front_coverage import cov_sim_names
import front_coverage as cov
from coevol import coevol_sim_names
import coevol

def get_simulation(sim_name):
    if sim_name in gp_sim_names:
        return getattr(gp, sim_name)
    if sim_name in nsga2_sim_names:
        return getattr(nsga2, sim_name)
    if sim_name in cov_sim_names:
        return getattr(cov, sim_name)
    if sim_name in coevol_sim_names:
        return getattr(coevol, sim_name)
    return None

# postprocessing 
def compute_run_stats(file_name: str):
    with open(file_name, 'r') as f:
        json_lines = f.readlines()
    metrics = [json.loads(l) for l in json_lines]
    stats = {}
    for m in metrics:
        game = m['game_name']
        sim = m['sim_name']
        fitness0 = m['hamming_distance_fitness']
        best_found = m['best_found']
        if best_found:
            assert fitness0[-1] == 0
        game_sim_m = stats.setdefault((game, sim), {})
        conv_count = game_sim_m.setdefault("conv_count", 0) 
        total_count = game_sim_m.setdefault("total_count", 0)
        game_sim_m["conv_count"] = conv_count + (1 if best_found else 0)
        game_sim_m["total_count"] = total_count + 1
    for (game, sim), m in stats.items():
        m["conv_rate"] = m["conv_count"] / m["total_count"]    

    # stats_maps = {}
    game_conv = {}
    sim_conv = {}
    for (game, sim), m in stats.items():
        # game_m = stats_maps.setdefault(sim, {})
        # game_m[game] = m 
        gd = game_conv.setdefault(game, 0)
        game_conv[game] = gd + m["conv_count"]
        sd = sim_conv.setdefault(sim, 0)
        sim_conv[sim] = sd + m["conv_count"]

    # games_s = sorted(game_conv.keys(), key = lambda x: game_conv[x], reverse=True)
    # rows = []
    # for sim in sorted(sim_conv.keys(), key = lambda x: sim_conv[x], reverse=True):    
    #     sim_rates = [('' if r == 0 else r) for g in games_s for r in [round(stats[(g, sim)]['conv_rate'] * 100)]]
    #     rows.append([sim, *sim_rates])

    col_names = sorted(sim_conv.keys(), key = lambda x: sim_conv[x], reverse=True)
    # col_names = ['gp', 'ifs', 'do_rand', 'doc', 'doc_p', 'doc_d'] #'cov_ht_bp', 'doc_wh_2_80', 'doc_w_2_80', 'doc_w_2', 'doc_wh_2',  'cov_rt_bp', 'do_fo', 'cov_et_bp', 'doc_w_3_80', 'doc_wh_3_80', 'doc_w_3', 'doc_wh_3', 'do_pca_diff_3', 'do_nsga', 'do_pca_diff_2', 'cov_ht_rp', 'do_pca_abs_2', 'do_pca_abs_3', 'cov_et_rp', 'cov_rt_rp']
    rows = []
    # row_names = ['cmp6', 'cmp8', 'disc1', 'disc2', 'disc3', 'disc4', 'disc5', 'maj6', 'malcev1', 'malcev2', 'malcev3', 'malcev4', 'malcev5', 'mux6', 'par5']
    row_names = sorted(game_conv.keys(), key = lambda x: game_conv[x], reverse=True)
    for rn in row_names:
        sim_rates = [('' if v == 0 else v) for cn in col_names for v in [round(stats[(rn, cn)]['conv_rate'] * 100)]]
        rows.append([rn, *sim_rates])

    print(tabulate(rows, headers=["", *col_names], tablefmt='grid', numalign="center", stralign="center"), file=open("./dmp.txt", "w"))

    return stats

sim_names = [*gp_sim_names, *nsga2_sim_names, *cov_sim_names, *coevol_sim_names]

sim_names = coevol_sim_names #["do_rand", "do_fo", "doc_p", "doc_d"]
from gp_benchmarks import benchmark_map
if __name__ == "__main__":
    print("testing evo runs")
    cnt = 0
    f = open("setups.txt", "w")
    for sim_name in sim_names:
        for b_name in benchmark_map.keys():
            _, bm = all_benchmarks[benchmark_map[b_name]]
            gold, _, _ = bm()
            print(sim_name, b_name, len(gold))
            print(f"'{sim_name}:{b_name}'", file=f)
            cnt += 1
    print(f"total setups: {cnt}")
    # cov_ht_bp(idx = 11)
    # compute_run_stats("data/test-based-gp/main-gp-objs.jsonlist")
    pass