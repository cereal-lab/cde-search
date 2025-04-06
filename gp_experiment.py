''' Running of DOC, DOF and CSE on benchmarks from bool and alg0 '''

from collections import defaultdict
import json
from matplotlib import pyplot as plt
import numpy as np
from tabulate import tabulate
from gp import gp_sim_names
import gp
from nsga2 import nsga2_sim_names
import nsga2
from front_coverage import cov_sim_names
import front_coverage as cov
from coevol import coevol_sim_names
import coevol
from scipy import stats as sci_stats
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{times}"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = ['Times']    


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
def compute_run_stats(file_name: str, sim_names = None, tablefmt = "latex", transpose = False, custom_prefix = ""):
    import os 
    out_name = os.path.basename(file_name).split('.')[0]
    out_dir = os.path.dirname(file_name)
    with open(file_name, 'r') as f:
        json_lines = f.readlines()
    metrics = [json.loads(l) for l in json_lines]
    stats = {}
    for m in metrics:
        game = m['game_name']
        sim = m['sim_name']
        if sim_names is not None and sim not in sim_names:
            continue
        fitness0 = m['hamming_distance_fitness']
        depth = m['depth_fitness']
        best_found = m['best_found']
        gen = m['gen']
        if best_found:
            assert fitness0[-1] == 0            
        game_sim_m = stats.setdefault((game, sim), {})
        conv_count = game_sim_m.setdefault("conv_count", 0) 
        total_count = game_sim_m.setdefault("total_count", 0)
        game_sim_m["conv_count"] = conv_count + (1 if best_found else 0)
        game_sim_m["total_count"] = total_count + 1
        game_sim_m.setdefault("err", []).append(fitness0[-1])
        if best_found:
            best_depth = depth[-1]
            game_sim_m.setdefault("best_depth", []).append(best_depth)
            game_sim_m.setdefault("cpu_time", []).append(m["cpu_time"])
            game_sim_m.setdefault("gen", []).append(gen)
    for (game, sim), m in stats.items():
        if m["total_count"] != 30:
            print(f"Warning: {game} {sim} count is {m['total_count']}")
        m["conv_rate"] = m["conv_count"] / m["total_count"]  
        if 'gen' in m:
            gen_std = round(np.ceil(np.std(m["gen"])))
            gen_mean = round(np.ceil(np.mean(m["gen"])))
            m['gen_m'] = f"${gen_mean} \\pm {gen_std}$"
        else:
            m['gen_m'] = ""
        err_mean = round(np.ceil(np.mean(m["err"])))
        err_std = round(np.ceil(np.std(m["err"])))
        m["err_m"] = f"${err_mean} \\pm {err_std}$"
        if "cpu_time" in m:
            cpu_time_mean = round(np.mean(m["cpu_time"]))
            cpu_time_std = round(np.std(m["cpu_time"]))
            m["cpu_time"] = f"${cpu_time_mean} \\pm {cpu_time_std}$"      
        else:
            m["cpu_time"] = ""  

    # stats_maps = {}
    game_conv = {}
    game_depth = {}
    game_total_count = {}
    sim_total_count = {}
    sim_conv = {}
    for (game, sim), m in stats.items():
        # game_m = stats_maps.setdefault(sim, {})
        # game_m[game] = m 
        gd = game_conv.setdefault(game, 0)
        game_conv[game] = gd + m["conv_count"]
        game_depth.setdefault(game, []).extend(m.get("best_depth", []))
        game_total_count[game] = game_total_count.setdefault(game, 0) + m["total_count"]
        sim_total_count[sim] = sim_total_count.setdefault(sim, 0) + m["total_count"]
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
    rows2 = []
    rows3 = []
    rows4 = []
    rows5 = []
    # row_names = ['cmp6', 'cmp8', 'disc1', 'disc2', 'disc3', 'disc4', 'disc5', 'maj6', 'malcev1', 'malcev2', 'malcev3', 'malcev4', 'malcev5', 'mux6', 'par5']
    row_names = sorted(game_conv.keys(), key = lambda x: game_conv[x], reverse=True)
    rows.append(["Total", "", "",  *[round(sim_conv[cn] * 100/sim_total_count[cn]) for cn in col_names ]])
    for rn in row_names:
        game_cnv = round(game_conv[rn] * 100 / game_total_count[rn])
        if len(game_depth[rn]) > 0:
            game_dpth_m = np.mean(game_depth[rn])
            game_dpth_std = np.std(game_depth[rn])
            game_dpth = f"${round(game_dpth_m)} \\pm {round(game_dpth_std)}$"
        else:
            game_dpth = ""
        sim_rates = [('' if v == 0 else v) for cn in col_names for v in [round(stats.get((rn, cn), {}).get('conv_rate', 0) * 100)]]
        rows.append([rn, game_cnv, game_dpth, *sim_rates])
        sim_gen = [v for cn in col_names for v in [stats.get((rn, cn), {}).get('gen_m', "")]]
        rows2.append([rn, *sim_gen])
        sim_err = [v for cn in col_names for v in [stats.get((rn, cn), {}).get('err_m', "")]]
        rows3.append([rn, *sim_err])
        sim_cpu_time = [v for cn in col_names for v in [stats.get((rn, cn), {}).get('cpu_time', "")]]
        rows4.append([rn, *sim_cpu_time])        
        total_count = [v for cn in col_names for v in [stats.get((rn, cn), {}).get('total_count', "")]]
        rows5.append([rn, *total_count])           

    suffix = ""
    col_names1 = ["", "Total", "depth", *col_names]
    col_names2 = ["", *col_names]
    col_names3 = ["", *col_names]
    col_names4 = ["", *col_names]    
    col_names5 = ["", *col_names]    
    if transpose:
        full_rows1 = [col_names1, *rows]
        full_rows2 = [col_names2, *rows2]
        full_rows3 = [col_names3, *rows3]
        full_rows4 = [col_names4, *rows4]
        full_rows5 = [col_names4, *rows5]
        col_names1, *rows = list(map(list, zip(*full_rows1)))
        col_names2, *rows2 = list(map(list, zip(*full_rows2)))
        col_names3, *rows3 = list(map(list, zip(*full_rows3)))
        col_names4, *rows4 = list(map(list, zip(*full_rows4)))
        col_names5, *rows5 = list(map(list, zip(*full_rows5)))
        suffix = "-T"
        pass


    with open(os.path.join(out_dir, f"{custom_prefix}{out_name}-conv-{tablefmt}{suffix}.txt"), "w") as f:
        print(tabulate(rows, headers=col_names1, tablefmt=tablefmt, numalign="center", stralign="center"), file=f)

    with open(os.path.join(out_dir, f"{custom_prefix}{out_name}-gen-{tablefmt}{suffix}.txt"), "w") as f:
        print(tabulate(rows2, headers=col_names2, tablefmt=tablefmt, numalign="center", stralign="center"), file=f)

    with open(os.path.join(out_dir, f"{custom_prefix}{out_name}-err-{tablefmt}{suffix}.txt"), "w") as f:
        print(tabulate(rows3, headers=col_names3, tablefmt=tablefmt, numalign="center", stralign="center"), file=f)
    
    with open(os.path.join(out_dir, f"{custom_prefix}{out_name}-cpu-{tablefmt}{suffix}.txt"), "w") as f:
        print(tabulate(rows4, headers=col_names4, tablefmt=tablefmt, numalign="center", stralign="center"), file=f)

    with open(os.path.join(out_dir, f"{custom_prefix}{out_name}-count-{tablefmt}{suffix}.txt"), "w") as f:
        print(tabulate(rows5, headers=col_names5, tablefmt=tablefmt, numalign="center", stralign="center"), file=f)

    return stats

def get_sol_dupl(file_name):
    with open(file_name, 'r') as f:
        json_lines = f.readlines()
    metrics = [json.loads(l) for l in json_lines]
    stats = {}
    for m in metrics:
        game = m['game_name']
        # sim = m['sim_name']
        best_ind = m['best']
        best_found = m['best_found']
        if best_found:
            game_s = stats.setdefault(game, {})
            game_s[best_ind] = game_s.get(best_ind, 0) + 1
    final_stats = {}
    for game, m in stats.items():
        cnts = sorted(m.values(), reverse=True)
        final_stats[game] = cnts
    return final_stats


def draw_line_plot(file_name, game_name, metric_name, sim_names, out_file, metric_label = None, sim_labels = None, pad_value = 0, max_metric_value = None,
                    plot_settings = {}, norm_metric = None):
    with open(file_name, 'r') as f:
        json_lines = f.readlines()
    metrics = [json.loads(l) for l in json_lines]
    stats = {}
    if metric_label is None:
        metric_label = metric_name
    for m in metrics:
        game = m['game_name']
        sim = m['sim_name']
        if game != game_name or sim not in sim_names:
            continue
        metric = m[metric_name]
        if norm_metric is not None:
            norm_metric_values = m[norm_metric]
            metric = [round(m * 100.0 / n) for m, n in zip(metric, norm_metric_values) ]
        stats.setdefault(sim, []).append(np.pad(metric, (0, 100 - len(metric)), 'constant', constant_values=pad_value))

    sim_labels_map = {sim_name:(sim_name if sim_labels is None else sim_labels[i]) for i, sim_name in enumerate(sim_names)}

    # if norm_metric is not None:
    #     max_metric_value = 100
    #     min_metric_value = 0
    # else:
    min_metric_value = -1
    should_compute_max = max_metric_value is None
    if should_compute_max:
        max_metric_value = 0
    plt.ioff()
    for sim, m in stats.items():
        values = np.array(m) # each gen is a col, each run is a row
        mean = np.mean(values, axis=0)
        confidence_level = 0.95
        degrees_freedom = values.shape[0] - 1
        sample_standard_error = sci_stats.sem(values, axis=0)
        confidence_interval = sci_stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
        min_v = confidence_interval[0]
        where_is_nan = np.isnan(min_v)
        min_v[where_is_nan] = mean[where_is_nan]
        max_v = confidence_interval[1]
        max_v[where_is_nan] = mean[where_is_nan]
        if should_compute_max:
            max_metric_value = max(max_metric_value, np.max(max_v))
        label = sim_labels_map[sim]
        plot_s = plot_settings.get(sim, {})
        plot_s = {**dict(linewidth=1), **plot_s}
        pl, = plt.plot(np.arange(0, 100) + 1, mean, label=label, **plot_s) # marker='o', markersize=5, linewidth=1)
        plt.fill_between(np.arange(0, 100) + 1, min_v, max_v, alpha=.1, linewidth=0, color = pl.get_color())
    plt.xlabel('Generation')
    plt.ylabel(metric_label)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.xlim(0, 101)
    # if norm_metric is None:
    #     max_metric_value = max_metric_value + 1
    plt.ylim(min_metric_value, max_metric_value + 1)
    plt.legend(fontsize='small')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    # plt.title(f'Best-of-run', fontsize=15)
    plt.tight_layout()
    plt.savefig(out_file, format='pdf')  
    plt.clf()  

def draw_line_plot2(file_name, game_name, metric_name, sim_names, out_file, metric_label = None, sim_labels = None, max_metric_value = None,
                    plot_settings = {}, norm_metric = None, max_gen = 0, min_metric_value = -1, log_scale = False):
    ''' like draw_line_plot but without padding '''
    with open(file_name, 'r') as f:
        json_lines = f.readlines()
    metrics = [json.loads(l) for l in json_lines]
    stats = {}
    if metric_label is None:
        metric_label = metric_name
    for m in metrics:
        game = m['game_name']
        sim = m['sim_name']
        if game != game_name or sim not in sim_names:
            continue
        metric = m[metric_name]
        if norm_metric is not None:
            norm_metric_values = m[norm_metric]
            metric = [round(m * 100.0 / n) for m, n in zip(metric, norm_metric_values) ]
        metric = np.array(metric, dtype=float)
        stats.setdefault(sim, []).append(np.pad(metric, (0, 100 - len(metric)), 'constant', constant_values=np.nan))

    # transpose unpadded  
    for sim, m in stats.items():
        stats[sim] = np.array(m).T

    sim_labels_map = {sim_name:(sim_name if sim_labels is None else sim_labels[i]) for i, sim_name in enumerate(sim_names)}

    # if norm_metric is not None:
    #     max_metric_value = 100
    #     min_metric_value = 0
    # else:
    should_compute_max = max_metric_value is None
    if should_compute_max:
        max_metric_value = 0
    plt.ioff()
    for sim, m in stats.items():
        values = np.array(m) # each gen is a col, each run is a row
        mean = np.nanmean(values, axis=1)
        non_nan_b = ~np.isnan(mean)
        non_nan_ids = np.where(non_nan_b)[0]
        mean = mean[non_nan_b]
        max_gen = max(max_gen, len(mean))
        values = values[non_nan_ids, :]
        confidence_level = 0.95
        degrees_freedom = np.sum(~np.isnan(values), axis=1) - 1
        sample_standard_error = sci_stats.sem(values, axis=1, nan_policy='omit')
        confidence_interval = sci_stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
        min_v = confidence_interval[0]
        max_v = confidence_interval[1]
        no_degrees = degrees_freedom < 10
        min_v[no_degrees] = mean[no_degrees] - sample_standard_error[no_degrees]
        max_v[no_degrees] = mean[no_degrees] + sample_standard_error[no_degrees]
        where_is_nan = np.isnan(min_v)
        min_v[where_is_nan] = mean[where_is_nan]
        where_is_nan = np.isnan(max_v)
        max_v[where_is_nan] = mean[where_is_nan]
        if should_compute_max:
            max_metric_value = max(max_metric_value, np.max(max_v))
        label = sim_labels_map[sim]
        plot_s = plot_settings.get(sim, {})
        plot_s = {**dict(linewidth=1), **plot_s}
        pl, = plt.plot(np.arange(0, len(mean)) + 1, mean, label=label, **plot_s) # marker='o', markersize=5, linewidth=1)
        plt.fill_between(np.arange(0, len(mean)) + 1, min_v, max_v, alpha=(.05 if sim == "gp" else .1), linewidth=0, color = pl.get_color())
    plt.xlabel('Generation')
    plt.ylabel(metric_label)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.xlim(0, max_gen + 1)
    if log_scale:
        plt.yscale('log')
    # if norm_metric is None:
    #     max_metric_value = max_metric_value + 1
    plt.ylim(min_metric_value, max_metric_value + 1)
    plt.legend(fontsize='small')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    # plt.title(f'Best-of-run', fontsize=15)
    plt.tight_layout()
    plt.savefig(out_file, format='pdf')  
    plt.clf()  

def draw_line_plot3(file_name, game_name, metric_name1, metric_name2, sim_names, out_file, metric_label = None, sim_labels = None, max_metric_value = None,
                    plot_settings = {}, norm_value = None, max_gen = 0, min_metric_value = -1, log_scale = False,
                    colors = None, suffix1 = "", suffix2 = ""):
    ''' like draw_line_plot but without padding '''
    with open(file_name, 'r') as f:
        json_lines = f.readlines()
    metrics = [json.loads(l) for l in json_lines]
    stats1 = {}
    stats2 = {}
    if metric_label is None:
        metric_label = metric_name1
    for m in metrics:
        game = m['game_name']
        sim = m['sim_name']
        if game != game_name or sim not in sim_names:
            continue
        metric = np.array(m[metric_name1], dtype=float)
        if norm_value is not None:
            metric = metric * 100 / norm_value
        stats1.setdefault(sim, []).append(np.pad(metric, (0, 100 - len(metric)), 'constant', constant_values=np.nan))
        metric = np.array(m[metric_name2], dtype=float)
        if norm_value is not None:
            metric = metric * 100 / norm_value
        stats2.setdefault(sim, []).append(np.pad(metric, (0, 100 - len(metric)), 'constant', constant_values=np.nan))

    # transpose unpadded  
    for sim, m in stats1.items():
        stats1[sim] = np.array(m).T[1:]

    for sim, m in stats2.items():
        stats2[sim] = np.array(m).T[1:]     

    sim_labels_map = {sim_name:(sim_name if sim_labels is None else sim_labels[i]) for i, sim_name in enumerate(sim_names)}

    # if norm_metric is not None:
    #     max_metric_value = 100
    #     min_metric_value = 0
    # else:
    should_compute_max = max_metric_value is None
    if should_compute_max:
        max_metric_value = 0
    plt.ioff()
    for sim_id, sim in enumerate(sim_names):
        values1 = stats1[sim]
        values2 = stats2[sim]
        def process(values):
            # values = np.array(m) # each gen is a col, each run is a row
            mean = np.nanmean(values, axis=1)
            non_nan_b = ~np.isnan(mean)
            non_nan_ids = np.where(non_nan_b)[0]
            mean = mean[non_nan_b]            
            values = values[non_nan_ids, :]
            confidence_level = 0.95
            degrees_freedom = np.sum(~np.isnan(values), axis=1) - 1
            sample_standard_error = sci_stats.sem(values, axis=1, nan_policy='omit')
            confidence_interval = sci_stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
            min_v = confidence_interval[0]
            max_v = confidence_interval[1]
            no_degrees = degrees_freedom < 10
            min_v[no_degrees] = mean[no_degrees] - sample_standard_error[no_degrees]
            max_v[no_degrees] = mean[no_degrees] + sample_standard_error[no_degrees]
            where_is_nan = np.isnan(min_v)
            min_v[where_is_nan] = mean[where_is_nan]
            where_is_nan = np.isnan(max_v)
            max_v[where_is_nan] = mean[where_is_nan]
            return min_v, mean, max_v         
        min_v, mean, max_v = process(values1)
        max_gen = max(max_gen, len(mean))
        if should_compute_max:
            max_metric_value = max(max_metric_value, np.max(max_v))        
        label1 = f"{sim_labels_map[sim]}, {suffix1}"
        plot_s = plot_settings.get(sim, {})
        plot_s = {**dict(linewidth=1), **plot_s}
        if colors is not None:
            plot_s.update(color = colors[sim_id])
        pl, = plt.plot(np.arange(0, len(mean)) + 1, mean, label=label1, **plot_s) # marker='o', markersize=5, linewidth=1)
        plt.fill_between(np.arange(0, len(mean)) + 1, min_v, max_v, alpha=(.05 if sim == "gp" else .05), linewidth=0, color = pl.get_color())
        min_v, mean, max_v = process(values2)
        if should_compute_max:
            max_metric_value = max(max_metric_value, np.max(max_v))
        plot_s.update(color = pl.get_color(), linestyle = '--', linewidth=0.5)
        label2= f"{sim_labels_map[sim]}, {suffix2}"
        pl, = plt.plot(np.arange(0, len(mean)) + 1, mean, label=label2, **plot_s) # marker='o', markersize=5, linewidth=1)
        plt.fill_between(np.arange(0, len(mean)) + 1, min_v, max_v, alpha=(.05 if sim == "gp" else .05), linewidth=0, color = pl.get_color())

    plt.xlabel('Generation')
    plt.ylabel(metric_label)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    plt.xlim(0, max_gen + 1)
    if log_scale:
        plt.yscale('log')
    # if norm_metric is None:
    #     max_metric_value = max_metric_value + 1
    plt.ylim(min_metric_value, max_metric_value + 1)
    plt.legend(fontsize='small')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    # plt.title(f'Best-of-run', fontsize=15)
    plt.tight_layout()
    plt.savefig(out_file, format='pdf')  
    plt.clf()  

def draw_streamgraph_of_breeding(file_name, game_name, sim_name, out_file, breed_size = 1000):
    ''' Draws stacked areas for different categories of offspring.
        Bad childred (area E) is at the top, good children (area A) of at least one parent - at the bottom,
        good dom children (area B) of at least one parent - at the bottom of A,
        good dom children (area C) of all parents - at the bottom of B.
        Middle area - non-comparable children (area D).
        breed_size - normalization 
        We plot only mean of these values.
    '''
    with open(file_name, 'r') as f:
        json_lines = f.readlines()
    metrics = [json.loads(l) for l in json_lines]
    stats = {}
    for m in metrics:
        game = m['game_name']
        sim = m['sim_name']
        if game != game_name or sim != sim_name:
            continue
        bad_children = np.array(m['bad_children'], dtype=float) * 100 / breed_size # area E
        good_children = np.array(m['good_children'], dtype=float) * 100 / breed_size # area A
        good_dom_children = np.array(m['good_dom_children'], dtype=float) * 100 / breed_size # area B
        best_children = np.array(m['best_children'], dtype=float) * 100 / breed_size #ploted as dashed line
        best_dom_children = np.array(m['best_dom_children'], dtype=float) * 100 / breed_size # area C        
        stats.setdefault("bad_children", []).append(np.pad(bad_children, (0, 100 - len(bad_children)), 'constant', constant_values=np.nan))
        stats.setdefault("good_children", []).append(np.pad(good_children, (0, 100 - len(good_children)), 'constant', constant_values=np.nan))
        stats.setdefault("good_dom_children", []).append(np.pad(good_dom_children, (0, 100 - len(good_dom_children)), 'constant', constant_values=np.nan))
        stats.setdefault("best_children", []).append(np.pad(best_children, (0, 100 - len(best_children)), 'constant', constant_values=np.nan))
        stats.setdefault("best_dom_children", []).append(np.pad(best_dom_children, (0, 100 - len(best_dom_children)), 'constant', constant_values=np.nan))


    max_gen = 0 
    for k, v in stats.items():
        vT = np.array(v).T
        vTmeans = np.nanmean(vT, axis=1)
        stats[k] = vTmeans[~np.isnan(vTmeans)]
        max_gen = max(max_gen, len(stats[k]))

    # stats["noncomparable_children"] = 100.0 - stats["bad_children"] - stats["good_children"]
    # assert np.all(stats["noncomparable_children"] >= 0), "Sum of areas A, D, E should be 100"    

    plt.ioff()
    gens = np.arange(0, max_gen) + 1
    # plt.fill_between(gens, 100.0 - stats["bad_children"], 100.0, label='Bad', color='#2F4F4F')
    plt.fill_between(gens, stats["good_children"], 100, label='Same or worse', color='#111111')
    plt.fill_between(gens, 0, stats["good_children"], label='Better than some parent', color='#2F4F4F')
    plt.fill_between(gens, 0, stats["good_dom_children"], label='Dominates some parents', color='#C0C0C0')
    plt.fill_between(gens, 0, stats["best_dom_children"], label='Dominates all parents', color='#E3E3E3')
    # plt.fill_between(gens, 100, 100 - E_smooth, label='E', color='red')

    plt.xlabel('Generation')
    plt.ylabel('Offspring split by quality, \\%')
    # plt.title('Stacked Area Chart with Smooth Lines')
    plt.legend(loc='upper left')
    plt.ylim(0, 100)
    plt.xlim(1, max_gen)
    plt.tight_layout()
    plt.savefig(out_file, format='pdf')  
    plt.clf()  


def draw_box_chart_for_breeding(file_name, game_name, sim_names, out_file, breed_size = 1000, breed_metric = "good_children", sim_labels = None, y_label = None):
    ''' Aggregates number of good children (or other breed_metric) accross all run and compute bar chart on runs '''
    with open(file_name, 'r') as f:
        json_lines = f.readlines()
    metrics = [json.loads(l) for l in json_lines]
    stats = {}
    for m in metrics:
        game = m['game_name']
        sim = m['sim_name']
        if sim not in sim_names or game != game_name:
            continue
        metric = m[breed_metric]
        metric_sum = np.sum(metric)
        num_breeds = breed_size * len(metric)
        stats.setdefault(sim, []).append(metric_sum * 100.0 / num_breeds)


    plt.ioff()
    if sim_labels is None:
        sim_labels = sim_names
    
    data = [stats[sim] for sim in sim_names]
    v = plt.violinplot(data, showmedians=True, showextrema=False)
    for pc in v['bodies']:
        pc.set_facecolor("#777")
        pc.set_edgecolor('black')
        # pc.set_alpha(1) 
        # 
    v['cmedians'].set_color("#777")
    v['cmedians'].set_alpha(1)
    # add_label(v, params['label'])

    # plt.set_xticks(np.arange(1, len(sim_labels) + 1))
    plt.xticks(np.arange(1, len(sim_labels) + 1), sim_labels)

    # plt.xlabel('Generation')
    plt.ylabel(y_label or (breed_metric + ', \\%'))
    # plt.title('Stacked Area Chart with Smooth Lines')
    # plt.legend(loc='upper left')
    # plt.ylim(0, 100)
    # plt.xlim(1, max_gen)
    plt.tight_layout()
    plt.savefig(out_file, format='pdf')  
    plt.clf()  

def draw_bar_charts_of_breeding(file_name, game_names, sim_names, out_file, breed_size = 1000, breed_metric = "good_children", sim_labels = None, y_label = None,
                                    delta = 3, prec = 0):
    with open(file_name, 'r') as f:
        json_lines = f.readlines()
    metrics = [json.loads(l) for l in json_lines]
    stats = defaultdict(int)
    num_breeds = defaultdict(int)
    for m in metrics:
        game = m['game_name']
        sim = m['sim_name']
        if sim not in sim_names or game not in game_names:
            continue
        metric = np.array(m[breed_metric], dtype=float)[1:]
        num_breeds[(game, sim)] = num_breeds.get((game, sim), 0) + len(metric) * breed_size
        stats[(game, sim)] = stats.get((game, sim), 0) + np.sum(metric)


    x = np.arange(len(game_names))
    width = 0.18
    multiplier = 0

    sim_stats = {}
    for sim in sim_names:
        sim_stats[sim] = [round(stats[(game, sim)] * 100.0 / num_breeds[(game, sim)], prec) for game in game_names]

    fig, ax = plt.subplots(layout='constrained') #figsize=(10, chart_height)

    plt.ioff()
    max_y = 0
    for sim_id, sim in enumerate(sim_stats):
        game_metrics = sim_stats[sim]
        offset = width * multiplier
        rects = ax.bar(x + offset, game_metrics, width, label=sim_labels[sim_id])
        max_y = max(max_y, max(game_metrics))
        ax.bar_label(rects, padding=1, fmt=lambda x: ("0" if x < 0.1 else f"{x:.1f}"), fontsize=8)
        multiplier += 1    

    ax.set_ylabel(y_label)
    # ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + 2 * width, game_names)
    ax.legend(loc='upper left', ncols=len(sim_names))
    ax.set_ylim(0, max_y + delta)

    fig.tight_layout()
    fig.savefig(out_file, format='pdf')  
    plt.clf()  


# sim_names = [*gp_sim_names, *nsga2_sim_names, *cov_sim_names, *coevol_sim_names]

def compute_breedings(file_name: str, out_file, game_names, sim_names = None,
                      breed_metric_names = ['good_children', 'best_children', 'good_dom_children', 'best_dom_children',
                                        'bad_children'], col_names = None,
                                        subrow_names = None,
                      breed_size = 1000,
                      aggregate = None):
    with open(file_name, 'r') as f:
        json_lines = f.readlines()
    metrics = [json.loads(l) for l in json_lines]
    stats = {}
    num_breedings = defaultdict(int)
    for m in metrics:
        game = m['game_name']
        sim = m['sim_name']
        if game not in game_names or (sim_names is not None and sim not in sim_names):
            continue
        is_first = True
        for breed_metric in breed_metric_names:
            metric = np.array(m[breed_metric])[1:]
            if is_first:
                num_breedings[(game, sim)] += len(metric) * breed_size
                is_first = False
            stats[(game, sim, breed_metric)] = stats.get((game, sim, breed_metric), 0) + np.sum(metric)

    if col_names is None:
        col_names = sim_names
    if subrow_names is None:
        subrow_names = breed_metric_names

    if aggregate == "problems":
        new_stats = defaultdict(int)
        new_num_breedings = defaultdict(int)
        for (g, s, m), v in stats.items():
            new_stats[(s, m)] += v
        for (g, s), v in num_breedings.items():
            new_num_breedings[s] += v

        stats = new_stats
        num_breedings = new_num_breedings

        for (s, m), v in stats.items():
            res = v * 100.0 / num_breedings[s]
            prec = 2 if res < 10 else 0
            res = round(res, prec)
            stats[(s, m)] = res

        rows = []
        for breed_metric, breed_metric_name in zip(breed_metric_names, subrow_names):
            row = []
            row.append(breed_metric_name)
            for sim, sim_name in zip(sim_names, col_names):
                row.append(stats.get((sim, breed_metric), ""))


            rows.append(row)

        with open(out_file, 'w') as f:
            print(tabulate(rows, headers=["Breeding", *col_names], tablefmt='latex', numalign="center", stralign="center"), file=f)

    elif aggregate == "simulations":

        new_stats = defaultdict(int)
        new_num_breedings = defaultdict(int)
        for (g, s, m), v in stats.items():
            new_stats[(g, m)] += v
        for (g, s), v in num_breedings.items():
            new_num_breedings[g] += v

        stats = new_stats
        num_breedings = new_num_breedings

        for (g, m), v in stats.items():
            res = v * 100.0 / num_breedings[g]
            prec = 2 if res < 10 else 0
            res = round(res, prec)
            stats[(g, m)] = res

        rows = []
        for breed_metric, breed_metric_name in zip(breed_metric_names, subrow_names):
            row = []
            row.append(breed_metric_name)
            for game in game_names:
                row.append(stats.get((game, breed_metric), ""))


            rows.append(row)

        col_names = game_names

        with open(out_file, 'w') as f:
            print(tabulate(rows, headers=["Breeding", *col_names], tablefmt='latex', numalign="center", stralign="center"), file=f)

    else:
        for (g, s, m), v in stats.items():
            res = v * 100.0 / num_breedings[(g, s)]
            prec = 2 if res < 10 else 0
            res = round(res, prec)
            stats[(g, s, m)] = res

        rows = []
        for game in game_names:        

            is_first = True
            for breed_metric, breed_metric_name in zip(breed_metric_names, subrow_names):
                row = []
                if is_first:
                    row.append(game)
                    is_first = False
                else:
                    row.append("")
                row.append(breed_metric_name)
                for sim, sim_name in zip(sim_names, col_names):
                    row.append(stats.get((game, sim, breed_metric), ""))


                rows.append(row)

        with open(out_file, 'w') as f:
            print(tabulate(rows, headers=["Problem", "Breeding", *col_names], tablefmt='latex', numalign="center", stralign="center"), file=f)

    # return stats


# sim_names = ["ifs"] #gp_sim_names #["do_rand", "do_fo", "doc_p", "doc_d"]
# sel_sim_names = ["coevol_uo_40", "coevol_uo2_50", "ifs", "ifs_0", "doc_d_0", "doc_p_0", "dof_w_3_0", "dof_wh_3_0", "cov_ht_bp", "cov_rt_bp", "gp", "gp_0", "doc_d", "doc_p", "dof_w_3", "dof_wh_3", 'do_nsga', 'do_nsga_0']
if __name__ == "__main__":
    print("testing evo runs")
    compute_run_stats("data/test-based-gp/coevol6.jsonlist", 
                      sim_names=None, tablefmt="github", transpose=False) #['do_rand', 'do_nsga', 'doc', 'doc_p', 'doc_d', 'dof_w_2', 'dof_w_3', 'dof_wh_2', 'dof_wh_3', 'dof_w_2_80', 'dof_w_3_80', 'dof_wh_2_80', 'dof_wh_3_80', 'do_fo', 'do_pca_abs_2', 'do_pca_abs_3', 'do_pca_diff_2', 'do_pca_diff_3',])
    # compute_run_stats("data/test-based-gp/all-all-circe.jsonlist", 
    #                   sim_names=None, tablefmt="github", transpose=True) #, custom_prefix="sel-")
    # pass
    # cnt = 0
    # from gp_benchmarks import all_discrete_benchmarks
    # f = open("setups.txt", "w")
    # for sim_name in coevol.coevol_sim_names:
    #     for b_name in all_discrete_benchmarks().keys():
    #         print(sim_name, b_name)
    #         print(f"'{sim_name}:{b_name}'", file=f)
    #         cnt += 1
    # print(f"total setups: {cnt}")
    # cov_ht_bp(idx = 11)
    # get_sol_dupl("data/test-based-gp/all.jsonlist")
    # compute_run_stats("data/test-based-gp/all.jsonlist", sim_names=["gp","ifs","coevol_uo_40","doc_p","doc_d","dof_w_3","dof_wh_3","do_nsga","cov_ht_bp","cov_rt_bp"])
    # for game_name in ["cmp6", "cmp8", "disc1", "disc2", "disc3", "disc4", "disc5", "maj6", "malcev1", "malcev2", "malcev3", "malcev4", "malcev5", "mux6", "par5"]:
    #     draw_line_plot("data/test-based-gp/all.jsonlist", game_name, "hamming_distance_fitness", 
    #                     sim_names=["coevol_uo_40", "ifs", "doc_p", "doc_d", "gp"], 
    #                     out_file=f"data/test-based-gp/hamming-distance-{game_name}.pdf", 
    #                     metric_label="Hamming distance",
    #                     sim_labels=["$CSE_{40}$", "$IFS$", "$DOC_{p}$", "$DOC_{d}$", "$GP$"],
    #                     max_metric_value=None,
    #                     plot_settings= {"gp": {"color": "black", "linestyle": "--", "linewidth": 0.5}})

    # for game_name in ["cmp6", "cmp8", "disc1", "disc2", "disc3", "disc4", "disc5", "maj6", "malcev1", "malcev2", "malcev3", "malcev4", "malcev5", "mux6", "par5"]:
    #     draw_line_plot2("data/test-based-gp/all.jsonlist", game_name, "syntax_dupl", 
    #                     sim_names=["coevol_uo_40", "ifs", "doc_p", "doc_d", "gp"], 
    #                     out_file=f"data/test-based-gp/syntax-dupl-{game_name}.pdf", 
    #                     metric_label="Syntax duplicates, \\%",
    #                     sim_labels=["$CSE_{40}$", "$IFS$", "$DOC_{p}$", "$DOC_{d}$", "$GP$"],
    #                     max_metric_value=None, norm_metric="stats_nodes",
    #                     plot_settings= {"gp": {"color": "black", "linestyle": "--", "linewidth": 0.5}})    

    # for game_name in ["cmp6", "cmp8", "disc1", "disc2", "disc3", "disc4", "disc5", "maj6", "malcev1", "malcev2", "malcev3", "malcev4", "malcev5", "mux6", "par5"]:
    #     draw_line_plot2("data/test-based-gp/all.jsonlist", game_name, "sem_dupl", 
    #                     sim_names=["coevol_uo_40", "ifs", "doc_p", "doc_d", "gp"], 
    #                     out_file=f"data/test-based-gp/sem-dupl-{game_name}.pdf", 
    #                     metric_label="Semantic duplicates, \\%",
    #                     sim_labels=["$CSE_{40}$", "$IFS$", "$DOC_{p}$", "$DOC_{d}$", "$GP$"],
    #                     max_metric_value=None, norm_metric="stats_nodes",
    #                     plot_settings= {"gp": {"color": "black", "linestyle": "--", "linewidth": 0.5}})    

    # for game_name in ["cmp6","cmp8", "disc1", "disc2", "disc3", "disc4", "disc5", "maj6", "malcev1", "malcev2", "malcev3", "malcev4", "malcev5", "mux6", "par5"]:
    #     draw_line_plot2("data/test-based-gp/all.jsonlist", game_name, "sem_repr_rate", 
    #                     sim_names=["coevol_uo_40","ifs", "doc_p", "doc_d", "gp"], 
    #                     out_file=f"data/test-based-gp/num-syntax-per-sem-{game_name}.pdf", 
    #                     metric_label="Num. syntaxes per semantics",
    #                     sim_labels=["$CSE_{40}$", "$IFS$", "$DOC_{p}$", "$DOC_{d}$", "$GP$"],
    #                     max_metric_value=None, norm_metric=None, min_metric_value = 1,
    #                     plot_settings= {"gp": {"color": "black", "linestyle": "--", "linewidth": 0.5}},
    #                     log_scale=True)   

    # for game_name in ["cmp6", "cmp8", "disc1", "disc2", "disc3", "disc4", "disc5", "maj6", "malcev1", "malcev2", "malcev3", "malcev4", "malcev5", "mux6", "par5"]:
    #     draw_line_plot2("data/test-based-gp/all.jsonlist", game_name, "num_uniq_syntaxes", 
    #                     sim_names=["coevol_uo_40", "ifs", "doc_p", "doc_d", "gp"], 
    #                     out_file=f"data/test-based-gp/uniq-syntax-{game_name}.pdf", 
    #                     metric_label="Unique syntaxes, \\%",
    #                     sim_labels=["$CSE_{40}$", "$IFS$", "$DOC_{p}$", "$DOC_{d}$", "$GP$"],
    #                     max_metric_value=None, norm_metric="stats_nodes",
    #                     plot_settings= {"gp": {"color": "black", "linestyle": "--", "linewidth": 0.5}})    

    # for game_name in ["cmp6", "cmp8", "disc1", "disc2", "disc3", "disc4", "disc5", "maj6", "malcev1", "malcev2", "malcev3", "malcev4", "malcev5", "mux6", "par5"]:
    #     draw_line_plot2("data/test-based-gp/all.jsonlist", game_name, "num_uniq_sems", 
    #                     sim_names=["coevol_uo_40", "ifs", "doc_p", "doc_d", "gp"], 
    #                     out_file=f"data/test-based-gp/uniq-sem-{game_name}.pdf", 
    #                     metric_label="Unique semantics, \\%",
    #                     sim_labels=["$CSE_{40}$", "$IFS$", "$DOC_{p}$", "$DOC_{d}$", "$GP$"],
    #                     max_metric_value=None, norm_metric="stats_nodes",
    #                     plot_settings= {"gp": {"color": "black", "linestyle": "--", "linewidth": 0.5}})     
    # 
    # for game_name in ["cmp6", "cmp8", "disc1", "disc2", "disc3", "disc4", "disc5", "maj6", "malcev1", "malcev2", "malcev3", "malcev4", "malcev5", "mux6", "par5"]:
    #     draw_line_plot2("data/test-based-gp/all.jsonlist", game_name, "best_children", 
    #                     sim_names=["coevol_uo_40", "ifs", "doc_p", "doc_d", "gp"], 
    #                     out_file=f"data/test-based-gp/best-children-{game_name}.pdf", 
    #                     metric_label="Num. better children than all parents",
    #                     sim_labels=["$CSE_{40}$", "$IFS$", "$DOC_{p}$", "$DOC_{d}$", "$GP$"],
    #                     max_metric_value=None, norm_metric=None,
    #                     plot_settings= {"gp": {"color": "black", "linestyle": "--", "linewidth": 0.5}}) #   

    # for game_name in ["cmp6", "cmp8", "disc1", "disc2", "disc3", "disc4", "disc5", "maj6", "malcev1", "malcev2", "malcev3", "malcev4", "malcev5", "mux6", "par5"]:
    #     draw_line_plot2("data/test-based-gp/all.jsonlist", game_name, "good_children", 
    #                     sim_names=["coevol_uo_40", "ifs", "doc_p", "doc_d", "gp"], 
    #                     out_file=f"data/test-based-gp/good-children-{game_name}.pdf", 
    #                     metric_label="Num. better children than any parent",
    #                     sim_labels=["$CSE_{40}$", "$IFS$", "$DOC_{p}$", "$DOC_{d}$", "$GP$"],
    #                     max_metric_value=None, norm_metric=None,
    #                     plot_settings= {"gp": {"color": "black", "linestyle": "--", "linewidth": 0.5}}) #       
        
    # for game_name in ["cmp6", "cmp8", "disc1", "disc2", "disc3", "disc4", "disc5", "maj6", "malcev1", "malcev2", "malcev3", "malcev4", "malcev5", "mux6", "par5"]:
    #     draw_line_plot2("data/test-based-gp/all.jsonlist", game_name, "bad_children", 
    #                     sim_names=["coevol_uo_40", "ifs", "doc_p", "doc_d", "gp"], 
    #                     out_file=f"data/test-based-gp/bad-children-{game_name}.pdf", 
    #                     metric_label="Num. children, worse than parents",
    #                     sim_labels=["$CSE_{40}$", "$IFS$", "$DOC_{p}$", "$DOC_{d}$", "$GP$"],
    #                     max_metric_value=None, norm_metric=None,
    #                     plot_settings= {"gp": {"color": "black", "linestyle": "--", "linewidth": 0.5}}) #         

    # cov_et_bp
    # for game_name in ["cmp6", "cmp8", "disc1", "disc2", "disc3", "disc4", "disc5", "maj6", "malcev1", "malcev2", "malcev3", "malcev4", "malcev5", "mux6", "par5"]:
    #     draw_box_chart_for_breeding("data/test-based-gp/all.jsonlist", game_name, sim_names=["coevol_uo_40", "ifs", "doc_p", "doc_d", "gp"], 
    #                                 out_file=f"data/test-based-gp/box-bad-children-{game_name}.pdf", 
    #                                 breed_size=1000, breed_metric="bad_children", sim_labels=["$CSE_{40}$", "$IFS$", "$DOC_{p}$", "$DOC_{d}$", "$GP$"], y_label="Bad children, \\%")

    # for game_name in ["cmp6", "cmp8", "disc1", "disc2", "disc3", "disc4", "disc5", "maj6", "malcev1", "malcev2", "malcev3", "malcev4", "malcev5", "mux6", "par5"]:
    #     draw_line_plot3("data/test-based-gp/all.jsonlist", game_name, "good_children", "bad_children", 
    #                     norm_value=1000,
    #                     sim_names=["coevol_uo_40", "ifs", "doc_p", "doc_d", "gp"], 
    #                     out_file=f"data/test-based-gp/pair-good-bad-children-{game_name}.pdf", 
    #                     metric_label="Good and suboptimal breedings, \\%",
    #                     sim_labels=["$CSE_{40}$", "$IFS$", "$DOC_{p}$", "$DOC_{d}$", "$GP$"],
    #                     colors = ["red", "blue", "green", "orange", "black"],
    #                     max_metric_value=None,
    #                     suffix1 = "good", suffix2 = "suboptimal",)
    #                     # plot_settings= {"gp": {"color": "black", "linestyle": "--", "linewidth": 0.5}}) #         

    # draw_bar_charts_of_breeding("data/test-based-gp/all.jsonlist", 
    #         ["cmp8", "maj6", "mux6", "par5", "disc4", "malcev4"],
    #         sim_names=["coevol_uo_40", "ifs", "doc_p", "doc_d", "gp"],
    #         out_file=f"data/test-based-gp/bar-good-children.pdf",
    #         breed_size=1000, breed_metric="good_children",
    #         sim_labels=["$CSE_{40}$", "$IFS$", "$DOC_{p}$", "$DOC_{d}$", "$GP$"],
    #         y_label="Good breedings, \\%")

    # draw_bar_charts_of_breeding("data/test-based-gp/all.jsonlist", 
    #         ["cmp8", "maj6", "mux6", "par5", "disc4", "malcev4"],
    #         sim_names=["coevol_uo_40", "ifs", "doc_p", "doc_d", "gp"],
    #         out_file=f"data/test-based-gp/bar-bad-children.pdf",
    #         breed_size=1000, breed_metric="bad_children",
    #         sim_labels=["$CSE_{40}$", "$IFS$", "$DOC_{p}$", "$DOC_{d}$", "$GP$"],
    #         y_label="Suboptimal breedings, \\%")    

    # draw_bar_charts_of_breeding("data/test-based-gp/all.jsonlist", 
    #         ["cmp8", "maj6", "mux6", "par5", "disc4", "malcev4"],
    #         sim_names=["coevol_uo_40", "ifs", "doc_p", "doc_d", "gp"],
    #         out_file=f"data/test-based-gp/bar-good-dom-children.pdf",
    #         breed_size=1000, breed_metric="good_dom_children",
    #         sim_labels=["$CSE_{40}$", "$IFS$", "$DOC_{p}$", "$DOC_{d}$", "$GP$"],
    #         y_label="Breeding of dominating child, \\%",
    #         delta = 0.5, prec = 3)

    # compute_breedings("data/test-based-gp/all.jsonlist", "data/test-based-gp/breedings.tex", 
    #                   ["mux6", "maj6", "cmp6", "malcev5", "malcev3", "par5", "malcev2", "malcev1", "disc3", "malcev4", "cmp8", "disc2", "disc1", "disc5", "disc4"],
    #                     sim_names = ["coevol_uo_40", "ifs", "doc_p", "doc_d", "dof_w_3", "cov_ht_bp", "dof_wh_3", "cov_rt_bp", "gp", "do_nsga"],
    #                     breed_metric_names = ["good_children", "best_children", "good_dom_children", "best_dom_children", "bad_children"],
    #                     col_names = ["$CSE_{40}$", "$IFS$", "$DOC_{p}$", "$DOC_{d}$", "$DOF_{w}$", "$HTBP$", "$DOF_{wh}$", "$RTBP$", "$GP$", "$NSGA$"],
    #                     subrow_names = ["Good", "Best", "Good dom","Best dom", "Bad"])                        

    # compute_breedings("data/test-based-gp/all.jsonlist", "data/test-based-gp/breedings-sim.tex", 
    #                   ["mux6", "maj6", "cmp6", "malcev5", "malcev3", "par5", "malcev2", "malcev1", "disc3", "malcev4", "cmp8", "disc2", "disc1", "disc5", "disc4"],
    #                     sim_names = ["coevol_uo_40", "ifs", "doc_p", "doc_d", "dof_w_3", "cov_ht_bp", "dof_wh_3", "cov_rt_bp", "gp", "do_nsga"],
    #                     breed_metric_names = ["good_children", "best_children", "good_dom_children", "best_dom_children", "bad_children"],
    #                     col_names = ["$CSE_{40}$", "$IFS$", "$DOC_{p}$", "$DOC_{d}$", "$DOF_{w}$", "$HTBP$", "$DOF_{wh}$", "$RTBP$", "$GP$", "$NSGA$"],
    #                     subrow_names = ["Good", "Best", "Good dom","Best dom", "Bad"],
    #                     aggregate = "problems")   

    # compute_breedings("data/test-based-gp/all.jsonlist", "data/test-based-gp/breedings-prob.tex", 
    #                   ["mux6", "maj6", "cmp6", "malcev5", "malcev3", "par5", "malcev2", "malcev1", "disc3", "malcev4", "cmp8", "disc2", "disc1", "disc5", "disc4"],
    #                     sim_names = ["coevol_uo_40", "ifs", "doc_p", "doc_d", "dof_w_3", "cov_ht_bp", "dof_wh_3", "cov_rt_bp", "gp", "do_nsga"],
    #                     breed_metric_names = ["good_children", "best_children", "good_dom_children", "best_dom_children", "bad_children"],
    #                     col_names = ["$CSE_{40}$", "$IFS$", "$DOC_{p}$", "$DOC_{d}$", "$DOF_{w}$", "$HTBP$", "$DOF_{wh}$", "$RTBP$", "$GP$", "$NSGA$"],
    #                     subrow_names = ["Good", "Best", "Good dom","Best dom", "Bad"],
    #                     aggregate = "simulations")       
    pass