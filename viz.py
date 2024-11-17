''' Module dedicated to visualization of dynamics of simulation process '''
# First we would like to see how individuals change in number games - an image of 2d plane with individuals as dots 
#     Candidate-parent --> candidate-child AND test-parent --> test-child = test log of them and who wins 
#     Outputs pngs on given folder to combine them in gif later 

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.font_manager import FontProperties

import json
from scipy import stats
import scikit_posthocs as sp
import numpy as np

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{times}"
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = ['Times']    

# legend_font = FontProperties(family='monospace', size=6)
param_img_folder = "img"

def draw_populations(point_groups, xrange = None, yrange = None, name="fig", fmt = "png", title = None, matrix = None):
    ''' creates an image of populaltion of 2-d numbers to track the dynamics '''
    # classes = {
    #     "prev": dict(marker='o', s=8, c='#cfcfcf', alpha=0.5),
    #     "ind": dict(marker='o', s=30, c='#151fd6'),
    #     "child": dict(marker='o', s=10, c='#28a4c9'),
    #     "cand": dict(marker='o', s=30, c='#151fd6'),
    #     "cand_child": dict(marker='o', s=10, c='#28a4c9'),
    #     "test": dict(marker='H', s=30, c='#159e1b'),
    #     "test_child": dict(marker='H', s=10, c='#85ba6a'),
    #     "target": dict(marker='x', s=20, c='#bf5a17')
    # }
    # data = [(g['xy'], g.get('legend', []), classes[g.get('class', 'ind')], 0 if g.get('class', 'ind') == 'prev' else 1) 
    #             for g in point_groups if len(g.get('xy', [])) > 0]
    # data.sort(key=lambda x: x[-1])
    plt.ioff()
    handles = []
    labels = []
    point_groups.sort(key=lambda g:0 if "bg" in g else 1)
    if matrix is not None:
        plt.imshow(matrix, aspect='auto', interpolation='nearest')
    for g in point_groups:
        if len(g["xy"]) == 0:
            continue
        x, y = zip(*g["xy"])
        classes = g.get("class", {})
        if "bg" in g:
            classes = {**dict(marker='o', s=20, c='gray', alpha=0.5, edgecolor="white"), **classes}
        scatter = plt.scatter(x, y, **classes)
        for label in g.get("legend", []):
            handles.append(scatter)
            labels.append(label)
    if xrange is not None: 
        plt.xlim(xrange[0], xrange[1])
    if yrange is not None: 
        plt.ylim(yrange[0], yrange[1])  
    plt.legend(handles = handles, labels = labels, loc='upper left', bbox_to_anchor=(1, 1), handletextpad = 0, labelspacing=0.5, markerscale=1, fontsize="small")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{param_img_folder}/{name}.{fmt}", format=fmt)    
    plt.clf()

def draw_metrics(metrics_file: str, metrics = ["DC", "ARR", "ARRA", "Dup", "R"],
                    aggregation = "last", sim_names = [], game_names = [], fixed_max = {}, fixed_mins = {}, rename = {}, stats_file = None):
    ''' metrix_file is in jsonlist format'''
    with open(metrics_file, "r") as f:
        lines = f.readlines()
    runs = [json.loads(line) for line in lines ]
    groups = {}
    present_sim_name = set()
    present_game_name = set()
    for run in runs: 
        sim_name = run["sim_name"]
        game_name = run["game_name"]
        if game_name == "IntransitiveRegionGame" and run["param_sel_size"] == 5:
            continue
        if len(sim_names) > 0 and sim_name not in sim_names:
            continue
        if len(game_names) > 0 and game_name not in game_names:
            continue
        present_sim_name.add(sim_name)
        present_game_name.add(game_name)
        for m in metrics:
            key = (game_name, m)
            groups.setdefault(key, {}).setdefault(sim_name, []).append(run["metric_" + m])
    present_sim_name = list(present_sim_name) if len(sim_names) == 0 else sim_names
    if len(game_names) == 0:
        game_names = list(present_game_name)
    for (game_name, metric_name), sim_values in groups.items():
        for sim_name, values in sim_values.items():
            new_values = []
            if aggregation == "last": #collect final metric value - bar or violinplot chart
                for i in range(len(values)):
                    new_values.append(values[i][-1])
            else: #for line plot and area between with each batch
                zipped = list(zip(*values))
                for i in range(len(zipped)):
                    i_values = zipped[i]
                    mean = np.mean(i_values)
                    if aggregation == "all":
                        confidence_level = 0.95
                        degrees_freedom = len(i_values) - 1
                        sample_standard_error = stats.sem(i_values)
                        confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
                        min_v = mean if np.isnan(confidence_interval[0]) else confidence_interval[0]
                        max_v = mean if np.isnan(confidence_interval[1]) else confidence_interval[1]
                        new_values.append((min_v, mean, max_v))
                    else:
                        new_values.append(mean)
            sim_values[sim_name] = new_values

    for metric_name in metrics:
        for game_name in game_names:
            sim_values = groups[(game_name, metric_name)]
            plt.ioff()
            fig, ax = plt.subplots() 
            if aggregation == "last": #violin plot
                # plt.axhline(y = 50, color = 'tab:gray', linewidth=0.5, linestyle = 'dashed')
                import matplotlib.patches as mpatches
                labels = []
                colors = colormaps.get_cmap("Pastel1").colors
                # for sim_id, sim_name in enumerate(present_sim_name):
                present_sim_name.sort(key = lambda x: np.mean(sim_values[x]))
                data = [[v * 100 for v in sim_values[sim_name]] for sim_name in present_sim_name]
                # v = ax.violinplot(data, showextrema = False, showmedians=True)
                v = ax.boxplot(data)
                # for sim_id, pc in enumerate(v['bodies']):
                #     color = colors[sim_id % len(colors)]
                #     pc.set_facecolor(color)
                #     pc.set_edgecolor('black')
                #     labels.append((mpatches.Patch(color=color), present_sim_name[sim_id]))
                # v['cmedians'].set_color('black')
                # v['cmedians'].set_alpha(1)
                # plt.legend(*zip(*labels), loc='upper left', bbox_to_anchor=(1, 1))
                # plt.xlabel('Spaces', size=14)
                ax.set_ylabel(f'{metric_name}, \%', size=14)   
                # ax.tick_params(axis='both', which='major', labelsize=14)            
                ax.yaxis.grid(True)
                # ax.set_xticklabels(["", *present_sim_name])  
                ax.set_xticks([y + 1 for y in range(len(data))],
                    labels=[rename.get(x, x) for x in present_sim_name])#      
                fig.set_tight_layout(True)      
                fig.savefig(f"data/plots/{aggregation}-{metric_name}-{game_name}.pdf", format='pdf')    
                stats_data = np.array([sim_values[sim_name] for sim_name in present_sim_name])
                friedman_res = stats.friedmanchisquare(*stats_data)
                nemenyi_res = sp.posthoc_nemenyi_friedman(stats_data.T) 
                print(f"\n----------------------------------", file=stats_file)
                print(f"Stat result for {game_name}, metric {metric_name}", file=stats_file)
                print(f"Friedman: {friedman_res}", file=stats_file)
                from tabulate import tabulate
                names = present_sim_name
                rows = []
                for i in range(len(names)):
                    row = []
                    row.append(names[i])
                    for j in range(len(names)):
                        row.append(nemenyi_res[i][j])
                    rows.append(row)
                print(tabulate(rows, headers=["", *names], tablefmt="grid", numalign="center", stralign="center"), file=stats_file)
                        
            else: #line and area chart for confidence interval     
                min_y = fixed_mins.get(metric_name, 100)
                max_y = fixed_max.get(metric_name, 0)
                for sim_name in present_sim_name:
                    values = sim_values[sim_name]
                    if aggregation == "all":
                        data = [v[1] * 100 for v in values]
                        lower = [v[0] * 100 for v in values]
                        upper = [v[2] * 100 for v in values]
                        cur_min = min(lower)
                        if cur_min < min_y:
                            min_y = cur_min
                        cur_max = max(upper)
                        if cur_max > max_y:
                            max_y = cur_max
                        ax.fill_between(range(1, len(data) + 1), lower, upper, alpha=.1, linewidth=0)
                    else:                    
                        data = [v * 100 for v in values]
                        cur_min = np,min(data)
                        if cur_min < min_y:
                            min_y = cur_min    
                        cur_max = max(data)
                        if cur_max > max_y:
                            max_y = cur_max           
                    ax.plot(range(1, len(data) + 1), data, label=rename.get(sim_name, sim_name), linewidth=1, markersize=1, marker='o')
                ax.set_xlim(0, 100)
                ax.set_ylim(min_y, max_y)
                plt.legend(loc='lower right', handletextpad = 0.2)
                plt.xlabel('Simulated time', size=14)
                plt.ylabel(f'{metric_name}, \%', size=14)
                ax.tick_params(axis='both', which='major', labelsize=14)            
                fig.set_tight_layout(True)
                fig.savefig(f"data/plots/{aggregation}-{metric_name}-{game_name}.pdf", format='pdf')
            plt.clf()


def draw_setups(metrics_file: str, sim_name = "rand", metric_name = "ARRA", prefix = "", threshold = 0.9,
                    game_names = [], fixed_max = None, fixed_min = None, rename = {}):
    ''' metrics_file is in jsonlist format'''
    with open(metrics_file, "r") as f:
        lines = f.readlines()
    runs = [json.loads(line) for line in lines ]
    values = [[] for _ in game_names]
    game_names_map = {game_name: i for i, game_name in enumerate(game_names)}
    for run in runs: 
        cur_sim_name = run["sim_name"]
        game_name = run["game_name"]
        if game_name == "IntransitiveRegionGame" and run["param_sel_size"] == 5:
            continue        
        if cur_sim_name != sim_name or game_name not in game_names:
            continue
        cur_values = run["metric_" + metric_name]
        values[game_names_map[game_name]].append(cur_values)
    
    for i in range(len(values)):
        setup_values = values[i]
        new_values = []
        for i_values in zip(*setup_values):
            confidence_level = 0.95
            mean = np.mean(i_values)
            degrees_freedom = len(i_values) - 1
            sample_standard_error = stats.sem(i_values)
            confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
            min_v = mean if np.isnan(confidence_interval[0]) else confidence_interval[0]
            max_v = mean if np.isnan(confidence_interval[1]) else confidence_interval[1]
            new_values.append((min_v, mean, max_v))
        values[i] = new_values

    plt.ioff()
    fig, ax = plt.subplots() 
    min_y = fixed_min or 100
    max_y = fixed_max or 0
    # draw horizontal dash line on level of threshold
    plt.axhline(y = threshold * 100, color = 'tab:gray', linewidth=0.5, linestyle = 'dashed')
    for game_name, game_values in zip(game_names, values):
        data = [v[1] * 100 for v in game_values]
        lower = [v[0] * 100 for v in game_values]
        upper = [v[2] * 100 for v in game_values]
        cur_min = min(lower)
        if cur_min < min_y:
            min_y = cur_min
        cur_max = max(upper)
        if cur_max > max_y:
            max_y = cur_max
        ax.fill_between(range(1, len(data) + 1), lower, upper, alpha=.1, linewidth=0)
        ax.plot(range(1, len(data) + 1), data, label=rename.get(game_name, game_name), linewidth=1, markersize=1, marker='o')
    ax.set_xlim(0, 100)
    ax.set_ylim(min_y, max_y)
    plt.legend(loc='lower right', handletextpad = 0.2)
    plt.xlabel('Simulated time', size=14)
    plt.ylabel(f'{sim_name}, {metric_name}, \%', size=14)
    ax.tick_params(axis='both', which='major', labelsize=14)            
    fig.set_tight_layout(True)
    fig.savefig(f"data/plots/{prefix}-{metric_name}-{sim_name}.pdf", format='pdf')
    plt.clf()


def draw_latex_mean_std_tbl(metrics_file: str, metric_name = "ARRA", sim_names = [], game_names = ["CompareOnOneGame"],
                                table_file=None, p_value = 0.05, name_remap = {},
                                exclude_sim_names = ["pl-d-2", "pl-d-5", "pl-d-100", "pl-l-2", "pl-l-5", "pl-l-100"]):
    ''' metrix_file is in jsonlist format'''
    with open(metrics_file, "r") as f:
        lines = f.readlines()
    runs = [json.loads(line) for line in lines ]
    groups = {}
    sim_names = set(sim_names)
    present_sim_names = set()
    exclude_sim_names = set(exclude_sim_names)
    for run in runs: 
        sim_name = run["sim_name"]
        game_name = run["game_name"]
        if game_name == "IntransitiveRegionGame" and run["param_sel_size"] == 5:
            continue      
        if sim_name in exclude_sim_names:
            continue  
        if len(sim_names) > 0 and sim_name not in sim_names:
            continue
        if game_name not in game_names:
            continue
        key = (sim_name, game_name)
        present_sim_names.add(sim_name)
        groups.setdefault(key, []).append(run["metric_" + metric_name])    
    present_sim_names = list(present_sim_names)
    group_by_sims = {}
    for (sim_name, game_name), all_values in groups.items():
        new_values = [run_values[-1] for run_values in all_values]
        mean = np.mean(new_values)
        std = np.std(new_values)
        group_by_sims.setdefault(sim_name, {})[game_name] = (mean, std, new_values)
        # groups[(sim_name, game_name, metric_name)] = (mean, std)
    for game_name in game_names:
        data = [group_by_sims[sim_name][game_name][2] for sim_name in present_sim_names]
        diffs = [[1 for _ in present_sim_names] for _ in present_sim_names]
        for i in range(len(diffs)):
            for j in range(i + 1, len(diffs)):
                _, pvalue = stats.mannwhitneyu(data[i], data[j])
                if pvalue < p_value:
                    diffs[i][j] = pvalue
                    diffs[j][i] = pvalue
        sim_ranks = {}
        # building domination relation by nemenyi test p-value and means 
        # to rank we need to build all domination chains and average ranks in them 
        dominations = {sim_name: set() for sim_name in present_sim_names}
        dominated_by = {sim_name: set() for sim_name in present_sim_names}
        # NOTE: sign depends on metric - ARRA >, Dup <, R <, DC >, ARR >
        f = (lambda x: x[0] > x[1]) if metric_name in [ "ARRA", "DC", "ARR"] else (lambda x: x[0] < x[1])
        for i in range(len(present_sim_names)):
            sim_name = present_sim_names[i]
            for j in range(i + 1, len(present_sim_names)):
                sim_name2 = present_sim_names[j]
                if diffs[i][j] < p_value:
                    if f((group_by_sims[sim_name][game_name][0], group_by_sims[sim_name2][game_name][0])):
                        dominations[sim_name].add(sim_name2)
                        dominated_by[sim_name2].add(sim_name)
                    else:
                        dominations[sim_name2].add(sim_name)
                        dominated_by[sim_name].add(sim_name2)  

        rank = 0
        while len(dominations) > 0:
            sources = [sim_name for sim_name, v in dominated_by.items() if len(v) == 0]  
            for sim_name in sources:
                sim_ranks[sim_name] = rank
                for sim_name2 in dominations.get(sim_name, []):
                    dominated_by[sim_name2].remove(sim_name)
                if sim_name in dominations:
                    del dominations[sim_name]
                del dominated_by[sim_name]
            rank += 1

        max_rank = rank

        for sim_name in present_sim_names:
            sim_rank = sim_ranks.get(sim_name, 1)
            mean, std, _ = group_by_sims[sim_name][game_name]
            group_by_sims[sim_name][game_name] = (mean, std, sim_rank, max_rank)
            
    avg_sim_ranks = {}
    avg_sim_mean = {}
    for sim_name in present_sim_names:
        avg_sim_ranks[sim_name] = np.mean([group_by_sims[sim_name][game_name][2] for game_name in game_names])
        avg_sim_mean[sim_name] = np.mean([group_by_sims[sim_name][game_name][0] for game_name in game_names])
        if metric_name in ["ARRA", "DC", "ARR"]:
            avg_sim_mean[sim_name] = -avg_sim_mean[sim_name]
    avg_rank = 0 
    avg_rank_value = None 
    for sim_name, sim_rank_value in sorted(avg_sim_ranks.items(), key=lambda x: x[1]):
        if avg_rank_value is None or abs(sim_rank_value - avg_rank_value) < 0.001:
            avg_rank_value = sim_rank_value
            avg_sim_ranks[sim_name] = avg_rank
        else:
            avg_rank += 1
            avg_rank_value = sim_rank_value
            avg_sim_ranks[sim_name] = avg_rank
    # NOTE: should be sorted by general rank
    rows = sorted(group_by_sims.items(), key=lambda x: (avg_sim_ranks[x[0]], avg_sim_mean[x[0]])) # from best to worst
    print(f"\\begin{{tabular}}{{ l | { '|'.join(['c c'] * len(game_names)) } | c }}", file = table_file)
    # print(f"\\\\\\hline", file = table_file)
    # header for games
    print(f"\\multirow{{2}}{{*}}{{Algo}}", end = "", file = table_file)
    for game_name in game_names:    
        gn = name_remap.get(game_name, game_name)
        print(f"& \\multicolumn{{2}}{{c|}}{{{gn}}} ", end = "", file = table_file)
    print(f"& \\multirow{{2}}{{*}}{{Rank}}", end = "", file = table_file)
    print(f"\\\\\\cline{{2-{2 * len(game_names) + 1}}}", file = table_file)
    #subheader for metrics
    # print(f" & ", end="", file = table_file)
    for game_name in game_names:
        print(f"& {metric_name} & Rank ", end = "", file = table_file)
    print(f" & ", end="", file = table_file)
    print(f"\\\\\\hline", file = table_file)
    # body of the table 
    for row_id, (sim_name, game_groups) in enumerate(rows):
        print(f"{sim_name} ", end = "", file = table_file)
        for game_name in game_names:
            mean, std, rank, _ = game_groups[game_name]
            print(" & {0:.1f} $\\pm$ {1:.1f} & {2}".format(mean * 100, std * 100, rank + 1), end = "", file = table_file)
        print(f"& {avg_sim_ranks[sim_name] + 1}", end = "", file = table_file)
        if row_id != len(rows) - 1:
            print(f"\\\\\\hline", file = table_file)
        else:
            print("", file = table_file)
    print(f"\\end{{tabular}}", file = table_file)

def draw_latex_threshold_convergence_tbl(metrics_file: str, metric_name = "ARRA", sim_names = [], game_names = ["CompareOnOneGame"],
                                table_file=None, name_remap = {}, min_threshold = 0.8, max_threshold = 0.9,
                                exclude_sim_names = ["des-med", "pl-l-1", "pl-d-1", "pl-d-2", "pl-d-5", "pl-d-100", "pl-l-2", "pl-l-5", "pl-l-100"],
                                rename = {"pl-d-0": "pl-d", "pl-l-0": "pl-l"}):
    ''' We compute number of runs in which algo crossed the threshold and did not exit it  '''
    with open(metrics_file, "r") as f:
        lines = f.readlines()
    runs = [json.loads(line) for line in lines ]
    groups = {}
    sim_names = set(sim_names)
    present_sim_names = set()
    exclude_sim_names = set(exclude_sim_names)
    for run in runs: 
        sim_name = run["sim_name"]
        game_name = run["game_name"]
        if game_name == "IntransitiveRegionGame" and run["param_sel_size"] == 5:
            continue      
        if sim_name in exclude_sim_names:
            continue  
        if len(sim_names) > 0 and sim_name not in sim_names:
            continue
        if game_name not in game_names:
            continue
        key = (sim_name, game_name)
        present_sim_names.add(sim_name)
        groups.setdefault(key, []).append(run["metric_" + metric_name])    
    present_sim_names = list(present_sim_names)
    group_by_sims = {}
    def score(v):
        if v >= max_threshold:
            return 1
        if v >= min_threshold:
            return 0.5
        return 0
    for (sim_name, game_name), all_values in groups.items():
        n = sum(score(run_values[-1]) for run_values in all_values)
        s = [0 if np.isnan(s) else s for run_values in all_values for s, _ in [stats.spearmanr(run_values, range(len(run_values)))]]

        ttrts = []
        # num_thr = 0
        # num_thr_reached = 0
        for run_values in all_values:
            ttrt = None # never reached                
            # is_below = True
            for i, v in enumerate(run_values):
                if v >= max_threshold and ttrt is None :
                    ttrt = i 
            if ttrt is not None:
                ttrts.append(ttrt)
        # v = 101 if len(ttrts) == 0 else np.median(ttrts)        
        group_by_sims.setdefault(sim_name, {})[game_name] = (n, ttrts, s)
    total_conv = {}
    total_spearman = {}
    total_ttrt = {}
    for sim_name in present_sim_names:
        total_conv[sim_name] = sum([group_by_sims[sim_name][game_name][0] for game_name in game_names])
        ttrts = [ttrt for game_name in game_names for ttrt in group_by_sims[sim_name][game_name][1]]
        total_ttrt[sim_name] = 101 if len(ttrts) == 0 else np.mean(ttrts)
        total_spearman[sim_name] = np.mean([s for game_name in game_names for s in group_by_sims[sim_name][game_name][2]])
    # NOTE: should be sorted by general rank
    rows = sorted(group_by_sims.items(), key=lambda x: total_conv[x[0]], reverse=True) # from best to worst
    print(f"\\begin{{tabular}}{{ | l | { ' '.join(['c'] * len(game_names)) } | c | { ' '.join(['c'] * len(game_names)) } | c | }}", file = table_file)
    # print(f"\\\\\\hline", file = table_file)
    # header for games
    print("\\hline", file = table_file)
    print(f"Algo", end = "", file = table_file)
    for game_name in game_names:    
        gn = name_remap.get(game_name, game_name)
        print(f"& {gn} ", end = "", file = table_file)
    print(f"& Rate", end = "", file = table_file)
    # for game_name in game_names:    
    #     gn = name_remap.get(game_name, game_name)
    #     print(f"& {gn} ", end = "", file = table_file)
    # print(f"& TTRT", end = "", file = table_file) 
    for game_name in game_names:    
        gn = name_remap.get(game_name, game_name)
        print(f"& {gn} ", end = "", file = table_file)
    print(f"& S", end = "", file = table_file)         
    print(f"\\\\\\hline", file = table_file)
    #subheader for metrics
    # print(f" & ", end="", file = table_file)
    # for game_name in game_names:
    #     print(f"& {metric_name} & Rank ", end = "", file = table_file)
    # print(f" & ", end="", file = table_file)
    # print(f"\\\\\\hline", file = table_file)
    # body of the table 
    for row_id, (sim_name, game_groups) in enumerate(rows):
        print(f"{sim_name} ", end = "", file = table_file)
        for game_name in game_names:
            num, _, _ = game_groups[game_name]
            print(" & {0:.0f} ".format(num * 100 / 30), end = "", file = table_file)
        print(f"& {total_conv[sim_name] * 100 / (30 * len(game_names)):.1f}", end = "", file = table_file)
        # for game_name in game_names:
        #     _, ttrts, _ = game_groups[game_name]
        #     ttrt = 101 if len(ttrts) == 0 else np.mean(ttrts)
        #     if ttrt == 101:
        #         print(" & $\infty$ ", end = "", file = table_file)
        #     else: 
        #         print(" & {0:.0f} ".format(ttrt), end = "", file = table_file)
        # if total_ttrt[sim_name] == 101:
        #     print(f"& $\infty$", end = "", file = table_file)
        # else:
        #     print(f"& {total_ttrt[sim_name]:.0f}", end = "", file = table_file)   
        for game_name in game_names:
            _, _, s = game_groups[game_name]
            sv = np.mean(s)
            print(" & {0:.2f} ".format(sv), end = "", file = table_file)
        print(f"& {total_spearman[sim_name]:.2f}", end = "", file = table_file)                 
        # if row_id != len(rows) - 1:
        print(f"\\\\\\hline", file = table_file)
        # else:
        #     print("", file = table_file)
    
    print(f"\\end{{tabular}}", file = table_file)


def draw_latex_thr_conv_tbl(metrics_file: str, metric_name = "ARRA", sim_names = [], game_names = ["CompareOnOneGame"],
                                table_file=None, name_remap = {}, min_threshold = 0.8, max_threshold = 0.9,
                                exclude_sim_names = ["pl-l-1", "pl-d-1", "pl-d-2", "pl-d-5", "pl-d-100", "pl-l-2", "pl-l-5", "pl-l-100"],
                                rename = {"pl-d-0": "pl-d", "pl-l-0": "pl-l"}):
    ''' We compute number of runs in which algo crossed the threshold and did not exit it  '''
    with open(metrics_file, "r") as f:
        lines = f.readlines()
    runs = [json.loads(line) for line in lines ]
    groups = {}
    sim_names = set(sim_names)
    present_sim_names = set()
    exclude_sim_names = set(exclude_sim_names)
    for run in runs: 
        sim_name = run["sim_name"]
        game_name = run["game_name"]
        if game_name == "IntransitiveRegionGame" and run["param_sel_size"] == 5:
            continue      
        if sim_name in exclude_sim_names:
            continue  
        if len(sim_names) > 0 and sim_name not in sim_names:
            continue
        if game_name not in game_names:
            continue
        key = (sim_name, game_name)
        present_sim_names.add(sim_name)
        groups.setdefault(key, []).append(run["metric_" + metric_name])    
    present_sim_names = list(present_sim_names)
    group_by_sims = {}
    def score(v):
        if v >= max_threshold:
            return 1
        if v >= min_threshold:
            return 0.5
        return 0
    for (sim_name, game_name), all_values in groups.items():
        n = sum(score(run_values[-1]) for run_values in all_values)
        s = [0 if np.isnan(s) else s for run_values in all_values for s, _ in [stats.spearmanr(run_values, range(len(run_values)))]]

        ttrts = []
        # num_thr = 0
        # num_thr_reached = 0
        for run_values in all_values:
            ttrt = None # never reached                
            # is_below = True
            for i, v in enumerate(run_values):
                if v >= max_threshold and ttrt is None :
                    ttrt = i 
            if ttrt is not None:
                ttrts.append(ttrt)
        # v = 101 if len(ttrts) == 0 else np.median(ttrts)        
        group_by_sims.setdefault(sim_name, {})[game_name] = (n, ttrts, s)
    total_conv = {}
    total_spearman = {}
    total_ttrt = {}
    for sim_name in present_sim_names:
        total_conv[sim_name] = sum([group_by_sims[sim_name][game_name][0] for game_name in game_names])
        ttrts = [ttrt for game_name in game_names for ttrt in group_by_sims[sim_name][game_name][1]]
        total_ttrt[sim_name] = 101 if len(ttrts) == 0 else np.mean(ttrts)
        total_spearman[sim_name] = np.mean([s for game_name in game_names for s in group_by_sims[sim_name][game_name][2]])
    # NOTE: should be sorted by general rank
    rows = sorted(group_by_sims.items(), key=lambda x: total_conv[x[0]], reverse=True) # from best to worst
    print(f"\\begin{{tabular}}{{ | l | { ' '.join(['c'] * len(game_names)) } | c | }}", file = table_file)
    # print(f"\\\\\\hline", file = table_file)
    # header for games
    print("\\hline", file = table_file)
    print(f"Algo", end = "", file = table_file)
    for game_name in game_names:    
        gn = name_remap.get(game_name, game_name)
        print(f"& {gn} ", end = "", file = table_file)
    print(f"& Rate", end = "", file = table_file)
    # for game_name in game_names:    
    #     gn = name_remap.get(game_name, game_name)
    #     print(f"& {gn} ", end = "", file = table_file)
    # print(f"& TTRT", end = "", file = table_file) 
    # for game_name in game_names:    
    #     gn = name_remap.get(game_name, game_name)
    #     print(f"& {gn} ", end = "", file = table_file)
    # print(f"& S", end = "", file = table_file)         
    print(f"\\\\\\hline", file = table_file)
    #subheader for metrics
    # print(f" & ", end="", file = table_file)
    # for game_name in game_names:
    #     print(f"& {metric_name} & Rank ", end = "", file = table_file)
    # print(f" & ", end="", file = table_file)
    # print(f"\\\\\\hline", file = table_file)
    # body of the table 
    for row_id, (sim_name, game_groups) in enumerate(rows):
        print(f"{sim_name} ", end = "", file = table_file)
        for game_name in game_names:
            num, _, _ = game_groups[game_name]
            print(" & {0:.0f} ".format(num * 100 / 30), end = "", file = table_file)
        print(f"& {total_conv[sim_name] * 100 / (30 * len(game_names)):.1f}", end = "", file = table_file)
        # for game_name in game_names:
        #     _, ttrts, _ = game_groups[game_name]
        #     ttrt = 101 if len(ttrts) == 0 else np.mean(ttrts)
        #     if ttrt == 101:
        #         print(" & $\infty$ ", end = "", file = table_file)
        #     else: 
        #         print(" & {0:.0f} ".format(ttrt), end = "", file = table_file)
        # if total_ttrt[sim_name] == 101:
        #     print(f"& $\infty$", end = "", file = table_file)
        # else:
        #     print(f"& {total_ttrt[sim_name]:.0f}", end = "", file = table_file)   
        # for game_name in game_names:
        #     _, _, s = game_groups[game_name]
        #     sv = np.mean(s)
        #     print(" & {0:.2f} ".format(sv), end = "", file = table_file)
        # print(f"& {total_spearman[sim_name]:.2f}", end = "", file = table_file)                 
        # if row_id != len(rows) - 1:
        print(f"\\\\\\hline", file = table_file)
        # else:
        #     print("", file = table_file)
    
    print(f"\\end{{tabular}}", file = table_file)


def formatFloat(fmt, val):
  ret = fmt % val
  if ret.startswith("0."):
    return ret[1:]
  if ret.startswith("-0."):
    return "-" + ret[2:]
  return ret

def draw_latex_thr_conv_grp_tbl(metrics_file: str, metric_name = "ARRA", sim_names = [], game_name_groups = [["CompareOnOneGame"]],
                                table_file=None, name_remap = {}, min_threshold = 0.8, max_threshold = 0.9,
                                exclude_sim_names = ["pl-l-1", "pl-d-1", "pl-d-2", "pl-d-5", "pl-d-100", "pl-l-2", "pl-l-5", "pl-l-100"],
                                rename = {"pl-d-0": "pl-d", "pl-l-0": "pl-l"}):
    ''' We compute number of runs in which algo crossed the threshold and did not exit it  '''
    with open(metrics_file, "r") as f:
        lines = f.readlines()
    runs = [json.loads(line) for line in lines ]
    groups = {}
    sim_names = set(sim_names)
    present_sim_names = set()
    exclude_sim_names = set(exclude_sim_names)
    all_game_names = set([g for gng in game_name_groups for g in gng])
    groups_map = {game_name: i for i, g in enumerate(game_name_groups) for game_name in g}
    for run in runs: 
        sim_name = run["sim_name"]
        game_name = run["game_name"]
        if game_name == "IntransitiveRegionGame" and run["param_sel_size"] == 5:
            continue      
        if sim_name in exclude_sim_names:
            continue  
        if len(sim_names) > 0 and sim_name not in sim_names:
            continue
        if game_name not in all_game_names:
            continue
        key = (sim_name, groups_map[game_name])
        present_sim_names.add(sim_name)
        groups.setdefault(key, []).append(run["metric_" + metric_name])    
    present_sim_names = list(present_sim_names)
    group_by_sims = {}
    def score(v):
        if v >= max_threshold:
            return 1
        if v >= min_threshold:
            return 0.5
        return 0
    for (sim_name, game_group), all_values in groups.items():
        n = sum(score(run_values[-1]) for run_values in all_values)
        s = [0 if np.isnan(s) else s for run_values in all_values for s, _ in [stats.spearmanr(run_values, range(len(run_values)))]]

        ttrts = []
        # num_thr = 0
        # num_thr_reached = 0
        for run_values in all_values:
            ttrt = None # never reached                
            # is_below = True
            for i, v in enumerate(run_values):
                if v >= max_threshold and ttrt is None :
                    ttrt = i 
            if ttrt is not None and run_values[-1] >= max_threshold:
                ttrts.append(ttrt)
        # v = 101 if len(ttrts) == 0 else np.median(ttrts)        
        group_by_sims.setdefault(sim_name, {})[game_group] = (n, ttrts, s)
    total_conv = {}
    total_spearman = {}
    total_ttrt = {}
    for sim_name in present_sim_names:
        total_conv[sim_name] = sum([group_by_sims[sim_name][i][0] for i in range(len(game_name_groups))])
        ttrts = [ttrt for i in range(len(game_name_groups)) for ttrt in group_by_sims[sim_name][i][1]]
        total_ttrt[sim_name] = 101 if len(ttrts) == 0 else np.mean(ttrts)
        total_spearman[sim_name] = np.mean([s for i in range(len(game_name_groups)) for s in group_by_sims[sim_name][i][2]])
    # NOTE: should be sorted by general rank
    rows = sorted(group_by_sims.items(), key=lambda x: total_conv[x[0]], reverse=True) # from best to worst
    print(f"\\begin{{tabular}}{{ | l | { ' '.join(['c'] * len(game_name_groups)) }  c |  { ' '.join(['c'] * len(game_name_groups)) } c |  { ' '.join(['c'] * len(game_name_groups)) } c |}}", file = table_file)
    # print(f"\\\\\\hline", file = table_file)
    # header for games
    print("\\hline", file = table_file)
    print(f"\\multirow{{2}}{{*}}{{Algo}}", end = "", file = table_file)
    print(f"& \\multicolumn{{{len(game_name_groups) + 1}}}{{c|}}{{Convergence rate, \%}}", end = "", file = table_file)
    print(f"& \\multicolumn{{{len(game_name_groups) + 1}}}{{c|}}{{Time to threshold}}", end = "", file = table_file)
    print(f"& \\multicolumn{{{len(game_name_groups) + 1}}}{{c|}}{{Monotonicity}}", end = "", file = table_file)
    # print(f"& \\multicolumn{{{len(game_names)}}}{{c}}{{Miss}}", end = "", file = table_file)
    print("\\\\", file = table_file)
    # print("&", end = "", file = table_file)
    for i in range(len(game_name_groups)):
        print(f"& {(i + 1)} ", end = "", file = table_file)
    print(f"& $\sum$ ", end = "", file = table_file)
    for i in range(len(game_name_groups)):
        print(f"& {(i + 1)} ", end = "", file = table_file)
    print(f"& $\sum$ ", end = "", file = table_file) 
    for i in range(len(game_name_groups)):
        print(f"& {(i + 1)} ", end = "", file = table_file)
    print(f"& $\sum$ ", end = "", file = table_file)         
    print(f"\\\\\\hline", file = table_file)
    #subheader for metrics
    # print(f" & ", end="", file = table_file)
    # for game_name in game_names:
    #     print(f"& {metric_name} & Rank ", end = "", file = table_file)
    # print(f" & ", end="", file = table_file)
    # print(f"\\\\\\hline", file = table_file)
    # body of the table 
    for row_id, (sim_name, game_groups) in enumerate(rows):
        print(f"{sim_name} ", end = "", file = table_file)
        for i in range(len(game_name_groups)):
            num, _, _ = game_groups[i]
            print(" & {0:.0f} ".format(num * 100 / (30 * len(game_name_groups[i]))), end = "", file = table_file)
        print(f"& {total_conv[sim_name] * 100 / (30 * sum(len(g) for g in game_name_groups)):.1f}", end = "", file = table_file)
        for i in range(len(game_name_groups)):
            _, ttrts, _ = game_groups[i]
            ttrt = 101 if len(ttrts) == 0 else np.mean(ttrts)
            if ttrt == 101:
                print(" & $\infty$ ", end = "", file = table_file)
            else: 
                print(" & {0:.0f} ".format(ttrt), end = "", file = table_file)
        if total_ttrt[sim_name] == 101:
            print(f"& $\infty$", end = "", file = table_file)
        else:
            print(f"& {total_ttrt[sim_name]:.0f}", end = "", file = table_file)   
        for i in range(len(game_name_groups)):
            _, _, s = game_groups[i]
            sv = np.mean(s)
            vv = formatFloat("%.2f", sv)
            print(f" & {vv} ", end = "", file = table_file)
        vv = formatFloat("%.2f", total_spearman[sim_name])
        print(f"& {vv}", end = "", file = table_file)                 
        # if row_id != len(rows) - 1:
        print(f"\\\\\\hline", file = table_file)
        # else:
        #     print("", file = table_file)
    
    print(f"\\end{{tabular}}", file = table_file)

def draw_latex_ranks_tbl(metrics_file: str, metric_name = "ARRA", sim_names = [], game_names = ["CompareOnOneGame"],
                                table_file=None, p_value = 0.05, name_remap = {},
                                exclude_sim_names = ["pl-d-2", "pl-d-5", "pl-d-100", "pl-l-2", "pl-l-5", "pl-l-100"]):
    ''' metrix_file is in jsonlist format'''
    with open(metrics_file, "r") as f:
        lines = f.readlines()
    runs = [json.loads(line) for line in lines ]
    groups = {}
    sim_names = set(sim_names)
    present_sim_names = set()
    exclude_sim_names = set(exclude_sim_names)
    for run in runs: 
        sim_name = run["sim_name"]
        game_name = run["game_name"]
        if game_name == "IntransitiveRegionGame" and run["param_sel_size"] == 5:
            continue        
        if sim_name in exclude_sim_names:
            continue
        if len(sim_names) > 0 and sim_name not in sim_names:
            continue
        if game_name not in game_names:
            continue
        key = (sim_name, game_name)
        present_sim_names.add(sim_name)
        groups.setdefault(key, []).append(run["metric_" + metric_name])    
    present_sim_names = list(present_sim_names)
    group_by_sims = {}
    for (sim_name, game_name), all_values in groups.items():
        new_values = [run_values[-1] for run_values in all_values]
        mean = np.mean(new_values)
        std = np.std(new_values)
        group_by_sims.setdefault(sim_name, {})[game_name] = (mean, std, new_values)
        # groups[(sim_name, game_name, metric_name)] = (mean, std)
    for game_name in game_names:
        data = [group_by_sims[sim_name][game_name][2] for sim_name in present_sim_names]
        # friedman_res = stats.friedmanchisquare(*data)
        diffs = [[1 for _ in present_sim_names] for _ in present_sim_names]
        for i in range(len(diffs)):
            for j in range(i + 1, len(diffs)):
                _, pvalue = stats.mannwhitneyu(data[i], data[j])
                if pvalue < p_value:
                    diffs[i][j] = pvalue
                    diffs[j][i] = pvalue
        sim_ranks = {}

        # from tabulate import tabulate
        # names = present_sim_names
        # rows = []
        # for i in range(len(names)):
        #     row = []
        #     row.append(names[i])
        #     for j in range(len(names)):
        #         row.append(diffs[i][j])
        #     rows.append(row)
        # print(tabulate(rows, headers=["", *names], tablefmt="grid", numalign="center", stralign="center"))

        # building domination relation by nemenyi test p-value and means 
        # to rank we need to build all domination chains and average ranks in them 
        dominations = {sim_name: set() for sim_name in present_sim_names}
        dominated_by = {sim_name: set() for sim_name in present_sim_names}
        # NOTE: sign depends on metric - ARRA >, Dup <, R <, DC >, ARR >
        f = (lambda x: x[0] > x[1]) if metric_name in [ "ARRA", "DC", "ARR"] else (lambda x: x[0] < x[1])
        for i in range(len(present_sim_names)):
            sim_name = present_sim_names[i]
            for j in range(i + 1, len(present_sim_names)):
                sim_name2 = present_sim_names[j]
                if diffs[i][j] < p_value:
                    if f((group_by_sims[sim_name][game_name][0], group_by_sims[sim_name2][game_name][0])):
                        dominations[sim_name].add(sim_name2)
                        dominated_by[sim_name2].add(sim_name)
                    else:
                        dominations[sim_name2].add(sim_name)
                        dominated_by[sim_name].add(sim_name2)  

        rank = 0
        while len(dominations) > 0:
            sources = [sim_name for sim_name, v in dominated_by.items() if len(v) == 0]  
            for sim_name in sources:
                sim_ranks[sim_name] = rank
                for sim_name2 in dominations.get(sim_name, []):
                    dominated_by[sim_name2].remove(sim_name)
                if sim_name in dominations:
                    del dominations[sim_name]
                del dominated_by[sim_name]
            rank += 1

        max_rank = rank
        for sim_name in present_sim_names:
            sim_rank = sim_ranks.get(sim_name, 1)
            mean, std, _ = group_by_sims[sim_name][game_name]
            group_by_sims[sim_name][game_name] = (mean, std, sim_rank, max_rank)

    # test_scores = [(sim_name, game_name, game_r) for sim_name, sim_g in group_by_sims.items() for game_name, game_r in sim_g.items()]
    # test_scores = {}
    # for sim_name in present_sim_names:
    #     rank_map = {}
    #     for game_name, game_r in group_by_sims[sim_name].items():
    #         if game_r[2] in [1,2,3]:
    #             rank_map.setdefault(game_r[2], []).append(game_name)
    #     for game_name in game_names:                    
    #         r = group_by_sims[sim_name][game_name][2]
    #         if r in [1,2,3]:
    #             test_scores[(sim_name, game_name)] = len(rank_map[r])

    # test_score_groups = {}
    # for (sim_name, game_name), cnt in test_scores.items():
    #     test_score_groups.setdefault((game_name, cnt), []).append(sim_name)
    # ts = sorted(test_score_groups.items(), key=lambda x: (x[0][1], -len(x[1])))
    # print(ts)
            
    avg_sim_ranks = {}
    avg_sim_mean = {}
    for sim_name in present_sim_names:
        avg_sim_ranks[sim_name] = np.mean([group_by_sims[sim_name][game_name][2] for game_name in game_names])
        avg_sim_mean[sim_name] = np.mean([group_by_sims[sim_name][game_name][0] for game_name in game_names])
        if metric_name in ["ARRA", "DC", "ARR"]:
            avg_sim_mean[sim_name] = -avg_sim_mean[sim_name]

    avg_rank = 0 
    avg_rank_value = None 
    for sim_name, sim_rank_value in sorted(avg_sim_ranks.items(), key=lambda x: x[1]):
        if avg_rank_value is None or abs(sim_rank_value - avg_rank_value) < 0.001:
            avg_rank_value = sim_rank_value
            avg_sim_ranks[sim_name] = avg_rank
        else:
            avg_rank += 1
            avg_rank_value = sim_rank_value
            avg_sim_ranks[sim_name] = avg_rank            
    # NOTE: should be sorted by general rank
    rows = sorted(group_by_sims.items(), key=lambda x: (avg_sim_ranks[x[0]], avg_sim_mean[x[0]])) # from best to worst
    print(f"\\begin{{tabular}}{{ l | { '|'.join(['c'] * len(game_names)) } | c }}", file = table_file)
    # print(f"\\\\\\hline", file = table_file)
    # header for games
    print(f"Algo", end = "", file = table_file)
    for gid, game_name in enumerate(game_names):    
        print(f"& {str(gid + 1).rjust(2, '0')} ", end = "", file = table_file)
    print(f"& r", end = "", file = table_file)
    print(f"\\\\\\hline", file = table_file)
    # body of the table 
    for row_id, (sim_name, game_groups) in enumerate(rows):
        print(f"{sim_name} ", end = "", file = table_file)
        for game_name in game_names:
            mean, std, rank, _ = game_groups[game_name]
            print(" & {0}".format(rank + 1), end = "", file = table_file)
        print(f"& {avg_sim_ranks[sim_name] + 1}", end = "", file = table_file)
        if row_id != len(rows) - 1:
            print(f"\\\\\\hline", file = table_file)
        else:
            print("", file = table_file)
    print(f"\\end{{tabular}}", file = table_file)

def monotonicity_index(series):
    n = len(series)
    increasing_pairs = 0
    decreasing_pairs = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if series[i] < series[j]:
                increasing_pairs += 1
            elif series[i] > series[j]:
                decreasing_pairs += 1

    total_pairs = n * (n - 1) / 2

    if increasing_pairs > decreasing_pairs:
        return increasing_pairs / total_pairs
    else:
        return decreasing_pairs / total_pairs


def draw_monotonicity_evaluation(metrics_file: str, metric_name = "ARRA", sim_names = [], game_names = ["CompareOnOneGame"],
                                table_file=None, p_value = 0.05, name_remap = {}):
    ''' Computes Spearman's Rank Correlation Coefficient and Monotonicity Index for each run and avg the evaluation metrics '''
    with open(metrics_file, "r") as f:
        lines = f.readlines()
    runs = [json.loads(line) for line in lines ]
    groups = {}
    sim_names = set(sim_names)
    present_sim_names = set()
    for run in runs: 
        sim_name = run["sim_name"]
        game_name = run["game_name"]
        if game_name == "IntransitiveRegionGame" and run["param_sel_size"] == 5:
            continue        
        if len(sim_names) > 0 and sim_name not in sim_names:
            continue
        if game_name not in game_names:
            continue
        key = (sim_name, game_name)
        present_sim_names.add(sim_name)
        groups.setdefault(key, []).append(run["metric_" + metric_name])    
    present_sim_names = list(present_sim_names)
    scores = {}
    for game_name in game_names:
        for sim_name in present_sim_names:
            all_values = groups[(sim_name, game_name)]
            spearmans = []
            mixes = []
            for run_values in all_values:
                score, _ = stats.spearmanr(run_values, range(len(run_values)))
                if not np.isnan(score):
                    spearmans.append(score)
                mix = 0 # monotonicity_index(run_values)
                mixes.append(mix)
            scores.setdefault(sim_name, {})[game_name] = ((0 if len(spearmans) == 0 else np.mean(spearmans)), np.mean(mixes))

    avg_spearman = {}
    avg_mix = {}
    for sim_name in present_sim_names:
        spearmans = []
        mixes = []
        for game_name in game_names:
            spearman, mix = scores[sim_name][game_name]
            spearmans.append(spearman)
            mixes.append(mix)        
        avg_spearman[sim_name] = np.mean(spearmans)
        avg_mix[sim_name] = np.mean(mixes)

    # NOTE: should be sorted by general rank
    rows = sorted(scores.items(), key=lambda x: (avg_spearman[x[0]], avg_mix[x[0]]), reverse=True) # from best to worst
    print(f"\\begin{{tabular}}{{ l | { '|'.join(['c'] * len(game_names)) } | c }}", file = table_file)
    # print(f"\\\\\\hline", file = table_file)
    # header for games
    print(f"Algo", end = "", file = table_file)
    for game_name in game_names:    
        gn = name_remap.get(game_name, game_name)
        print(f"& {gn} ", end = "", file = table_file)
    print(f"& S", end = "", file = table_file)
    print(f"\\\\\\hline", file = table_file)
    # body of the table 
    for row_id, (sim_name, game_groups) in enumerate(rows):
        print(f"{sim_name} ", end = "", file = table_file)
        for game_name in game_names:
            spearman, mix = game_groups[game_name]
            print(" & {0:.2f}".format(spearman), end = "", file = table_file)
        print(f"& {avg_spearman[sim_name]:.2f}", end = "", file = table_file)
        if row_id != len(rows) - 1:
            print(f"\\\\\\hline", file = table_file)
        else:
            print("", file = table_file)
    print(f"\\end{{tabular}}", file = table_file)


def draw_ttrt(metrics_file: str, metric_name = "ARRA", sim_names = [], game_names = ["CompareOnOneGame"],
                                table_file=None, threshold = 0.9, name_remap = {},
                                exclude_sim_names = ["pl-d-2", "pl-d-5", "pl-d-100", "pl-l-2", "pl-l-5", "pl-l-100"]):
    ''' Computes Time to Reach Threashold. Note: threshold should be adjustedbe game and sim. The latest time is taken '''
    with open(metrics_file, "r") as f:
        lines = f.readlines()
    runs = [json.loads(line) for line in lines ]
    groups = {}
    sim_names = set(sim_names)
    present_sim_names = set()
    exclude_sim_names = set(exclude_sim_names)
    for run in runs: 
        sim_name = run["sim_name"]
        game_name = run["game_name"]
        if game_name == "IntransitiveRegionGame" and run["param_sel_size"] == 5:
            continue      
        if sim_name in exclude_sim_names:
            continue  
        if len(sim_names) > 0 and sim_name not in sim_names:
            continue
        if game_name not in game_names:
            continue
        key = (sim_name, game_name)
        present_sim_names.add(sim_name)
        groups.setdefault(key, []).append(run["metric_" + metric_name])    
    present_sim_names = list(present_sim_names)
    scores = {}
    for game_name in game_names:
        for sim_name in present_sim_names:
            all_values = groups[(sim_name, game_name)]
            ttrts = []
            num_thr = 0
            num_thr_reached = 0
            for run_values in all_values:
                ttrt = None # never reached                
                is_below = True
                for i, v in enumerate(run_values):
                    if v >= threshold:
                        if is_below:
                            num_thr += 1
                            if ttrt is None:
                                num_thr_reached += 1
                            ttrt = i 
                        is_below = False
                    else:
                        is_below = True
                if ttrt is not None:
                    ttrts.append(ttrt)
            v = 101 if len(ttrts) == 0 else np.median(ttrts)
            # if num_thr_reached >= len(all_values) / 2:
            if num_thr_reached == len(all_values):
                num_thr_avg = num_thr / num_thr_reached
                scores.setdefault(sim_name, {})[game_name] = (v, num_thr_avg, len(all_values) - num_thr_reached)

    # NOTE: should be sorted by general rank
    rows = sorted(scores.items(), key=lambda x: (sum(x[1].get(g, (101, 0, 30))[2] for g in reversed(game_names)), sum(x[1].get(g, (101, 0, 30))[0] for g in reversed(game_names)))) # from best to worst
    print(f"\\begin{{tabularx}}{{\\textwidth}}{{ l | { 'Y ' * len(game_names) } | { 'Y ' * len(game_names) } }}", file = table_file)
    # print(f"\\\\\\hline", file = table_file)
    # header for games
    print(f"\\multirow{{2}}{{*}}{{Algo}}", end = "", file = table_file)
    print(f"& \\multicolumn{{{len(game_names)}}}{{c|}}{{TTRT}}", end = "", file = table_file)
    print(f"& \\multicolumn{{{len(game_names)}}}{{c}}{{Osc}}", end = "", file = table_file)
    # print(f"& \\multicolumn{{{len(game_names)}}}{{c}}{{Miss}}", end = "", file = table_file)
    print("\\\\", file = table_file)
    # print(f"\\\\\\cline{{2-{3 * len(game_names) + 1}}}", file = table_file)
    # print(f" & ", end="", file = table_file)
    for game_name in game_names:    
        gn = name_remap.get(game_name, game_name)
        print(f"& {gn}", end = "", file = table_file)
    for game_name in game_names:    
        gn = name_remap.get(game_name, game_name)
        print(f"& {gn}", end = "", file = table_file)
    # for game_name in game_names:    
    #     gn = name_remap.get(game_name, game_name)
    #     print(f"& {gn}", end = "", file = table_file)        
    print(f"\\\\\\hline", file = table_file)
    # body of the table 
    for row_id, (sim_name, game_groups) in enumerate(rows):
        print(f"{sim_name} ", end = "", file = table_file)
        for game_name in game_names:
            ttrt, _, _ = game_groups.get(game_name, (101, 0, 30))
            if ttrt == 101:
                print(" & $\\infty$ ", end = "", file = table_file)
            else:
                print(" & {0} ".format(round(ttrt)), end = "", file = table_file)
        for game_name in game_names:
            _, osc, _ = game_groups.get(game_name, (101, 0, 30))
            if osc < 0.1:
                print(" & 0 ", end = "", file = table_file)
            else:
                print(f" & {osc:.1f} ", end = "", file = table_file)       
        # for game_name in game_names:
        #     _, _, miss = game_groups[game_name]
        #     print(f" & {miss} ", end = "", file = table_file)                           
        if row_id != len(rows) - 1:
            print(f"\\\\\\hline", file = table_file)
        else:
            print("", file = table_file)
    print(f"\\end{{tabularx}}", file = table_file)


if __name__ == "__main__":
    ''' Test drawings '''
    # draw_populations([(1,2)], [(2,3)], [(6,7)], [(5,7)], xrange=(0, 100), yrange=(0, 100))
    
    # HC experiments
    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "all", \
    #                 sim_names=["rand", "hc-pmo-i", "hc-pmo-p", "hc-r-i", "hc-r-p"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})    
    
    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "last", \
    #                 sim_names=["rand", "hc-pmo-i", "hc-pmo-p", "hc-r-i", "hc-r-p"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})   

    # DE experiments
    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "all", \
    #                 sim_names=["rand", "de-l", "de-d-0", "de-d-1", "de-d-m", "de-d-g", "de-d-s", "de-d-d-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0}, rename={"de-d-d-100":"de-d-d"})    
    
    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "last", \
    #                 sim_names=["rand", "de-l", "de-d-0", "de-d-1", "de-d-m", "de-d-g", "de-d-s", "de-d-d-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0}, rename={"de-d-d-100":"de-d-d"}) 
    
    # DES
    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "all", \
    #                 sim_names=["rand", "des-mea", "des-med", "des-mea-0", "des-med-0", "des-mea-100", "des-med-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0}, rename={"des-mea-100":"des-mea-1", "des-med-100":"des-med-1"})    
    
    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "last", \
    #                 sim_names=["rand", "des-mea", "des-med", "des-mea-0", "des-med-0", "des-mea-100", "des-med-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0}, rename={"des-mea-100":"des-mea-1", "des-med-100":"des-med-1"})     
    
    #PL
    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "all", \
    #                 sim_names=["rand", "pl-l-0", "pl-l-100", "pl-d-0", "pl-d-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0}, rename={"pl-l-100":"pl-l-1", "pl-d-100":"pl-d-1"})    
    
    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "last", \
    #                 sim_names=["rand", "pl-l-0", "pl-l-100", "pl-d-0", "pl-d-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0}, rename={"pl-l-100":"pl-l-1", "pl-d-100":"pl-d-1"})     
    

    #Best
    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "all", \
    #                 sim_names=["rand", "hc-r-p", "de-d-g", "de-d-d-100", "des-mea-100", "des-med-100", "pl-d-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0}, rename={"de-d-d-100":"de-d-d-1", "des-mea-100":"des-mea-1", "des-med-100":"des-med-1", "pl-d-100":"pl-d-1"})
    
    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "last", \
    #                 sim_names=["rand", "hc-r-p", "de-d-g", "de-d-d-100", "des-mea-100", "des-med-100", "pl-d-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0}, rename={"de-d-d-100":"de-d-d-1", "des-mea-100":"des-mea-1", "des-med-100":"des-med-1", "pl-d-100":"pl-d-1"})     

    # Spaces
    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "all", \
    #                 sim_names=["rand", "hc-r-p", "de-d-g", "de-d-s", "de-d-d-100", "des-mea-100", "des-med-100", "pl-d-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0}, rename={"de-d-d-100":"de-d-d-1", "des-mea-100":"des-mea-1", "des-med-100":"des-med-1", "pl-d-100":"pl-d-1"})
    
    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "last", \
    #                 sim_names=["rand", "hc-r-p", "de-d-g", "de-d-s", "de-d-d-100", "des-mea-100", "des-med-100", "pl-d-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0}, rename={"de-d-d-100":"de-d-d-1", "des-mea-100":"des-mea-1", "des-med-100":"des-med-1", "pl-d-100":"pl-d-1"}, \
    #                 stats_file = open("data/plots/stats.txt", "w"))     
     
    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "all", \
    #                 sim_names=["rand", "hc-pmo-p", "hc-r-p", "de-l", "de-d-0", "de-d-1", "de-d-d-100", "des-mea-100", "des-med-100", "pl-l-100", "pl-d-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})

    # with open("data/plots/tables.tex", "w") as f:
    #     draw_latex_mean_std_tbl("data/metrics/spaces.jsonlist", metric_name = "ARRA", 
    #                                 sim_names = [], 
    #                                 game_names = ["GreaterThanGame", "CompareOnOneGame", "FocusingGame", "IntransitiveRegionGame"], 
    #                                 table_file = f,
    #                                 name_remap={"CompareOnOneGame": "Cmp1", "FocusingGame": "Focus", "IntransitiveRegionGame": "Intr", "GreaterThanGame":"GrTh"})
    

    space_games = ["ideal", 
                   "dependant-all-1", "dependant-all-2",
                   "skew-p-1", "skew-p-2", "skew-p-3", "skew-p-4", 
                   "trivial-5", "trivial-25", "trivial-50",
                    "skew-t-1", "skew-t-2", "skew-t-3", "skew-t-4", "skew-t-5", 
                    "skew-c-1", "skew-c-2", "skew-c-3", "skew-c-4", "skew-c-5", 
                    "dupl-t-5", "dupl-t-10", "dupl-t-50", 
                    "dupl-c-5", "dupl-c-10", "dupl-c-50",
                    "span-all-ends-1", "span-all-ends-5", "span-all-ends-10", "span-all-ends-20",
                    "span-pairs-1", "span-pairs-5", "span-pairs-10", "span-pairs-20"
                    ]
    # with open("data/plots/tables.tex", "w") as f:
    #     draw_latex_ranks_tbl("data/metrics/spaces.jsonlist", metric_name = "ARRA", 
    #                                 sim_names = [], 
    #                                 game_names = space_games, 
    #                                 table_file = f)   


    # space_games = ["ideal", "skew-p-4", "trivial-50",
    #                     "skew-t-1", "skew-t-5", "skew-c-1", "skew-c-5", 
    #                     "span-all-ends-1", "span-all-ends-10", "span-pairs-1", "span-pairs-10", 
    #                     "dupl-t-50", "dupl-c-50", "dependant-all-2"]
    # with open("data/plots/tables.tex", "w") as f:
    #     draw_latex_thr_conv_tbl("data/metrics/spaces.jsonlist", metric_name = "ARRA", 
    #                                 sim_names = [], 
    #                                 game_names = space_games, 
    #                                 table_file = f,
    #                                 name_remap={space_games:i for i, space_games in enumerate(space_games)})       
    

    # space_games = [["ideal"], 
    #                ["dependant-all-1", "dependant-all-2"],
    #                ["skew-p-1", "skew-p-2", "skew-p-3", "skew-p-4"], 
    #                ["trivial-5", "trivial-25", "trivial-50"],
    #                 ["skew-t-1", "skew-t-2", "skew-t-3", "skew-t-4", "skew-t-5"], 
    #                 ["skew-c-1", "skew-c-2", "skew-c-3", "skew-c-4", "skew-c-5"], 
    #                 ["dupl-t-5", "dupl-t-10", "dupl-t-50"], 
    #                 ["dupl-c-5", "dupl-c-10", "dupl-c-50"],
    #                 ["span-all-ends-1", "span-all-ends-5", "span-all-ends-10", "span-all-ends-20"], 
    #                 ["span-pairs-1", "span-pairs-5", "span-pairs-10", "span-pairs-20"], 
    #                 ]
    # with open("data/plots/tables.tex", "w") as f:
    #     draw_latex_thr_conv_grp_tbl("data/metrics/spaces.jsonlist", metric_name = "ARRA", 
    #                                 sim_names = [], 
    #                                 game_name_groups = space_games, 
    #                                 table_file = f)      

    # space_games = ["span-all-pairs-1"]
    # with open("data/plots/tables.tex", "w") as f:
    #     draw_latex_ranks_tbl("data/metrics/spaces.jsonlist", metric_name = "ARRA", 
    #                                 sim_names = [], 
    #                                 game_names = space_games, 
    #                                 table_file = f)        
     

    # Best of Best
    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["DC", "ARR", "AmultiRRA", "Dup", "R"], aggregation = "all", \
    #                 sim_names=["de-d-d-0", "de-d-d-1", "de-d-d-2", "de-d-d-5", "de-d-d-100", "de-d-g", "de-d-s", "de-d-0", "de-d-m"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})
    # 

    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["ARRA"], aggregation = "all", \
    #                 game_names=['span-all-ends-5', 'dupl-c-10', 'dupl-c-100'],
    #                 sim_names=["de-d-d-0", "de-d-d-1", "de-d-d-100", "de-d-g", "de-d-m", "de-d-0"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})      

    # DE-D-D-X spanned preservation 
    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["ARRA"], aggregation = "all", \
    #                 sim_names=["de-d-d-0", "de-d-d-1", "de-d-d-2", "de-d-d-5", "de-d-d-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})        

    # draw_metrics("data/metrics/spaces2.jsonlist", metrics = ["ARRA"], aggregation = "all", \
    #                 game_names=[], \
    #                 sim_names=["de-d-g", "de-d-s"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})       
    
    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["ARRA", "DC", "ARR", "Dup", "R"], aggregation = "all", \
    #                 game_names=[ "dupl-100"], \
    #                 sim_names=["de-l", "de-d-0", "de-d-1", "de-d-m", "de-d-g", "de-d-d-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0},
    #                 rename={"de-d-d-100":"de-d-d"})      
        
    # for sim_name in ["de-d-d-1", "pl-d-100", "de-d-g", "des-mea-5", "des-mea-100"]:
    #     draw_setups("data/metrics/spaces.jsonlist", sim_name=sim_name, prefix='skew', 
    #                     game_names=["ideal", "skew-p-1", "skew-p-2", "skew-p-3", "skew-p-4"],
    #                     rename = {"ideal": (5,5), "skew-p-1": (6,4), "skew-p-2": (7,3), "skew-p-3": (8,2), "skew-p-4": (9,1)})
        
    # for sim_name in ["de-d-d-1"]:
    #     draw_setups("data/metrics/spaces.jsonlist", sim_name=sim_name, prefix='trivial', 
    #                     game_names=["ideal", "trivial-1", "trivial-5", "trivial-10", "trivial-15", "trivial-20", "trivial-25"],
    #                     rename = {"ideal": (5,5), "skew-p-1": (6,4), "skew-p-2": (7,3), "skew-p-3": (8,2), "skew-p-4": (9,1)})        

    # for sim_name in ["de-d-d-1"]:
    #     draw_setups("data/metrics/spaces.jsonlist", sim_name=sim_name, prefix='skew-t', 
    #                     game_names=["ideal", "skew-t-1", "skew-t-2", "skew-t-3", "skew-t-4", "skew-t-5"],
    #                     rename = {"ideal": (1,1,1,1,1), "skew-t-1": (6,1,1,1,1), "skew-t-2": (1,6,1,1,1), "skew-t-3": (1,1,6,1,1), "skew-t-4": (1,1,1,6,1), "skew-t-5": (1,1,1,1,6)})        

    # for sim_name in ["de-d-d-1"]:
    #     draw_setups("data/metrics/spaces.jsonlist", sim_name=sim_name, prefix='skew-c', 
    #                     game_names=["ideal", "skew-c-1", "skew-c-2", "skew-c-3", "skew-c-4", "skew-c-5"],
    #                     rename = {"ideal": (1,1,1,1,1), "skew-c-1": (6,1,1,1,1), "skew-c-2": (1,6,1,1,1), "skew-c-3": (1,1,6,1,1), "skew-c-4": (1,1,1,6,1), "skew-c-5": (1,1,1,1,6)})        

    # for sim_name in ["de-d-d-1"]:
    #     draw_setups("data/metrics/spaces.jsonlist", sim_name=sim_name, prefix='span', 
    #                     game_names=["ideal", "span-all-ends-1", "span-all-ends-5", "span-one-pair-1", "span-all-pairs-1"],
    #                     rename = {"ideal": "no spanned", "span-all-ends-1": "all axis, 1", "span-all-ends-5": "all axis, 5", "span-one-pair-1": "axis pair, 1", "span-all-pairs-1": "axis pairs, 10"})        

    # for sim_name in ["de-d-d-1"]:
    #     draw_setups("data/metrics/spaces.jsonlist", sim_name=sim_name, prefix='dupl-t', 
    #                     game_names=["ideal", "dupl-t-2", "dupl-t-3", "dupl-t-4", "dupl-t-5", "dupl-t-10", "dupl-t-100"],
    #                     rename = {"ideal": "no dupl", "dupl-t-2": "2 dupl. tests", "dupl-t-3": "3 dupl. tests", "dupl-t-4": "4 dupl. tests", "dupl-t-5": "5 dupl. tests", "dupl-t-10": "10 dupl. tests", "dupl-t-100": "100 dupl. tests"})        

    # for sim_name in ["de-d-d-1"]:
    #     draw_setups("data/metrics/spaces.jsonlist", sim_name=sim_name, prefix='dupl-c', 
    #                     game_names=["ideal", "dupl-c-2", "dupl-c-3", "dupl-c-4", "dupl-c-5", "dupl-c-10", "dupl-c-100"],
    #                     rename = {"ideal": "no dupl", "dupl-c-2": "2 dupl. tests", "dupl-c-3": "3 dupl. tests", "dupl-c-4": "4 dupl. tests", "dupl-c-5": "5 dupl. tests", "dupl-c-10": "10 dupl. tests", "dupl-c-100": "100 dupl. tests"})        

    # for sim_name in ["de-d-d-1"]:
    #     draw_setups("data/metrics/spaces.jsonlist", sim_name=sim_name, prefix='dep', 
    #                     game_names=["ideal", "dependant-all-1", "dependant-all-2"],
    #                     rename = {"ideal": "no dep", "dependant-all-1": "1 dep", "dependant-all-2": "2 deps"})       
    

    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["ARRA", "DC", "ARR", "Dup", "R"], aggregation = "all", \
    #             game_names=['IntransitiveRegionGame'], \
    #             sim_names=["rand", "de-d-d-0", "de-d-d-1", "de-d-d-100", "de-d-g", "de-d-m", "de-d-0"], \
    #             fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})   #  

    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["ARRA", "DC", "ARR", "Dup", "R"], aggregation = "all", \
    #             game_names=['dupl-t-100'], \
    #             sim_names=["rand", "de-d-d-0", "de-d-d-1", "de-d-d-100", "de-d-g", "de-d-m", "de-d-0"], \
    #             fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})   #
    # 

    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["ARRA"], aggregation = "all", \
    #             game_names=[], \
    #             sim_names=["rand", "de-d-d-0", "de-d-d-1", "de-d-d-100", "de-d-g", "de-d-m", "de-d-0"], \
    #             fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})   #            

    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["ARRA"], aggregation = "all", \
    #             game_names=["dupl-c-100"], \
    #             sim_names=["rand", "de-d-d-1", "de-d-d-100", "de-d-g", "de-d-m", "de-d-0", "pl-d-100"], \
    #             fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})   #   
    # 

    # space_games2 = ["ideal", "skew-p-4", "trivial-25", "skew-t-5", 
    #                 "skew-c-5", "span-all-ends-1", "span-all-ends-5", "span-pairs-1", 
    #                 "span-pairs-5", "dupl-t-50", "dupl-c-50"]
    
    # map_names = {name:i for i, name in enumerate(space_games) if name in space_games2 }

    # with open("data/plots/tables.tex", "w") as f:
    #     draw_monotonicity_evaluation("data/metrics/spaces.jsonlist", metric_name = "ARRA", 
    #                                 sim_names = [], 
    #                                 game_names = ["GreaterThanGame", "CompareOnOneGame", "FocusingGame", "IntransitiveRegionGame", *space_games2], 
    #                                 table_file = f,
    #                                 name_remap={**map_names, "CompareOnOneGame": "Cmp1", "FocusingGame": "Focus", "IntransitiveRegionGame": "Intr", "GreaterThanGame":"GrTh"})
        
    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["ARRA"], aggregation = "all", \
    #             game_names=["ideal", "skew-p-1", "skew-p-2", "skew-p-3", "skew-p-4"], \
    #             sim_names=["des-mea", "des-mea-2", "des-mea-5", "des-mea-100"], \
    #             fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})   #           

    # draw_ttrt("data/metrics/spaces.jsonlist", metric_name = "ARRA", sim_names = [], 
    #             game_names = ["ideal", "skew-p-1", "skew-p-2", "skew-p-3", "skew-p-4" ],
    #             table_file = open("data/plots/tables.tex", "w"), threshold = 0.9, 
    #             name_remap = {"ideal": "5,5", "skew-p-1": "6,4", "skew-p-2": "7,3", "skew-p-3": "8,2", "skew-p-4": "9,1"})

    # draw_ttrt("data/metrics/spaces.jsonlist", metric_name = "ARRA", sim_names = [], 
    #             game_names=["ideal", "trivial-5", "trivial-25", "trivial-50"],
    #             table_file = open("data/plots/tables.tex", "w"), threshold = 0.9, 
    #             name_remap = {"ideal": "0", "trivial-1": "1", "trivial-5": "5", "trivial-10": "10", "trivial-15": "15", "trivial-20": "20", "trivial-25": "25"})    
    

    # draw_ttrt("data/metrics/spaces.jsonlist",
    #                 game_names=["skew-t-1", "skew-t-2", "skew-t-3", "skew-t-4", "skew-t-5"],
    #                 table_file = open("data/plots/tables.tex", "w"), threshold = 0.9, 
    #                 name_remap = {"skew-t-1": "x=1", "skew-t-2": "x=2", "skew-t-3": "x=3", "skew-t-4": "x=4", "skew-t-5": "x=5"})

    # draw_ttrt("data/metrics/spaces.jsonlist",
    #                 game_names=["skew-c-1", "skew-c-2", "skew-c-3", "skew-c-4", "skew-c-5"],
    #                 table_file = open("data/plots/tables.tex", "w"), threshold = 0.9, 
    #                 name_remap = {"skew-c-1": "x=1", "skew-c-2": "x=2", "skew-c-3": "x=3", "skew-c-4": "x=4", "skew-c-5": "x=5"}) 

    # draw_ttrt("data/metrics/spaces.jsonlist",
    #                 game_names=["ideal", "span-all-ends-1", "span-all-ends-5", "span-all-ends-10", "span-all-ends-20"],
    #                 table_file = open("data/plots/tables.tex", "w"), threshold = 0.9, 
    #                 name_remap = {"ideal": "0", "span-all-ends-1": "all", "span-all-ends-5": "all 5", "span-all-ends-10": "all 10", "span-all-ends-20": "all 20", "span-one-pair-1": "pair", "span-all-pairs-1": "pairs"})        
    
    # draw_ttrt("data/metrics/spaces.jsonlist",
    #                 game_names=["ideal", "span-pairs-1", "span-pairs-5", "span-pairs-10", "span-pairs-20"],
    #                 table_file = open("data/plots/tables.tex", "w"), threshold = 0.9, 
    #                 name_remap = {"ideal": "0", "span-pairs-1": "1 p", "span-pairs-5": "5 p", "span-pairs-10": "10 p", "span-pairs-20": "20 p"})        

    # draw_ttrt("data/metrics/spaces.jsonlist",
    #                 game_names=["ideal", "dupl-t-5", "dupl-t-10", "dupl-t-50"],
    #                 table_file = open("data/plots/tables.tex", "w"), threshold = 0.9, 
    #                 name_remap = {"ideal": "0", "dupl-t-5": "d=5", "dupl-t-10": "d=10", "dupl-t-50": "d=50"})        

    # draw_ttrt("data/metrics/spaces.jsonlist",
    #                 game_names=["ideal", "dupl-c-5", "dupl-c-10", "dupl-c-50"],
    #                 table_file = open("data/plots/tables.tex", "w"), threshold = 0.9, 
    #                 name_remap = {"ideal": "0", "dupl-c-5": "d=5", "dupl-c-10": "d=10", "dupl-c-50": "d-50"})        

    # draw_ttrt("data/metrics/spaces.jsonlist",
    #                 game_names=["ideal", "dependant-all-1", "dependant-all-2"],
    #                 table_file = open("data/plots/tables.tex", "w"), threshold = 0.9, 
    #                 name_remap = {"ideal": "no dep", "dependant-all-1": "1 dep", "dependant-all-2": "2 deps"})       

    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["ARRA"], aggregation = "all", \
    #             game_names=["ideal", "skew-p-1", "skew-p-2", "skew-p-3", "skew-p-4"], \
    #             sim_names=["des-mea", "des-mea-2", "des-mea-5", "des-mea-100"], \
    #             fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})   #         

    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "all", \
    #                 game_names=[], \
    #                 sim_names=["rand", "des-med", "des-mea"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})       

    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "all", \
    #                 sim_names=["rand", "de-d-0", "de-d-1", "de-d-m", "de-d-g", "de-d-d"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})  

    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["ARRA", "DC", "ARR", "Dup", "R"], aggregation = "all", \
    #                 game_names=["ideal", "span-all-ends-1", "span-all-ends-5", "span-all-ends-10", "span-all-ends-20", "span-pairs-1", "span-pairs-5", "span-pairs-10", "span-pairs-20"], \
    #                 sim_names=["rand", "des-mod", "des-mea", "hc-r-p", "pl-d-0", "de-l", "de-d-1", "de-d-m", "de-d-g", "de-d-d-1", "de-d-d-5"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0},
    #                 rename={"pl-d-0": "pl-d"})  

    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["ARRA", "DC", "ARR", "Dup", "R"], aggregation = "all", \
    #                 game_names=[], \
    #                 sim_names=["rand", "des-mod", "des-mea", "hc-r-p", "pl-d-0", "de-l", "de-d-1", "de-d-m", "de-d-g", "de-d-d-1", "de-d-d-5"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0},
    #                 rename={"pl-d-0": "pl-d"})  

    # with open("data/plots/tables.tex", "w") as f:
    #     draw_latex_threshold_convergence_tbl("data/metrics/spaces.jsonlist", metric_name = "ARRA", 
    #                                 sim_names = [], 
    #                                 game_names = ["GreaterThanGame", "CompareOnOneGame", "FocusingGame", "IntransitiveRegionGame"], 
    #                                 table_file = f,
    #                                 name_remap={"CompareOnOneGame": "Cmp1", "FocusingGame": "Focus", "IntransitiveRegionGame": "Intr", "GreaterThanGame":"GrTh"})
    
    pass


# import numpy as np
# from scipy.stats import spearmanr

# # Example data
# x = np.array([10, 20, 19, 40, 18])
# y = np.array([5, 15, 25, 35, 45])

# # Calculate Spearman rank-order correlation coefficient
# correlation, p_value = spearmanr(x, y)

# print(f"Spearman correlation coefficient: {correlation}")
# print(f"P-value: {p_value}")