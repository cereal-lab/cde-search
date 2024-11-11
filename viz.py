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
    sim_names = set(sim_names)
    game_names = set(game_names)
    with open(metrics_file, "r") as f:
        lines = f.readlines()
    runs = [json.loads(line) for line in lines ]
    groups = {}
    present_sim_name = set()
    for run in runs: 
        sim_name = run["sim_name"]
        game_name = run["game_name"]
        if len(sim_names) > 0 and sim_name not in sim_names:
            continue
        if len(game_names) > 0 and game_name not in game_names:
            continue
        present_sim_name.add(sim_name)
        for m in metrics:
            key = (game_name, m)
            groups.setdefault(key, {}).setdefault(sim_name, []).append(run["metric_" + m])
    present_sim_name = list(present_sim_name)
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

    for (game_name, metric_name), sim_values in groups.items():
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
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.xlabel('Simulated time', size=14)
            plt.ylabel(f'{metric_name}, \%', size=14)
            ax.tick_params(axis='both', which='major', labelsize=14)            
            fig.set_tight_layout(True)
            fig.savefig(f"data/plots/{aggregation}-{metric_name}-{game_name}.pdf", format='pdf')
        plt.clf()

def draw_latex_mean_std_tbl(metrics_file: str, metric_name = "ARRA", sim_names = [], game_names = ["CompareOnOneGame"],
                                table_file=None, p_value = 0.05, name_remap = {}):
    ''' metrix_file is in jsonlist format'''
    with open(metrics_file, "r") as f:
        lines = f.readlines()
    runs = [json.loads(line) for line in lines ]
    groups = {}
    sim_names = set(sim_names)
    present_sim_names = set()
    for run in runs: 
        sim_name = run["sim_name"]
        game_name = run["game_name"]
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
        data = np.array([group_by_sims[sim_name][game_name][2] for sim_name in present_sim_names])
        friedman_res = stats.friedmanchisquare(*data)
        sim_ranks = {}
        if friedman_res.pvalue < p_value:
            nemenyi_res = sp.posthoc_nemenyi_friedman(data.T)

            # from tabulate import tabulate
            # names = present_sim_names
            # rows = []
            # for i in range(len(names)):
            #     row = []
            #     row.append(names[i])
            #     for j in range(len(names)):
            #         row.append(nemenyi_res[i][j])
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
                    if nemenyi_res[i][j] < p_value:
                        if f((group_by_sims[sim_name][game_name][0], group_by_sims[sim_name2][game_name][0])):
                            dominations[sim_name].add(sim_name2)
                            dominated_by[sim_name2].add(sim_name)
                        else:
                            dominations[sim_name2].add(sim_name)
                            dominated_by[sim_name].add(sim_name2)  

            rank = 1
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

        for sim_name in present_sim_names:
            sim_rank = sim_ranks.get(sim_name, 1)
            mean, std, _ = group_by_sims[sim_name][game_name]
            group_by_sims[sim_name][game_name] = (mean, std, sim_rank)
            
    avg_sim_ranks = {}
    avg_sim_mean = {}
    for sim_name in present_sim_names:
        avg_sim_ranks[sim_name] = np.mean([group_by_sims[sim_name][game_name][2] for game_name in game_names])
        avg_sim_mean[sim_name] = np.mean([group_by_sims[sim_name][game_name][0] for game_name in game_names])
        if metric_name in ["ARRA", "DC", "ARR"]:
            avg_sim_mean[sim_name] = -avg_sim_mean[sim_name]
    # NOTE: should be sorted by general rank
    rows = sorted(group_by_sims.items(), key=lambda x: (avg_sim_ranks[x[0]], avg_sim_mean[x[0]])) # from best to worst
    print(f"\\begin{{tabular}}{{ r | { '|'.join(['c c'] * len(game_names)) } | c }}", file = table_file)
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
            mean, std, rank = game_groups[game_name]
            print(" & {0:.1f} $\\pm$ {1:.1f} & {2}".format(mean * 100, std * 100, rank), end = "", file = table_file)
        print(f"& {avg_sim_ranks[sim_name]:.1f}", end = "", file = table_file)
        if row_id != len(rows) - 1:
            print(f"\\\\\\hline", file = table_file)
        else:
            print("", file = table_file)
    print(f"\\end{{tabular}}", file = table_file)

def draw_latex_ranks_tbl(metrics_file: str, metric_name = "ARRA", sim_names = [], game_names = ["CompareOnOneGame"],
                                table_file=None, p_value = 0.05, name_remap = {}):
    ''' metrix_file is in jsonlist format'''
    with open(metrics_file, "r") as f:
        lines = f.readlines()
    runs = [json.loads(line) for line in lines ]
    groups = {}
    sim_names = set(sim_names)
    present_sim_names = set()
    for run in runs: 
        sim_name = run["sim_name"]
        game_name = run["game_name"]
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
        data = np.array([group_by_sims[sim_name][game_name][2] for sim_name in present_sim_names])
        friedman_res = stats.friedmanchisquare(*data)
        sim_ranks = {}
        if friedman_res.pvalue < p_value:
            nemenyi_res = sp.posthoc_nemenyi_friedman(data.T)

            # from tabulate import tabulate
            # names = present_sim_names
            # rows = []
            # for i in range(len(names)):
            #     row = []
            #     row.append(names[i])
            #     for j in range(len(names)):
            #         row.append(nemenyi_res[i][j])
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
                    if nemenyi_res[i][j] < p_value:
                        if f((group_by_sims[sim_name][game_name][0], group_by_sims[sim_name2][game_name][0])):
                            dominations[sim_name].add(sim_name2)
                            dominated_by[sim_name2].add(sim_name)
                        else:
                            dominations[sim_name2].add(sim_name)
                            dominated_by[sim_name].add(sim_name2)  

            rank = 1
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

        for sim_name in present_sim_names:
            sim_rank = sim_ranks.get(sim_name, 1)
            mean, std, _ = group_by_sims[sim_name][game_name]
            group_by_sims[sim_name][game_name] = (mean, std, sim_rank)
            
    avg_sim_ranks = {}
    avg_sim_mean = {}
    for sim_name in present_sim_names:
        avg_sim_ranks[sim_name] = np.mean([group_by_sims[sim_name][game_name][2] for game_name in game_names])
        avg_sim_mean[sim_name] = np.mean([group_by_sims[sim_name][game_name][0] for game_name in game_names])
        if metric_name in ["ARRA", "DC", "ARR"]:
            avg_sim_mean[sim_name] = -avg_sim_mean[sim_name]
    # NOTE: should be sorted by general rank
    rows = sorted(group_by_sims.items(), key=lambda x: (avg_sim_ranks[x[0]], avg_sim_mean[x[0]])) # from best to worst
    print(f"\\begin{{tabular}}{{ r | { '|'.join(['c'] * len(game_names)) } | c }}", file = table_file)
    # print(f"\\\\\\hline", file = table_file)
    # header for games
    print(f"Algo", end = "", file = table_file)
    for gid, game_name in enumerate(game_names):    
        print(f"& {gid + 1} ", end = "", file = table_file)
    print(f"& r", end = "", file = table_file)
    print(f"\\\\\\hline", file = table_file)
    # body of the table 
    for row_id, (sim_name, game_groups) in enumerate(rows):
        print(f"{sim_name} ", end = "", file = table_file)
        for game_name in game_names:
            mean, std, rank = game_groups[game_name]
            print(" & {0}".format(rank), end = "", file = table_file)
        print(f"& {avg_sim_ranks[sim_name]:.1f}", end = "", file = table_file)
        if row_id != len(rows) - 1:
            print(f"\\\\\\hline", file = table_file)
        else:
            print("", file = table_file)
    print(f"\\end{{tabular}}", file = table_file)


if __name__ == "__main__":
    ''' Test drawings '''
    # draw_populations([(1,2)], [(2,3)], [(6,7)], [(5,7)], xrange=(0, 100), yrange=(0, 100))
    
    # HC experiments
    # draw_metrics("data/metrics/num-games.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "all", \
    #                 sim_names=["rand", "hc-pmo-i", "hc-pmo-p", "hc-r-i", "hc-r-p"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})    
    
    # draw_metrics("data/metrics/num-games.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "last", \
    #                 sim_names=["rand", "hc-pmo-i", "hc-pmo-p", "hc-r-i", "hc-r-p"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})   

    # DE experiments
    # draw_metrics("data/metrics/num-games.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "all", \
    #                 sim_names=["rand", "de-l", "de-d-0", "de-d-1", "de-d-m", "de-d-g", "de-d-s", "de-d-d-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0}, rename={"de-d-d-100":"de-d-d"})    
    
    # draw_metrics("data/metrics/num-games.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "last", \
    #                 sim_names=["rand", "de-l", "de-d-0", "de-d-1", "de-d-m", "de-d-g", "de-d-s", "de-d-d-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0}, rename={"de-d-d-100":"de-d-d"}) 
    
    # DES
    # draw_metrics("data/metrics/num-games.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "all", \
    #                 sim_names=["rand", "des-mea", "des-med", "des-mea-0", "des-med-0", "des-mea-100", "des-med-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0}, rename={"des-mea-100":"des-mea-1", "des-med-100":"des-med-1"})    
    
    # draw_metrics("data/metrics/num-games.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "last", \
    #                 sim_names=["rand", "des-mea", "des-med", "des-mea-0", "des-med-0", "des-mea-100", "des-med-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0}, rename={"des-mea-100":"des-mea-1", "des-med-100":"des-med-1"})     
    
    #PL
    # draw_metrics("data/metrics/num-games.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "all", \
    #                 sim_names=["rand", "pl-l-0", "pl-l-100", "pl-d-0", "pl-d-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0}, rename={"pl-l-100":"pl-l-1", "pl-d-100":"pl-d-1"})    
    
    # draw_metrics("data/metrics/num-games.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "last", \
    #                 sim_names=["rand", "pl-l-0", "pl-l-100", "pl-d-0", "pl-d-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0}, rename={"pl-l-100":"pl-l-1", "pl-d-100":"pl-d-1"})     
    

    #Best
    # draw_metrics("data/metrics/num-games.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "all", \
    #                 sim_names=["rand", "hc-r-p", "de-d-g", "de-d-d-100", "des-mea-100", "des-med-100", "pl-d-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0}, rename={"de-d-d-100":"de-d-d-1", "des-mea-100":"des-mea-1", "des-med-100":"des-med-1", "pl-d-100":"pl-d-1"})
    
    # draw_metrics("data/metrics/num-games.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "last", \
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
     
    # draw_metrics("data/metrics/num-games.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "all", \
    #                 sim_names=["rand", "hc-pmo-p", "hc-r-p", "de-l", "de-d-0", "de-d-1", "de-d-d-100", "des-mea-100", "des-med-100", "pl-l-100", "pl-d-100"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})

    # with open("data/plots/tables.tex", "w") as f:
    #     draw_latex_mean_std_tbl("data/metrics/num-games.jsonlist", metric_name = "ARRA", 
    #                                 sim_names = [], 
    #                                 game_names = ["GreaterThanGame", "CompareOnOneGame", "FocusingGame", "IntransitiveRegionGame"], 
    #                                 table_file = f,
    #                                 name_remap={"CompareOnOneGame": "Cmp1", "FocusingGame": "Focus", "IntransitiveRegionGame": "Intr", "GreaterThanGame":"GrTh"})
    
    # with open("data/plots/tables.tex", "w") as f:
    #     draw_latex_ranks_tbl("data/metrics/spaces.jsonlist", metric_name = "ARRA", 
    #                                 sim_names = [], 
    #                                 game_names = ["ideal", "skew-p-1", "skew-p-2", "skew-p-3", "skew-p-4", "trivial-25", "skew-t-1", 
    #                                                 "skew-t-5", "skew-c-1", "skew-c-5", "span-all-ends-1", "span-all-ends-5", "span-one-pair-1", 
    #                                                 "span-all-pairs-1", "dupl-t-2", "dupl-t-3", "dupl-t-4", "dupl-t-5", "dupl-c-2", "dupl-c-3", "dupl-c-4", "dupl-c-5",
    #                                                 "dependant-all-1", "dependant-all-2"], 
    #                                 table_file = f)

    # Best of Best
    # draw_metrics("data/metrics/spaces.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "all", \
    #                 sim_names=["de-d-d-0", "de-d-d-1", "de-d-d-2", "de-d-d-5", "de-d-d-100", "de-d-g", "de-d-s", "de-d-0", "de-d-m"], \
    #                 fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})
    # 

    # DE-D-D-X spanned preservation 
    draw_metrics("data/metrics/spaces.jsonlist", metrics = ["ARRA"], aggregation = "all", \
                    sim_names=["de-d-d-0", "de-d-d-1", "de-d-d-2", "de-d-d-5", "de-d-d-100"], \
                    fixed_max = {"DC": 100, "ARR": 100, "ARRA": 100}, fixed_mins={"Dup": 0, "R": 0})        
    
    pass