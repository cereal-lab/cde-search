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
    plt.legend(handles = handles, labels = labels, loc='upper left', bbox_to_anchor=(1, 1))
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{param_img_folder}/{name}.{fmt}", format=fmt)    
    plt.clf()

def draw_metrics(metrics_file: str, metrics = ["DC", "ARR", "ARRA", "Dup", "R"],
                    aggregation = "last", ignore_sim = [], ignore_game = []):
    ''' metrix_file is in jsonlist format'''
    ignore_sim = set(ignore_sim)
    ignore_game = set(ignore_game)
    with open(metrics_file, "r") as f:
        lines = f.readlines()
    runs = [json.loads(line) for line in lines ]
    groups = {}
    for run in runs: 
        sim_name = run["sim_name"]
        game_name = run["game_name"]
        if sim_name in ignore_sim or game_name in ignore_game:
            continue
        for m in metrics:
            key = (game_name, m)
            groups.setdefault(key, {}).setdefault(sim_name, []).append(run["metric_" + m])
    for (game_name, metric_name), sim_values in groups.items():
        for sim_name, values in sim_values.items():
            if aggregation == "last": #collect final metric value - bar or violinplot chart
                for i in range(len(values)):
                    values[i] = values[i][-1]
            else: #for line plot and area between with each batch
                zipped = zip(len(values))
                for i in range(len(values)):
                    new_values = zipped[i]
                    mean = np.mean(new_values)
                    if aggregation == "all":
                        confidence_level = 0.95
                        degrees_freedom = len(new_values) - 1
                        sample_standard_error = stats.sem(new_values)
                        confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean, sample_standard_error)
                        values[i] = (confidence_interval[0], mean, confidence_interval[1])
                    else:
                        values[i] = mean

    for (game_name, metric_name), sim_values in groups.items():
        fig, ax = plt.subplots() 
        if aggregation == "last": #violin plot
            sorted_sim_values = sorted(sim_values.items(), key=lambda x:np.mean(x[1]))
            plt.axhline(y = 50, color = 'tab:gray', linewidth=0.5, linestyle = 'dashed')
            import matplotlib.patches as mpatches
            labels = []
            colors = colormaps.get_cmap("Pastel1").colors
            for sim_id, (sim_name, values) in enumerate(sorted_sim_values):
                data = [v * 100 for v in values]
                v = ax.violinplot(data, showextrema = False, showmedians=True)
                color = colors[sim_id % len(colors)]
                for pc in v['bodies']:
                    pc.set_facecolor(color)
                    pc.set_edgecolor('black')
                v['cmedians'].set_color(color)
                v['cmedians'].set_alpha(1)
                labels.append((mpatches.Patch(color=color), sim_name))
            plt.legend(*zip(*labels), loc=4)
            # plt.xlabel('Spaces', size=14)
            plt.ylabel(f'{metric_name}, \%', size=14)   
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_xticklabels([sim_name for sim_name, _ in sorted_sim_values])       
            fig.set_tight_layout(True)      
            fig.savefig(f"{game_name}-{metric_name}-{aggregation}.pdf", format='pdf')    
            stats_data = np.array([samples for sim_name, samples in sorted_sim_values])
            friedman_res = stats.friedmanchisquare(*stats_data)
            nemenyi_res = sp.posthoc_nemenyi_friedman(stats_data.T) 
            print(f"\n----------------------------------")
            print(f"Stat result for {game_name}, metric {metric_name}")
            print(f"Friedman: {friedman_res}")
            from tabulate import tabulate
            names = [sim_name for sim_name, _ in sorted_sim_values]
            rows = []
            for i in range(len(names)):
                row = []
                row.append(names[i])
                for j in range(len(names)):
                    row.append(nemenyi_res[i][j])
                rows.append(row)
            print(tabulate(rows, headers=["", *names], tablefmt="grid", numalign="center", stralign="center"))
                    
        else: #line and area chart for confidence interval 
            sorted_sim_values = sorted(sim_values.items(), key=lambda x:x[1][-1][1])
            for sim_name, values in sorted_sim_values:
                data = [v[1] for v in values]
                if aggregation == "all":
                    lower = [v[0] for v in values]
                    upper = [v[2] for v in values]
                    ax.fill_between(range(len(data)), lower, upper, alpha=.1, linewidth=0)
                ax.plot(range(len(data)), data, label=sim_name, linewidth=1, markersize=5, marker='o')
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            plt.legend(loc=3)
            plt.xlabel('Simulated time', size=14)
            plt.ylabel(f'{metric_name}, \%', size=14)
            ax.tick_params(axis='both', which='major', labelsize=14)            
            fig.set_tight_layout(True)
            fig.savefig(f"{game_name}-{metric_name}-{aggregation}.pdf", format='pdf')
        plt.clf()

if __name__ == "__main__":
    ''' Test drawings '''
    # draw_populations([(1,2)], [(2,3)], [(6,7)], [(5,7)], xrange=(0, 100), yrange=(0, 100))
    draw_metrics("metrics.jsonlist", metrics = ["DC", "ARR", "ARRA", "Dup", "R"], aggregation = "last")
    pass



    
