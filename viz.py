''' Module dedicated to visualization of dynamics of simulation process '''
# First we would like to see how individuals change in number games - an image of 2d plane with individuals as dots 
#     Candidate-parent --> candidate-child AND test-parent --> test-child = test log of them and who wins 
#     Outputs pngs on given folder to combine them in gif later 

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

legend_font = FontProperties(family='monospace', size=6)
param_img_folder = "img"

def draw_populations(candidate_parents, candidate_children, test_parents = [], test_children = [], prev = [],
                        xrange = None, yrange = None, name="fig", fmt = "png", title = None, legend = []):
    ''' creates an image of populaltion of 2-d numbers to track the dynamics '''
    data = [(candidate_parents, 'o', '#151fd6', 30, 'C'), 
            (candidate_children, 'o', '#28a4c9', 10, None), 
            (test_parents, 'H', '#159e1b', 30, 'T'), 
            (test_children, 'H', '#85ba6a', 10, None)]
    plt.ioff()
    if len(prev) > 0:
        x, y = zip(*prev)
        plt.scatter(x, y, marker='o', s=10, c='#cfcfcf', alpha=0.5)
    # legend_labels = []
    # legend_handles = []
    handle_index = {}
    for xy, marker, color, scale, key in data:
        if len(xy) == 0:
            continue
        x, y = zip(*xy)
        # if xy2 is not None:
        #     for p1, p2 in list(zip(xy, xy2))[:num_legend]:
        #         label = f"{p1} vs {p2}"
        #         legend_labels.append(label)
        #         arrow_properties = dict(arrowstyle='->', linestyle='dashed', linewidth=1, color='gray', alpha = 0.5)
        #         plt.annotate('', p2, p1, arrowprops=arrow_properties)                
        #         # plt.arrow(p1[0], p1[1], dxy[0], dxy[1], width=0.05, length_includes_head=True)
        scatter = plt.scatter(x, y, marker=marker, s=scale, c=color)
        handle_index[key] = scatter
        # if xy2 is not None:
        #     for _ in range(num_legend):
        #         legend_handles.append(scatter)
    if xrange is not None: 
        plt.xlim(xrange[0], xrange[1])
    if yrange is not None: 
        plt.ylim(yrange[0], yrange[1])  
    handles = [handle_index[l[0]] for l in legend]
    legend = [l[2:] for l in legend]
    plt.legend(handles = handles, labels = legend, prop=legend_font, loc='upper left', bbox_to_anchor=(1, 1))
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{param_img_folder}/{name}.{fmt}", format=fmt)    
    plt.clf()

if __name__ == "__main__":
    ''' Test drawings '''
    draw_populations([(1,2)], [(2,3)], [(6,7)], [(5,7)], xrange=(0, 100), yrange=(0, 100))



    
