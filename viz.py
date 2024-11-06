''' Module dedicated to visualization of dynamics of simulation process '''
# First we would like to see how individuals change in number games - an image of 2d plane with individuals as dots 
#     Candidate-parent --> candidate-child AND test-parent --> test-child = test log of them and who wins 
#     Outputs pngs on given folder to combine them in gif later 

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

legend_font = FontProperties(family='monospace', size=6)
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
        plt.imshow(matrix, cmap='gray', interpolation='nearest')
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
    plt.legend(handles = handles, labels = labels, prop=legend_font, loc='upper left', bbox_to_anchor=(1, 1))
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{param_img_folder}/{name}.{fmt}", format=fmt)    
    plt.clf()

if __name__ == "__main__":
    ''' Test drawings '''
    draw_populations([(1,2)], [(2,3)], [(6,7)], [(5,7)], xrange=(0, 100), yrange=(0, 100))



    
