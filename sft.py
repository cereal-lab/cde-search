''' Semantic Fluency Test (SFT) and application of dimension extraction onto this game '''

from collections import deque
from functools import partial
from itertools import combinations
import json
import os
from typing import Optional
from gensim.models import KeyedVectors

from de import extract_dims_np
from nsga2 import get_pareto_front_indexes
os.environ['GENSIM_DATA_DIR'] = os.path.join(os.getcwd(), '.cache')
os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.cache')
import gensim.downloader as gsl
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
# distance measures


pretrained_w2v = None # cache, loaded on first request

#problems with 'date plum', 'alimentive' and 'granny_smith_apple'
def w2v_embedding(phrase):
    ''' Word2Vec distance between two phrases '''
    global pretrained_w2v
    if not pretrained_w2v:
        w2v_path = gsl.load('word2vec-google-news-300', return_path=True)
        pretrained_w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    p = phrase.replace(' ', '_')
    if p in pretrained_w2v:
        v = pretrained_w2v[p]
    else:
        words = p.split('_')
        vects = np.array([pretrained_w2v[w] for w in words if w in pretrained_w2v])
        v = np.mean(vects, axis=0)
    # c = np.zeros_like(v) if context is None else pretrained_w2v[context]
    # if p2 in pretrained_w2v:
    #     v2 = pretrained_w2v[p2]
    # else:
    #     v2 = get_avg(p2, pretrained_w2v)
    # v1 = v1 - c
    # v2 = v2 - c
    # res = 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return v

# def diff_v(context, phrase1, phrase2):
#     global pretrained_w2v
#     if not pretrained_w2v:
#         w2v_path = gsl.load('word2vec-google-news-300', return_path=True)
#         pretrained_w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
#     c = pretrained_w2v[context]
#     p1 = phrase1.replace(' ', '_')
#     if p1 in pretrained_w2v:
#         v1 = pretrained_w2v[p1]
#     else:
#         v1 = get_avg(p1, pretrained_w2v)
#     return v1 - c

loaded_feature_extractor = None # cached pipeline
def lm_embedding(checkpoint, phrase):
    global loaded_feature_extractor
    if loaded_feature_extractor is None:
        loaded_feature_extractor = SentenceTransformer(checkpoint)
    e = loaded_feature_extractor.encode(phrase)
    # c = np.zeros_like(e) if context is None else loaded_feature_extractor.encode(context)
    # e2 = loaded_feature_extractor.encode(phrase2)
    # res = loaded_feature_extractor.similarity(e1 - c, e2 - c)
    # rv = res.item()
    return e.numpy()

# def lm_encoding(checkpoint, prefix, phrase):
#     global loaded_feature_extractor
#     if loaded_feature_extractor is None:
#         loaded_feature_extractor = SentenceTransformer(checkpoint)
#     e = loaded_feature_extractor.encode(prefix + phrase)
#     return e.tolist()

sent_embedding = partial(lm_embedding, 'all-MiniLM-L6-v2')
# sent_lm_encoding = partial(lm_encoding, 'all-MiniLM-L6-v2')

# build interaction matrix from the dataset - there could be different ways to build interactions 
def no_cut(lists):
    return lists 

def cut_list(max_len, list):
    return list[:max_len]

def inv_cos_dist(v1, v2):
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def agg_cumavg(dists, *_):
    return (np.cumsum(dists) / np.arange(1, len(dists) + 1)).tolist()

def no_agg(dists, *_):
    return dists

# def agg_cat(dists, context):
#     whole_seq = " ".join(dists)

def agg_i_cos_centroid(v, prev, c):
    if len(prev) == 0:
        return []
    if c is None:
        c = np.zeros_like(v)
    v1 = np.mean(prev, axis=0)
    return [inv_cos_dist(v - c, v1 - c)]

def agg_i_cos_mean(v, prev, c):
    if len(prev) == 0:
        return [] 
    if c is None: 
        c = np.zeros_like(v)
    vc = v - c
    dists = [inv_cos_dist(vc, p - c) for p in prev]
    return [np.mean(dists)]

def agg_i_cos_cat(v, prev, c):
    if c is None:
        c = np.zeros_like(v)
    vc = v - c
    dists = [inv_cos_dist(vc, p - c) for p in prev]
    return dists

def agg_i_cos_ctx(v, prev, c):
    return [inv_cos_dist(v, c)]

def agg_i_v(v, prev, c):
    return [v]

def get_interactions(context, cut_list, mapper, agg_i, final_agg, s: list[str]):
    ''' Build interactions from SFT sequence s in different ways '''
    context_repr = None if context is None else mapper(context)
    s_repr = [mapper(el) for el in cut_list(s)]
    res = []
    for i in range(len(s_repr)):
        i_prev = [s_repr[j] for j in range(i)]
        res.extend(agg_i(s_repr[i], i_prev, context_repr))
    final_res = final_agg(res, context_repr)
    return final_res

# def abs_interactions(cut_lists, final_agg, dist_measure, s: list[str]):
#     ''' Relative to category word usually '''
#     s = cut_lists(s)
#     res = [ dist_measure(s[i]) for i in range(len(s)) ]
#     final_res = final_agg(res)
#     return final_res

def whole_interaction(cut_lists, dist_measure, s: list[str]):
    ''' Build whole string and computes its interaction representation '''
    s = cut_lists(s)
    whole = ' '.join(s)
    return dist_measure(whole)

def load_dataset(group: Optional[str] = None):
    ''' group could be Experiment1, Experiment2, Experiment3 '''
    dataset = pd.read_csv('./data/sft/snafu_sample_cleaned.csv')
    # # filtering - select all id, listnum and category columns where RT column is 0 in the dataset 
    # to_filter_out = dataset[dataset['RT'] == 0][['id', 'listnum', 'category']].drop_duplicates()
    # # Merge with indicator
    # merged = dataset.merge(to_filter_out, on=['id', 'listnum', 'category'], how='left', indicator=True)

    # # Filter out rows that are in to_filter_out
    # filtered_dataset = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    filtered_dataset = dataset
    if group is not None:
        filtered_dataset = filtered_dataset[filtered_dataset['group'] == group]
    res = {} # category to participant id to list id to list of words
    for _, row in filtered_dataset.iterrows():
        res.setdefault(row['category'], {}).setdefault(row['id'], {}) \
            .setdefault(row['listnum'], []).append((row['item'], row['RT'], row['RTstart'])) 
    res = {category:{pid:[[(w, t) for w, t, _ in sorted(l, key=lambda x: x[-1])] for l in lists.values()] for pid, lists in participants.items()} for category, participants in res.items() }
    return res

loaded_dataset = None # cached SFT dataset (postprocessed)
def get_from_dataset(category: str, lid: int):
    ''' Get SFT list from the dataset '''
    global loaded_dataset
    if loaded_dataset is None:
        loaded_dataset = load_dataset()
    participans = list(loaded_dataset[category].keys())
    lists = [[w for w, _ in loaded_dataset[category][p][lid]] for p in participans]
    return lists, participans

def compute_and_save_ints(group = 'fruits', max_seq_len = 8, list_num = 1):
    group = "fruits"
    get_ints = partial(get_interactions, group)
    loaded_lists, _ = get_from_dataset(group, list_num)
    min_l = min(len(l) for l in loaded_lists)
    int_fs = {
        # "w2v_flow_all": partial(get_ints, partial(cut_list, 8), w2v_embedding, agg_i_cos_mean, no_agg),
        # "w2v_flow": partial(get_ints, partial(cut_list, 8), w2v_embedding, agg_i_cos_mean, agg_cumavg),
        # "w2v_all": partial(get_ints, partial(cut_list, 8), w2v_embedding, agg_i_cos_cat, no_agg),
        # "w2v_cdot": partial(get_ints, partial(cut_list, 8), w2v_embedding, agg_i_cos_centroid, agg_cumavg),
        # "w2v_abs":  partial(get_ints, partial(cut_list, 8), w2v_embedding, agg_i_cos_ctx, no_agg),
        # "w2v_abs_avg":  partial(get_ints, partial(cut_list, 8), w2v_embedding, agg_i_cos_ctx, agg_cumavg),

        # "lm_flow": partial(get_ints, partial(cut_list, min_l), sent_embedding, agg_i_cos_mean, agg_cumavg),
        # "lm_all": partial(get_ints, partial(cut_list, min_l), sent_embedding, agg_i_cos_cat, no_agg),
        # "lm_cdot": partial(get_ints, partial(cut_list, min_l), sent_embedding, agg_i_cos_centroid, agg_cumavg),
        # "lm_abs":  partial(get_ints, partial(cut_list, min_l), sent_embedding, agg_i_cos_ctx, no_agg),
        # "lm_abs_avg":  partial(get_ints, partial(cut_list, min_l), sent_embedding, agg_i_cos_ctx, agg_cumavg),

        # "repr": partial(whole_interaction, no_cut, partial(sent_lm_encoding, '')),
        # "repr_g": partial(whole_interaction, no_cut, partial(sent_lm_encoding, group + " ")),
    }
    for int_f_name, int_f in int_fs.items():
        ints = []
        for p in loaded_lists:
            i = int_f(p)
            ints.append(i)
        print(f"dumping interactions on {group}, {int_f_name}")
        with open(f'./data/sft/interactions/{int_f_name}-{group}-{list_num}.json', 'w') as f:
            f.write(json.dumps(ints))

def compute_word_cloud(group = 'fruits', max_seq_len = 8, list_num = 1):
    group = "fruits"
    loaded_lists, _ = get_from_dataset(group, list_num)
    all_words = set() 
    for p in loaded_lists:
        for w in p[:max_seq_len]:
            all_words.add(w)
    emb = w2v_embedding
    c = emb(group)
    all_words = list(all_words)
    all_ints = []
    for w in all_words:
        ints = []
        v1 = emb(w) - c
        for w2 in all_words:
            v2 = emb(w2) - c
            i = inv_cos_dist(v1, v2)
            ints.append(i)
        all_ints.append(ints)
    with open(f'./data/sft/interactions/cloud-w2v-{group}-{list_num}.json', 'w') as f:
        f.write(json.dumps({"ints": all_ints, "vocab": all_words}))

def compute_dims(file_name):
    ''' Compute dimensions from interactions '''
    with open(file_name, 'r') as f:
        ints = json.loads(f.read())
    ints = np.round(np.array(ints), decimals=2)
    for i in range(2, len(ints[0])+1):
        dims, orig, spanned = extract_dims_np(ints[:, :i])
        print(dims)
    return ints

def compute_all_fronts(file_name):
    with open(file_name, 'r') as f:
        ints = json.loads(f.read())
    ints = np.round(np.array(ints), decimals=2)
    for i in range(2, len(ints[0]) + 1):
        front_ids = get_pareto_front_indexes(ints[:, :i])
        print(front_ids)
    return ints    


import networkx as nx
import matplotlib.pyplot as plt
def compute_and_draw_fronts(file_name, step = None):
    with open(file_name, 'r') as f:
        ints = json.loads(f.read())
    ints = np.round(np.array(ints), decimals=2)
    if step is not None:
        ints = ints[:, :step]
    fronts = []
    all_fronts_indicies = np.array([], dtype=int)
    while True:
        front_ids = get_pareto_front_indexes(ints, exclude_indexes = all_fronts_indicies)
        if len(front_ids) == 0:
            break
        fronts.append(front_ids)
        all_fronts_indicies = np.concatenate([all_fronts_indicies, front_ids])
    front_dominations = []
    for prev_front, next_front in zip(fronts, fronts[1:]):
        d = np.all(ints[prev_front][:, np.newaxis] >= ints[next_front], axis=-1)
        front_dominations.append(d)
    sorted_fronts = []
    for front, front_d in zip(fronts, front_dominations):
        num_doms = np.sum(front_d, axis=-1)
        nf = [f for f, _ in sorted(list(zip(front, num_doms)), key=lambda x: x[-1], reverse=True)]
        sorted_fronts.append(nf)
    sorted_fronts.append(fronts[-1])
    front_dominations = []
    for prev_front, next_front in zip(sorted_fronts, sorted_fronts[1:]):
        d = np.all(ints[prev_front][:, np.newaxis] >= ints[next_front], axis=-1)
        front_dominations.append(d) 
    G = nx.DiGraph()
    pos = {}
    for i, front in enumerate(sorted_fronts):
        num_nodes = len(front)
        y_positions = np.linspace(-num_nodes / 2, num_nodes / 2, num_nodes)
        for j, node in enumerate(front):
            G.add_node(node)
            pos[node] = (i, y_positions[j])
            if i < len(sorted_fronts) - 1:
                for k, prev_node in enumerate(sorted_fronts[i + 1]):
                    if front_dominations[i][j][k]:
                        G.add_edge(prev_node, node)

    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=8, font_color='black', font_weight='bold', edge_color='gray', linewidths=0.5, arrowsize=10)
    plt.title("Pareto Fronts and domination")
    plt.savefig("./data/sft/interactions/fronts-w2v_flow-fruits-1.png")
    print(fronts)
    return ints    

def compute_and_draw_trends(file_name, step = None, num_fronts = 2):
    with open(file_name, 'r') as f:
        ints = json.loads(f.read())
    ints = np.round(np.array(ints), decimals=2)
    if step is not None:
        ints = ints[:, :step]
    fronts = []
    all_fronts_indicies = np.array([], dtype=int)
    while True:
        front_ids = get_pareto_front_indexes(ints, exclude_indexes = all_fronts_indicies)
        if len(front_ids) == 0:
            break
        fronts.append(front_ids)
        all_fronts_indicies = np.concatenate([all_fronts_indicies, front_ids])
    front_dominations = []
    for prev_front, next_front in zip(fronts, fronts[1:]):
        d = np.all(ints[prev_front][:, np.newaxis] >= ints[next_front], axis=-1)
        front_dominations.append(d)
    sorted_fronts = []
    for front, front_d in zip(fronts, front_dominations):
        num_doms = np.sum(front_d, axis=-1)
        nf = [f for f, _ in sorted(list(zip(front, num_doms)), key=lambda x: x[-1], reverse=True)]
        sorted_fronts.append(nf)
    sorted_fronts.append(fronts[-1])
    plt.figure(figsize=(10, 6))
    for i, front in enumerate(sorted_fronts[:num_fronts]):
        linestyle = '-' if i == 0 else '--'
        marker = 'o' if i == 0 else 'x'
        for j, ind in enumerate(front):
            plt.plot([i + 2 for i in range(ints.shape[1])], ints[ind].tolist(), label=f"{ind}", linestyle = linestyle, marker=marker, markersize=8)  
    plt.title('Forward flow change with new words')
    plt.xlabel('Num words')
    plt.ylabel('Forward flow')   
    plt.legend()
    plt.savefig("./data/sft/interactions/trends-w2v_flow-fruits-1.png")

def get_cloud_words_ordered(file_name, start_word):
    with open(file_name, 'r') as f:
        cloud = json.loads(f.read())
    ints = cloud['ints']
    vocab = cloud['vocab']
    idx = vocab.index(start_word)
    res = sorted(list(zip(vocab, ints[idx])), key=lambda x: x[-1], reverse=True)
    print(res)
    return res

def exec_ideal_participant(file_name, start_word, max_len = 8, num_beams = 5):
    with open(file_name, 'r') as f:
        cloud = json.loads(f.read())
    ints = np.array(cloud['ints'])
    vocab = cloud['vocab']
    selected_words = []
    cur_idx = vocab.index(start_word)
    # mask = np.ones(len(vocab), dtype=bool) 
    # mask[cur_idx] = False 
    indexes = np.arange(len(vocab))
    selected_words.append([cur_idx])
    # cur_len = 1
    # memo = {}
    q = deque(selected_words)
    res = []
    while len(q) > 0:
        words = q.popleft()
        if len(words) == max_len:
            res.append(words)
            continue
        words_set = set(words)
        possible_words = [i for i in indexes if i not in words_set]
        new_word_scores = [ (new_wid, np.mean(ints[new_wid, words])) for new_wid in possible_words ]
        new_word_scores.sort(key=lambda x: x[-1], reverse=True)
        for new_wid, new_wd in reversed(new_word_scores[:num_beams]):
            new_words = [*words, new_wid]
            q.appendleft(new_words)
    

    seq_flows = [] 
    words_seqs = []
    for s in res:
        words_seqs.append([vocab[i] for i in s])
        means = [np.mean(ints[s[l-1], s[:l-1]]) for l in range(2, len(s) + 1)]
        ds = agg_cumavg(means)
        seq_flows.append(ds)
    # print(words_seqs)
    # print(seq_flows)
    words_seqs = np.array(words_seqs)
    all_strategies = np.array(seq_flows)
    best_strategy_idx = get_pareto_front_indexes(all_strategies)
    print(best_strategy_idx)
    best_strategies = all_strategies[best_strategy_idx]
    best_seqs = words_seqs[best_strategy_idx]
    return best_strategies, best_seqs

if __name__ == "__main__":
    # compute_word_cloud("fruits")
    exec_ideal_participant("./data/sft/interactions/cloud-w2v-fruits-1.json", "apple", max_len=8, num_beams=3)
    # compute_and_draw_trends("./data/sft/interactions/w2v_flow-fruits-1.json")
    pass


# w1, w2, .... w8
# w1: --> [  ]
# w2: --> [  ]

# w1 w2 w3 w4 w5 

i = 0
from time import time
start = time()
for _ in range(100000000):
    i += 1
print(time() - start)

# IDEAl Pareto comparison with n-objectives relaxed (allowing non pareto relation by n objectives)