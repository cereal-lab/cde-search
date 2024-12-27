''' Running of DOC, DOF and CSE on benchmarks from bool and alg0 '''

from itertools import combinations, cycle, product, zip_longest
from typing import Optional
import numpy as np
from rnd import default_rnd, seed

from derivedObj import matrix_factorization, xmean_cluster
from nsga2 import run_nsga2, run_front_coverage
from utils import write_metrics
from functools import partial

num_runs = 30

from domain_alg0 import build_vars, disc, f_a1, f_a2, f_a3, f_a4, f_a5, malcev
from gp import gp_evaluate, ramped_half_and_half, run_koza, subtree_breed, subtree_mutation, subtree_crossover, tournament_selection
from domain_bool import cmp, maj, mul, par, default_funcs, build_lines

# included setups: GP - classic Koza
#   initialization - ramped half-and-half
#   mutation: subtree replace mutation with 0.1 probability
#   crossover: subtree crossover with 0.9 probability
#   fitness_fn: distance to ideal L1 or L2 norm 

archive_size = 10
max_generations = 100
population_size = 1000

# included setups: NSGA2
#   initialization - ramped half-and-half
#   breed: same as above for GP
#   fitness_fn: is multiobjective and is defined by either clustering or factoriation from interaction semantics

# included setups: CSE - TODO: CSE based evolutionary algo vs NSGA2 

bool_benchmark_problems = [
    ("cmp6", partial(cmp, 6)), ("cmp8", partial(cmp, 8)), ("maj6", partial(maj, 6)), ("mux6", partial(mul, 2)), ("par5", partial(par, 5))
]

def build_bool_bench_problem(problem): 
    inputs, outputs = problem()
    return outputs, default_funcs, build_lines(inputs)

bool_benchmark = [ (name, partial(build_bool_bench_problem, problem)) for name, problem in bool_benchmark_problems]

alg0_benchmark_problems = [ disc, malcev ]

def build_alg0_bench_problem(problem, f_a): 
    inputs, outputs = problem()
    return outputs, [f_a], build_vars(inputs)

alg0_benchmark = [ (problem.__name__ + str(i + 1), partial(build_alg0_bench_problem, problem, f_a)) for problem in alg0_benchmark_problems for i, f_a in enumerate([f_a1, f_a2, f_a3, f_a4, f_a5])]

benchmark = bool_benchmark + alg0_benchmark

def depth_fitness(prev_fitness, interactions, *, population = [], **kwargs):
    for i in range(len(population)):
        prev_fitness[i] = population[i].get_depth()

def hamming_distance_fitness(prev_fitness, interactions, *, eval_mask = None, **kwargs):
    if eval_mask is None:
        eval_mask = np.ones(interactions.shape[0], dtype=bool)
    prev_fitness[eval_mask] = np.sum(1 - interactions[eval_mask], axis = 1)

def ifs_fitness(prev_fitness, interactions, **kwargs):
    counts = np.sum((interactions[:, None] == interactions) & (interactions == 1), axis = 0).astype(float)
    counts[counts > 0] = 1 / counts[counts > 0]
    ifs = np.sum(counts, axis=1)
    prev_fitness[:] = -ifs

# ifs_fitness(0, np.array([[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 1]]))

def hypervolume_fitness(prev_fitness, interactions, *, derived_objectives = [], **kwargs):
    prev_fitness[:] = -np.prod(derived_objectives, axis = 1)

def weighted_hypervolume_fitness(prev_fitness, interactions, *, derived_objectives = [], test_clusters = [], **kwargs):
    weights = np.array([np.sum(interactions[:, tc] == 1, axis=1) for tc in test_clusters]).T
    weighted_objs = weights * derived_objectives
    prev_fitness[:] = np.prod(weighted_objs, axis = 1)

# a = np.array([ 1, 0, 1])
# aa = np.array([[ 1, 1, 1], [0, 0, 1], [1, 0, 1]])    
# np.sum((a == aa)[:, a == 1], axis=0)

def program_test_interactions(outputs, program_outputs):
    return (program_outputs == outputs).astype(int)

def get_metrics(main_fitness_index: int, fitnesses, population):
    ''' Get the best program in the population '''
    fitness_order = np.lexsort(fitnesses.T[::-1])
    best_index = fitness_order[0]
    best_fitness = fitnesses[best_index]
    best = population[best_index]
    is_best = best_fitness[main_fitness_index] == 0
    return is_best, (best, *best_fitness)

# np.lexsort(np.array([np.array([1, 2, 3]), np.array([3, 2, 1])]).T[::-1])

# np.stack([np.array([1, 2, 3]), np.array([3, 2, 1])], axis=0)

def full_objectives(interactions):
    return interactions, dict()

def doc_objectives(interactions):
    test_clusters, centers = xmean_cluster(interactions.T.tolist(), 4)
    derived_objectives = np.array(centers).T
    return derived_objectives, dict(test_clusters = test_clusters)

def rand_objectives(interactions):
    rand_ints = default_rnd.random(interactions.shape)
    res = doc_objectives(rand_ints)
    return res

def dof_w_objectives(k, alpha, interactions):
    if alpha < 1: 
        num_columns_to_take = int(alpha * interactions.shape[1])
        random_column_indexes = default_rnd.choice(interactions.shape[1], num_columns_to_take, replace = False)
        interactions = interactions[:, random_column_indexes]
    W, _, _ = matrix_factorization(interactions.tolist(), k)
    return W, {}

def dof_wh_objectives(k, alpha, interactions):
    if alpha < 1: 
        num_columns_to_take = int(alpha * interactions.shape[1])
        random_column_indexes = default_rnd.choice(interactions.shape[1], num_columns_to_take, replace = False)
        interactions = interactions[:, random_column_indexes]    
    W, H, _ = matrix_factorization(interactions.tolist(), k)
    objs = np.sum(W[:, :, None] * H, axis = -1)
    return objs, {}

def do_pca_abs_objectives(num_components, interactions):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=num_components)
    components = pca.fit_transform(interactions)
    return np.abs(components), {}

def do_pca_diff_objectives(num_components, interactions):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=num_components)
    components = pca.fit_transform(interactions)
    min_components = np.min(components, axis = 0)
    return (components - min_components + 0.1), {}

def do_feature_objectives(interactions):
    ''' we extract derived objectives based on program features: 
        - how many tests the program passes 
        - the maximal difficulty of the passed test by the program 
    '''
    test_stats = np.sum(interactions, axis = 0)
    whole_program_weights = interactions * test_stats
    whole_program_weights[whole_program_weights == 0] = np.inf
    program_stats = np.sum(interactions, axis = 1)
    program_min_test_difficulty = np.min(whole_program_weights, axis = 1)
    res = np.stack([program_stats, interactions.shape[0] - program_min_test_difficulty], axis = 1)
    return res, {}

def select_test_hardest(ints: np.ndarray, allowed_test_mask: np.ndarray):
    allowed_test_indexes = np.where(allowed_test_mask)[0]
    test_stats = np.sum(ints[:, allowed_test_mask], axis = 0) #sum accross columns to see how many programs pass each test    
    test_stats[test_stats == 0] = np.inf
    test_id_id = np.argmin(test_stats)    
    test_id = allowed_test_indexes[test_id_id]
    return test_id

def select_test_easiest(ints: np.ndarray, allowed_test_mask: np.ndarray):
    allowed_test_indexes = np.where(allowed_test_mask)[0]
    test_stats = np.sum(ints[:, allowed_test_mask], axis = 0) #sum accross columns to see how many programs pass each test    
    test_id_id = np.argmax(test_stats)
    test_id = allowed_test_indexes[test_id_id]
    return test_id

def select_test_random(ints: np.ndarray, allowed_test_mask: np.ndarray):
    allowed_test_indexes = np.where(allowed_test_mask)[0]
    return default_rnd.choice(allowed_test_indexes)

def select_program_best_by_test(test_selection, ints: np.ndarray, allowed_program_mask: np.ndarray, allowed_test_mask: np.ndarray):
    selected_test_id = test_selection(ints, allowed_test_mask)
    allowed_program_indexes = np.where(allowed_program_mask)[0]
    program_candidate_id_ids = np.where(ints[allowed_program_indexes, selected_test_id] != 0)[0]
    program_candidate_ids = allowed_program_indexes[program_candidate_id_ids]
    allowed_test_indexes = np.where(allowed_test_mask)[0]
    program_stats = np.sum(ints[program_candidate_ids][:, allowed_test_indexes], axis = 1)
    if len(program_stats) == 0:
        return None, None
    best_program_solved_test_num = np.max(program_stats)
    best_program_id_ids = np.where(program_stats == best_program_solved_test_num)[0]
    best_program_ids = program_candidate_ids[best_program_id_ids] #the program that passes the most tests including selected rare test
    best_program_id = default_rnd.choice(best_program_ids, 1)[0]
    allowed_program_mask[best_program_id] = False
    solved_test_ids_by_best = np.where(ints[best_program_id] != 0)[0]
    allowed_test_mask[solved_test_ids_by_best] = False
    return best_program_id, selected_test_id

def select_program_random(test_selection, ints: np.ndarray, allowed_program_mask: np.ndarray, allowed_test_mask: np.ndarray):
    selected_test_id = test_selection(ints, allowed_test_mask)
    allowed_program_indexes = np.where(allowed_program_mask)[0]
    program_candidate_id_ids = np.where(ints[allowed_program_indexes, selected_test_id] != 0)[0]
    if len(program_candidate_id_ids) == 0:
        return None, None
    program_candidate_id_id = default_rnd.choice(program_candidate_id_ids, 1)[0]
    best_program_id = allowed_program_indexes[program_candidate_id_id]
    allowed_program_mask[best_program_id] = False
    solved_test_ids_by_best = np.where(ints[best_program_id] != 0)[0]
    allowed_test_mask[solved_test_ids_by_best] = False
    return best_program_id, selected_test_id

# def select_program_totally_best(ints: np.ndarray, allowed_program_mask: np.ndarray, allowed_test_mask: np.ndarray, num_hardest_tests = 2):
#     # selected_test_id = test_selection(ints, allowed_test_mask)
#     allowed_test_indexes = np.where(allowed_test_mask)[0]
#     test_stats = np.sum(ints[:, allowed_test_mask], axis = 0) #sum accross columns to see how many programs pass each test    
#     test_stats[test_stats == 0] = np.inf

#     allowed_program_indexes = np.where(allowed_program_mask)[0]
#     whole_program_weights = ints[allowed_program_indexes][:, allowed_test_indexes] * test_stats
#     whole_program_weights[whole_program_weights == 0] = np.inf
#     whole_program_weights_sorted = np.sort(whole_program_weights, axis = 1)
#     whole_program_weights_subset = whole_program_weights_sorted[:, :num_hardest_tests]
#     program_indexes = np.lexsort(whole_program_weights_subset.T)

#     # program_win_counts = np.sum(whole_program_weights != 0, axis = 1)
#     # program_weights = np.sum(whole_program_weights, axis = 1) 
#     # program_weights[program_win_counts == 0] = np.inf
#     # program_weights[program_win_counts != 0] /= program_win_counts[program_win_counts != 0]
#     # best_program_min_w = np.min(program_weights)
#     # best_program_id_ids = np.where(program_weights == best_program_min_w)[0]
#     # best_program_ids = allowed_program_indexes[best_program_id_ids] #the program that passes the most tests including selected rare test
#     best_program_id_id = program_indexes[0] #default_rnd.choice(best_program_ids, 1)[0]
#     best_program_id = allowed_program_indexes[best_program_id_id]
#     allowed_program_mask[best_program_id] = False
#     solved_test_ids_by_best = np.where(ints[best_program_id] != 0)[0]
#     allowed_test_mask[solved_test_ids_by_best] = False
#     best_test_stats = np.copy(whole_program_weights[best_program_id])
#     best_test_stats[best_test_stats == 0] = np.inf
#     selected_test_id_id = np.argmin(best_test_stats)
#     selected_test_id = allowed_test_indexes[selected_test_id_id]
#     return best_program_id, selected_test_id


# def select_program_shallow(ints: np.ndarray, selected_test_id: int, population: list, allowed_program_mask: np.ndarray, allowed_test_mask: np.ndarray):
#     allowed_program_indexes = np.where(allowed_program_mask)[0]
#     program_candidate_id_ids = np.where(ints[allowed_program_indexes, selected_test_id] != 0)[0]
#     program_candidate_ids = allowed_program_indexes[program_candidate_id_ids]
#     program_depths = np.array([population[i].get_depth() for i in program_candidate_ids])
#     best_program_id = program_candidate_ids[np.argmin(program_depths)]
#     allowed_program_mask[best_program_id] = False
#     solved_test_ids_by_best = np.where(ints[best_program_id] != 0)[0]
#     allowed_test_mask[solved_test_ids_by_best] = False
#     return best_program_id    

full_coverage_subgroups = iter([])#cyclic iterator
full_coverage_selected = iter([])
def full_coverage_selection(population, selection_size = 2):
    global full_coverage_subgroups
    global full_coverage_selected
    if (next_index := next(full_coverage_selected, None)) is None:
        subgroup_start, subgroup_size = next(full_coverage_subgroups)
        full_coverage_selected = iter(subgroup_start + default_rnd.choice(subgroup_size, min(subgroup_size, selection_size), replace = False))
        next_index = next(full_coverage_selected)
    best = population[next_index]
    return best

def select_full_test_coverage(test_program_selection, selection_size, interactions):
    ''' There are different ways to have full test coverage    
    Does the selection of test metter?
    Hypothesis 1. selects hardest test and then best program that solves it 
    Hypothesis 2. selects easiest test and then best program that solves it
    Hypothesis 3. selects random test and then best program that solves it'''
    global full_coverage_subgroups
    ints = interactions #np.copy(interactions)    
    # non_zero_test_ids  = np.where(test_stats != 0)[0]
    selected = []
    allowed_test_mask = np.any(ints, axis = 0) #solvable tests currently
    allowed_test_mask_all = np.copy(allowed_test_mask) #only have False for really picked tests, while allowed_test_mask set to false on program covering the test
    allowed_program_mask = np.any(ints, axis = 1) #programs that solve at least one test
    groups = [0]
    while (len(selected) < selection_size) and np.any(allowed_test_mask_all) and np.any(allowed_program_mask):
        # test_with_min_programs_id = test_selection(ints, allowed_test_mask)
        # best_program_id = program_selection(ints, test_with_min_programs_id, population, allowed_program_mask, allowed_test_mask)
        # allowed_test_mask_all[test_with_min_programs_id] = False
        # test_with_min_programs_id = test_selection(ints, allowed_test_mask)
        best_program_id, test_with_min_programs_id = test_program_selection(ints, allowed_program_mask, allowed_test_mask)
        if best_program_id is not None:
            allowed_test_mask_all[test_with_min_programs_id] = False
            groups[-1] += 1
            selected.append(best_program_id)
        if best_program_id is None or not np.any(allowed_test_mask): #try second, thrird etc test coverages
            allowed_test_mask = np.copy(allowed_test_mask_all)
            groups.append(0)
            # NOTE: at this point it makes sense to form coverage groups.
    if groups[-1] == 0:
        groups.pop()
    indexes = list(zip([0, *np.cumsum(groups).tolist()], groups))
    full_coverage_subgroups = cycle(indexes)
    return np.array(selected)

full_test_coverage_hardest_test_best_program = partial(select_full_test_coverage, partial(select_program_best_by_test, select_test_hardest)) 

full_test_coverage_easiest_test_best_program = partial(select_full_test_coverage, partial(select_program_best_by_test, select_test_easiest))

full_test_coverage_random_test_best_program = partial(select_full_test_coverage, partial(select_program_best_by_test, select_test_random))

full_test_coverage_hardest_test_rand_program = partial(select_full_test_coverage, partial(select_program_random, select_test_hardest)) 

full_test_coverage_easiest_test_rand_program = partial(select_full_test_coverage, partial(select_program_random, select_test_easiest))

full_test_coverage_random_test_rand_program = partial(select_full_test_coverage, partial(select_program_random, select_test_random))

# full_test_coverage_best_program = partial(select_full_test_coverage, select_program_totally_best)

# def select_rand_coverage(selection_size, interactions): 
#     ''' Randomly samples programs from interactions '''
#     program_ids = default_rnd.choice(interactions.shape[0], selection_size, replace = False)
#     return program_ids

# def sorting_test_pairs_easy_first(test_pairs, test_stats):
#     sorted_pairs = sorted(test_pairs, key = lambda x: test_stats[x[0]] + test_stats[x[1]], reverse=True)
#     return sorted_pairs

# def sorting_test_pairs_hard_first(test_pairs, test_stats):
#     test_stats[test_stats == 0] = np.inf
#     sorted_pairs = sorted(test_pairs, key = lambda x: test_stats[x[0]] + test_stats[x[1]])
#     return sorted_pairs

# def sorting_test_pairs_random(test_pairs, test_stats):
#     return default_rnd.permutation(test_pairs).tolist()

# pair_coverage_subgroups = iter([]) #pos and lengthes of each subgroup in population, should be cycling iterator
# pair_coverage_selected = iter([])

# def subgroup_selection(population):
#     global pair_coverage_subgroups
#     global pair_coverage_selected
#     if (next_index := next(pair_coverage_selected, None)) is None:
#         pos, (l1, l2) = next(pair_coverage_subgroups)
#         best_index1 = pos + default_rnd.choice(l1)
#         best_index2 = pos + l1 + default_rnd.choice(l2)
#         pair_coverage_selected = iter([best_index1, best_index2])
#         next_index = next(pair_coverage_selected)
#     best = population[next_index]
#     return best
    
# prev_test_pairs = []
# def select_pair_coverage(pair_sorting, selection_size, interactions, population, persist_test_pair = False):
#     ''' Picks two tests to be solved in parent programs combination 
#         There could be many ways to pick a pair, we have some hypothesis but need to check different pair selections
#         Hypothesis 1: tests that are easily solvable is easy to recombine
#         Hypothesis 2: solving first pairs of most difficult tests gives better convergence speed 
#         Hypothesis 3: random pair selection is good enough.
#         NOTE: NOTE: We need to measure average time to recombine targeted test pair         
#         - we need metric to estimate the effect
#     '''
#     global prev_test_pairs
#     global pair_coverage_subgroups
#     ints = interactions
#     selected = []
#     test_pairs = combinations(range(ints.shape[1]), 2)
#     filtered_test_pairs = [] # tests that are not currently solved together
#     prev_test_pairs_set = set(prev_test_pairs)
#     if persist_test_pair:
#         for test1, test2 in test_pairs:
#             # if np.all(np.logical_not(np.logical_and(ints[:, test1], ints[:, test2]))) and not((test1, test2) in prev_test_pairs_set):
#             if not((test1, test2) in prev_test_pairs_set):
#                 filtered_test_pairs.append((test1, test2))
#     else:
#         filtered_test_pairs = list(test_pairs)
#     test_stats = np.sum(ints, axis = 0)
#     # if persist_test_pair:
#     #     pair_sorting = partial(holder.pair_sorting, pair_sorting)
#     sorted_test_pairs = pair_sorting(filtered_test_pairs, test_stats)
#     sorted_test_pairs = prev_test_pairs + sorted_test_pairs
#     sorted_test_pairs_iter = iter(sorted_test_pairs)
#     group_lengths = []
#     new_test_pairs = []
#     while len(selected) < selection_size:
#         test_id1, test_id2 = next(sorted_test_pairs_iter, (None, None)) #we target first this pair of tests
#         if test_id1 is None:
#             break
#         new_test_pairs.append((test_id1, test_id2))
#         candidate_program_ids1 = np.where(ints[:, test_id1] != 0)[0]
#         candidate_program_ids2 = np.where(ints[:, test_id2] != 0)[0]
#         # fixed_candidate_program_ids1 = np.setdiff1d(candidate_program_ids1, candidate_program_ids2, assume_unique=True)
#         # fixed_candidate_program_ids2 = np.setdiff1d(candidate_program_ids2, candidate_program_ids1, assume_unique=True)
#         # candidate_program_ids1 = fixed_candidate_program_ids1
#         # candidate_program_ids2 = fixed_candidate_program_ids2
#         if len(candidate_program_ids1) > len(candidate_program_ids2):
#             candidate_program_ids1, candidate_program_ids2 = candidate_program_ids2, candidate_program_ids1
#         if len(candidate_program_ids1) == 0:
#             continue
#         half_avail_size = (selection_size - len(selected)) // 2
#         if len(candidate_program_ids1) > half_avail_size:
#             candidate_program_ids1 = default_rnd.choice(candidate_program_ids1, half_avail_size, replace=False)
#         other_avail_size = selection_size - len(selected) - len(candidate_program_ids1)
#         if len(candidate_program_ids2) > other_avail_size:
#             candidate_program_ids2 = default_rnd.choice(candidate_program_ids2, other_avail_size, replace=False)
#         group_lengths.append((len(candidate_program_ids1), len(candidate_program_ids2)))
#         selected.extend(candidate_program_ids1)
#         selected.extend(candidate_program_ids2)
#     if persist_test_pair:
#         prev_test_pairs = new_test_pairs
#     indexes = list(zip([0, *np.cumsum([x + y for x, y in group_lengths]).tolist()], group_lengths))
#     pair_coverage_subgroups = cycle(indexes)
#     return np.array(selected)

# pair_coverage_easy_first = partial(select_pair_coverage, sorting_test_pairs_easy_first)

# pair_coverage_hard_first = partial(select_pair_coverage, sorting_test_pairs_hard_first)

# pair_coverage_random = partial(select_pair_coverage, sorting_test_pairs_random)

# pair_coverage_easy_first_persist = partial(select_pair_coverage, sorting_test_pairs_easy_first, persist_test_pair = True)

# pair_coverage_hard_first_persist = partial(select_pair_coverage, sorting_test_pairs_hard_first, persist_test_pair = True)

# pair_coverage_random_persist = partial(select_pair_coverage, sorting_test_pairs_random, persist_test_pair = True)

def run_pipeline_on_benchmark(sim_name, 
                            pipeline_fn, idx, 
                            init_fn = partial(ramped_half_and_half, 1, 5), 
                            selection_fn = partial(tournament_selection, 7),
                            mutation_fn = partial(subtree_mutation, 0.1, 17),
                            crossover_fn = partial(subtree_crossover, 0.1, 17),
                            breed_fn = partial(subtree_breed, 0.1, 0.9),
                            get_metrics_fn = partial(get_metrics, 0),
                            fitness_fns = [hamming_distance_fitness, depth_fitness],
                            record_fitness_ids = [0, 1],
                            metrics_file = "data/metrics/objs.jsonlist"):
    name, problem = benchmark[idx]
    outputs, func_list, terminal_list = problem()
    int_fn = partial(program_test_interactions, outputs)
    initialization = partial(init_fn, func_list, terminal_list)
    selection = selection_fn
    mutation = partial(mutation_fn, func_list, terminal_list)
    crossover = crossover_fn
    breed = partial(breed_fn, selection, mutation, crossover)
    evaluate = partial(gp_evaluate, len(outputs), int_fn, fitness_fns)
    for run_id in range(num_runs):
        stats = pipeline_fn(population_size, max_generations, initialization, breed, evaluate, get_metrics_fn)
        best_inds, *fitness_metrics = zip(*stats)
        metrics = dict(game = name, sim = sim_name, seed = seed, run_id = run_id, best_ind = str(best_inds[-1]), best_ind_depth = best_inds[-1].get_depth())
        for i, metric in enumerate(fitness_metrics):
            if i in record_fitness_ids:
                i_i = record_fitness_ids.index(i)
                metrics["fitness" + str(i_i)] = metric
        write_metrics(metrics, metrics_file)
        pass

gp = partial(run_pipeline_on_benchmark, "gp", run_koza)
ifs = partial(run_pipeline_on_benchmark, "ifs", run_koza, 
              fitness_fns = [ifs_fitness, hamming_distance_fitness, depth_fitness], get_metrics_fn = partial(get_metrics, 1),
              record_fitness_ids = [1, 2, 0])

def build_do_pipeline(sim_name, derive_objectives):
    return partial(run_pipeline_on_benchmark, sim_name, partial(run_nsga2, archive_size, derive_objectives = derive_objectives), 
                        selection_fn = partial(tournament_selection, 3))

do_rand = build_do_pipeline("do_rand", rand_objectives)
do_nsga = build_do_pipeline("do_nsga", full_objectives)

doc = build_do_pipeline("doc", doc_objectives)
doc_p = partial(build_do_pipeline("doc_p", doc_objectives), 
                fitness_fns = [hypervolume_fitness, hamming_distance_fitness, depth_fitness], 
                get_metrics_fn = partial(get_metrics, 1), record_fitness_ids = [1, 2, 0])

doc_d = partial(build_do_pipeline("doc_d", doc_objectives),
                fitness_fns = [weighted_hypervolume_fitness, hamming_distance_fitness, depth_fitness], 
                get_metrics_fn = partial(get_metrics, 1), record_fitness_ids = [1, 2, 0])

dof_w_2 = build_do_pipeline("doc_w_2", partial(dof_w_objectives, 2, 1))
dof_w_3 = build_do_pipeline("doc_w_3", partial(dof_w_objectives, 3, 1))
dof_wh_2 = build_do_pipeline("doc_wh_2", partial(dof_wh_objectives, 2, 1))
dof_wh_3 = build_do_pipeline("doc_wh_3", partial(dof_wh_objectives, 3, 1))

dof_w_2_80 = build_do_pipeline("doc_w_2_80", partial(dof_w_objectives, 2, 0.8))
dof_w_3_80 = build_do_pipeline("doc_w_3_80", partial(dof_w_objectives, 3, 0.8))
dof_wh_2_80 = build_do_pipeline("doc_wh_2_80", partial(dof_wh_objectives, 2, 0.8))
dof_wh_3_80 = build_do_pipeline("doc_wh_3_80", partial(dof_wh_objectives, 3, 0.8))

do_fo = build_do_pipeline("do_fo", do_feature_objectives)

do_pca_abs_2 = build_do_pipeline("do_pca_abs_2", partial(do_pca_abs_objectives, 2))
do_pca_abs_3 = build_do_pipeline("do_pca_abs_3", partial(do_pca_abs_objectives, 3))

do_pca_diff_2 = build_do_pipeline("do_pca_diff_2", partial(do_pca_diff_objectives, 2))
do_pca_diff_3 = build_do_pipeline("do_pca_diff_3", partial(do_pca_diff_objectives, 3))

def build_cov_pipeline(sim_name, select_parents_fn, selection_fn):
    return partial(run_pipeline_on_benchmark, sim_name, partial(run_front_coverage, archive_size, select_parents = partial(select_parents_fn, archive_size)), 
                        selection_fn = selection_fn)

cov_ht_bp = build_cov_pipeline("cov_ht_bp", full_test_coverage_hardest_test_best_program, full_coverage_selection)

cov_et_bp = build_cov_pipeline("cov_et_bp", full_test_coverage_easiest_test_best_program, full_coverage_selection)

cov_rt_bp = build_cov_pipeline("cov_rt_bp", full_test_coverage_random_test_best_program, full_coverage_selection)

cov_ht_rp = build_cov_pipeline("cov_ht_rp", full_test_coverage_hardest_test_rand_program, full_coverage_selection)

cov_et_rp = build_cov_pipeline("cov_et_rp", full_test_coverage_easiest_test_rand_program, full_coverage_selection)

cov_rt_rp = build_cov_pipeline("cov_rt_rp", full_test_coverage_random_test_rand_program, full_coverage_selection)

# cov_bp = build_cov_pipeline("cov_bp", full_test_coverage_best_program, full_coverage_selection)

# pair_cov_ht = build_cov_pipeline("pair_cov_ht", pair_coverage_hard_first, subgroup_selection)

# pair_cov_et = build_cov_pipeline("pair_cov_et", pair_coverage_easy_first, subgroup_selection)

# pair_cov_rt = build_cov_pipeline("pair_cov_rt", pair_coverage_random, subgroup_selection)

# pair_cov_ht_p = build_cov_pipeline("pair_cov_ht", pair_coverage_hard_first_persist, subgroup_selection)

# pair_cov_et_p = build_cov_pipeline("pair_cov_et", pair_coverage_easy_first_persist, subgroup_selection),

# pair_cov_rt_p = build_cov_pipeline("pair_cov_rt", pair_coverage_random_persist, subgroup_selection)

benchmark_map = {name: i for i, (name, _) in enumerate(benchmark) }

sim_names = [ 'gp', 'ifs', 'do_rand', 'do_nsga', 'doc', 'doc_p', 'doc_d', 'dof_w_2', 'dof_w_3', 'dof_wh_2', 'dof_wh_3', 'dof_w_2_80', 'dof_w_3_80', 'dof_wh_2_80', 'dof_wh_3_80', 'do_fo', 'do_pca_abs_2', 'do_pca_abs_3', 'do_pca_diff_2', 'do_pca_diff_3', 'cov_ht_bp', 'cov_et_bp', 'cov_rt_bp', 'cov_ht_rp', 'cov_et_rp', 'cov_rt_rp' ]

if __name__ == "__main__":
    print("testing evo runs")
    for sim_name in sim_names:
        for b_name in benchmark_map.keys():
            print(f"{sim_name}:{b_name}")
    # gp(idx = 10)
    pass