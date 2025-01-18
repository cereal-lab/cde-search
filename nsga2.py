""" Implements NSGA-II algorithm. """
from functools import partial
import numpy as np
from derivedObj import matrix_factorization, xmean_cluster
from rnd import default_rnd, seed

from gp import depth_fitness, get_metrics, gp_evaluate, hamming_distance_fitness, init_each, ramped_half_and_half, subtree_breed, subtree_crossover, subtree_mutation, tournament_selection
from utils import write_metrics

def get_pareto_front_indexes(fitnesses: np.ndarray, exclude_indexes: np.array = []) -> np.ndarray:
    ''' Get the pareto front from a population. 
        NOTE: greater is better here. Invert your fitness if it is the opposite.
    '''
    mask = np.ones(fitnesses.shape[0], dtype=bool)
    mask[exclude_indexes] = False
    index_remap = np.where(mask)[0]
    domination_matrix = np.all(fitnesses[mask][:, None] <= fitnesses[mask], axis=2) & np.any(fitnesses[mask][:, None] < fitnesses[mask], axis=2)
    indexes = np.where(~np.any(domination_matrix, axis=1))[0]
    return index_remap[indexes]

def get_pareto_front_indexes_neg(fitnesses: np.ndarray, exclude_indexes: np.array = []) -> np.ndarray:
    ''' Get the pareto front from a population. 
        NOTE: greater is better here. Invert your fitness if it is the opposite.
    '''
    mask = np.ones(fitnesses.shape[0], dtype=bool)
    mask[exclude_indexes] = False
    index_remap = np.where(mask)[0]
    domination_matrix = np.all(fitnesses[mask][:, None] >= fitnesses[mask], axis=2) & np.any(fitnesses[mask][:, None] > fitnesses[mask], axis=2)
    indexes = np.where(~np.any(domination_matrix, axis=1))[0]
    return index_remap[indexes]    

def get_sparsity(fitnesses: np.ndarray) -> np.ndarray:
    ''' Get the sparsity of the pareto front. '''
    # sparsity = np.zeros_like(fitnesses[front_indexes], dtype=float)
    # objective_ranges[objective_ranges == 0] = 1
    orders_by_objectives = np.argsort(fitnesses, axis=0)
    obj_sorted = np.sort(fitnesses, axis=0)
    objective_ranges = obj_sorted[-1, :] - obj_sorted[0, :]
    objective_ranges[objective_ranges == 0] = 1
    # sparsity[(orders_by_objectives == 0) | (orders_by_objectives == fitnesses.shape[0] - 1)] = np.inf
    # sparsity[:, objective_ranges == 0] = np.inf
    # mask = np.isinf(sparsity)
    # indexes = np.tile(np.arange(fitnesses.shape[0])[:, None], (1, fitnesses.shape[1]))
    # prevs = (indexes - 1) % fitnesses.shape[0]
    # nexts = (indexes + 1) % fitnesses.shape[0]
    # prev_indexes = np.take_along_axis(orders_by_objectives, prevs, axis=0)
    # next_indexes = np.take_along_axis(orders_by_objectives, nexts, axis=0)
    prev_fitnesses = np.vstack((obj_sorted[-1], obj_sorted[:-1]))
    next_fitnesses = np.vstack((obj_sorted[1:], obj_sorted[0]))
    # mask = ~mask
    # prev_fitnesses = np.take_along_axis(fitnesses, prev_indexes, axis=0)
    # next_fitnesses = np.take_along_axis(fitnesses, next_indexes, axis=0)
    # all_ranges = np.tile(objective_ranges, (len(front_indexes), 1))
    sparsity_sorted = (next_fitnesses - prev_fitnesses) / objective_ranges
    sparsity = np.empty_like(sparsity_sorted)
    for column_id in range(sparsity.shape[1]):
        sparsity[orders_by_objectives[:, column_id], column_id] = sparsity_sorted[:, column_id]

    # sparsity = np.take_along_axis(sparsity_sorted, orders_by_objectives, axis=0)
    sparsity[sparsity < 0] = np.inf
    front_sparsity = np.sum(sparsity, axis=1)
    return front_sparsity

# a = np.zeros((4,3), dtype=bool)
# a
# a[0] = [True, False, True]
# a[1,1] = True
# f = np.array([[0,100],[10,90],[50, 50],[90,10],[100,0]])
# rnd = np.random.RandomState(0)
# f = rnd.random((1000,2))
# fx = get_pareto_front_indexes_neg(f)
# get_sparsity(f[fx])
# s = np.zeros_like(f, dtype = float)
# o = np.array([1, 0, 3])
# s[:, o == 0] = np.inf
# s
# np.min(f, axis=0)
# np.argsort(f, axis=0)
# f[None, :]
# f[:, None]
# f.shape
# f[:, None][0]
# f[0]
# f[:, None] <= f
# np.all(f <= f[2], axis=1) & np.any(f < f[2], axis=1)
# get_pareto_front_indexes(f) 
# get_pareto_front_indexes(f, [0,1,2,3])


def run_nsga2(archive_size, max_generations, 
              initialization_fn, breed, evaluate, get_metrics):
    """ Run NSGA-II algorithm. """
    # Create initial population
    population = initialization_fn()
    archive = []
    stats = []
    generation = 0
    while True:
        all_inds = population + archive
        fitnesses, _, _, derived_objectives = evaluate(all_inds) 
        new_archive = []
        all_fronts_indicies = np.array([], dtype=int)
        best_front_indexes = None
        while len(new_archive) < archive_size:
            new_front_indices = get_pareto_front_indexes(derived_objectives, exclude_indexes = all_fronts_indicies)
            if len(new_front_indices) == 0:
                break            
            new_front = [all_inds[i] for i in new_front_indices]
            if best_front_indexes is None:
                best_front_indexes = new_front_indices
            if len(new_archive) + len(new_front_indices) <= archive_size:
                all_fronts_indicies = np.concatenate([all_fronts_indicies, new_front_indices])
                new_archive += new_front
            else:
                left_to_take = archive_size - len(new_archive)
                front_sparsity = get_sparsity(derived_objectives[new_front_indices])
                front_id_idx = np.argsort(front_sparsity)[-left_to_take:]
                front_indexes = new_front_indices[front_id_idx]
                all_fronts_indicies = np.concatenate([all_fronts_indicies, front_indexes])
                new_archive += [all_inds[i] for i in front_indexes]
                break  
        best_front = [all_inds[i] for i in best_front_indexes]
        best_front_fitnesses = fitnesses[best_front_indexes]
        best_found, metrics = get_metrics(best_front_fitnesses, best_front)
        stats.append(metrics)
        if best_found or (generation >= max_generations):
            break
        generation += 1
        archive = new_archive
        archive_fitnesses = fitnesses[all_fronts_indicies]
        population = breed(archive, archive_fitnesses)
    return stats

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

def run_nsga2_experiment(sim_name, game_name, gold_outputs, func_list, terminal_list,
                            evaluate_fn = gp_evaluate,
                            init_fn = partial(init_each, partial(ramped_half_and_half, 1, 5)),
                            selection_fn = partial(tournament_selection, selection_size = 7),
                            mutation_fn = partial(subtree_mutation, 0.1, 17),
                            crossover_fn = partial(subtree_crossover, 0.1, 17),
                            breed_fn = partial(subtree_breed, mutation_rate = 0.1, crossover_rate = 0.9),
                            get_metrics_fn = partial(get_metrics, 0),
                            fitness_fns = [hamming_distance_fitness, depth_fitness],
                            derive_objectives_fn = full_objectives,
                            record_fitness_ids = [0, 1],
                            metrics_file = "data/metrics/objs.jsonlist",
                            archive_size = 1000,
                            max_generations = 100,
                            population_size = 1000,
                            num_runs = 30):
    initialization = partial(init_fn, func_list, terminal_list, population_size)
    mutation_fn = partial(mutation_fn, func_list, terminal_list)
    breed = partial(breed_fn, selection_fn = selection_fn, mutation_fn = mutation_fn, crossover_fn = crossover_fn, breed_size = population_size)
    evaluate = partial(evaluate_fn, gold_outputs, fitness_fns = fitness_fns, derive_objectives = derive_objectives_fn)
    for run_id in range(num_runs):
        stats = run_nsga2(archive_size, max_generations, initialization, breed, evaluate, get_metrics_fn)
        best_inds, *fitness_metrics = zip(*stats)
        metrics = dict(game = game_name, sim = sim_name, seed = seed, run_id = run_id, best_ind = str(best_inds[-1]), best_ind_depth = best_inds[-1].get_depth())
        for i, metric in enumerate(fitness_metrics):
            if i in record_fitness_ids:
                i_i = record_fitness_ids.index(i)
                metrics["fitness" + str(i_i)] = metric
        write_metrics(metrics, metrics_file)
        pass

def hypervolume_fitness(interactions, derived_objectives = [], **kwargs):
    return -np.prod(derived_objectives, axis = 1)

def weighted_hypervolume_fitness(interactions, derived_objectives = [], test_clusters = [], **kwargs):
    weights = np.array([np.sum(interactions[:, tc] == 1, axis=1) for tc in test_clusters]).T
    weighted_objs = weights * derived_objectives
    return np.prod(weighted_objs, axis = 1)


do_nsga = partial(run_nsga2_experiment, "do_nsga")
do_rand = partial(run_nsga2_experiment, "do_rand", derive_objectives_fn = rand_objectives)

doc = partial(run_nsga2_experiment, "doc", derive_objectives_fn = doc_objectives)
doc_p = partial(run_nsga2_experiment, "doc_p", derive_objectives_fn = doc_objectives, 
                fitness_fns = [hypervolume_fitness, hamming_distance_fitness, depth_fitness], 
                get_metrics_fn = partial(get_metrics, 1), record_fitness_ids = [1, 2, 0])

doc_d = partial(run_nsga2_experiment, "doc_d", derive_objectives_fn = doc_objectives,
                fitness_fns = [weighted_hypervolume_fitness, hamming_distance_fitness, depth_fitness], 
                get_metrics_fn = partial(get_metrics, 1), record_fitness_ids = [1, 2, 0])

dof_w_2 = partial(run_nsga2_experiment, "dof_w_2", derive_objectives_fn = partial(dof_w_objectives, 2, 1))
dof_w_3 = partial(run_nsga2_experiment, "dof_w_3", derive_objectives_fn = partial(dof_w_objectives, 3, 1))
dof_wh_2 = partial(run_nsga2_experiment, "dof_wh_2", derive_objectives_fn = partial(dof_wh_objectives, 2, 1))
dof_wh_3 = partial(run_nsga2_experiment, "dof_wh_3", derive_objectives_fn = partial(dof_wh_objectives, 3, 1))

dof_w_2_80 = partial(run_nsga2_experiment, "dof_w_2_80", derive_objectives_fn = partial(dof_w_objectives, 2, 0.8))
dof_w_3_80 = partial(run_nsga2_experiment, "dof_w_3_80", derive_objectives_fn = partial(dof_w_objectives, 3, 0.8))
dof_wh_2_80 = partial(run_nsga2_experiment, "dof_wh_2_80", derive_objectives_fn = partial(dof_wh_objectives, 2, 0.8))
dof_wh_3_80 = partial(run_nsga2_experiment, "dof_wh_3_80", derive_objectives_fn = partial(dof_wh_objectives, 3, 0.8))

do_fo = partial(run_nsga2_experiment, "do_fo", derive_objectives_fn = do_feature_objectives)

do_pca_abs_2 = partial(run_nsga2_experiment, "do_pca_abs_2", derive_objectives_fn = partial(do_pca_abs_objectives, 2))
do_pca_abs_3 = partial(run_nsga2_experiment, "do_pca_abs_3", derive_objectives_fn = partial(do_pca_abs_objectives, 3))

do_pca_diff_2 = partial(run_nsga2_experiment, "do_pca_diff_2", derive_objectives_fn = partial(do_pca_diff_objectives, 2))
do_pca_diff_3 = partial(run_nsga2_experiment, "do_pca_diff_3", derive_objectives_fn = partial(do_pca_diff_objectives, 3))

nsga2_sim_names = [ 'do_rand', 'do_nsga', 'doc', 'doc_p', 'doc_d', 'dof_w_2', 'dof_w_3', 'dof_wh_2', 'dof_wh_3', 'dof_w_2_80', 'dof_w_3_80', 'dof_wh_2_80', 'dof_wh_3_80', 'do_fo', 'do_pca_abs_2', 'do_pca_abs_3', 'do_pca_diff_2', 'do_pca_diff_3' ]