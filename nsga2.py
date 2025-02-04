""" Implements NSGA-II algorithm. """
import numpy as np
from derivedObj import matrix_factorization, xmean_cluster
from rnd import default_rnd
from functools import partial

from gp import analyze_population, create_runtime_context, gp_eval, depth_fitness, hamming_distance_fitness, identity_map, init_each, subtree_breed

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


def nsga2_loop(archive_size, population_size, max_gens, 
              init_fn, map_fn, breed_fn, eval_fn, analyze_pop_fn):
    """ Run NSGA-II algorithm. """
    # Create initial population
    population = init_fn(population_size)
    archive = []
    gen = 0
    best_ind = None
    while gen < max_gens:
        all_inds = population + archive
        all_inds = map_fn(all_inds)
        outputs, fitnesses, _, derived_objectives = eval_fn(all_inds) 
        new_archive = []
        all_fronts_indicies = np.array([], dtype=int)
        # best_front_indexes = None
        while len(new_archive) < archive_size:
            new_front_indices = get_pareto_front_indexes(derived_objectives, exclude_indexes = all_fronts_indicies)
            if len(new_front_indices) == 0:
                break            
            new_front = [all_inds[i] for i in new_front_indices]
            # if best_front_indexes is None:
            #     best_front_indexes = new_front_indices
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
        archive = new_archive
        archive_fitnesses = fitnesses[all_fronts_indicies]
        archive_outputs = outputs[all_fronts_indicies]
        # best_front = [all_inds[i] for i in best_front_indexes]
        # best_front_fitnesses = fitnesses[best_front_indexes]
        # best_front_outcomes = outputs[best_front_indexes]
        best_ind = analyze_pop_fn(archive, archive_outputs, archive_fitnesses)
        if best_ind is not None:
            break
        population = breed_fn(population_size, archive, archive_fitnesses)
        gen += 1
    return best_ind, gen

def full_objectives(interactions):
    return interactions, dict()

def doc_objectives(interactions):
    test_clusters, centers = xmean_cluster(interactions.T.tolist(), 4)
    # derived_objectives = np.column_stack([np.mean(interactions[:, tc], axis = 1) for tc in test_clusters])
    derived_objectives = np.array(centers).T
    # NOTE: centroid === to np.mean above.
    return derived_objectives, dict(test_clusters = test_clusters)

def rand_objectives(interactions):
    rand_ints = default_rnd.randint(2, size = interactions.shape)
    res = doc_objectives(rand_ints)
    return res

def dof_w_objectives(interactions, k, alpha):
    if alpha < 1: 
        num_columns_to_take = int(alpha * interactions.shape[1])
        random_column_indexes = default_rnd.choice(interactions.shape[1], num_columns_to_take, replace = False)
        interactions = interactions[:, random_column_indexes]
    W, _, _ = matrix_factorization(interactions, k)
    return W, {}

def dof_wh_objectives(interactions, k, alpha):
    if alpha < 1: 
        num_columns_to_take = int(alpha * interactions.shape[1])
        random_column_indexes = default_rnd.choice(interactions.shape[1], num_columns_to_take, replace = False)
        interactions = interactions[:, random_column_indexes]    
    W, H, _ = matrix_factorization(interactions, k)
    objs = np.sum(W[:, :, None] * H, axis = -1)
    return objs, {}

# import numpy as np
# W = np.array([[0.96, 1.51], [0.39, 1.84], [0.86, 0.38]])
# H = np.array([[2.16, 1.2, 0.72, 0.72], [0.05, 0.35, 0.9, 0.9]])
# np.sum(W[:, :, None] * H, axis = -1)

def do_pca_abs_objectives(interactions, k):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=k)
    components = pca.fit_transform(interactions)
    return np.abs(components), {}

def do_pca_diff_objectives(interactions, k):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=k)
    components = pca.fit_transform(interactions)
    min_components = np.min(components, axis = 0)
    return (components - min_components + 0.1), {}

def do_feature_objectives(interactions):
    ''' we extract derived objectives based on program features: 
        - how many tests the program passes 
        - the maximal difficulty of the passed test by the program 
    '''
    test_stats = np.sum(interactions, axis = 0)
    whole_program_weights = (interactions * test_stats).astype(float)
    whole_program_weights[whole_program_weights == 0] = np.inf
    program_stats = np.sum(interactions, axis = 1)
    program_min_test_difficulty = np.min(whole_program_weights, axis = 1)
    res = np.stack([program_stats, interactions.shape[0] - program_min_test_difficulty], axis = 1)
    return res, {}

def gp_nsga2(problem_init, *,
                        population_size = 1000, max_gens = 100, archive_size = 1000,
                        fitness_fns = [hamming_distance_fitness, depth_fitness], main_fitness_fn = hamming_distance_fitness,
                        select_fitness_ids = None, init_fn = init_each, map_fn = identity_map, breed_fn = subtree_breed, 
                        eval_fn = gp_eval, analyze_pop_fn = analyze_population, derive_objs_fn = full_objectives):
    eval_fn = partial(eval_fn, derive_objs_fn = derive_objs_fn)
    runtime_context = create_runtime_context(fitness_fns, main_fitness_fn, select_fitness_ids)
    problem_init(runtime_context = runtime_context)
    evo_funcs = [init_fn, map_fn, breed_fn, eval_fn, analyze_pop_fn]
    evo_funcs_bound = [partial(fn, runtime_context = runtime_context) for fn in evo_funcs]    
    best_ind, gen = nsga2_loop(archive_size, population_size, max_gens, *evo_funcs_bound)
    runtime_context.stats["gen"] = gen
    runtime_context.stats["best_found"] = best_ind is not None
    return best_ind, runtime_context.stats

def hypervolume_fitness(interactions, outputs, derived_objectives = [], **_):
    if derived_objectives is None or len(derived_objectives) == 0:
        return np.zeros(interactions.shape[0], dtype=float)
    return -np.prod(derived_objectives, axis = 1)

def weighted_hypervolume_fitness(interactions, outputs, derived_objectives = [], test_clusters = [], **_):
    if derived_objectives is None or len(derived_objectives) == 0:
        return np.zeros(interactions.shape[0], dtype=float)
    weights = np.array([np.sum(interactions[:, tc] == 1, axis=1) for tc in test_clusters]).T
    weighted_objs = weights * derived_objectives
    return -np.prod(weighted_objs, axis = 1)


do_nsga = gp_nsga2
do_nsga_0 = partial(gp_nsga2, select_fitness_ids = [0])

# do_rand = partial(gp_nsga2, derive_objs_fn = rand_objectives)

# doc = partial(gp_nsga2, derive_objs_fn = doc_objectives)
doc_p = partial(gp_nsga2, derive_objs_fn = doc_objectives, 
                fitness_fns = [hypervolume_fitness, hamming_distance_fitness, depth_fitness])

doc_p_0 = partial(doc_p, select_fitness_ids = [0, 1])

doc_d = partial(gp_nsga2, derive_objs_fn = doc_objectives,
                fitness_fns = [weighted_hypervolume_fitness, hamming_distance_fitness, depth_fitness])

doc_d_0 = partial(doc_d, select_fitness_ids = [0, 1])

# dof_w_2 = partial(gp_nsga2, derive_objs_fn = partial(dof_w_objectives, k = 2, alpha = 1))
dof_w_3 = partial(gp_nsga2, derive_objs_fn = partial(dof_w_objectives, k = 3, alpha = 1))
dof_w_3_0 = partial(dof_w_3, select_fitness_ids = [0])
# dof_wh_2 = partial(gp_nsga2, derive_objs_fn = partial(dof_wh_objectives, k = 2, alpha = 1))
dof_wh_3 = partial(gp_nsga2, derive_objs_fn = partial(dof_wh_objectives, k = 3, alpha = 1))
dof_wh_3_0 = partial(dof_wh_3, select_fitness_ids = [0])

# dof_w_2_80 = partial(gp_nsga2, derive_objs_fn = partial(dof_w_objectives, k = 2, alpha = 0.8))
# dof_w_3_80 = partial(gp_nsga2, derive_objs_fn = partial(dof_w_objectives, k = 3, alpha = 0.8))
# dof_wh_2_80 = partial(gp_nsga2, derive_objs_fn = partial(dof_wh_objectives, k = 2, alpha = 0.8))
# dof_wh_3_80 = partial(gp_nsga2, derive_objs_fn = partial(dof_wh_objectives, k = 3, alpha = 0.8))

# do_fo = partial(gp_nsga2, derive_objs_fn = do_feature_objectives)

# do_pca_abs_2 = partial(gp_nsga2, derive_objs_fn = partial(do_pca_abs_objectives, k = 2))
# do_pca_abs_3 = partial(gp_nsga2, derive_objs_fn = partial(do_pca_abs_objectives, k = 3))

# do_pca_diff_2 = partial(gp_nsga2, derive_objs_fn = partial(do_pca_diff_objectives, k = 2))
# do_pca_diff_3 = partial(gp_nsga2, derive_objs_fn = partial(do_pca_diff_objectives, k = 3))

# nsga2_sim_names = [ 'do_rand', 'do_nsga', 'doc', 'doc_p', 'doc_d', 'dof_w_2', 'dof_w_3', 'dof_wh_2', 'dof_wh_3', 'dof_w_2_80', 'dof_w_3_80', 'dof_wh_2_80', 'dof_wh_3_80', 'do_fo', 'do_pca_abs_2', 'do_pca_abs_3', 'do_pca_diff_2', 'do_pca_diff_3' ]
nsga2_sim_names = [ 'do_nsga', 'doc_p', 'doc_d', 'dof_w_3', 'dof_wh_3', 'do_nsga_0', 'doc_p_0', 'doc_d_0', 'dof_w_3_0', 'dof_wh_3_0' ]

if __name__ == '__main__':
    import gp_benchmarks
    problem_builder = gp_benchmarks.get_benchmark('cmp6')
    best_prog, stats = doc_d_0(problem_builder)
    print(best_prog)
    print(stats)
    pass    
