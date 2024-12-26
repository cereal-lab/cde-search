""" Implements NSGA-II algorithm. """
import numpy as np

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


def run_nsga2(archive_size, population_size, max_generations, 
              initialization, breed, evaluate, get_metrics, derive_objectives = None):
    """ Run NSGA-II algorithm. """
    # Create initial population
    population = [initialization() for _ in range(population_size)]
    archive = []
    stats = []
    generation = 0
    while True:
        all_inds = population + archive
        fitnesses, _, _, derived_objectives = evaluate(all_inds, derive_objectives = derive_objectives) 
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
        archive = new_archive
        # archive_fitnesses = fitnesses[all_fronts_indicies]
        best_front = [all_inds[i] for i in best_front_indexes]
        best_front_fitnesses = fitnesses[best_front_indexes]
        best_found, metrics = get_metrics(best_front_fitnesses, best_front)
        stats.append(metrics)
        if best_found or (generation >= max_generations):
            break
        generation += 1
        population = breed(population_size, archive)
    return stats


def run_front_coverage(archive_size, population_size, max_generations, 
              initialization, breed, evaluate, get_metrics, select_parents = None):
    # Create initial population
    population = [initialization() for _ in range(population_size)]
    stats = []
    generation = 0
    while True:
        fitnesses, interactions, *_ = evaluate(population) 
        all_fronts_indexes = np.array([], dtype=int)
        best_front_indexes = None
        while len(all_fronts_indexes) < archive_size:
            new_front_indices = get_pareto_front_indexes(interactions, exclude_indexes = all_fronts_indexes)
            if len(new_front_indices) == 0:
                break
            if best_front_indexes is None:
                best_front_indexes = new_front_indices
            all_fronts_indexes = np.concatenate([all_fronts_indexes, new_front_indices])
        best_front = [population[i] for i in best_front_indexes]
        best_front_fitnesses = fitnesses[best_front_indexes]
        best_found, metrics = get_metrics(best_front_fitnesses, best_front)
        stats.append(metrics)
        if best_found or (generation >= max_generations):
            break
        # all_front_inds = [population[i] for i in all_fronts_indexes]
        selected_indexes_ids = select_parents(interactions[all_fronts_indexes]) #, fitnesses[all_fronts_indexes])
        selected_indexes = all_fronts_indexes[selected_indexes_ids]
        parents = [population[i] for i in selected_indexes]
        new_population = breed(population_size, parents)
        new_population.extend(parents)
        generation += 1
        population = new_population
    return stats