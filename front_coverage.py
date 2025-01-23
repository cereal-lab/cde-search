

from functools import partial
from itertools import cycle
import numpy as np

from gp import BreedingStats, analyze_population, gp_eval, cached_node_builder, depth_fitness, hamming_distance_fitness, identity_map, init_each, simple_node_builder, subtree_breed
from nsga2 import get_pareto_front_indexes
from rnd import default_rnd
import utils


def pareto_front_loop(archive_size, population_size, max_gens, 
              init_fn, map_fn, breed_fn, eval_fn, analyze_pop_fn, select_parents_fn):
    # Create initial population
    population = init_fn(population_size)
    gen = 0
    best_ind = None
    while gen < max_gens:
        population = map_fn(population)
        outputs, fitnesses, interactions = eval_fn(population) 
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
        best_front_outcomes = outputs[best_front_indexes]
        best_ind = analyze_pop_fn(best_front, best_front_outcomes, best_front_fitnesses)
        if best_ind is not None:
            break
        # all_front_inds = [population[i] for i in all_fronts_indexes]
        selected_indexes_ids = select_parents_fn(archive_size, interactions[all_fronts_indexes]) #, fitnesses[all_fronts_indexes])
        selected_indexes = all_fronts_indexes[selected_indexes_ids]
        parents = [population[i] for i in selected_indexes]
        parents_fitnesses = fitnesses[selected_indexes]
        new_population = breed_fn(population_size, parents, parents_fitnesses)
        new_population.extend(parents)
        population = new_population
        gen += 1
    return best_ind, gen

class FullTestCoverageIterators():
    def __init__(self):
        self.subgroups = iter([])
        self.selected = iter([])

def full_coverage_selection(population, fitnesses, coverage_selection_size = 2, *, full_coverage: FullTestCoverageIterators):
    if (next_index := next(full_coverage.selected, None)) is None:
        subgroup_start, subgroup_size = next(full_coverage.subgroups)
        choices = default_rnd.choice(subgroup_size, size = min(subgroup_size, coverage_selection_size), replace = False)
        indexes = subgroup_start + choices
        full_coverage.selected = iter(indexes)
        next_index = next(full_coverage.selected)
        if len(population) >= 6 and str(population[5]) == "x6":
            print(f"selected: ss {subgroup_start}, ssize {subgroup_size} choice {choices} indexes {indexes} next {next_index}")
    # best = population[next_index]
    return next_index

def select_test_hardest(ints: np.ndarray, allowed_program_mask: np.ndarray, allowed_test_mask: np.ndarray):
    allowed_test_indexes = np.where(allowed_test_mask)[0]
    test_stats = np.sum(ints[allowed_program_mask][:, allowed_test_mask], axis = 0).astype(float) #sum accross columns to see how many programs pass each test    
    test_stats[test_stats == 0] = np.inf
    test_id_id = np.argmin(test_stats)    
    test_id = allowed_test_indexes[test_id_id]
    return test_id

def select_test_easiest(ints: np.ndarray, allowed_program_mask: np.ndarray, allowed_test_mask: np.ndarray):
    allowed_test_indexes = np.where(allowed_test_mask)[0]
    test_stats = np.sum(ints[allowed_program_mask][:, allowed_test_indexes], axis = 0) #sum accross columns to see how many programs pass each test    
    test_id_id = np.argmax(test_stats)
    test_id = allowed_test_indexes[test_id_id]
    return test_id

def select_test_random(ints: np.ndarray, allowed_program_mask: np.ndarray, allowed_test_mask: np.ndarray):
    allowed_test_indexes = np.where(allowed_test_mask)[0]
    return default_rnd.choice(allowed_test_indexes)

def select_program_best_by_test(ints: np.ndarray, allowed_program_mask: np.ndarray, allowed_test_mask: np.ndarray, test_selection = select_test_hardest):
    selected_test_id = test_selection(ints, allowed_program_mask, allowed_test_mask)
    allowed_program_indexes = np.where(allowed_program_mask)[0]
    program_candidate_id_ids = np.where(ints[allowed_program_indexes, selected_test_id] != 0)[0]
    if len(program_candidate_id_ids) == 0:
        return None, None
    program_candidate_ids = allowed_program_indexes[program_candidate_id_ids]
    allowed_test_indexes = np.where(allowed_test_mask)[0]
    program_stats = np.sum(ints[program_candidate_ids][:, allowed_test_indexes], axis = 1)
    best_program_solved_test_num = np.max(program_stats)
    best_program_id_ids = np.where(program_stats == best_program_solved_test_num)[0]
    best_program_ids = program_candidate_ids[best_program_id_ids] #the program that passes the most tests including selected rare test
    best_program_id = default_rnd.choice(best_program_ids, 1)[0]
    allowed_program_mask[best_program_id] = False
    solved_test_ids_by_best = np.where(ints[best_program_id] != 0)[0]
    allowed_test_mask[solved_test_ids_by_best] = False
    return best_program_id, selected_test_id

def select_program_random(ints: np.ndarray, allowed_program_mask: np.ndarray, allowed_test_mask: np.ndarray, test_selection = select_test_hardest):
    selected_test_id = test_selection(ints, allowed_program_mask, allowed_test_mask)
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

def select_full_test_coverage(front_selection_size, interactions, test_program_selection = select_program_best_by_test, *, full_coverage: FullTestCoverageIterators):
    ''' There are different ways to have full test coverage    
    Does the selection of test metter?
    Hypothesis 1. selects hardest test and then best program that solves it 
    Hypothesis 2. selects easiest test and then best program that solves it
    Hypothesis 3. selects random test and then best program that solves it'''
    ints = interactions #np.copy(interactions)    
    # non_zero_test_ids  = np.where(test_stats != 0)[0]
    selected = []
    allowed_test_mask = np.any(ints, axis = 0) #solvable tests currently
    allowed_test_mask_all = np.copy(allowed_test_mask) #only have False for really picked tests, while allowed_test_mask set to false on program covering the test
    allowed_program_mask = np.any(ints, axis = 1) #programs that solve at least one test
    groups = []
    cur_counter = 0 #number of selected programs in current group
    # should_break_next_time = False
    while (len(selected) < front_selection_size) and np.any(allowed_test_mask_all) and np.any(allowed_program_mask):
        # test_with_min_programs_id = test_selection(ints, allowed_test_mask)
        # best_program_id = program_selection(ints, test_with_min_programs_id, population, allowed_program_mask, allowed_test_mask)
        # allowed_test_mask_all[test_with_min_programs_id] = False
        # test_with_min_programs_id = test_selection(ints, allowed_test_mask)
        best_program_id, test_with_min_programs_id = test_program_selection(ints, allowed_program_mask, allowed_test_mask)
        if best_program_id is not None:
            allowed_test_mask_all[test_with_min_programs_id] = False
            cur_counter += 1
            selected.append(best_program_id)
            # should_break_next_time = False
        if best_program_id is None or not np.any(allowed_test_mask): #try second, thrird etc test coverages
            if cur_counter == 0:
                break
            allowed_test_mask = np.copy(allowed_test_mask_all)
            groups.append(cur_counter)
            cur_counter = 0
            # NOTE: at this point it makes sense to form coverage groups.
    if cur_counter > 0:
        groups.append(cur_counter)
    if len(selected) == 0:
        selected = default_rnd.choice(ints.shape[0], size = min(front_selection_size, ints.shape[0]), replace = False)
    if len(groups) == 0:
        groups.append(len(selected))
    indexes = list(zip([0, *np.cumsum(groups).tolist()], groups))
    full_coverage.subgroups = cycle(indexes)
    return np.array(selected)

full_test_coverage_hardest_test_best_program = partial(select_full_test_coverage, test_program_selection = partial(select_program_best_by_test, test_selection = select_test_hardest)) 

full_test_coverage_easiest_test_best_program = partial(select_full_test_coverage, test_program_selection = partial(select_program_best_by_test, test_selection = select_test_easiest))

full_test_coverage_random_test_best_program = partial(select_full_test_coverage, test_program_selection = partial(select_program_best_by_test, test_selection = select_test_random))

full_test_coverage_hardest_test_rand_program = partial(select_full_test_coverage, test_program_selection = partial(select_program_random, test_selection = select_test_hardest)) 

full_test_coverage_easiest_test_rand_program = partial(select_full_test_coverage, test_program_selection = partial(select_program_random, test_selection = select_test_easiest))

full_test_coverage_random_test_rand_program = partial(select_full_test_coverage, test_program_selection = partial(select_program_random, test_selection = select_test_random))


def front_evolve(gold_outputs, func_list, terminal_list, *,
                    population_size = 1000, max_gens = 100, archive_size = 10,
                    fitness_fns = [hamming_distance_fitness, depth_fitness], main_fitness_fn = hamming_distance_fitness,
                    init_fn = init_each, map_fn = identity_map, breed_fn = partial(subtree_breed, breed_select_fn = full_coverage_selection),
                    eval_fn = gp_eval, analyze_pop_fn = analyze_population,
                    select_parents_fn = full_test_coverage_hardest_test_best_program):
    stats = {}
    syntax_cache = {}
    node_builder = partial(cached_node_builder, syntax_cache = syntax_cache, node_builder = simple_node_builder, stats = stats)
    full_coverage = FullTestCoverageIterators()
    shared_context = dict(
        gold_outputs = gold_outputs, func_list = func_list, terminal_list = terminal_list,
        fitness_fns = fitness_fns, main_fitness_fn = main_fitness_fn, node_builder = node_builder,
        syntax_cache = syntax_cache, stats = stats, int_cache = {}, out_cache = {}, full_coverage = full_coverage,
        breeding_stats = BreedingStats())
    evol_fns = utils.bind_fns(shared_context, init_fn, map_fn, breed_fn, eval_fn, analyze_pop_fn, select_parents_fn)    
    best_ind, gen = pareto_front_loop(archive_size, population_size, max_gens, *evol_fns)
    stats["gen"] = gen
    stats["best_found"] = best_ind is not None
    return best_ind, stats

cov_ht_bp = partial(front_evolve, select_parents_fn = full_test_coverage_hardest_test_best_program)

cov_et_bp = partial(front_evolve, select_parents_fn = full_test_coverage_easiest_test_best_program)

cov_rt_bp = partial(front_evolve, select_parents_fn = full_test_coverage_random_test_best_program)

cov_ht_rp = partial(front_evolve, select_parents_fn = full_test_coverage_hardest_test_rand_program)

cov_et_rp = partial(front_evolve, select_parents_fn = full_test_coverage_easiest_test_rand_program)

cov_rt_rp = partial(front_evolve, select_parents_fn = full_test_coverage_random_test_rand_program)

cov_sim_names = [ 'cov_ht_bp', 'cov_et_bp', 'cov_rt_bp', 'cov_ht_rp', 'cov_et_rp', 'cov_rt_rp' ]

if __name__ == '__main__':
    import gp_benchmarks
    game_name, (gold_outputs, func_list, terminal_list) = gp_benchmarks.get_benchmark('cmp8')
    best_prog, stats = cov_rt_bp(gold_outputs, func_list, terminal_list)
    print(best_prog)
    print(stats)
    pass    

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
