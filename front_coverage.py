from dataclasses import dataclass, field
from functools import partial
from itertools import cycle
import numpy as np

from gp import RuntimeContext, analyze_population, create_runtime_context, gp_eval, depth_fitness, hamming_distance_fitness, identity_map, init_each, subtree_breed
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
        parents_interactions = interactions[selected_indexes]
        new_population = breed_fn(population_size, parents, parents_fitnesses, parents_interactions)
        new_population.extend(parents)
        population = new_population
        gen += 1
    return best_ind, gen

class FullTestCoverageIterators():
    def __init__(self):
        self.subgroups = iter([])
        self.selected = iter([])

@dataclass 
class FrontCovRuntimeContext(RuntimeContext):
    full_coverage: FullTestCoverageIterators = field(default_factory=FullTestCoverageIterators)

def full_coverage_selection(population, fitnesses, interactions, coverage_selection_size = 2, *, runtime_context: FrontCovRuntimeContext):
    if (next_index := next(runtime_context.full_coverage.selected, None)) is None:
        subgroup_start, subgroup_size = next(runtime_context.full_coverage.subgroups)
        if subgroup_size < coverage_selection_size:
            choices = default_rnd.choice(subgroup_size, size = coverage_selection_size, replace = True)
        else:
            choices = default_rnd.choice(subgroup_size, size = coverage_selection_size, replace = False)
        indexes = subgroup_start + choices
        runtime_context.full_coverage.selected = iter(indexes)
        next_index = next(runtime_context.full_coverage.selected)
        # if len(population) >= 6 and str(population[5]) == "x6":
        #     print(f"selected: ss {subgroup_start}, ssize {subgroup_size} choice {choices} indexes {indexes} next {next_index}")
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

def select_full_test_coverage(front_selection_size, interactions, test_program_selection = select_program_best_by_test, *, runtime_context: FrontCovRuntimeContext):
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
            if cur_counter == 1 and best_program_id is None:
                allowed_program_mask = np.any(ints, axis = 1) #reset program mask to allow same programs but for different allowed_test_mask_all
            cur_counter = 0
            # NOTE: at this point it makes sense to form coverage groups.
    if cur_counter > 0:
        groups.append(cur_counter)
    if len(selected) == 0:
        selected = default_rnd.choice(ints.shape[0], size = min(front_selection_size, ints.shape[0]), replace = False)
    if len(groups) == 0:
        groups.append(len(selected))
    indexes = list(zip([0, *np.cumsum(groups).tolist()], groups))
    runtime_context.full_coverage.selected = iter([])
    runtime_context.full_coverage.subgroups = cycle(indexes)
    return np.array(selected)

full_test_coverage_hardest_test_best_program = partial(select_full_test_coverage, test_program_selection = partial(select_program_best_by_test, test_selection = select_test_hardest)) 

full_test_coverage_easiest_test_best_program = partial(select_full_test_coverage, test_program_selection = partial(select_program_best_by_test, test_selection = select_test_easiest))

full_test_coverage_random_test_best_program = partial(select_full_test_coverage, test_program_selection = partial(select_program_best_by_test, test_selection = select_test_random))

full_test_coverage_hardest_test_rand_program = partial(select_full_test_coverage, test_program_selection = partial(select_program_random, test_selection = select_test_hardest)) 

full_test_coverage_easiest_test_rand_program = partial(select_full_test_coverage, test_program_selection = partial(select_program_random, test_selection = select_test_easiest))

full_test_coverage_random_test_rand_program = partial(select_full_test_coverage, test_program_selection = partial(select_program_random, test_selection = select_test_random))


def front_evolve(problem_init, *,
                    population_size = 1000, max_gens = 100, archive_size = 100,
                    fitness_fns = [hamming_distance_fitness, depth_fitness], main_fitness_fn = hamming_distance_fitness,
                    select_fitness_ids = None, init_fn = init_each, map_fn = identity_map, breed_fn = partial(subtree_breed, breed_select_fn = full_coverage_selection),
                    eval_fn = gp_eval, analyze_pop_fn = analyze_population, 
                    select_parents_fn = full_test_coverage_hardest_test_best_program):
    runtime_context = create_runtime_context(fitness_fns, main_fitness_fn, select_fitness_ids, 
                                                context_class=FrontCovRuntimeContext)
    problem_init(runtime_context = runtime_context)
    evo_funcs = [init_fn, map_fn, breed_fn, eval_fn, analyze_pop_fn, select_parents_fn]
    evo_funcs_bound = [partial(fn, runtime_context = runtime_context) for fn in evo_funcs]
    best_ind, gen = pareto_front_loop(archive_size, population_size, max_gens, *evo_funcs_bound)
    runtime_context.stats["gen"] = gen
    runtime_context.stats["best_found"] = best_ind is not None
    return best_ind, runtime_context.stats

cov_ht_bp = partial(front_evolve, select_parents_fn = full_test_coverage_hardest_test_best_program)

cov_et_bp = partial(front_evolve, select_parents_fn = full_test_coverage_easiest_test_best_program)

cov_rt_bp = partial(front_evolve, select_parents_fn = full_test_coverage_random_test_best_program)

cov_ht_rp = partial(front_evolve, select_parents_fn = full_test_coverage_hardest_test_rand_program)

cov_et_rp = partial(front_evolve, select_parents_fn = full_test_coverage_easiest_test_rand_program)

cov_rt_rp = partial(front_evolve, select_parents_fn = full_test_coverage_random_test_rand_program)

# cov_sim_names = [ 'cov_ht_bp', 'cov_et_bp', 'cov_rt_bp', 'cov_ht_rp', 'cov_et_rp', 'cov_rt_rp' ]
cov_sim_names = [ 'cov_ht_bp']

if __name__ == '__main__':
    import gp_benchmarks
    problem_builder = gp_benchmarks.get_benchmark('cmp6')
    best_prog, stats = cov_rt_bp(problem_builder)
    print(best_prog)
    print(stats)
    pass    