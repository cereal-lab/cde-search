''' Reimplements interaction game defined in games.py run_game in more functional way 
    Used in epxerimentation with test-based GP
'''
from functools import partial
import numpy as np
from de import extract_dims_np
from gp import depth_fitness, get_metrics, gp_evaluate, hamming_distance_fitness, init_each, tournament_selection, ramped_half_and_half, random_selection, subtree_breed, subtree_crossover, subtree_mutation
from rnd import default_rnd, seed
from utils import write_metrics

def run_coevolution(max_num_interactions, 
              initialization_fn, selection_fns, interract_fn, update_fn, get_metric_fn):
    """ General coevolution loop that interracts programs and tests. """
    # Create initial population
    populations = initialization_fn() # list of populations that should interract
    stats = []
    step = 0
    while step < max_num_interactions:

        selections = [s(pop) for s, pop in zip(selection_fns, populations)] # selecting who should interract from populations

        fitnesses, interactions = interract_fn(selections) # return list of interactions, one per population. In zero-sum game, sum of interactions = const (e.g. 0, 1)
        # test-based GP we consider - zero-sum game where sum = 1, win = 1 (program solved test), 0

        best_found, metrics = get_metric_fn(fitnesses, selections) 
        stats.append(metrics)
        if best_found:
            break

        populations = update_fn(selections, interactions) # update populations based on interactions

        step += 1

    return stats

def get_program_metrics(main_fitness_index, fitnesses, populations):
    return get_metrics(main_fitness_index, fitnesses, populations[0])

# def init_programs_tests(init_programs, test_pool_size, test_fraction = 1.0):
#     programs = init_programs()
#     num_tests = int(test_fraction * test_pool_size)
#     tests = default_rnd.choice(test_pool_size, num_tests, replace=False)
#     return [programs, tests]

def zero_init():
    return [[], []]

#                                     init_fn = partial(ramped_half_and_half, 1, 5),
#                                     selection_fn = random_selection,
#                                     mutation_fn = partial(subtree_mutation, 0.1, 17),
#                                     crossover_fn = partial(subtree_crossover, 0.1, 17),                               
#                                     breed_fn = partial(subtree_breed, mutation_rate = 0.1, crossover_rate = 0.9),
# # 

fitness_cache = {}
current_good_programs = []
interaction_cache = {}
breed_groups = {} # for program, defines good candidates for breeding

previously_selected = None

def select_based_on_breeding_group(population, fitnesses, one_selection = tournament_selection):
    global previously_selected
    if previously_selected is not None:
        res = previously_selected
        previously_selected = None 
        return res
    selected_one = one_selection(population, fitnesses)
    breeding_group = breed_groups.get(selected_one, [])
    if len(breeding_group) == 0:
        previously_selected = one_selection(population, fitnesses)
    else:
        next_id = default_rnd.choice(len(breeding_group))
        previously_selected = breeding_group[next_id]
    return selected_one

# def tournament_selection(population, fitnesses, comp_fn = min, selection_size = 7):
#     ''' Select parents using tournament selection '''
#     selected = default_rnd.choice(len(population), selection_size, replace=True)
#     best_index = comp_fn(selected, key=lambda i: (*fitnesses[i].tolist(),))
#     best = population[best_index]
#     return best    

def explore_exploit_programs(good_programs, init_fn, breed_fn, explore_size = 0, exploit_size = 100):
    ''' select those programs who previously distinguished tests of underlying objectives and then add new programs to explore'''
    global current_good_programs
    current_good_programs = good_programs # we store them to use later for coordinate system extraction
    if len(good_programs) == 0: # nobody can breed
        all_programs = [init_fn() for _ in range(explore_size + exploit_size)]
    else:
        # all_programs = list(discriminating_programs)
        all_programs = []
        inited_programs = [ init_fn() for _ in range(explore_size)]
        all_programs.extend(inited_programs)
        fitnesses = [fitness_cache[p] for p in good_programs]
        breed_programs = breed_fn(breed_size = exploit_size, population = good_programs, fitnesses = fitnesses)
        all_programs.extend(breed_programs)
    return all_programs

def explore_exploit_tests(good_tests, gold_outputs, selection_size_fraction = 1):
    ''' selecting tests found to be underlying objectives '''
    if selection_size_fraction == 1:
        return np.arange(len(gold_outputs))
    
    to_select_num = int(selection_size_fraction * len(gold_outputs)) - len(good_tests)
    if to_select_num <= 0:
        return good_tests
    
    good_test_set = set(good_tests)
    possible_new_tests = [i for i in range(len(gold_outputs)) if i not in good_test_set]
    selected_new_tests = default_rnd.choice(possible_new_tests, to_select_num, replace=False)
    return np.concatenate([good_tests, selected_new_tests])

def prog_test_interractions(populations, evaluate_fn = gp_evaluate):
    ''' Test-based interactions, zero-sum game '''
    programs, tests = populations
    fitnesses, interactions, *_ = evaluate_fn(programs)
    for i, p in enumerate(programs):
        fitness_cache[p] = fitnesses[i]
        interaction_cache[p] = interactions[i]
    all_programs = current_good_programs + programs
    all_fitnesses = np.array([fitness_cache[p] for p in all_programs])
    all_interactions = np.array([interaction_cache[p] for p in all_programs])
    return all_fitnesses, all_interactions[:, tests]

# def get_discriminating_set(selected_other: np.ndarray, interactions: np.ndarray):
#     ''' Returns candidates that could distinguish selected individuals based on seen interactions 
#         selected_other - set of selected rows from interactions 
#         interactions - matrix of interactions, we pick discriminating columns
#     '''
#     test_pairs = [ (selected_other[i], selected_other[j]) for i in range(len(selected_other)) for j in range(i + 1, len(selected_other)) ]
#     discriminating_lists = {}
#     for (tid1, tid2) in test_pairs:
#         test1 = interactions[tid1]
#         test2 = interactions[tid2]
#         ps = np.where(np.logical_xor(test1, test2))[0]
#         for cid in ps:
#             tid1, tid2 = min(tid1, tid2), max(tid1, tid2)
#             discriminating_lists.setdefault(cid, set()).add((tid1, tid2))
#     discriminating_set = set()
#     while len(discriminating_lists) > 0: 
#         max_cid = max(discriminating_lists.keys(), key = lambda x: len(discriminating_lists[x]))
#         max_cid_set = discriminating_lists[max_cid]
#         discriminating_set.add(max_cid)
#         del discriminating_lists[max_cid]
#         to_delete = set()
#         for cid, cid_set in discriminating_lists.items():
#             cid_set -= max_cid_set
#             if len(cid_set) == 0:
#                 to_delete.add(cid)
#         for cid in to_delete:
#             del discriminating_lists[cid]
#     return discriminating_set

# def update_test_underlying_objectives(populations, interactions, sample_size = 20, spanned_num = 10):
#     """ Finds underlying objectives tests and discriminating programs """
#     programs, tests = populations
#     test_ints = (1 - interactions).T
#     dims, origin, spanned = extract_dims_np(test_ints)
#     spanned_ends = { }
#     for test_id, position in spanned.items():
#         for dim_id, point_id in position.items():
#             spanned_ends.setdefault((dim_id, point_id), []).append(test_id)
#     dim_ends = [ (dim_id, len(d) - 1) for dim_id, d in enumerate(dims) ]
#     preserved_spanned = set()
#     for dim_end in dim_ends:
#         if dim_end in spanned_ends:
#             preserved_spanned.update(spanned_ends[dim_end])
#     preserved_spanned = list(preserved_spanned)
#     if len(preserved_spanned) > spanned_num:
#         preserved_spanned_ids = default_rnd.choice(len(preserved_spanned), spanned_num, replace=False)
#         preserved_spanned = [preserved_spanned[i] for i in preserved_spanned_ids]
#     selected_tests = preserved_spanned
#     axes = [[[test_id for test_id in point] for point in dim] for dim in dims]
#     # axes.sort(key = lambda x: x[-1][1])
#     axe_id = 0
#     while len(selected_tests) < sample_size and len(axes) > 0:
#         axis = axes[axe_id]
#         point = axis[-1]
#         el_id = default_rnd.randint(0, len(point))
#         test = point.pop(el_id)
#         selected_tests.append(test)
#         if len(point) == 0:
#             axis.pop()
#         if len(axis) == 0:
#             axes.pop(axe_id)
#             if len(axes) == 0:
#                 break
#         else:
#             axe_id += 1
#         if axe_id == len(axes):
#             axe_id = 0
    
#     selected_program_ids = get_discriminating_set(selected_tests, test_ints) #computing most distinguishing programs 
#     selected_program = [programs[i] for i in selected_program_ids]

#     selected_tests_ = [tests[i] for i in selected_tests]
#     return [selected_program, selected_tests_]

def select_all_tests(tests, cand_interactions):
    return tests

def select_discriminative_tests(tests, prog_interactions: np.ndarray):
    # test_interations = 1 - cand_interactions.T
    prog_pairs = [ (i, j) for i in range(prog_interactions.shape[0]) for j in range(i + 1, len(prog_interactions.shape[0])) ]
    discriminating_lists = {}
    for (i, j) in prog_pairs:
        int1 = prog_interactions[i]
        int2 = prog_interactions[j]
        discriminating_test_ids_for_pair = np.where(np.logical_xor(int1, int2))[0]
        for test_id in discriminating_test_ids_for_pair:
            discriminating_lists.setdefault(test_id, set()).add((i, j))
    discriminating_test_ids = []
    while len(discriminating_lists) > 0: 
        max_discriminating_test_id = max(discriminating_lists.keys(), key = lambda x: len(discriminating_lists[x]))
        distinguished_program_pairs = discriminating_lists.pop(max_discriminating_test_id)
        discriminating_test_ids.append(max_discriminating_test_id)
        new_discriminating_lists = {}
        for test_id, test_prog_pairs in discriminating_lists.items():
            test_prog_pairs -= distinguished_program_pairs
            if len(test_prog_pairs) > 0:
                new_discriminating_lists[test_id] = test_prog_pairs
        discriminating_lists = new_discriminating_lists
    discriminating_tests = [tests[i] for i in discriminating_test_ids]
    return discriminating_tests

def update_cand_underlying_objectives(populations, interactions, test_selection_strategy = select_all_tests, max_obj_size = 20, ignore_same_dim = True):
    """ Finds underlying objectives for programs, samples most discriminating tests (if necessary)
        Extracted coordinate system preserve relation of what individuals could be combined in breeding 
        Each underlying objective could be combined with other one, or spanned point which is at the end but does not fulfill this objective 
        Each spanned point could be combined with underlying objectives or their spanned points if it does not fulfill them 
        
        Underlying objectives could be filtered out if new population is too big:
            - in case if size of spanned is too small, we just randomly sample underlying objectives to be preserved
            - otherwise, we discard all of the objectives from interactions and build new objectives from spanned points and repeat size analysis 
    """
    global breed_groups    
    ps, tests = populations
    programs = current_good_programs + ps

    selected_programs = programs
    selected_interactions = interactions
    
    need_to_sample_objs = False
    add_all_objs = False
    while True:
        dims, origin, spanned = extract_dims_np(selected_interactions) # program underlying objectives
        objs = [list(dim[-1]) for dim in dims] # ends of axes
        objs_count = sum(len(point) for point in objs)
        dim_ends = [ (dim_id, len(d) - 1) for dim_id, d in enumerate(dims) ]
        spanned_ends = { }
        for test_id, position in spanned.items():
            for dim_id, point_id in position.items():
                spanned_ends.setdefault((dim_id, point_id), []).append(test_id)
        preserved_spanned = set()
        for dim_end in dim_ends:
            if dim_end in spanned_ends:
                preserved_spanned.update(spanned_ends[dim_end])
        if len(preserved_spanned) + objs_count <= max_obj_size:
            add_all_objs = True
            break
        if max_obj_size >= len(preserved_spanned):
            need_to_sample_objs = True 
            break
        else: 
            spanned_ids = list(spanned.keys())
            selected_programs = [selected_programs[i] for i in spanned_ids]
            selected_interactions = selected_interactions[spanned_ids]
            # repeat dimension extraction on spanned points
    
    prog_poss = {test_id: {dim_id: len(dim) - 1} for dim_id, dim in enumerate(dims) for test_id in dim[-1] }
    prog_poss.update(spanned)
    selected_program_ids = list(preserved_spanned)
    if add_all_objs:
        for obj in objs:
            for prog_id in obj:
                selected_program_ids.append(prog_id)
    elif need_to_sample_objs:
        # resort to random sampling of objectives
        while (len(selected_program_ids) < max_obj_size) and len(objs) > 0:
            obj_idxs = list(range(0, len(objs)))
            default_rnd.shuffle(obj_idxs)
            idx_to_delete = set()
            for obj_id in obj_idxs:
                obj = objs[obj_id]
                prog_id_id = default_rnd.choice(len(obj))
                prog_id = obj.pop(prog_id_id)
                if len(obj) == 0:
                    idx_to_delete.add(obj_id)
                selected_program_ids.append(prog_id)
                if len(selected_program_ids) == max_obj_size:
                    break
            objs = [obj for i, obj in enumerate(objs) if i not in idx_to_delete]

    all_selected_programs = [selected_programs[i] for i in selected_program_ids]

    # test_interactions = 1 - interactions[all_selected_programs].T
    all_selected_tests = test_selection_strategy(tests, selected_interactions[selected_program_ids])
    new_breed_groups = {}
    for prog_id in selected_program_ids:
        prog_dims = prog_poss[prog_id]
        prog_breed_group = []
        prog_keys = set(prog_dims.keys())
        for prog_id2 in selected_program_ids:
            prog_dims2 = prog_poss[prog_id2]
            prog_keys2 = set(prog_dims2.keys())
            common_keys = prog_keys & prog_keys2
            new_keys = prog_keys2 - prog_keys
            # second part of eq - could we ignore it??
            if (len(new_keys) > 0) or (not ignore_same_dim and any(prog_dims[dim_id] < prog_dims2[dim_id] for dim_id in common_keys)):
                prog_breed_group.append(selected_programs[prog_id2])
        new_breed_groups[selected_programs[prog_id]] = prog_breed_group
    breed_groups = new_breed_groups
    return [all_selected_programs, all_selected_tests]

def run_coevolution_experiment(sim_name, game_name, gold_outputs, func_list, terminal_list,
                        evaluate_fn = gp_evaluate,
                        init_fn = partial(ramped_half_and_half, 1, 5),
                        mutation_fn = partial(subtree_mutation, 0.1, 17),
                        breed_fn = partial(subtree_breed, mutation_rate = 0.1, crossover_rate = 0.9,                                            
                                            crossover_fn = partial(subtree_crossover, 0.1, 17),
                                            selection_fn = select_based_on_breeding_group),                        
                        program_selection_fn = 
                            partial(explore_exploit_programs, explore_size = 0, exploit_size = 100),
                        test_selection_fn = partial(explore_exploit_tests, selection_size_fraction = 1),
                        interract_fn = prog_test_interractions,
                        update_fn = partial(update_cand_underlying_objectives, test_selection_strategy = select_all_tests, 
                                                max_obj_size = 20, ignore_same_dim = True),
                        get_metrics_fn = partial(get_program_metrics, 0),
                        fitness_fns = [hamming_distance_fitness, depth_fitness],
                        record_fitness_ids = [0, 1],
                        metrics_file = "data/metrics/objs.jsonlist",
                        max_generations = 100,
                        num_runs = 30):
    mutation = partial(mutation_fn, func_list, terminal_list)
    breed = partial(breed_fn, mutation_fn = mutation)
    program_selection = partial(program_selection_fn, 
                                init_fn = partial(init_fn, func_list = func_list, terminal_list = terminal_list), 
                                breed_fn = breed)
    test_selection = partial(test_selection_fn, gold_outputs = gold_outputs)
    selection_fns = [ program_selection, test_selection ]
    evaluate = partial(evaluate_fn, gold_outputs, fitness_fns = fitness_fns)
    interract = partial(interract_fn, evaluate_fn = evaluate)
    for run_id in range(num_runs):
        stats = run_coevolution(max_generations, zero_init, selection_fns, interract, update_fn, get_metrics_fn)
        best_inds, *fitness_metrics = zip(*stats)
        metrics = dict(game = game_name, sim = sim_name, seed = seed, run_id = run_id, best_ind = str(best_inds[-1]), best_ind_depth = best_inds[-1].get_depth())
        for i, metric in enumerate(fitness_metrics):
            if i in record_fitness_ids:
                i_i = record_fitness_ids.index(i)
                metrics["fitness" + str(i_i)] = metric
        write_metrics(metrics, metrics_file)
        pass

coevol_uo = partial(run_coevolution_experiment, "coevol_uo")

coevol_sim_names = ["coevol_uo"]

if __name__ == "__main__":
    import gp_benchmarks
    game_name, (gold_outcomes, func_list, terminal_list) = gp_benchmarks.get_benchmark('cmp6')
    coevol_uo(game_name, gold_outcomes, func_list, terminal_list)
    pass