''' Reimplements interaction game defined in games.py run_game in more functional way 
    Used in epxerimentation with test-based GP
'''
from functools import partial
import numpy as np
from de import extract_dims_np
from gp import analyze_population, cached_eval, cached_node_builder, depth_fitness, hamming_distance_fitness, init_each, simple_node_builder, subtree_breed, tournament_selection
from rnd import default_rnd
import utils

def coevol_loop(max_gens, population_sizes, init_fn, select_fns, interract_fn, update_fn, analyze_pop_fn):
    """ General coevolution loop """
    # Create initial population
    populations = init_fn(population_sizes) # list of populations that should interract
    gen = 0
    best_ind = None
    fitnesses = [None for _ in populations]
    while gen < max_gens:

        selections = [s(pop, fitnesses) for s, pop in zip(select_fns, populations, fitnesses)] # selecting who should interract from populations

        outputs, fitnesses, interactions = interract_fn(selections) # return list of interactions, one per population. In zero-sum game, sum of interactions = const (e.g. 0, 1)
        # test-based GP we consider - zero-sum game where sum = 1, win = 1 (program solved test), 0

        best_ind = analyze_pop_fn(selections[0], outputs, fitnesses, all_populations = selections)
        if best_ind is not None:
            break

        populations, fitnesses = update_fn(selections, interactions, fitnesses) # update populations based on interactions

        gen += 1

    return best_ind

# def init_programs_tests(init_programs, test_pool_size, test_fraction = 1.0):
#     programs = init_programs()
#     num_tests = int(test_fraction * test_pool_size)
#     tests = default_rnd.choice(test_pool_size, num_tests, replace=False)
#     return [programs, tests]

def zero_init(population_sizes, num_pops = 2):
    return [[] for _ in range(num_pops)]

class BreedingGroups():
    ''' Contains relation of what inds are good for breedins, computed from coordinate system extraction '''
    def __init__(self):
        # group has relation of ind_id to other ind_ids. np.ndarray
        self.groups = {}
        self.selected_ind_id = None
        self.selected_group_ids = None


# fitness_cache = {}
# current_good_programs = []
# interaction_cache = {}
# breed_groups = {} # for program, defines good candidates for breeding

# previously_selected = None

def breeding_group_selection(population, fitnesses, select_fn = tournament_selection, *, breeding_groups: BreedingGroups):
    if breeding_groups.selected_ind_id is not None:
        if breeding_groups.selected_group_ids is None:
            subpop = population
            subpop_fitnesses = fitnesses
        else: 
            subpop = [population[i] for i in breeding_groups.selected_group_ids]
            subpop_fitnesses = fitnesses[breeding_groups.selected_group_ids]
        new_ind_id = select_fn(subpop, subpop_fitnesses)
        # new_ind = subpop[new_ind_id]
        breeding_groups.selected_ind_id = None 
        breeding_groups.selected_group_ids = None
        if breeding_groups.selected_group_ids is not None:
            new_ind_id = breeding_groups.selected_group_ids[new_ind_id]
        return new_ind_id 
    selected_one_id = select_fn(population, fitnesses)
    breeding_groups.selected_ind_id = selected_one_id
    breeding_groups.selected_group_ids = breeding_groups.groups.get(selected_one_id, None)
    return selected_one_id

def select_explore_exploit_programs(good_programs, fitnesses, init_fn = init_each, 
                                    breed_fn = partial(subtree_breed, breed_select_fn = breeding_group_selection), 
                                    explore_size = 0, exploit_size = 100):
    ''' select those programs who previously distinguished tests of underlying objectives and then add new programs to explore'''
    global current_good_programs
    current_good_programs = good_programs # we store them to use later for coordinate system extraction
    if len(good_programs) == 0: # nobody can breed
        all_programs = init_fn(explore_size + exploit_size)
    else:
        # all_programs = []
        all_programs = list(good_programs)
        inited_programs = init_fn(explore_size)
        all_programs.extend(inited_programs)
        breed_programs = breed_fn(exploit_size, good_programs, fitnesses)
        all_programs.extend(breed_programs)
    return all_programs

def select_explore_exploit_tests(good_tests, fitnesses, fraction = 1, *, gold_outputs):
    ''' selecting tests found to be underlying objectives '''
    if fraction == 1:
        return np.arange(len(gold_outputs))
    
    to_select_num = int(fraction * len(gold_outputs)) - len(good_tests)
    if to_select_num <= 0:
        return good_tests
    
    good_test_set = set(good_tests)
    possible_new_tests = [i for i in range(len(gold_outputs)) if i not in good_test_set]
    selected_new_tests = default_rnd.choice(possible_new_tests, to_select_num, replace=False)
    return np.concatenate([good_tests, selected_new_tests])

def prog_test_interractions(populations, eval_fn = cached_eval, *, gold_outputs):
    ''' Test-based interactions, zero-sum game '''
    programs, tests = populations
    fitnesses, interactions, outputs = eval_fn(programs)
    if len(gold_outputs) == len(tests):
        return fitnesses, interactions    
    subset_interactions = interactions[:, tests]
    return outputs, fitnesses, subset_interactions

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

def update_cand_underlying_objectives(populations, interactions, fitnesses, test_selection_strategy = select_all_tests, max_obj_size = 20, ignore_same_dim = True, *, breeding_groups: BreedingGroups):
    """ Finds underlying objectives for programs, samples most discriminating tests (if necessary)
        Extracted coordinate system preserve relation of what individuals could be combined in breeding 
        Each underlying objective could be combined with other one, or spanned point which is at the end but does not fulfill this objective 
        Each spanned point could be combined with underlying objectives or their spanned points if it does not fulfill them 
        
        Underlying objectives could be filtered out if new population is too big:
            - in case if size of spanned is too small, we just randomly sample underlying objectives to be preserved
            - otherwise, we discard all of the objectives from interactions and build new objectives from spanned points and repeat size analysis 
    """
    programs, tests = populations

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
        # filtering out spanned points which are not ends of axes
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
    
    selected_program_ids = list(preserved_spanned)
    if add_all_objs:
        for obj in objs:
            for prog_id in obj:
                selected_program_ids.append(prog_id)
    elif need_to_sample_objs:
        # resort to random sampling of objectives
        objs_wins = np.sum(selected_interactions[[obj[0] for obj in objs]], axis = 1)
        objs_with_wins = sorted(zip(objs, objs_wins), key = lambda x: x[1], reverse = True)
        objs = [obj for obj, _ in objs_with_wins]
        while (len(selected_program_ids) < max_obj_size) and len(objs) > 0:
            obj_idxs = list(range(0, len(objs)))
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
    selected_fitnesses = fitnesses[selected_program_ids]

    prog_poss = {test_id: {dim_id: len(dim) - 1} for dim_id, dim in enumerate(dims) for test_id in dim[-1] }
    prog_poss.update(spanned)

    # test_interactions = 1 - interactions[all_selected_programs].T
    all_selected_tests = test_selection_strategy(tests, selected_interactions[selected_program_ids])
    new_breed_groups = {}
    for new_id, prog_id in enumerate(selected_program_ids):
        prog_dims = prog_poss[prog_id]
        prog_breed_group = []
        prog_keys = set(prog_dims.keys())
        for new_id2, prog_id2 in enumerate(selected_program_ids):
            prog_dims2 = prog_poss[prog_id2]
            prog_keys2 = set(prog_dims2.keys())
            common_keys = prog_keys & prog_keys2
            new_keys = prog_keys2 - prog_keys
            # second part of eq - could we ignore it??
            if (len(new_keys) > 0) or (not ignore_same_dim and any(prog_dims[dim_id] < prog_dims2[dim_id] for dim_id in common_keys)):
                prog_breed_group.append(new_id2)
        new_breed_groups[new_id] = np.array(prog_breed_group)
    breeding_groups.groups = new_breed_groups
    breeding_groups.selected_group_ids = None 
    breeding_groups.selected_ind_id = None
    return [all_selected_programs, all_selected_tests], [selected_fitnesses, None]
    
def gp_coevolve2(gold_outputs, func_list, terminal_list,
                 population_sizes = [0, 0], max_gens = 100,
                fitness_fns = [hamming_distance_fitness, depth_fitness], main_fitness_fn = None,
                init_fn = zero_init, first_select_fn = select_explore_exploit_programs,
                second_select_fn = select_explore_exploit_tests,
                interract_fn = prog_test_interractions,
                update_fn = update_cand_underlying_objectives,
                analyze_pop_fn = analyze_population):
    stats = {}
    syntax_cache = {}
    node_builder = partial(cached_node_builder, syntax_cache = syntax_cache, node_builder = simple_node_builder)
    breeding_groups = BreedingGroups()
    shared_context = dict(
        gold_outputs = gold_outputs, func_list = func_list, terminal_list = terminal_list,
        fitness_fns = fitness_fns, main_fitness_fn = main_fitness_fn, node_builder = node_builder,
        syntax_cache = syntax_cache, eval_cache = {}, stats = stats, breeding_groups = breeding_groups)
    init_fn, first_select_fn, second_select_fn, interract_fn, update_fn, analyze_pop_fn = utils.bind_fns(shared_context, init_fn, first_select_fn, second_select_fn, interract_fn, update_fn, analyze_pop_fn)
    best_ind = coevol_loop(max_gens, population_sizes, max_gens, init_fn, [first_select_fn, second_select_fn], interract_fn, update_fn, analyze_pop_fn)
    return best_ind, stats

coevol_uo = gp_coevolve2

coevol_sim_names = ["coevol_uo"]

# if __name__ == "__main__":
#     import gp_benchmarks
#     game_name, (gold_outputs, func_list, terminal_list) = gp_benchmarks.get_benchmark('cmp6')
#     coevol_uo(game_name, gold_outputs, func_list, terminal_list)
#     pass