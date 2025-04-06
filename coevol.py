''' Reimplements interaction game defined in games.py run_game in more functional way 
    Used in epxerimentation with test-based GP
'''
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
import numpy as np
from de import extract_dims_np_b
from gp import RuntimeContext, analyze_population, create_runtime_context, gp_eval, depth_fitness, hamming_distance_fitness, identity_map, init_each, random_selection, subtree_breed, tournament_selection
from rnd import default_rnd
import utils

def coevol_loop(max_gens, population_sizes, init_fn, map_fns, select_fns, interract_fn, update_fn, analyze_pop_fn):
    """ General coevolution loop """
    # Create initial population
    populations = init_fn(population_sizes) # list of populations that should interract
    gen = 0
    best_ind = None
    while gen < max_gens:

        selections = [s(pop) for s, pop in zip(select_fns, populations)] # selecting who should interract from populations

        selections = [m(pop) for m, pop in zip(map_fns, selections)]

        outputs, fitnesses, interactions, selections = interract_fn(selections) # return list of interactions, one per population. In zero-sum game, sum of interactions = const (e.g. 0, 1)
        # test-based GP we consider - zero-sum game where sum = 1, win = 1 (program solved test), 0

        best_ind = analyze_pop_fn(selections[0], outputs, fitnesses, all_populations = selections)
        if best_ind is not None:
            break

        populations = update_fn(selections, interactions) # update populations based on interactions
        gen += 1

    return best_ind, gen

# def init_programs_tests(init_programs, test_pool_size, test_fraction = 1.0):
#     programs = init_programs()
#     num_tests = int(test_fraction * test_pool_size)
#     tests = default_rnd.choice(test_pool_size, num_tests, replace=False)
#     return [programs, tests]

def zero_init(population_sizes, num_pops = 2, *, runtime_context: RuntimeContext):
    return [[] for _ in range(num_pops)]

class NondominantGroups():
    ''' From extracted u.o we compute nondominant groups for breeding '''
    def __init__(self):
        self.groups = [] # lsit of list of ids from population
        # self.ind_groups = {} # ind_id to group_id
        self.spanned_groups = {} # int -> set of int, to exclude as breeding candidate
        self.selected_group_id = None
        self.group_interactions = None
        self.group_fitnesses = None
        self.uo_reprs = 0

@dataclass
class CoevolRuntimeContext(RuntimeContext):
    nondominant: NondominantGroups = field(default_factory=NondominantGroups)

# fitness_cache = {}
# current_good_programs = []
# interaction_cache = {}
# breed_groups = {} # for program, defines good candidates for breeding

# previously_selected = None

def breeding_group_selection(population, fitnesses, select_fn = random_selection, *, runtime_context: CoevolRuntimeContext):
    nondominant = runtime_context.nondominant
    if nondominant.selected_group_id is not None:
        exclude_groups = nondominant.spanned_groups.get(nondominant.selected_group_id, set())
        exclude_groups.add(nondominant.selected_group_id)
        candidate_groups = [group_id for group_id in range(len(nondominant.groups)) if group_id not in exclude_groups]
        if len(candidate_groups) == 0:
            selected_one_id = default_rnd.choice(len(population))
            nondominant.selected_group_id = None
            return selected_one_id
        selected_group_id = select_fn(candidate_groups, nondominant.group_fitnesses[candidate_groups], runtime_context=runtime_context)
        selected_group = nondominant.groups[candidate_groups[selected_group_id]]
        selected_program_id_id = default_rnd.choice(len(selected_group))
        selected_program_id = selected_group[selected_program_id_id]
        nondominant.selected_group_id = None 
        return selected_program_id
    selected_group_id = select_fn(nondominant.groups, nondominant.group_fitnesses, runtime_context=runtime_context)
    prog_ids = nondominant.groups[selected_group_id]
    prog_id_id = default_rnd.choice(len(prog_ids))
    prog_id = prog_ids[prog_id_id]
    nondominant.selected_group_id = selected_group_id
    return prog_id

def select_explore_exploit_programs(good_programs, init_fn = init_each, 
                                    breed_fn = partial(subtree_breed, breed_select_fn = breeding_group_selection), 
                                    explore_size = 0, exploit_size = 500, uo_repr = 2, *, runtime_context: CoevolRuntimeContext):
    ''' select those programs who previously distinguished tests of underlying objectives and then add new programs to explore'''
    if len(good_programs) == 0: # nobody can breed
        all_programs = init_fn(explore_size + exploit_size, runtime_context = runtime_context)
    else:
        # all_programs = []
        all_programs = [] #list(good_programs)
        inited_programs = init_fn(explore_size, runtime_context = runtime_context)
        all_programs.extend(inited_programs)
        breed_programs = breed_fn(exploit_size, good_programs, None, runtime_context=runtime_context)
        all_programs.extend(breed_programs)

        #from good programs we only preserve at max uo_repr inds 
        runtime_context.nondominant.uo_reprs = 0
        for group in runtime_context.nondominant.groups:
            if len(group) > uo_repr:
                group_ids = default_rnd.choice(len(group), uo_repr, replace=False)
                selected_group = [group[i] for i in group_ids]
                all_programs.extend([good_programs[i] for i in selected_group])
                runtime_context.nondominant.uo_reprs += len(group_ids)
            else:
                all_programs.extend([good_programs[i] for i in group])
                runtime_context.nondominant.uo_reprs += len(group)
    return all_programs

def select_explore_exploit_tests(good_tests, rand_fraction = 0, *, runtime_context: RuntimeContext):
    ''' selecting tests found to be underlying objectives '''
    if rand_fraction == 1 or (len(good_tests) == 0 and rand_fraction == 0):
        return np.arange(len(runtime_context.gold_outputs))
    
    to_select_num = int(rand_fraction * len(runtime_context.gold_outputs)) - len(good_tests)
    if to_select_num <= 0:
        return good_tests
    
    good_test_set = set(good_tests)
    possible_new_tests = [i for i in range(len(runtime_context.gold_outputs)) if i not in good_test_set]
    selected_new_tests = default_rnd.choice(possible_new_tests, to_select_num, replace=False)
    return np.concatenate([good_tests, selected_new_tests])

def prog_test_interractions(populations, eval_fn = gp_eval, *, runtime_context: RuntimeContext):
    ''' Test-based interactions, zero-sum game '''
    programs, tests = populations
    outputs, fitnesses, interactions = eval_fn(programs, runtime_context = runtime_context)
    return outputs, fitnesses, interactions, populations

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

def select_hardest_tests(tests, prog_interactions: np.ndarray, fraction = 0.5, selection_size = 10, *, runtime_context: RuntimeContext):
    test_stats = np.sum(prog_interactions, axis=0)
    rand_test_vals = default_rnd.rand(len(test_stats))
    sorted_test_ids = sorted(enumerate(test_stats), key=lambda x: (x[1], rand_test_vals[x[0]]))
    filtered_test_ids = [i for i, c in sorted_test_ids if c > 0]
    fraction_size = max(1, int(fraction * prog_interactions.shape[1]))
    min_size = round(min(fraction_size, selection_size))
    runtime_context.stats.setdefault('obj_size', []).append(min_size)
    res = filtered_test_ids[:min_size]
    return res

def select_hardest_tests_anneal(tests, prog_interactions: np.ndarray, anneal_time = 100, max_fraction = 0.5, max_selection_size = 50, min_fraction = 0.1, min_selection_size = 10, *, runtime_context: RuntimeContext):
    test_stats = np.sum(prog_interactions, axis=0)
    rand_test_vals = default_rnd.rand(len(test_stats))
    sorted_test_ids = sorted(enumerate(test_stats), key=lambda x: (x[1], rand_test_vals[x[0]]))
    filtered_test_ids = [i for i, c in sorted_test_ids if c > 0]
    time_i = runtime_context.anneal_context.get(select_hardest_tests_anneal.__name__, 0)
    fraction = max_fraction - (max_fraction - min_fraction) * min(time_i, anneal_time) / anneal_time
    selection_size = max_selection_size - (max_selection_size - min_selection_size) * min(time_i, anneal_time) / anneal_time
    fraction_size = max(1, int(fraction * prog_interactions.shape[1]))
    min_size = round(min(fraction_size, selection_size))
    runtime_context.stats.setdefault('obj_size', []).append(min_size)
    res = filtered_test_ids[:min_size]
    runtime_context.anneal_context[select_hardest_tests_anneal.__name__] = time_i + 1
    return res

import time 

def dedupl_prog_ids(prog_ids, population):
    res = [] 
    present = {}
    for prog_id in prog_ids:
        prog = population[prog_id]
        if prog in present:
            continue
        present[prog] = prog_id 
        res.append(prog_id)
    return res

def update_cand_underlying_objectives(populations, interactions, test_selection_strategy = select_hardest_tests, *, runtime_context: CoevolRuntimeContext):
    """ Finds underlying objectives for programs, samples most discriminating tests (if necessary)
        Extracted coordinate system preserve relation of what individuals could be combined in breeding 
        Each underlying objective could be combined with other one, or spanned point which is at the end but does not fulfill this objective 
        Each spanned point could be combined with underlying objectives or their spanned points if it does not fulfill them 
        
        Underlying objectives could be filtered out if new population is too big:
            - in case if size of spanned is too small, we just randomly sample underlying objectives to be preserved
            - otherwise, we discard all of the objectives from interactions and build new objectives from spanned points and repeat size analysis 
    """
    programs, tests = populations

    selected_interactions = interactions[:, tests]

    # if weighted_interractions:
    #     # idea as in ifs - we count number of programs that solve tests
    #     test_difficulty = np.sum(selected_interactions, axis=0)
    #     solvable_tests_mask = test_difficulty > 0
    #     if not(np.all(solvable_tests_mask)):
    #         selected_interactions = selected_interactions[:, solvable_tests_mask]
    #         test_difficulty = test_difficulty[solvable_tests_mask]
    #     test_weight = 1.0 / test_difficulty # 0 --> many programs solve it, 1 --> one program solves it
    #     selected_interactions = selected_interactions * test_weight

    # TODO: handle case when all tests are difficult!!!

    origin_zero = np.packbits(np.zeros_like(selected_interactions[0], dtype = bool))
    packed_ints = np.packbits(selected_interactions.astype(bool), axis=1)
    start_ms = time.process_time()
    dims, origin, spanned_points = extract_dims_np_b(packed_ints, origin_outcomes=origin_zero) # program underlying objectives
    end_ms = time.process_time()
    dims_time = end_ms - start_ms
    runtime_context.stats.setdefault('dims_time', []).append(dims_time)
    # print(f"Extract dims time: {dims_time}")
    # pass 
    # dims0, _, spanned0 = extract_dims([list(t) for t in selected_interactions])
    # pass
    runtime_context.stats.setdefault('dims', []).append(len(dims))
    dim_points = { (dim_id, len(d) - 1): list(d[-1]) for dim_id, d in enumerate(dims) }
    dim_points_coords = frozenset(dim_points.keys())
    spanned_groups = {} 
    preserved_points =  [dedupl_prog_ids(prog_ids, programs) for (_, _), prog_ids in sorted(dim_points.items(), key = lambda x: x[0][0])]
    span_id_to_covered_dims = {}
    for span_coords, prog_ids in spanned_points.items():
        span_dims = frozenset.intersection(span_coords, dim_points_coords)
        if len(span_dims) == 0:
            continue
        span_id = len(preserved_points)
        preserved_points.append(dedupl_prog_ids(prog_ids, programs))
        span_id_to_covered_dims[span_id] = span_dims
        spanned_groups[span_id] = set(dim_id for (dim_id, _) in span_dims)
    for span_id1, span_dims1 in span_id_to_covered_dims.items():
        for span_id2, span_dims2 in span_id_to_covered_dims.items():
            if (span_id1 != span_id2) and (span_dims1.issubset(span_dims2) or span_dims2.issubset(span_dims1)):
                spanned_groups[span_id1].add(span_id2)
    runtime_context.stats.setdefault('spanned', []).append(len(spanned_groups))
    group_interactions = interactions[[p[0] for p in preserved_points]]
    group_fitnesses = len(runtime_context.gold_outputs) - np.sum(group_interactions, axis=1)
    selected_group_ids = set()
    selected_group_ids_list = []
    for test_id in tests:
        subgroup_ids = np.where(group_interactions[:, test_id] == 1)[0] # group solves tests
        filtered_subgroup_ids = [i for i in subgroup_ids if i not in selected_group_ids]
        if len(filtered_subgroup_ids) == 0:
            continue
        selected_group_id = min(filtered_subgroup_ids, key = lambda x: group_fitnesses[x])
        selected_group_ids.add(selected_group_id)
        selected_group_ids_list.append(selected_group_id)

    selected_groups = []
    selected_group_remap = {}
    for i in selected_group_ids_list:
        pp = preserved_points[i]
        selected_group_remap[i] = len(selected_groups)
        selected_groups.append(pp)
    selected_spanned_groups = {}
    for group_id, excluded_groups in spanned_groups.items():
        if group_id in selected_group_ids:
            new_excluded_groups = set(selected_group_remap[eg] for eg in excluded_groups if eg in selected_group_ids)
            if len(new_excluded_groups) > 0:
                selected_spanned_groups[selected_group_remap[group_id]] = new_excluded_groups

    selected_ids = [i for p in selected_groups for i in p]
    prog_id_remap = {old_id:new_id for new_id, old_id in enumerate(selected_ids)}
    selected_groups_remapped = [[prog_id_remap[i] for i in p] for p in selected_groups]
    # ind_groups = {prog_id: group_id for group_id, prog_ids in enumerate(preserved_points) for prog_id in prog_ids}

    runtime_context.nondominant.groups = selected_groups_remapped
    # nondominant.ind_groups = ind_groups
    runtime_context.nondominant.spanned_groups = selected_spanned_groups
    runtime_context.nondominant.selected_group_id = None
    runtime_context.nondominant.group_interactions = group_interactions[selected_group_ids_list]
    runtime_context.nondominant.group_fitnesses = group_fitnesses[selected_group_ids_list][:, None]

    selected_programs = [programs[i] for i in selected_ids]
    selected_tests = test_selection_strategy(tests, runtime_context.nondominant.group_interactions, runtime_context = runtime_context)
    return [selected_programs, selected_tests]

def update_cand_underlying_objectives2(populations, interactions, test_selection_strategy = select_hardest_tests, *, runtime_context: CoevolRuntimeContext):
    """ See update_cand_underlying_objectives, applies CSE two times, one on original interactions, second on spanned points """
    programs, tests = populations

    selected_interactions = interactions[:, tests]

    # if weighted_interractions:
    #     # idea as in ifs - we count number of programs that solve tests
    #     test_difficulty = np.sum(selected_interactions, axis=0)
    #     solvable_tests_mask = test_difficulty > 0
    #     if not(np.all(solvable_tests_mask)):
    #         selected_interactions = selected_interactions[:, solvable_tests_mask]
    #         test_difficulty = test_difficulty[solvable_tests_mask]
    #     test_weight = 1.0 / test_difficulty # 0 --> many programs solve it, 1 --> one program solves it
    #     selected_interactions = selected_interactions * test_weight

    # TODO: handle case when all tests are difficult!!!

    def filter_spanned_points(spanned_points, dim_points_coords):
        filtered_spanned_points = {}
        all_spanned_dims = set()
        for span_coords, prog_ids in spanned_points.items():
            span_dims = frozenset.intersection(span_coords, dim_points_coords)
            if len(span_dims) == 0:
                continue
            filtered_spanned_points[span_coords] = prog_ids
            all_spanned_dims.update(span_dims)
        return filtered_spanned_points, all_spanned_dims
    
    origin_zero = np.packbits(np.zeros_like(selected_interactions[0], dtype = bool))
    packed_ints = np.packbits(selected_interactions.astype(bool), axis=1)
    start_ms = time.process_time()
    dims, origin, spanned_points = extract_dims_np_b(packed_ints, origin_outcomes=origin_zero) # program underlying objectives

    dim_points = { (dim_id, len(d) - 1): list(d[-1]) for dim_id, d in enumerate(dims) }
    dim_points_coords = frozenset(dim_points.keys())

    filtered_spanned_points, all_spanned_dims = filter_spanned_points(spanned_points, dim_points_coords)

    filtered_dims = {k: v for k, v in dim_points.items() if k not in all_spanned_dims }

    filtered_spanned_points_int_ids = [ v[0] for v in filtered_spanned_points.values() ]
    filtered_spanned_points_ints = packed_ints[filtered_spanned_points_int_ids]
    remap_coord = {v[0]: k for k, v in filtered_spanned_points.items()}
    remap_group = {v[0]: v for k, v in filtered_spanned_points.items()}
    dims2, _, spanned_points2 = extract_dims_np_b(filtered_spanned_points_ints, origin_outcomes=origin_zero) # program underlying objectives    

    end_ms = time.process_time()
    dims_time = end_ms - start_ms
    runtime_context.stats.setdefault('dims_time', []).append(dims_time)    
    
    dim_points2 = { (dim_id, len(d) - 1): list(d[-1]) for dim_id, d in enumerate(dims2) }
    dim_points_coords2 = frozenset(dim_points2.keys())

    filtered_spanned_points2, all_spanned_dims2 = filter_spanned_points(spanned_points2, dim_points_coords2)

    filtered_dims2 = {remap_coord[filtered_spanned_points_int_ids[v[0]]]:remap_group[filtered_spanned_points_int_ids[v[0]]] for k, v in dim_points2.items() if k not in all_spanned_dims2 }

    spanned_points_remapped2 = {remap_coord[filtered_spanned_points_int_ids[v[0]]]:remap_group[filtered_spanned_points_int_ids[v[0]]] for k, v in filtered_spanned_points2.items()}

    left_spanned_points = {**filtered_dims2, **spanned_points_remapped2}

    runtime_context.stats.setdefault('dims', []).append(len(dims))
    runtime_context.stats.setdefault('dims2', []).append(len(dims2))
    runtime_context.stats.setdefault('spanned', []).append(len(filtered_spanned_points))
    runtime_context.stats.setdefault('spanned2', []).append(len(filtered_spanned_points2))

    preserved_points =  []
    preserved_points_dims = [] 
    
    for coors, prog_ids in filtered_dims.items():
        preserved_points.append(dedupl_prog_ids(prog_ids, programs))
        preserved_points_dims.append(frozenset([coors]))

    for coords, prog_ids in left_spanned_points.items():
        preserved_points.append(dedupl_prog_ids(prog_ids, programs))
        preserved_points_dims.append(coords)

    excluded_breed_groups = defaultdict(set)

    # spanned_groups = {}    
    # preserved_points =  [prog_ids for (_, _), prog_ids in sorted(dim_points.items(), key = lambda x: x[0][0])]
    # span_id_to_covered_dims = {}

    # print(f"Extract dims time: {dims_time}")
    # pass 
    # dims0, _, spanned0 = extract_dims([list(t) for t in selected_interactions])
    # pass
    # dim_points = { (dim_id, len(d) - 1): list(d[-1]) for dim_id, d in enumerate(dims) }
    # dim_points_coords = frozenset(dim_points.keys())
    # spanned_groups = {}    
    # preserved_points =  [prog_ids for (_, _), prog_ids in sorted(dim_points.items(), key = lambda x: x[0][0])]
    # span_id_to_covered_dims = {}
    # for span_coords, prog_ids in left_spanned_points.items():
    #     preserved_points.append(prog_ids)
        # span_id_to_covered_dims[span_id] = span_dims
        # spanned_groups[span_id] = set(dim_id for (dim_id, _) in span_dims)
    for point_id1 in range(len(preserved_points_dims)):
        point_dims1 = preserved_points_dims[point_id1]
        excluded_breed_groups[point_id1].add(point_id1)
        for point_id2 in range(point_id1 + 1, len(preserved_points_dims)):
            point_dims2 = preserved_points_dims[point_id2]
            if not frozenset.isdisjoint(point_dims1, point_dims2):
                excluded_breed_groups[point_id1].add(point_id2)
                excluded_breed_groups[point_id2].add(point_id1)
    # stats.setdefault('spanned', []).append(len(spanned_groups))
    group_interactions = interactions[[p[0] for p in preserved_points]]
    group_fitnesses = len(runtime_context.gold_outputs) - np.sum(group_interactions, axis=1)
    selected_group_ids = set()
    selected_group_ids_list = []
    for test_id in tests:
        subgroup_ids = np.where(group_interactions[:, test_id] == 1)[0] # group solves tests
        filtered_subgroup_ids = [i for i in subgroup_ids if i not in selected_group_ids]
        if len(filtered_subgroup_ids) == 0:
            continue
        selected_group_id = min(filtered_subgroup_ids, key = lambda x: group_fitnesses[x])
        selected_group_ids.add(selected_group_id)
        selected_group_ids_list.append(selected_group_id)

    selected_groups = []
    selected_group_remap = {}
    for i in selected_group_ids_list:
        pp = preserved_points[i]
        selected_group_remap[i] = len(selected_groups)
        selected_groups.append(pp)
    selected_spanned_groups = {}
    for group_id, excluded_groups in excluded_breed_groups.items():
        if group_id in selected_group_ids:
            new_excluded_groups = set(selected_group_remap[eg] for eg in excluded_groups if eg in selected_group_ids)
            if len(new_excluded_groups) > 0:
                selected_spanned_groups[selected_group_remap[group_id]] = new_excluded_groups

    selected_ids = [i for p in selected_groups for i in p]
    prog_id_remap = {old_id:new_id for new_id, old_id in enumerate(selected_ids)}
    selected_groups_remapped = [[prog_id_remap[i] for i in p] for p in selected_groups]
    # ind_groups = {prog_id: group_id for group_id, prog_ids in enumerate(preserved_points) for prog_id in prog_ids}

    runtime_context.nondominant.groups = selected_groups_remapped
    # nondominant.ind_groups = ind_groups
    runtime_context.nondominant.spanned_groups = selected_spanned_groups
    runtime_context.nondominant.selected_group_id = None
    runtime_context.nondominant.group_interactions = group_interactions[selected_group_ids_list]
    runtime_context.nondominant.group_fitnesses = group_fitnesses[selected_group_ids_list][:, None]

    selected_programs = [programs[i] for i in selected_ids]
    selected_tests = test_selection_strategy(tests, runtime_context.nondominant.group_interactions, runtime_context = runtime_context)
    return [selected_programs, selected_tests]

def find_discriminative_tests(interactions: np.ndarray):
    ''' interactions are underlying objectives (rows) vs tests (columns) '''
    uo_pairs = [ (i, j) for i in range(interactions.shape[0]) for j in range(i + 1, interactions.shape[0]) ]
    discriminating_lists = {}
    for (uo1, uo2) in uo_pairs:
        uo_ints1 = interactions[uo1]
        uo_ints2 = interactions[uo2]
        tids = np.where(uo_ints1 != uo_ints2)[0]
        for tid in tids:
            if uo_ints1[tid] < uo_ints2[tid]:
                discriminating_lists.setdefault(tid, set()).add((uo1, uo2))
            else:
                discriminating_lists.setdefault(tid, set()).add((uo2, uo1))
    discriminating_set = set()
    while len(discriminating_lists) > 0: 
        max_tid = max(discriminating_lists.keys(), key = lambda x: len(discriminating_lists[x]))
        max_tid_set = discriminating_lists[max_tid]
        discriminating_set.add(max_tid)
        del discriminating_lists[max_tid]
        to_delete = set()
        for tid, tid_set in discriminating_lists.items():
            tid_set -= max_tid_set
            if len(tid_set) == 0:
                to_delete.add(tid)
        for tid in to_delete:
            del discriminating_lists[tid]
    return discriminating_set

def select_dicriminating_then_hardest_tests(uo_interactions: np.ndarray, all_interactions: np.ndarray, fraction = 0.5, selection_size = 10, *, runtime_context: RuntimeContext):
    fraction_size = max(1, int(fraction * all_interactions.shape[1]))
    min_size = round(min(fraction_size, selection_size))
    discriminating_tests = find_discriminative_tests(uo_interactions)
    min_size -= len(discriminating_tests)
    if min_size < 0:
        runtime_context.stats.setdefault('obj_size', []).append(len(discriminating_tests))
        return list(discriminating_tests)
    other_test_ids = [tid for tid in range(all_interactions.shape[1]) if tid not in discriminating_tests]
    left_interactions = all_interactions[:, other_test_ids]
    test_stats = np.sum(left_interactions, axis=0)
    rand_test_vals = default_rnd.rand(len(test_stats))
    sorted_test_ids = sorted(enumerate(test_stats), key=lambda x: (x[1], rand_test_vals[x[0]]))
    filtered_test_ids = [i for i, c in sorted_test_ids if c > 0]
    res = filtered_test_ids[:min_size]
    runtime_context.stats.setdefault('obj_size', []).append(min_size + len(discriminating_tests))
    discriminating_tests.update(res)
    return list(discriminating_tests)

def update_cand_underlying_objectives_discr_tests(populations, interactions, fraction = 0.5, selection_size = 10, *, runtime_context: CoevolRuntimeContext):
    """ See update_cand_underlying_objectives,
        As a test selection strategy, we select tests that discriminate extracted underlying objectives
    """
    programs, tests = populations

    selected_interactions = interactions[:, tests]

    origin_zero = np.packbits(np.zeros_like(selected_interactions[0], dtype = bool))
    packed_ints = np.packbits(selected_interactions.astype(bool), axis=1)
    start_ms = time.process_time()
    dims, _, spanned_points = extract_dims_np_b(packed_ints, origin_outcomes=origin_zero) # program underlying objectives
    end_ms = time.process_time()
    dims_time = end_ms - start_ms
    runtime_context.stats.setdefault('dims_time', []).append(dims_time)
    # print(f"Extract dims time: {dims_time}")
    # pass 
    # dims0, _, spanned0 = extract_dims([list(t) for t in selected_interactions])
    # pass
    runtime_context.stats.setdefault('dims', []).append(len(dims))
    dim_points = { (dim_id, len(d) - 1): list(d[-1]) for dim_id, d in enumerate(dims) }
    dim_points_coords = frozenset(dim_points.keys())
    spanned_groups = {} 
    preserved_points = [dedupl_prog_ids(prog_ids, programs) for (_, _), prog_ids in sorted(dim_points.items(), key = lambda x: x[0][0])]
    uo_interactions = interactions[[p[0] for p in preserved_points]]
    span_id_to_covered_dims = {}
    for span_coords, prog_ids in spanned_points.items():
        span_dims = frozenset.intersection(span_coords, dim_points_coords)
        if len(span_dims) == 0:
            continue
        span_id = len(preserved_points)
        preserved_points.append(dedupl_prog_ids(prog_ids, programs))
        span_id_to_covered_dims[span_id] = span_dims
        spanned_groups[span_id] = set(dim_id for (dim_id, _) in span_dims)
    for span_id1, span_dims1 in span_id_to_covered_dims.items():
        for span_id2, span_dims2 in span_id_to_covered_dims.items():
            if (span_id1 != span_id2) and (span_dims1.issubset(span_dims2) or span_dims2.issubset(span_dims1)):
                spanned_groups[span_id1].add(span_id2)
    runtime_context.stats.setdefault('spanned', []).append(len(spanned_groups))
    group_interactions = interactions[[p[0] for p in preserved_points]]

    new_tests = select_dicriminating_then_hardest_tests(uo_interactions, group_interactions, 
                                                        fraction=fraction, selection_size=selection_size, runtime_context=runtime_context)

    group_fitnesses = len(runtime_context.gold_outputs) - np.sum(group_interactions, axis=1)
    selected_group_ids = set()
    selected_group_ids_list = []
    for test_id in new_tests:
        subgroup_ids = np.where(group_interactions[:, test_id] == 1)[0] # group solves tests
        filtered_subgroup_ids = [i for i in subgroup_ids if i not in selected_group_ids]
        if len(filtered_subgroup_ids) == 0:
            continue
        selected_group_id = min(filtered_subgroup_ids, key = lambda x: group_fitnesses[x])
        selected_group_ids.add(selected_group_id)
        selected_group_ids_list.append(selected_group_id)

    selected_groups = []
    selected_group_remap = {}
    for i in selected_group_ids_list:
        pp = preserved_points[i]
        selected_group_remap[i] = len(selected_groups)
        selected_groups.append(pp)
    selected_spanned_groups = {}
    for group_id, excluded_groups in spanned_groups.items():
        if group_id in selected_group_ids:
            new_excluded_groups = set(selected_group_remap[eg] for eg in excluded_groups if eg in selected_group_ids)
            if len(new_excluded_groups) > 0:
                selected_spanned_groups[selected_group_remap[group_id]] = new_excluded_groups

    selected_ids = [i for p in selected_groups for i in p]
    prog_id_remap = {old_id:new_id for new_id, old_id in enumerate(selected_ids)}
    selected_groups_remapped = [[prog_id_remap[i] for i in p] for p in selected_groups]
    # ind_groups = {prog_id: group_id for group_id, prog_ids in enumerate(preserved_points) for prog_id in prog_ids}

    runtime_context.nondominant.groups = selected_groups_remapped
    # nondominant.ind_groups = ind_groups
    runtime_context.nondominant.spanned_groups = selected_spanned_groups
    runtime_context.nondominant.selected_group_id = None
    runtime_context.nondominant.group_interactions = group_interactions[selected_group_ids_list]
    runtime_context.nondominant.group_fitnesses = group_fitnesses[selected_group_ids_list][:, None]

    selected_programs = [programs[i] for i in selected_ids]
    return [selected_programs, new_tests]

def gp_coevolve2(problem_init, *,
                 population_sizes = [0, 0], max_gens = 200,
                fitness_fns = [hamming_distance_fitness, depth_fitness], main_fitness_fn = None,
                select_fitness_ids = None, init_fn = zero_init, map_fn = identity_map, 
                first_select_fn = select_explore_exploit_programs,
                second_select_fn = select_explore_exploit_tests,
                interract_fn = prog_test_interractions,
                update_fn = update_cand_underlying_objectives,
                analyze_pop_fn = analyze_population):
    runtime_context = create_runtime_context(fitness_fns, main_fitness_fn, select_fitness_ids, context_class=CoevolRuntimeContext)
    problem_init(runtime_context = runtime_context)
    evo_funcs = [init_fn, map_fn, first_select_fn, second_select_fn, interract_fn, update_fn, analyze_pop_fn]
    init_fn, map_fn, first_select_fn, second_select_fn, interract_fn, update_fn, analyze_pop_fn = [partial(fn, runtime_context = runtime_context) for fn in evo_funcs]
    best_ind, gen = coevol_loop(max_gens, population_sizes, init_fn, [ map_fn, identity_map ], [first_select_fn, second_select_fn], interract_fn, update_fn, analyze_pop_fn)
    runtime_context.stats['gen'] = gen
    runtime_context.stats["best_found"] = best_ind is not None
    return best_ind, runtime_context.stats

def update_cand_uo_builder(frac_or_size = 10, main_fn = update_cand_underlying_objectives):
    return partial(main_fn, 
                   test_selection_strategy = partial(select_hardest_tests, fraction = frac_or_size / 100.0, selection_size = frac_or_size))

def update_cand_uo_annel_builder(min_frac_or_size = 10, max_frac_or_size = 50, anneal_time = 100, main_fn = update_cand_underlying_objectives):
    return partial(main_fn, 
                   test_selection_strategy = partial(select_hardest_tests_anneal, anneal_time = anneal_time,
                                                     min_fraction = min_frac_or_size / 100.0, min_selection_size = min_frac_or_size,
                                                     max_fraction = max_frac_or_size / 100.0, max_selection_size = max_frac_or_size))

def update_cand_uo_discr_builder(frac_or_size = 10):
    return partial(update_cand_underlying_objectives_discr_tests, fraction = frac_or_size / 100.0, selection_size = frac_or_size)

coevol_uo_10 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(10))
coevol_uo_20 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(20))
coevol_uo_30 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(30))
coevol_uo_40 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(40))
coevol_uo_50 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(50))
coevol_uo_60 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(60))
coevol_uo_70 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(70))
coevol_uo_80 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(80))
coevol_uo_90 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(90))
coevol_uo_100 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(100))

# coevol_uo_d_10 = partial(gp_coevolve2, update_fn = update_cand_uo_discr_builder(10))
# coevol_uo_d_20 = partial(gp_coevolve2, update_fn = update_cand_uo_discr_builder(20))
coevol_uo_d_30 = partial(gp_coevolve2, update_fn = update_cand_uo_discr_builder(30))
# coevol_uo_d_40 = partial(gp_coevolve2, update_fn = update_cand_uo_discr_builder(40))
# coevol_uo_d_50 = partial(gp_coevolve2, update_fn = update_cand_uo_discr_builder(50))
# coevol_uo_d_60 = partial(gp_coevolve2, update_fn = update_cand_uo_discr_builder(60))
# coevol_uo_d_70 = partial(gp_coevolve2, update_fn = update_cand_uo_discr_builder(70))
# coevol_uo_d_80 = partial(gp_coevolve2, update_fn = update_cand_uo_discr_builder(80))
# coevol_uo_d_90 = partial(gp_coevolve2, update_fn = update_cand_uo_discr_builder(90))
# coevol_uo_d_100 = partial(gp_coevolve2, update_fn = update_cand_uo_discr_builder(100))

coevol_uo_50_10 = partial(gp_coevolve2, update_fn = update_cand_uo_annel_builder(10, 50, anneal_time=90))


# coevol_uo2_10 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(10, main_fn=update_cand_underlying_objectives2))
# coevol_uo2_20 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(20, main_fn=update_cand_underlying_objectives2))
# coevol_uo2_30 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(30, main_fn=update_cand_underlying_objectives2))
# coevol_uo2_40 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(40, main_fn=update_cand_underlying_objectives2))
coevol_uo2_50 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(50, main_fn=update_cand_underlying_objectives2))
# coevol_uo2_60 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(60, main_fn=update_cand_underlying_objectives2))
# coevol_uo2_70 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(70, main_fn=update_cand_underlying_objectives2))
# coevol_uo2_80 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(80, main_fn=update_cand_underlying_objectives2))
# coevol_uo2_90 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(90, main_fn=update_cand_underlying_objectives2))
# coevol_uo2_100 = partial(gp_coevolve2, update_fn = update_cand_uo_builder(100, main_fn=update_cand_underlying_objectives2))


coevol_sim_names = ["coevol_uo_50_10", "coevol_uo_10", "coevol_uo_20", "coevol_uo_30", "coevol_uo_40", "coevol_uo_50", "coevol_uo_60", "coevol_uo_70", "coevol_uo_80", "coevol_uo_90", "coevol_uo_100"]
# coevol_sim_names = ["coevol_uo_d_10", "coevol_uo_d_20", "coevol_uo_d_30", "coevol_uo_d_40", "coevol_uo_d_50", "coevol_uo_d_60", "coevol_uo_d_70", "coevol_uo_d_80", "coevol_uo_d_90", "coevol_uo_d_100"]
# coevol_sim_names2 = ["coevol_uo2_10", "coevol_uo2_20", "coevol_uo2_30", "coevol_uo2_40", "coevol_uo2_50", "coevol_uo2_60", "coevol_uo2_70", "coevol_uo2_80", "coevol_uo2_90", "coevol_uo2_100"]
# coevol_sim_names = [*coevol_sim_names1, *coevol_sim_names2]
# coevol_sim_names = ["coevol_uo_40", "coevol_uo2_50"]

if __name__ == '__main__':
    import gp_benchmarks
    problem_builder = gp_benchmarks.get_benchmark('cmp8')
    best_prog, stats = coevol_uo_40(problem_builder)
    print(best_prog)
    print(stats)
    pass    