''' Dimension extraction algorithms. 
    
    Thanks to Dr. Paul Wiegand 
    https://github.com/cereal-lab/EvoPIE/blob/dashboard-integration/evopie/datadashboard/analysislayer/deca.py

    The algorithm is modified in next manner:
       1. Spanned point is the points that can combined 2 or MORE axes (in contract to only 2 in original article)  
       2. Origin, spanned and duplicates are collected separatelly 
             TODO: they should be extanded with additional info: spanned - what [(axis_id, point_id)...] where combined
                                                                 duplicates - what test is already represents it and what (axis_id, point_id)
    As with original algo, there could be situation when new test can extend several axes - in this case first axis is selected
        - this is what makes the algo depending on order of tests, probably sorting at start should fixate the order more strictly
       t1 t2 t3
    c1 0   1  0
    c2 0   0  1
    c3 1   0  0
    c4 1   1  1
'''

from typing import Any, Optional

import numpy as np

def extract_dims(tests: list[list[int]]):
    ''' Ideas from article (with modifications)

    Bucci, Pollack, & de Jong (2004).  "Automated Extraction
    of Question Structure".  In Proceedings of the 2004 Genetic
    and Evolutionary Computation Conference.  
    
    Each test is list of interaction outcomes with same solutions across tests. 
    NOTE: in tests, we have 0-s for candidate to solve the test and 1-s when test wins
    '''
    origin = [] # tests that are placed at the origin of coordinate system
    spanned = {} # tests that are combinations of tests on axes (union of CFS)
    dimensions = [] # tests on axes 
    # duplicates = [] # same behavior (CFS) tests

    # iteration through tests from those that fail least students to those that fail most
    for test_id, test in sorted(enumerate(tests), key=lambda x: sum(x[1])):
        if all(t == 0 for t in test): #trivial tests 
            origin.append(test_id)
            continue
        test_dims = [] # set of dimensions to which current test could belong
        is_dup = False # is there already same test
        for dim_id, dim in enumerate(dimensions):
            # dim[-1] - last point on axis, dim[-1][1] - CFS of the last point
            if all(t == d for t, d in zip(test, dim[-1][1])): #match of performance of last point
                # duplicates.append(test_id)
                dim[-1][0].append(test_id) #dim[*][0] is set of tests that belongs to this space point
                is_dup = True
                break 
            else: #something unique for this dim
                i = len(dim) - 1 
                # we go from last point on axis to first one
                while i >= 0:
                    # check that current test dominates the point on at least one candidate
                    if all(t >= d for t, d in zip(test, dim[i][1])) and any(t > d for t, d in zip(test, dim[i][1])):
                        test_dims.append((dim, dim_id, i))   #the test dominates the point, so it can be a part of this axis
                        break 
                    i -= 1 # if nondominant - test would not have this dim in test_dims
        if is_dup: #for duplicates noop 
            continue
        # unique point should be created in the space, is this spanned point?
        all_ands = [max(el) for el in zip(*[dim[dim_pos][1] for (dim, dim_id, dim_pos) in test_dims])]
        # all_ands represent the union of CFS of all axes which were (minimally) dominated by the test
        # it is the check that the test belongs to spanned point
        if all_ands == test: #spanned, TODO: add identification of axes which were joined
            spanned[test_id] = {dim_id: dim_pos for (_, dim_id, dim_pos) in test_dims}
            continue 
        elif len(all_ands) == 0: #test is not a union of any axes, new dimension
            dimensions.append([([test_id], test)])
        else: # test can extends one of  the existend dimensions
            at_ends_dims = [dim for dim, dim_id, pos in test_dims if pos == len(dim) - 1]
            if len(at_ends_dims) > 0: #is there dimensions which could be extended at the end?
                dim = min(at_ends_dims, key=lambda dim: (len(dim), len(dim[-1][0])))
                dim.append(([test_id], test)) #at new end of axis
            else: # otherwise, the axis should fork, we cannot allow this, but create new axis instead
                dimensions.append([([test_id], test)])

    dims = [[test_ids for test_ids, _ in dim] for dim in dimensions] #pick only test sets
    # duplicates = [tid for dim in dimensions for test_ids, _ in dim for tid in test_ids[1:]]
    return dims, origin, spanned

def get_test_pos(test: list[Optional[int]], dimensions: list[list[tuple[list[int], list[int]]]]):
    test_pos = []
    dupl_point = None
    # check for position on axes from ends to origin
    for dim_id, dim in enumerate(dimensions):  
        found_dim_pos = False
        is_incompatable = False
        for point_id, point in reversed(list(enumerate(dim))):
            if all(t == d for t, d in zip(test, point[1]) if t is not None):
                dupl_point = (dim_id, point_id)
                found_dim_pos = True
                break
            if all(t >= d for t, d in zip(test, point[1]) if t is not None) and any(t > d for t, d in zip(test, point[1]) if t is not None):
                test_pos.append((dim_id, point_id + 1))  #the test dominates the point, so it can be a part of this axis
                found_dim_pos = True 
                break
            if any(t > d for t, d in zip(test, point[1]) if t is not None) and any(t < d for t, d in zip(test, point[1]) if t is not None):
                is_incompatable = True
                break
        if is_incompatable:
            continue
        if not found_dim_pos and all(t <= d for t, d in zip(test, dim[0][1]) if t is not None) and any(t < d for t, d in zip(test, dim[0][1]) if t is not None):
            test_pos.append((dim_id, 0))
    return test_pos, dupl_point

def get_test_pos_def(test: list[int], dimensions: list[list[tuple[list[int], list[int]]]]):
    test_pos = []
    dupl_point = None
    for dim_id, dim in enumerate(dimensions):  
        found_dim_pos = False
        is_incompatable = False
        for point_id, point in reversed(list(enumerate(dim))):
            if all(t == d for t, d in zip(test, point[1])):
                dupl_point = (dim_id, point_id)
                found_dim_pos = True
                break
            if all(t >= d for t, d in zip(test, point[1])) and any(t > d for t, d in zip(test, point[1])):
                test_pos.append((dim_id, point_id + 1))  #the test dominates the point, so it can be a part of this axis
                found_dim_pos = True 
                break
            if any(t > d for t, d in zip(test, point[1])) and any(t < d for t, d in zip(test, point[1])):
                is_incompatable = True
                break
        if is_incompatable:
            continue
        if not found_dim_pos and all(t <= d for t, d in zip(test, dim[0][1])) and any(t < d for t, d in zip(test, dim[0][1])):
            test_pos.append((dim_id, 0))
    return test_pos, dupl_point

def get_test_prev_pos(test: list[int], dimensions: list[list[tuple[list[int], list[int]]]]):
    test_pos = []
    for dim_id, dim in enumerate(dimensions):  
        for point_id, point in reversed(list(enumerate(dim))):
            if all(t >= d for t, d in zip(test, point[1])) and any(t > d for t, d in zip(test, point[1])):
                test_pos.append((dim_id, point_id))  #the test dominates the point, so it can be a part of this axis
                break
    return test_pos

def is_spanned_pos(test: list[Optional[int]], dimensions: list[list[tuple[list[int], list[int]]]]) -> bool:
    spanned_dims = []
    for dim in dimensions:  
        for point in reversed(dim):
            if all(t >= d for t, d in zip(test, point[1]) if t is not None) and any(t > d for t, d in zip(test, point[1]) if t is not None):
                spanned_dims.append(point[1])
                break
    if len(spanned_dims) > 1:
        all_ands = [max(el) for el in zip(*spanned_dims)]
        is_spanned = all(d == t for d, t in zip(all_ands, test) if t is not None)
        return is_spanned, all_ands
    return False, None

def is_spanned_pos_def(test: list[int], dimensions: list[list[tuple[list[int], list[int]]]]) -> bool:
    spanned_dims = []
    for dim in dimensions:  
        for point in reversed(dim):
            if all(t >= d for t, d in zip(test, point[1])) and any(t > d for t, d in zip(test, point[1])):
                spanned_dims.append(point[1])
                break    
    if len(spanned_dims) > 1:
        all_ands = [max(el) for el in zip(*spanned_dims)]
        is_spanned = all(d == t for d, t in zip(all_ands, test))
        return is_spanned
    return False

def is_spanned_prev_pos(test_prev_pos: list[tuple[int, int]], test: list[int], dimensions: list[list[tuple[list[int], list[int]]]]) -> bool:
    all_ands = [max(el) for el in zip(*[dimensions[dim_id][point_id][1] for (dim_id, point_id) in test_prev_pos])]
    is_spanned = len(all_ands) > 0 and all(d == t for d, t in zip(all_ands, test))
    return is_spanned

def set_unknown_to_const(test: list[Optional[int]], v: int):
    for i in range(len(test)):
        if test[i] is None:
            test[i] = v

def set_unknown_to_vect(test: list[Optional[int]], vect: list[int]):
    for i in range(len(test)):
        if test[i] is None:
            test[i] = vect[i]

def filter_spanned(dimensions, spanned_set: set[Any]):
    # still need collect spanned point that appear after point insertions - another pass through dimensions
    dim_filters = []
    for dim in dimensions:
        dim_spanned_points = []
        for point_id, point in enumerate(dim):
            if point_id == 0:
                continue
            point_prev_pos = get_test_prev_pos(point[1], dimensions)
            is_spanned = is_spanned_prev_pos(point_prev_pos, point[1], dimensions)
            if is_spanned:
                # spanned_dims = {dim_id: next(iter(dimensions[dim_id][point_id][0])) for dim_id, point_id in point_prev_pos}
                spanned_set.update(point[0])
                dim_spanned_points.append(point_id)
        dim_spanned_points.sort(reverse=True)
        dim_filters.append(dim_spanned_points)
    for dim, dim_filtrer in zip(dimensions, dim_filters):
        for point_id in dim_filtrer:
            del dim[point_id]

def extract_dims_fix(tests: list[list[int]]):
    ''' Fix case - see matrix-case.txt '''
    origin = [] # tests that are placed at the origin of coordinate system
    spanned_set = set()
    dimensions = [] # tests on axes     
    
    test_to_insert = []

    one_counts = {}

    for test_id, test in enumerate(tests):
        if all(t == 0 for t in test):
            origin.append(test_id)
        else:
            test_to_insert.append((test_id, test))
            one_counts[test_id] = sum(t for t in test if t is not None)

    while len(test_to_insert) > 0:
        new_dims = []
        expands_at_start = {}
        expands_at_end = {}
        other_expands = {}
        test_number_of_pos = {}
        test_outcomes = {}
        for test_id, test in test_to_insert:
            test_pos, dupl_pos = get_test_pos_def(test, dimensions)
            is_spanned = is_spanned_pos_def(test, dimensions)
            if is_spanned:
                spanned_set.add(test_id)
                continue
            if dupl_pos is not None:                 
                dim_id, point_id = dupl_pos
                dimensions[dim_id][point_id][0].add(test_id)
                continue
            test_outcomes[test_id] = test
            if len(test_pos) == 0:
                new_dims.append(test_id)
            else:
                test_number_of_pos[test_id] = len(test_pos)
                for dim_id, point_id in test_pos:
                    if point_id == 0:
                        expands_at_start.setdefault(test_id, []).append(dim_id)
                    elif point_id == len(dimensions[dim_id]):
                        expands_at_end.setdefault(test_id, []).append(dim_id)
                    else:
                        other_expands.setdefault(test_id, []).append((dim_id, point_id))
        if len(test_outcomes) == 0:
            test_to_insert = []
            continue            
        if len(new_dims) > 0:            
            min_test_id = min(new_dims, key=lambda x: one_counts[x])
            min_test = test_outcomes[min_test_id]
            dimensions.append([(set([min_test_id]), min_test)])
        else:
            tests_sorted_by_number_of_moves = sorted(test_number_of_pos.items(), key=lambda x: x[1])
            _, min_num_of_moves = tests_sorted_by_number_of_moves[0]
            min_group = [test_id for test_id, num_of_moves in tests_sorted_by_number_of_moves if num_of_moves == min_num_of_moves]
            can_expand_later = [test_id for test_id in min_group if test_id in other_expands]
            if len(can_expand_later) > 0:
                can_expand_later_with_score = [(test_id, one_counts[test_id]) for test_id in can_expand_later]
                expansion_poss = [(test_id, dim_id, point_id, (score, len(dimensions[dim_id]) - point_id, len(dimensions[dim_id]))) for test_id, score in can_expand_later_with_score for dim_id, point_id in other_expands[test_id]]
                min_test_id, min_dim_id, min_point_id, _ = min(expansion_poss, key=lambda x: x[-1])
                min_test = test_outcomes[min_test_id]
                dim = dimensions[min_dim_id]
                dimensions[min_dim_id] = [*dim[:min_point_id], (set([min_test_id]), min_test), *dim[min_point_id:]]
            else: #min_group expands only at start
                expand_group = [test_id for test_id in min_group if test_id in expands_at_start]
                if len(expand_group) > 0:
                    min_group_with_score = [(test_id, -one_counts[test_id]) for test_id in expand_group] 
                    expansion_poss = [(test_id, dim_id, (score, len(dimensions[dim_id])) ) for test_id, score in min_group_with_score for dim_id in expands_at_start[test_id]]
                    min_test_id, min_dim_id, _ = min(expansion_poss, key=lambda x: x[-1])
                    min_test = test_outcomes[min_test_id]
                    dim = dimensions[min_dim_id]
                    dimensions[min_dim_id] = [(set([min_test_id]), min_test), *dim]
                else:
                    min_group_with_score = [(test_id, (one_counts[test_id], )) for test_id in min_group] 
                    expansion_poss = [(test_id, dim_id, (*score, len(dimensions[dim_id])) ) for test_id, score in min_group_with_score for dim_id in expands_at_end[test_id]]
                    min_test_id, min_dim_id, _ = min(expansion_poss, key=lambda x: x[-1])
                    min_test = test_outcomes[min_test_id]
                    dimensions[min_dim_id].append((set([min_test_id]), min_test))
        del test_outcomes[min_test_id]
        test_to_insert = list(test_outcomes.items())
                
    # spanned refs contain dim_id and test that represent some point in dim. We need to find point id of the test from dimensions 
    filter_spanned(dimensions, spanned_set)
    spanned = {}
    for test_id in spanned_set:
        test_pos = get_test_prev_pos(tests[test_id], dimensions)
        spanned[test_id] = {dim_id: point_id for dim_id, point_id in test_pos}    

    dims = [[sorted(test_ids) for test_ids, _ in dim] for dim in dimensions] #pick only test sets
    return dims, origin, spanned

def extract_dims_approx(tests: list[list[Optional[int]]]):
    ''' Coordinate System Extraction algorithm (CSE) with approximations.'''
    origin = []
    spanned_set = set()
    dimensions = [] # tests on axes       

    test_to_insert = []

    one_counts = {}
    unknown_counts = {}    

    for test_id, test in enumerate(tests):
        if all(t is None or t == 0 for t in test):
            set_unknown_to_const(test, 0)
            origin.append(test_id)
        else:
            test_to_insert.append((test_id, test))
            one_counts[test_id] = sum(t for t in test if t is not None)
            unknown_counts[test_id] = sum(1 for t in test if t is None)            

    while len(test_to_insert) > 0:
        new_dims = []
        expands_at_start = {}
        expands_at_end = {}
        other_expands = {}
        test_number_of_pos = {}
        test_outcomes = {}
        for test_id, test in test_to_insert:
            test_pos, dupl_pos = get_test_pos(test, dimensions)
            is_spanned, spanned_approx = is_spanned_pos(test, dimensions)
            if is_spanned:
                set_unknown_to_vect(test, spanned_approx)
                spanned_set.add(test_id)
                continue
            if dupl_pos is not None: #is_duplicate                 
                dim_id, point_id = dupl_pos
                dimensions[dim_id][point_id][0].add(test_id)
                approx_values = dimensions[dim_id][point_id][1]
                set_unknown_to_vect(test, approx_values)
                continue
            test_outcomes[test_id] = test
            if len(test_pos) == 0:
                new_dims.append(test_id)
            else:
                test_number_of_pos[test_id] = len(test_pos)
                for dim_id, point_id in test_pos:
                    if point_id == 0:
                        expands_at_start.setdefault(test_id, []).append(dim_id)
                    elif point_id == len(dimensions[dim_id]):
                        expands_at_end.setdefault(test_id, []).append(dim_id)
                    else:
                        other_expands.setdefault(test_id, []).append((dim_id, point_id))
        if len(test_outcomes) == 0:
            test_to_insert = []
            continue                        
        if len(new_dims) > 0:
            min_test_id = min(new_dims, key=lambda x: (unknown_counts[x], one_counts[x]))
            min_test = test_outcomes[min_test_id]
            set_unknown_to_const(min_test, 0)
            dimensions.append([(set([min_test_id]), min_test)])
        else:
            tests_sorted_by_number_of_moves = sorted(test_number_of_pos.items(), key=lambda x: x[1])
            _, min_num_of_moves = tests_sorted_by_number_of_moves[0]
            min_group = [test_id for test_id, num_of_moves in tests_sorted_by_number_of_moves if num_of_moves == min_num_of_moves]
            can_expand_later = [test_id for test_id in min_group if test_id in other_expands]
            if len(can_expand_later) > 0:
                can_expand_later_with_score = [(test_id, (unknown_counts[test_id], one_counts[test_id])) for test_id in can_expand_later]
                expansion_poss = [(test_id, dim_id, point_id, (*score, len(dimensions[dim_id]) - point_id, len(dimensions[dim_id]))) for test_id, score in can_expand_later_with_score for dim_id, point_id in other_expands[test_id]]
                min_test_id, min_dim_id, min_point_id, _ = min(expansion_poss, key=lambda x: x[-1])
                min_test = test_outcomes[min_test_id]
                prev_point = dimensions[min_dim_id][min_point_id - 1]
                set_unknown_to_vect(min_test, prev_point[1])
                dim = dimensions[min_dim_id]
                dimensions[min_dim_id] = [*dim[:min_point_id], (set([min_test_id]), min_test), *dim[min_point_id:]]
            else: #min_group expands only at start or end, we prefer start
                expand_group = [test_id for test_id in min_group if test_id in expands_at_start]
                if len(expand_group) > 0:
                    min_group_with_score = [(test_id, (-one_counts[test_id], unknown_counts[test_id])) for test_id in expand_group] 
                    expansion_poss = [(test_id, dim_id, (*score, len(dimensions[dim_id])) ) for test_id, score in min_group_with_score for dim_id in expands_at_start[test_id]]
                    min_test_id, min_dim_id, _ = min(expansion_poss, key=lambda x: x[-1])
                    min_test = test_outcomes[min_test_id]
                    next_point = dimensions[min_dim_id][0]
                    set_unknown_to_vect(min_test, next_point[1])
                    dim = dimensions[min_dim_id]
                    dimensions[min_dim_id] = [(set([min_test_id]), min_test), *dim]
                else: #expand at end
                    min_group_with_score = [(test_id, (unknown_counts[test_id], one_counts[test_id])) for test_id in min_group] 
                    expansion_poss = [(test_id, dim_id, (*score, sum(-1 if d == o == 1 else 0 for d, o in zip(dimensions[dim_id][-1][1], test_outcomes[test_id])), len(dimensions[dim_id])) ) for test_id, score in min_group_with_score for dim_id in expands_at_end[test_id]]
                    min_test_id, min_dim_id, (unknown_values_count, ones_count, axis_match_score, dim_len) = min(expansion_poss, key=lambda x: x[-1])
                    min_test = test_outcomes[min_test_id]
                    if axis_match_score == 0: #no matching 1s, so better to ignore individual
                        # we know too little about this individual - but it is non-trivial. Create new axis?
                        # origin.append(min_test_id)
                        min_test = test_outcomes[min_test_id]
                        set_unknown_to_const(min_test, 0)
                        dimensions.append([(set([min_test_id]), min_test)])                        
                    else:                        
                        prev_point = dimensions[min_dim_id][-1]
                        set_unknown_to_vect(min_test, prev_point[1])
                        dimensions[min_dim_id].append((set([min_test_id]), min_test))
        del test_outcomes[min_test_id]
        test_to_insert = list(test_outcomes.items())

    filter_spanned(dimensions, spanned_set)
    spanned = {}
    for test_id in spanned_set:
        test_pos = get_test_prev_pos(tests[test_id], dimensions)
        spanned[test_id] = {dim_id: point_id for dim_id, point_id in test_pos}

    dims = [[sorted(test_ids) for test_ids, _ in dim] for dim in dimensions] #pick only test sets
    return dims, origin, spanned # here all unknown values are approximated

def filter_nonb_spanned(dimensions, spanned_set: set[Any]):
    # still need collect spanned point that appear after point insertions - another pass through dimensions
    dim_filters = []
    for dim in dimensions:
        dim_spanned_points = []
        for point_id, point in enumerate(dim):
            if point_id == 0:
                continue
            # point_prev_pos = get_test_nonb_pos(point[1], dimensions)
            is_spanned = is_spanned_nonb_pos(point[1], dimensions)
            if is_spanned:
                # spanned_dims = {dim_id: next(iter(dimensions[dim_id][point_id][0])) for dim_id, point_id in point_prev_pos}
                spanned_set.update(point[0])
                dim_spanned_points.append(point_id)
        dim_spanned_points.sort(reverse=True)
        dim_filters.append(dim_spanned_points)
    for dim, dim_filtrer in zip(dimensions, dim_filters):
        for point_id in dim_filtrer:
            del dim[point_id]

def get_test_nonb_pos(test: list[Optional[float]], dimensions: list[list[tuple[list[int], list[float]]]]):
    test_pos = []
    # dupl_point = None
    # check for position on axes from ends to origin
    for dim_id, dim in enumerate(dimensions):  
        found_dim_pos = False
        is_incompatable = False
        for point_id, point in reversed(list(enumerate(dim))):
            # if all(t == d for t, d in zip(test, point[1]) if t is not None):
            #     dupl_point = (dim_id, point_id)
            #     found_dim_pos = True
            #     break
            if all(t >= d for t, d in zip(test, point[1]) if t is not None):
                test_pos.append((dim_id, point_id + 1))  #the test dominates the point, so it can be a part of this axis
                found_dim_pos = True 
                break
            if any(t > d for t, d in zip(test, point[1]) if t is not None) and any(t < d for t, d in zip(test, point[1]) if t is not None):
                is_incompatable = True
                break
        if is_incompatable:
            continue
        if not found_dim_pos and all(t <= d for t, d in zip(test, dim[0][1]) if t is not None):
            test_pos.append((dim_id, 0))
    return test_pos

def is_spanned_nonb_pos(test: list[Optional[float]], dimensions: list[list[tuple[list[int], list[float]]]]) -> bool:
    spanned_dims = []
    for dim in dimensions:
        for point in reversed(dim):
            if all(t >= d for t, d in zip(test, point[1]) if t is not None):
                spanned_dims.append(point[1])
                break
    if len(spanned_dims) <= 1:
        return False, []
    same_dims = set()
    dims_approx = {}
    for spanned_dim in spanned_dims:
        for i, (t, d) in enumerate(zip(test, spanned_dim)):
            if t is not None and d == t:
                same_dims.add(i)
            elif t is None: 
                dims_approx[i] = d if i not in dims_approx else max(d, dims_approx[i])
    present_dims = [i for i, t in enumerate(test) if t is not None]
    if len(same_dims) == len(present_dims):
        approx = [dims_approx[i] if t is None else t for i, t in enumerate(test)]
        return True, approx
    return False, []

# NOTE: needs to be fixed spanned points -- check _np version
# NOTE: this version does not work correctly
def extract_nonb_dims_approx(tests: list[list[Optional[float]]]):
    ''' Generalizes CSE to nonbinary continuous outcomes '''    
    spanned_set = set()
    dimensions = [] # tests on axes       

    test_to_insert = []

    one_counts = {}
    unknown_counts = {}    

    
    for test_id, test in enumerate(tests):
    #     if all(t is None or t == 0 for t in test):
    #         set_unknown_to_const(test, 0)
    #         origin.append(test_id)
    #     else:
        test_to_insert.append((test_id, test))
        one_counts[test_id] = sum(t for t in test if t is not None)
        unknown_counts[test_id] = sum(1 for t in test if t is None)

    while len(test_to_insert) > 0:
        new_dims = []
        expands_at_start = {}
        expands_at_end = {}
        # dupls = {}
        other_expands = {}
        # test_number_of_pos = {}
        test_outcomes = {}
        test_pos_map = {}
        for test_id, test in test_to_insert:
            test_pos = get_test_nonb_pos(test, dimensions)
            # is_spanned, spanned_approx = is_spanned_nonb_pos(test_pos, test, dimensions)
            # if is_spanned:
            #     set_unknown_to_vect(test, spanned_approx)
            #     spanned_set.add(test_id)
            #     continue
            test_outcomes[test_id] = test
            if len(test_pos) == 0:
                new_dims.append(test_id)
            else:
                # test_number_of_pos[test_id] = len(test_pos)
                test_pos_map[test_id] = test_pos
                for dim_id, point_id in test_pos:
                    if point_id == 0:
                        expands_at_start.setdefault(test_id, []).append(dim_id)
                    elif point_id == len(dimensions[dim_id]):
                        expands_at_end.setdefault(test_id, []).append(dim_id)
                    else:
                        other_expands.setdefault(test_id, []).append((dim_id, point_id))
        # if len(test_outcomes) == 0:
        #     test_to_insert = []
        #     continue                        
        if len(new_dims) > 0:
            min_test_id = min(new_dims, key=lambda x: (unknown_counts[x], one_counts[x]))
            min_test = test_outcomes[min_test_id]
            set_unknown_to_const(min_test, 0)
            dimensions.append([(set([min_test_id]), min_test)])
        else:
            grouped_by_num_of_pos = {}
            for test_id, test_pos in test_pos_map.items():
                grouped_by_num_of_pos.setdefault(len(test_pos), []).append(test_id)
            number_of_moves = sorted(grouped_by_num_of_pos.keys())
            for min_num_of_moves in number_of_moves:
                min_group = grouped_by_num_of_pos[min_num_of_moves]
                new_min_group = []
                if min_num_of_moves > 1: # filter spanned 
                    for test_id in min_group:
                        test_pos = test_pos_map[test_id]
                        is_spanned, spanned_approx = is_spanned_nonb_pos(test_outcomes[test_id], dimensions)
                        if is_spanned:
                            spanned_set.add(test_id)
                            set_unknown_to_vect(test_outcomes[test_id], spanned_approx)
                            del test_outcomes[test_id]
                        else:
                            new_min_group.append(test_id)
                else:
                    new_min_group = min_group
                min_group, new_min_group = new_min_group, []
                # filter duplicates
                for test_id in min_group:
                    test_pos = test_pos_map[test_id]
                    is_dupl = False
                    for dim_id, pos_id in test_pos:
                        if pos_id > 0:
                            if all(t == d for t, d in zip(test_outcomes[test_id], dimensions[dim_id][pos_id - 1][1]) if t is not None):
                                dimensions[dim_id][pos_id - 1][0].add(test_id)
                                set_unknown_to_vect(test_outcomes[test_id], dimensions[dim_id][pos_id - 1][1])
                                is_dupl = True
                                del test_outcomes[test_id]
                                break
                    if not is_dupl:
                        new_min_group.append(test_id)
                min_group = new_min_group
                if len(min_group) > 0:
                    break
            
            if len(min_group) == 0:
                test_to_insert = []
                continue
            can_expand_later = [test_id for test_id in min_group if test_id in other_expands]
            if len(can_expand_later) > 0:
                can_expand_later_with_score = [(test_id, (unknown_counts[test_id], one_counts[test_id])) for test_id in can_expand_later]
                expansion_poss = [(test_id, dim_id, point_id, (*score, len(dimensions[dim_id]) - point_id, len(dimensions[dim_id]))) for test_id, score in can_expand_later_with_score for dim_id, point_id in other_expands[test_id]]
                min_test_id, min_dim_id, min_point_id, _ = min(expansion_poss, key=lambda x: x[-1])
                min_test = test_outcomes[min_test_id]
                prev_point = dimensions[min_dim_id][min_point_id - 1]
                set_unknown_to_vect(min_test, prev_point[1])
                dim = dimensions[min_dim_id]
                dimensions[min_dim_id] = [*dim[:min_point_id], (set([min_test_id]), min_test), *dim[min_point_id:]]
            else: #min_group expands only at start or end, we prefer start
                expand_group = [test_id for test_id in min_group if test_id in expands_at_start]
                if len(expand_group) > 0:
                    min_group_with_score = [(test_id, (-one_counts[test_id], unknown_counts[test_id])) for test_id in expand_group] 
                    expansion_poss = [(test_id, dim_id, (*score, len(dimensions[dim_id])) ) for test_id, score in min_group_with_score for dim_id in expands_at_start[test_id]]
                    min_test_id, min_dim_id, _ = min(expansion_poss, key=lambda x: x[-1])
                    min_test = test_outcomes[min_test_id]
                    next_point = dimensions[min_dim_id][0]
                    set_unknown_to_vect(min_test, next_point[1])
                    dim = dimensions[min_dim_id]
                    dimensions[min_dim_id] = [(set([min_test_id]), min_test), *dim]
                else: #expand at end
                    min_group_with_score = [(test_id, (unknown_counts[test_id], one_counts[test_id])) for test_id in min_group] 
                    expansion_poss = [(test_id, dim_id, (*score, sum(-1 if d == o else 0 for d, o in zip(dimensions[dim_id][-1][1], test_outcomes[test_id])), len(dimensions[dim_id])) ) for test_id, score in min_group_with_score for dim_id in expands_at_end[test_id]]
                    min_test_id, min_dim_id, (unknown_values_count, ones_count, axis_match_score, dim_len) = min(expansion_poss, key=lambda x: x[-1])
                    min_test = test_outcomes[min_test_id]
                    if axis_match_score == 0 and unknown_values_count > 0: #no matching 1s, so better to ignore individual
                        # we know too little about this individual - but it is non-trivial. Create new axis?
                        # origin.append(min_test_id)
                        min_test = test_outcomes[min_test_id]
                        set_unknown_to_const(min_test, 0)
                        dimensions.append([(set([min_test_id]), min_test)])                        
                    else:                        
                        prev_point = dimensions[min_dim_id][-1]
                        set_unknown_to_vect(min_test, prev_point[1])
                        dimensions[min_dim_id].append((set([min_test_id]), min_test))
        del test_outcomes[min_test_id]
        test_to_insert = list(test_outcomes.items())

    # detect origin - origin is a point which has outcomes <= outcomes of starts of fall dims 
    origin_dim_ids = [] # zeroes are tests that dominated by all other
    for dim_id, dim in enumerate(dimensions):
        start = dim[0][1]
        if all(t <= d for dim2 in dimensions for t, d in zip(start, dim2[0][1])):
            origin_dim_ids.append(dim_id)

    origin = []
    for dim_id in sorted(origin_dim_ids, reverse=True):
        origin.extend(dimensions[dim_id][0][0])
        del dimensions[dim_id][0]
        if len(dimensions[dim_id]) == 0:
            del dimensions[dim_id]            

    filter_spanned(dimensions, spanned_set)
    spanned = {}
    for test_id in spanned_set:
        test_pos = get_test_prev_pos(tests[test_id], dimensions)
        spanned[test_id] = {dim_id: point_id for dim_id, point_id in test_pos}

    dims = [[sorted(test_ids) for test_ids, _ in dim] for dim in dimensions] #pick only test sets
    return dims, origin, spanned # here all unknown values are approximated

# def get_pos_np(test_ids, interactions, dimension_outcomes):
#     ext_pos = {tid: [] for tid in test_ids}
#     span_pos = {tid: [] for tid in test_ids}
#     # test_pos_map = {tid: [] for tid in test_ids}
#     for dim_id, dim in enumerate(dimension_outcomes):  
#         idxs_to_find = np.array(test_ids)
#         fints = interactions[idxs_to_find]
#         all_indexes = np.arange(idxs_to_find.shape[0])
#         mask = np.ones_like(all_indexes, dtype=bool)
#         incompat = set()
#         for point_id, point in reversed(list(enumerate(dim))):
#             ints = fints[mask]
#             original_idxs = all_indexes[mask]
#             idx_idx = np.where(np.all(ints >= point, axis=1))[0]
#             idxs = original_idxs[idx_idx]
#             for idx in idxs:
#                 real_idx = idxs_to_find[idx]
#                 if point_id != 0:
#                     span_pos[real_idx].append((dim_id, point_id))
#                 if real_idx not in incompat:
#                     ext_pos[real_idx].append((dim_id, point_id))
#             mask[idxs] = False
#             ints = fints[mask]
#             original_idxs = all_indexes[mask]
#             incompat_idx_idx = np.where(np.any(ints > point, axis=1) & np.any(ints < point, axis = 1))[0]
#             incompat_idxs = original_idxs[incompat_idx_idx]
#             for idx in incompat_idxs:
#                 incompat.add(idxs_to_find[idx])
#             # mask[incompat_idxs] = False
#             if not(np.any(mask)):
#                 break   
#     span_pos = {tid: pos for tid, pos in span_pos.items() if len(pos) > 1}
#     return ext_pos, span_pos 


# def get_extension_pos(test_ids, interactions, dimension_outcomes):
#     test_pos_map = {tid: [] for tid in test_ids}
#     for dim_id, dim in enumerate(dimension_outcomes):  
#         idxs_to_find = np.array(test_ids)
#         fints = interactions[idxs_to_find]
#         all_indexes = np.arange(idxs_to_find.shape[0])
#         mask = np.ones_like(all_indexes, dtype=bool)
#         for point_id, point in reversed(list(enumerate(dim))):
#             ints = fints[mask]
#             original_idxs = all_indexes[mask]
#             idx_idx = np.where(np.all(ints >= point, axis=1))[0]
#             idxs = original_idxs[idx_idx]
#             for idx in idxs:
#                 test_pos_map[idxs_to_find[idx]].append((dim_id, point_id))
#             mask[idxs] = False
#             ints = fints[mask]
#             original_idxs = all_indexes[mask]
#             incompat_idx_idx = np.where(np.any(ints > point, axis=1) & np.any(ints < point, axis = 1))[0]
#             incompat_idxs = original_idxs[incompat_idx_idx]
#             mask[incompat_idxs] = False
#             if not(np.any(mask)):
#                 break   
#     return test_pos_map 

def get_extension_pos(test_interactions, dimension_outcomes):
    test_pos = []
    for dim_id, dim in enumerate(dimension_outcomes):  
        for point_id, point in reversed(list(enumerate(dim))):
            if np.all(test_interactions >= point):
                test_pos.append((dim_id, point_id))
                break
            if np.any(test_interactions > point) & np.any(test_interactions < point):
                break
    return test_pos 

def get_extension_pos_b(interactions: np.ndarray, dimension_outcomes: list[list[np.ndarray]]):
    test_poss = [[] for _ in range(interactions.shape[0])]
    for dim_id, dim in enumerate(dimension_outcomes):  
        active_test_ids = np.arange(interactions.shape[0]) #np.ones(interactions.shape[0], dtype=bool)
        for point_id, point in reversed(list(enumerate(dim))):
            ints = interactions[active_test_ids]
            diffs = ints ^ point
            are_non_dominant = np.any(diffs & point, axis=1) #same cannot be because we took unique first
            incompat = are_non_dominant & np.any(diffs & ints, axis=1)
            dominant = ~are_non_dominant
            compat = ~incompat
            id_ids = np.nonzero(dominant)[0]
            for id_id in id_ids:
                test_poss[active_test_ids[id_id]].append((dim_id, point_id))
            active_test_ids = active_test_ids[are_non_dominant & compat]
            if len(active_test_ids) == 0:
                break
    return test_poss 

# def get_pos_b2(interactions: np.ndarray, dimension_outcomes: np.ndarray):
#     int_view = interactions[:, np.newaxis, np.newaxis, :]
#     dim_view = dimension_outcomes[np.newaxis, :, :, :]
#     point_dominates = np.any((int_view ^ dim_view) & dim_view, axis = -1)
#     int_dominates = np.any((int_view ^ dim_view) & int_view, axis = -1)
#     dominations = int_dominates & ~point_dominates
#     poss = np.argmax(~dominations, axis = -1)
#     ext_poss = np.zeros_like(poss, dtype = bool)
#     # ext_poss[poss == dim_sizes[np.newaxis, :, :]] = True
#     return poss - 1

# D1 = np.array([[[0,0,0,0], [0,1,0,0], [0,1,0,1], [1,1,1,1]], 
#                [[0,0,0,0], [0,0,1,0], [1,1,1,1], [1,1,1,1]], 
#                [[0,0,0,0], [1,0,0,1], [1,1,1,1], [1,1,1,1]]], dtype=bool)

# poss = np.array([[1,1,0,1], [1,0,1,1], [0,0,1,1]], dtype = bool)

# get_pos_b2(poss, D1)

# def is_spanned_np(test_interactions, dimension_outcomes):
#     comb = np.zeros_like(test_interactions)
#     for dim_id, dim in enumerate(dimension_outcomes):  
#         for point_id, point in reversed(list(enumerate(dim))):
#             if np.all(test_interactions >= point):
#                 comb += point
#                 break
#     return np.all(comb >= test_interactions)

def get_span_pos(test_interactions: np.ndarray, dimension_outcomes: list[list[np.ndarray]], delta = 0):
    covered_objs = np.zeros_like(test_interactions, dtype=bool)
    # where_not_origin = ~covered_objs
    spanned_dims = {}
    for dim_id, dim in enumerate(dimension_outcomes):  
        for point_id, point in reversed(list(enumerate(dim))):
            if point_id != 0 and np.all(test_interactions >= point):
                covered_objs[point == test_interactions] = True
                spanned_dims[dim_id] = point_id - delta
                break
    is_spanned = np.all(covered_objs)
    if is_spanned:
        return frozenset(spanned_dims.items())
    return None

def get_span_pos_b(test_interactions: np.ndarray, dimension_outcomes: list[list[np.ndarray]], delta = 0):
    covered_objs = np.zeros_like(test_interactions)
    # where_not_origin = ~covered_objs
    spanned_dims = {}
    for dim_id, dim in enumerate(dimension_outcomes):  
        for point_id, point in reversed(list(enumerate(dim))):
            if point_id == 0:
                continue
            diffs = test_interactions ^ point
            dominant = ~np.any(diffs & point) #same cannot be because we took unique first
            if dominant:
                covered_objs = covered_objs | (point & test_interactions)
                spanned_dims[dim_id] = point_id - delta
                break
    is_spanned = np.all(covered_objs == test_interactions)
    if is_spanned:
        return frozenset(spanned_dims.items())
    return None

def get_span_pos_b2(test_interactions: np.ndarray, dimension_outcomes: list[list[np.ndarray]], delta = 0):
    covered_objs = np.zeros_like(test_interactions)
    # where_not_origin = ~covered_objs
    spanned_dims = {}
    for dim_id, dim in enumerate(dimension_outcomes):  
        for point_id, point in reversed(list(enumerate(dim))):
            if point_id == 0:
                continue
            diffs = test_interactions ^ point
            dominant = ~np.any(diffs & point) #same cannot be because we took unique first
            if dominant:
                covered_objs = covered_objs | (point & test_interactions)
                spanned_dims[dim_id] = point_id - delta
                break
    is_spanned = np.all(covered_objs == test_interactions)
    if is_spanned:
        return frozenset(spanned_dims.items())
    return None

# def get_span_pos(interactions: np.ndarray, dimension_outcomes: list[np.ndarray]):
#     ''' test_interactions - positions to detect in coordinate system 
#                             shape: (n_samples, n_interactions)
#         dimension_outcomes - current coordinate system of shape: list of dims of (n_points, n_interactions)
#     '''
#     covered_objs = np.zeros_like(test_interactions, dtype=bool)
#     covered_objs[test_interactions == 0] = True
#     where_ones = test_interactions == 1
#     spanned_dims = {}
#     for dim_id, dim in enumerate(dimension_outcomes):  
#         for point_id, point in reversed(list(enumerate(dim))):
#             if np.all(test_interactions >= point) and np.any(point == 1):
#                 covered_objs[(point == 1) & where_ones] = True
#                 spanned_dims[dim_id] = point_id
#                 break
#     is_spanned = np.all(covered_objs)
#     if is_spanned:
#         return frozenset(spanned_dims)
#     return None

# x = np.argmax(np.arange(10) > 6)

# span_pos = get_span_pos(np.array([0,1,1]), [[np.array([1,0,0])],[np.array([0,1,0])],[np.array([0,0,1])]])
# # span_pos = get_span_pos(np.array([[0,1,1],[1,1,1],[1,1,0]]), [[np.array([1,0,0])],[np.array([0,1,0])],[np.array([0,0,1])]])
# pass 

def extract_dims_np(tests: np.ndarray, origin_outcomes = None):
    ''' Generalizes CSE to nonbinary continuous outcomes, fully defined outcomes '''
    # origin = np.where(np.all(tests == 0, axis=1))[0]
    if  origin_outcomes is None:
        origin_outcomes = np.min(tests, axis=0)
    origin = np.where(np.all(tests == origin_outcomes, axis=1))[0]
    
    test_to_insert = np.where(np.any(tests != origin_outcomes, axis=1))[0]
    unique_tests, unique_indexes = np.unique(tests[test_to_insert], axis=0, return_index=True) 

    # spanned_set = set()
    # dimensions = [] # tests on axes 
    dimension_outcomes = []
    test_to_insert = np.arange(unique_tests.shape[0])


    one_counts = np.sum(unique_tests, axis=1)    
    
    global_test_pos_map = {}
    tests_to_update = test_to_insert
    while len(test_to_insert) > 0:
        for test_id in tests_to_update:
            test_pos = get_extension_pos(unique_tests[test_id], dimension_outcomes)
            global_test_pos_map[test_id] = test_pos
        test_pos_map = {tid: global_test_pos_map[tid] for tid in test_to_insert }
        # span_pos_map = {tid: pos for tid, pos in global_span_pos_map.items() if tid in test_to_insert}
        # test_pos_map = get_extension_pos(test_to_insert, unique_tests, dimension_outcomes)

        # pos_conflict_counts = {}
        # for poss in test_pos_map.values():
        #     for pos in poss:
        #         pos_conflict_counts[pos] = pos_conflict_counts.get(pos, 0) + 1
        
        # test_conflict_counts = {test_id: 0 if len(test_pos) == 0 else np.mean([pos_conflict_counts[pos] for pos in test_pos]) for test_id, test_pos in test_pos_map.items()}

        spanned_tests = []
        min_test_id = None
        while len(test_pos_map) > 0: 
            min_test_id = next((test_id for test_id, test_pos in test_pos_map.items() if len(test_pos) == 0), None)
            if (min_test_id is not None) and (get_span_pos(unique_tests[min_test_id], dimension_outcomes) is not None):
                spanned_tests.append(min_test_id)
                del test_pos_map[min_test_id]
            else:
                break

        if len(spanned_tests) > 0:
            test_to_insert = test_to_insert[~np.isin(test_to_insert, spanned_tests)]

        if len(test_pos_map) == 0:
            break

        if min_test_id is not None:
            dimension_outcomes.append([origin_outcomes, unique_tests[min_test_id]])
            test_to_insert = test_to_insert[test_to_insert != min_test_id]
            tests_to_update = test_to_insert #.copy()
            continue

        spanned_tests = []
        while len(test_pos_map) > 0:
            min_test_id, min_test_pos = min(test_pos_map.items(), key=lambda x: (len(x[1]), (0 if len(x[1]) == 0 else min(pos_id for _, pos_id in x[1])), one_counts[x[0]]))
            if get_span_pos(unique_tests[min_test_id], dimension_outcomes) is not None:
                spanned_tests.append(min_test_id)
                del test_pos_map[min_test_id]
            else:
                break

        if len(spanned_tests) > 0:
            test_to_insert = test_to_insert[~np.isin(test_to_insert, spanned_tests)]

        if len(test_pos_map) == 0:
            break
                                    
        min_dim_id, min_pos_id = min(min_test_pos, key=lambda x: x[1])
        dim = dimension_outcomes[min_dim_id]
        dimension_outcomes[min_dim_id] = [*dim[:min_pos_id+1], unique_tests[min_test_id], *dim[min_pos_id+1:]]
        test_to_insert = test_to_insert[test_to_insert != min_test_id]
        tests_to_update = [] 
        for test_id in test_to_insert:
            test_pos = test_pos_map[test_id]
            if any(dim_id == min_dim_id for dim_id, point_id in test_pos):
                tests_to_update.append(test_id)
        # tests_to_update = np.array(tests_to_update)

    dims = []    
    unplaced_test_mask = np.ones(len(tests), dtype=bool) #set(range(len(tests)))
    unplaced_test_mask[origin] = False
    # for tid in origin:
    #     unplaced_test_ids.remove(tid)
    for dim in dimension_outcomes:
        one_dim = []
        dims.append(one_dim)
        for point in dim[1:]:
            point_tests = np.where(np.all(tests == point, axis=1))[0]
            one_dim.append(point_tests)
            unplaced_test_mask[point_tests] = False
            # for tid in point_tests:
            #     unplaced_test_ids.remove(tid)

    spanned = {}
    unplaced_test_ids = np.where(unplaced_test_mask)[0]
    for test_id in unplaced_test_ids:
        test_interactions = tests[test_id]
        spanned_pos = get_span_pos(test_interactions, dimension_outcomes, delta = 1) #delta 1, as 0 point is origin
        assert spanned_pos is not None, f'Spanned point has None position for test {test_id}, {test_interactions}'
        spanned.setdefault(spanned_pos, []).append(test_id)
    return dims, origin, spanned # here all unknown values are approximated

# import numpy as np

# zo = np.random.randint(0, 2, size=(10, 32), dtype=bool)
# zo = np.vstack([np.zeros(32, dtype=bool), zo])
# z = np.packbits(zo, axis=1)
# np.unpackbits(z).reshape(zo.shape)
# np.unpackbits(z, axis=1)
# zoo = np.bitwise_and.reduce(z, axis=0)

def extract_dims_np_b(tests: np.ndarray, origin_outcomes = None):
    ''' Version of extract_dims_np on binary interactions 0/1 - bit arrays  '''
    # origin = np.where(np.all(tests == 0, axis=1))[0]
    if  origin_outcomes is None:
        origin_outcomes = np.packbits(np.logical_and.reduce(tests, axis=0))
    origin = np.where(np.all(tests == origin_outcomes, axis=1))[0]
    
    unique_tests = np.unique(tests[np.any(tests != origin_outcomes, axis=1)], axis=0, return_index=False) 

    # spanned_set = set()
    # dimensions = [] # tests on axes 
    dimension_outcomes = []
    test_to_insert = np.arange(unique_tests.shape[0])

    unique_tests_unpacked = np.unpackbits(unique_tests, axis=1)

    one_counts = np.sum(unique_tests_unpacked, axis=1)    
    
    global_test_pos_map = {}
    tests_to_update = test_to_insert
    while len(test_to_insert) > 0:
        test_poss = get_extension_pos_b(unique_tests[tests_to_update], dimension_outcomes)
        for test_id, test_pos in zip(tests_to_update, test_poss):
            global_test_pos_map[test_id] = test_pos
        test_pos_map = {tid: global_test_pos_map[tid] for tid in test_to_insert }
        # span_pos_map = {tid: pos for tid, pos in global_span_pos_map.items() if tid in test_to_insert}
        # test_pos_map = get_extension_pos(test_to_insert, unique_tests, dimension_outcomes)

        # pos_conflict_counts = {}
        # for poss in test_pos_map.values():
        #     for pos in poss:
        #         pos_conflict_counts[pos] = pos_conflict_counts.get(pos, 0) + 1
        
        # test_conflict_counts = {test_id: 0 if len(test_pos) == 0 else np.mean([pos_conflict_counts[pos] for pos in test_pos]) for test_id, test_pos in test_pos_map.items()}

        spanned_tests = []
        min_test_id = None
        while len(test_pos_map) > 0: 
            min_test_id = next((test_id for test_id, test_pos in test_pos_map.items() if len(test_pos) == 0), None)
            if (min_test_id is not None) and (get_span_pos_b(unique_tests[min_test_id], dimension_outcomes) is not None):
                spanned_tests.append(min_test_id)
                del test_pos_map[min_test_id]
            else:
                break

        if len(spanned_tests) > 0:
            test_to_insert = test_to_insert[~np.isin(test_to_insert, spanned_tests)]

        if len(test_pos_map) == 0:
            break

        if min_test_id is not None:
            dimension_outcomes.append([origin_outcomes, unique_tests[min_test_id]])
            test_to_insert = test_to_insert[test_to_insert != min_test_id]
            tests_to_update = test_to_insert #.copy()
            continue

        spanned_tests = []
        while len(test_pos_map) > 0:
            min_test_id, min_test_pos = min(test_pos_map.items(), key=lambda x: (len(x[1]), (0 if len(x[1]) == 0 else min(pos_id for _, pos_id in x[1])), one_counts[x[0]]))
            if get_span_pos_b(unique_tests[min_test_id], dimension_outcomes) is not None:
                spanned_tests.append(min_test_id)
                del test_pos_map[min_test_id]
            else:
                break

        if len(spanned_tests) > 0:
            test_to_insert = test_to_insert[~np.isin(test_to_insert, spanned_tests)]

        if len(test_pos_map) == 0:
            break
                                    
        min_dim_id, min_pos_id = min(min_test_pos, key=lambda x: x[1])
        dim = dimension_outcomes[min_dim_id]
        dimension_outcomes[min_dim_id] = [*dim[:min_pos_id+1], unique_tests[min_test_id], *dim[min_pos_id+1:]]
        test_to_insert = test_to_insert[test_to_insert != min_test_id]
        tests_to_update = [] 
        for test_id in test_to_insert:
            test_pos = test_pos_map[test_id]
            if any(dim_id == min_dim_id for dim_id, point_id in test_pos):
                tests_to_update.append(test_id)
        # tests_to_update = np.array(tests_to_update)

    # assert that all dim points are not spanned???
    # for dim_id, dim in enumerate(dimension_outcomes):
    #     for point_id, point in reversed(list(enumerate(dim))):
    #         if point_id == 0:
    #             continue
    #         new_dims = [*dimension_outcomes[:dim_id], dimension_outcomes[dim_id][:point_id], *dimension_outcomes[dim_id + 1:]]
    #         new_dims = [dim for dim in new_dims if len(dim) > 1]
    #         span_pos = get_span_pos_b(point, new_dims)
    #         if span_pos is not None:
    #             print("Span point in dimension", dim_id, point_id, point, span_pos)

    dims = []    
    unplaced_test_mask = np.ones(len(tests), dtype=bool) #set(range(len(tests)))
    unplaced_test_mask[origin] = False
    # for tid in origin:
    #     unplaced_test_ids.remove(tid)
    for dim in dimension_outcomes:
        one_dim = []
        dims.append(one_dim)
        for point in dim[1:]:
            point_tests = np.where(np.all(tests == point, axis=1))[0]
            one_dim.append(point_tests)
            unplaced_test_mask[point_tests] = False
            # for tid in point_tests:
            #     unplaced_test_ids.remove(tid)

    spanned = {}
    unplaced_test_ids = np.where(unplaced_test_mask)[0]
    for test_id in unplaced_test_ids:
        test_interactions = tests[test_id]
        spanned_pos = get_span_pos_b(test_interactions, dimension_outcomes, delta = 1) #delta 1, as 0 point is origin
        assert spanned_pos is not None, f'Spanned point has None position for test {test_id}, {test_interactions}'
        spanned.setdefault(spanned_pos, []).append(test_id)
    return dims, origin, spanned # here all unknown values are approximated

# TODO: organize archive variables as state accessible to populations - probably encapsulate into Archive instance 
# TODO: DECA as archive. By itself it is ocmputationally heavy operation to compute dimensions on all previous interactions 
#       as result, we resort to dimensions from most resent batch. 
#       But the intermediate approach could also consider archive individuals in addition to batch        
def cosa_extract_archive(tests: list[list[int]]):
    ''' Procedure to extract test basis based on COSA article: 
            
        Wojciech Ja ́skowski and Krzysztof Krawiec. “Coordinate System Archive
        for coevolution”. In: Aug. 2010, pp. 1-10. doi: 10 . 1109 / CEC . 2010 .
        5586066.        

        NOTE 1: As witht other methods, we assume that we do not control candidates population, therefore only form an archive for tests
        NOTE 2: in tests, we have 0-s for candidate to solve the test G(s, t) and 1-s when test wins (candidate fails) not(G(s, t))

    '''
    # first we remove duplicates and leave only "oldest" test with more previous interactions 
    def dedup(tests):
        ''' figures out duplicates by outcomes '''
        tests_dedup = {} 
        dup_map = {}
        for test_id, test in enumerate(tests):
            test_tuple = tuple(test)
            if test_tuple in tests_dedup:
                dup_map[tests_dedup[test_tuple]].add(test_id)
            else: 
                tests_dedup[test_tuple] = test_id
                dup_map[test_id] = set([test_id])
        return dup_map
    test_dups = dedup(tests)
    test_ids = list(test_dups.keys())
    #Pareto candidates - columns of given tests, we also enumerate them 
    candidates = [[tests[tid][i] for tid in test_ids] for i in range(len((tests[0])))]
    cand_dups = dedup(candidates)    
    candidate_ids = list(cand_dups.keys())
    candidate_map = {cid: candidates[cid] for cid in candidate_ids }
    candidates_ids_pareto = set() #computing pareto front 
    candidate_ids_sorted = sorted(candidate_map.keys(), key=lambda x: sum(candidate_map[x]))
    for cid in candidate_ids_sorted: #NOTE: s1 dominates s2 on T if exists t in T where G(s, t) (outcome is 0)
        if cid not in candidate_map:
            continue
        outcomes = candidate_map[cid]
        for cid2, outcomes2 in list(candidate_map.items()):
            if cid2 != cid and all(o1 <= o2 for o1, o2 in zip(outcomes, outcomes2)): #remove dominated or same
                del candidate_map[cid2]
        candidates_ids_pareto.add(cid)
        del candidate_map[cid] #done processing cid 
    #computing T_base - FindTests in the article 
    #   1. split set of tests onto chains 
    #      starting with building domination graph 
    test_dominated_by = {test_id: set() for test_id in test_ids}
    test_dominates = {test_id: set() for test_id in test_ids}
    tests_dedup_ordered = sorted(test_ids, key = lambda tid: sum(tests[tid]))
    for i in range(len(tests_dedup_ordered)):
        test_id = tests_dedup_ordered[i]
        test = tests[test_id]
        for j in range(i+1, len(tests_dedup_ordered)):
            test_id2 = tests_dedup_ordered[j]
            test2 = tests[test_id2]
            if all(o2 >= o1 for o2, o1 in zip(test2, test)):
                test_dominated_by[test_id].add(test_id2)
                test_dominates[test_id2].add(test_id)
    # building chains to each test in domination graph
    # https://www.geeksforgeeks.org/maximum-bipartite-matching/#
    # Max-flow on bipartite graph with unit capacities
    # each time we pick longest chain and add it to chains, then remove all added edges
    # but does it produce minimum partitioning (seems to be greedy)
    # conterexample: a -> c -> f, a -> d -> f, b -> d -> f. Splits [a -> d -> f, c, b] VS [a -> c -> f, b -> d]
    # THIS IS THE PROBLEM HERE ^^^^ - for now we do not use COSA till later
    
    chains = [] # we collect chains here    

    while True:
        sources = [test_id for test_id, dominates in test_dominates.items() if len(dominates) == 0]
        if len(sources) == 0: #no more sources, end of chains
            break 
        all_local_chains = {} # chains till test_id which is key in this dict
        for test_id in sources:
            all_local_chains[test_id] = [[test_id]]
        final_chains = {}
        while len(all_local_chains) > 0: #we try to increase each chain in all_local_chains by one, if no such chains - break
            new_local_chains = {}
            for test_id, test_chains in all_local_chains.items():
                for next_test_id in test_dominated_by[test_id]:
                    new_local_chains.setdefault(next_test_id, []).extend([[*chain, next_test_id] for chain in test_chains])
                final_chains.setdefault(test_id, []).extend(test_chains)
            all_local_chains = new_local_chains
        # we pick the chain with minimal number of branching of its nodes
        dests = [test_id for test_id, dominated_by in test_dominated_by.items() if len(dominated_by) == 0]
        dest_chains = [chain for d in dests for chain in final_chains[d]]
        dest_chains.sort(key = lambda chain: (-len(chain), sum(len(final_chains[test_id]) for test_id in chain)))
        selected_chain = dest_chains[0]
        chains.append(selected_chain)
        for test_id in selected_chain:
            del test_dominates[test_id]
            for dominates in test_dominates.values():
                dominates.discard(test_id)
            del test_dominated_by[test_id]
            for dominated_by in test_dominated_by.values():
                dominated_by.discard(test_id)            
    # now we have chains, we continue FindTests from article 
    n_dims = len(chains)
    def get_pos(candidate_id, chains):
        res = [len([test_id for test_id in c if tests[test_id][candidate_id] == 0]) for c in chains] #0 in outcome - candidate solves the test
        return res 
    candidates_ids_pareto_sorted = list(candidates_ids_pareto)
    candidates_ids_pareto_sorted.sort(key = lambda x: min(get_pos(x, chains)), reverse=True)
    test_ids_base = list(test_ids)
    for candidate_id in candidates_ids_pareto_sorted:
        filtered_tests = {test_id for test_id in test_ids if tests[test_id][candidate_id] == 0 } #candidate solves the test
        #finding greatest antichain - maximal set of incomparable (Pareto non-dominant) tests in filtered_tests
        #for this we can use chains filtered by filtered_tests and get ends of each chain/axis
        filtered_chains = [[test_id for test_id in chain if test_id in filtered_tests] for chain in chains]
        antichain_ids = [chain[-1] for chain in filtered_chains if len(chain) > 0]
        if n_dims == len(antichain_ids): #predicate "found" in the article - 3 criterias satisfied
            test_ids_base = antichain_ids
            break 
    # end of FindTests from the article     
    def pairSetCover(A_must, A, B, get_outcome):
        A_diff = set.difference(A, A_must)
        B_list = list(B)
        N_map = {} #map of canddiate_id to pairs it distinguish
        # for each element in candidate_ids_diff figure out distinguishible pairs from test_base_ids
        for i in range(len(B_list)):
            for j in range(i+1, len(B_list)):
                b1, b2 = (B_list[i], B_list[j])
                if any(get_outcome(a, b1) != get_outcome(a, b2) for a in A_must):
                    # the pair is already distinguishible
                    continue
                for a in A_diff:
                    if get_outcome(a, b1) != get_outcome(a, b2):
                        N_map.setdefault(a, set()).add((b1, b2))
        V_res = set(A_must)
        while True: 
            u, pairs = next((el for el in sorted(N_map.items(), key = lambda x: len(x[1]), reverse=True)), (None, None))
            if u is None: 
                break 
            V_res.add(u)
            for a, other_pairs in list(N_map.items()):
                other_pairs.difference_update(pairs)
                if len(other_pairs) == 0:
                    del N_map[a]
        return V_res
    # 2. PairSetCover - extend pareto set of candidates to distinguish all test_base
    # S_arch = pairSetCover(candidates_ids_pareto, set(candidate_ids), test_ids_base, 
    #             lambda candidate_id, test_id: tests[test_id][candidate_id])
    # 3. PairSetCover - extends test_base to distinguish all pareto set of candidates
    T_arch = pairSetCover(test_ids_base, set(test_ids), candidates_ids_pareto, 
                lambda test_id, candidate_id: tests[test_id][candidate_id])
    # we do not need candidates archive
    res = [tid2 for tid in T_arch for tid2 in test_dups[tid]]
    return res

def get_batch_pareto_layers(tests: list[list[int]], max_layers = 1):
    ''' Ideas taken from IPCA and LAPCA methods

        E. D. de Jong, “The Incremental Pareto-Coevolution Archive,” in
        Genetic and Evolutionary Computation - GECCO 2004. Proceedings of
        the Genetic and Evolutionary Computation Conference. Part I, K. D.
        et al., Ed. Seattle, Washington, USA: Springer-Verlag, Lecture Notes
        in Computer Science Vol. 3102, Jun. 2004, pp. 525-536.

        E. De Jong, “Towards a bounded Pareto-Coevolution archive,” in
        Proceedings of the Congress on Evolutionary Computation CEC-04,
        vol. 2. Portland, Oregon, USA: IEEE Service Center, Jun. 2004, pp.
        2341-2348.        

        NOTE 1: Because we do not control second population, we just resort to computation of pareto layers of tests

        We maintain the archive of tests.
        New test is added to the archive if it is non-dominated with respect to all tests already in the archive.
        Found layer then removed from tests.

        NOTE 2: We also just process last batch, and ignore whole interaction matrix. Otherwise, execution could be too computation-heavy
        NOTE 3: we filter out all trivial 0-s
    '''
    layers = []
    layer_num = 0 
    test_map = {test_id:test for test_id, test in enumerate(tests)}
    for test_id in list(test_map.keys()):
        if all(o == 0 for o in test_map[test_id]):
            del test_map[test_id]
    while layer_num < max_layers and len(test_map) > 0:  
        tests_archive = {} #map of test_id to set of semantically same tests 
        for test_id, test in test_map.items():
            dominates = set() 
            dominated_by = set()
            same_to = None
            for arch_id in tests_archive.keys():
                arch = test_map[arch_id]
                has_any_in_test = any(o1 > o2 for o1, o2 in zip(test, arch))
                has_any_in_arch = any(o1 > o2 for o1, o2 in zip(arch, test))
                if has_any_in_test and has_any_in_arch: # non-dominant 
                    pass 
                elif has_any_in_test and not has_any_in_arch: # remove from arch
                    dominates.add(arch_id)
                elif not has_any_in_test and has_any_in_arch: # weaker test 
                    dominated_by.add(arch_id)
                else: # same 
                    same_to = arch_id 
                    break 
            if same_to is not None:
                tests_archive[same_to].add(test_id)
            elif len(dominated_by) == 0:
                for arch_id in dominates:
                    del tests_archive[arch_id]
                tests_archive[test_id] = set([test_id])
            else: # dominated by archive - noop
                pass
        layers.append([test_id for test_ids in tests_archive.values() for test_id in test_ids])
        #filter out processed tests:
        for test_ids in tests_archive.values():
            for test_id in test_ids:
                del test_map[test_id]
        layer_num += 1

    return layers 

def get_batch_pareto_layers2(tests: list[list[Optional[int]]], max_layers = 1, discard_spanned = False):
    ''' Works with sparsed matrices '''
    layers = []
    layer_num = 0 
    test_map = {}
    discarded = set()
    duplicates = { }
    for test_id, test in enumerate(tests):
        if all(o == 0 for o in test if o is not None):
            discarded.add(test_id)
            continue
        dupl_of_test = None
        for test_id2 in range(test_id + 1, len(tests)):
            t2 = tests[test_id2]
            if all(o1 == o2 for o1, o2 in zip(test, t2) if o1 is not None and o2 is not None): #duplicate 
                dupl_of_test = test_id2 
                break 
        if dupl_of_test is not None:
            duplicates.setdefault(dupl_of_test, set()).add(test_id)
            continue
        if discard_spanned: # do spanned test
            dominated_tests = [t2 for t2 in tests if all(o1 >= o2 for o1, o2 in zip(test, t2) if o1 is not None and o2 is not None) and any(o1 > o2 for o1, o2 in zip(test, t2) if o1 is not None and o2 is not None)]
            and_all = [1 if any(o == 1 for o in el) else None if any(o is None for o in el) else 0 for el in zip(*dominated_tests)]
            if len(and_all) > 0 and all(o1 == o2 for o1, o2 in zip(test, and_all) if o1 is not None and o2 is not None):
                discarded.add(test_id)
                continue 
        test_map[test_id] = test
    while layer_num < max_layers and len(test_map) > 0: 
        tests_archive = {} #map of test_id to set of semantically same tests 
        for test_id, test in test_map.items():
            dominates = set() 
            dominated_by = set()
            same_to = None
            for arch_id in tests_archive.keys():
                arch = test_map[arch_id]
                has_any_in_test = any(o1 > o2 for o1, o2 in zip(test, arch) if o1 is not None and o2 is not None)
                has_any_in_arch = any(o1 > o2 for o1, o2 in zip(arch, test) if o1 is not None and o2 is not None)
                if has_any_in_test and has_any_in_arch: # non-dominant 
                    pass 
                elif has_any_in_test and not has_any_in_arch: # remove from arch
                    dominates.add(arch_id)
                elif not has_any_in_test and has_any_in_arch: # weaker test 
                    dominated_by.add(arch_id)
                else: # same 
                    same_to = arch_id 
                    break 
            if same_to is not None:
                tests_archive[same_to].add(test_id)
            elif len(dominated_by) == 0:
                for arch_id in dominates:
                    del tests_archive[arch_id]
                tests_archive[test_id] = set([test_id])
            else: # dominated by archive - noop
                pass
        layers.append([test_id for test_ids in tests_archive.values() for test_id in test_ids])
        #filter out processed tests:
        for test_ids in tests_archive.values():
            for test_id in test_ids:
                del test_map[test_id]
        layer_num += 1

    layers = [[el2 for el in layer for el2 in [el, *duplicates.get(el, [])]] for layer in layers]
    return layers, discarded

def test_dim_counts(num_cand: int, start_n: int, end_n: int, cnt = 1, seed: int = 119):
    ''' Computes stats for different methods of num of dimensions  '''
    stats = []
    rnd = np.random.RandomState(seed)
    for n in range(start_n, end_n + 1):
        for i in range(cnt):
            interactions = rnd.randint(0, 2, size=(num_cand, n), dtype=int)

            dims1, o1, s1 = extract_dims(interactions.tolist())
            dims2, o2, s2 = extract_dims_fix(interactions)
            # dims3, o3, s3 = extract_dims_approx(tests)
            # dims4, o4, s4 = extract_nonb_dims_approx(tests)
            z0 = np.packbits(np.zeros_like(interactions[0], dtype=bool))
            t0 = np.packbits(np.array(interactions), axis=1)
            dims5, o5, s5 = extract_dims_np_b(t0, origin_outcomes=z0)     
            if n < len(dims1):
                stats.append(dict(n = n, de_dim_count = len(dims1), cse_fix_count = len(dims2), cse_b_count = len(dims5),
                                    ints = interactions))
            pass
    return stats




if __name__ == "__main__":
    # stats = test_dim_counts(10, 6, 6, cnt=100, seed=119)
    # print(stats)
    # pass 

    tests = np.array([[0, 1, 0, 1, 0, 1],
       [1, 1, 1, 1, 0, 0],
       [1, 0, 1, 0, 1, 1],
    #    [1, 1, 1, 1, 0, 0],
       [1, 0, 0, 1, 0, 0],
       [1, 0, 1, 1, 1, 0],
       [0, 0, 1, 1, 0, 1],
    #    [0, 1, 0, 1, 1, 1],
       [1, 1, 0, 0, 0, 1],
       [0, 0, 0, 1, 1, 1]])

    tests = np.array([[1, 1, 0, 0],
                    [1, 0, 1, 0],
                    [1, 0, 0, 1],
                    [0, 1, 1, 0],
                    [0, 1, 0, 1]
                    ])
    
    dom_rel = []
    for i in range(len(tests)):
        iv = tests[i]
        for j in range(len(tests)):
            jv = tests[j]
            if np.all(iv >= jv) and np.any(iv > jv):
                dom_rel.append((i, j))
    print(dom_rel)
    
    dims2, o2, s2 = extract_dims_fix(tests)    
    vects = [[tests[p[0]] for p in d] for d in dims2]
    pass

    tests = np.array([
        [9,1,9],
        [9,2,9],
        [1,2,2],
        [1,2,3],
        [2,9,1],
        [3,9,2],
        [9,9,9],
        [1,1,1],
        [1,2,2],
        [9,2,3],
    ])
    res = extract_dims_np(tests)
    # dims4, orig4, span4 = extract_nonb_dims_approx(tests)
    pass
    #testing of de module functions 
    tests = [
        [0,0,0],
        [0,0,1],
        [1,0,0],
        [0,1,0],
        [0,1,1],
        [1,1,0],
        [1,0,1],
        [1,1,1],
    ]
    dims3, orig3, span3 = extract_dims_approx(tests)
    dims4, orig4, span4 = extract_nonb_dims_approx(tests)
    tests_np = np.array(tests, dtype=bool)
    dims5, orig5, span5 = extract_dims_np_b(np.packbits(tests_np, axis=1))
    pass

    tests = [[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
[1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
[1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
[1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
[0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
[1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1],
[1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
[0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1],
[1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1],
[0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
[1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]] 

    # tests = np.fromfile("test1.bin", dtype=np.int64).reshape((166,10))
    # kx = [30, 84, 119, 133, 137, 159, 149, 34, 165, 75, 106, 85, 161, 36, 62, 67, 153, 154, 10, 53, 69, 82, 157, 18, 39, 98, 92, 51, 160, 63, 26, 56, 131, 13, 31]
    # tests = tests[kx]
    # tests = [list(t) for t in tests]

    dims1, o1, s1 = extract_dims(tests)
    dims2, o2, s2 = extract_dims_fix(tests)
    dims3, o3, s3 = extract_dims_approx(tests)
    dims4, o4, s4 = extract_nonb_dims_approx(tests)
    z0 = np.packbits(np.zeros_like(tests[0], dtype=bool))
    t0 = np.packbits(np.array(tests), axis=1)
    dims5, o5, s5 = extract_dims_np_b(t0, origin_outcomes=z0)
    pass
    test2 = [[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 1, None, None, None, None, None, None, 0, 0, 0],
        [1, 1, None, None, None, None, None, None, 1, 1, 1],
        [1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
        [0, 1, None, None, None, None, None, None, 1, 1, 0],
        [1, 1, None, None, None, None, None, None, 1, 1, 1],
        [1, 1, None, 1, None, None, None, 0, 1, 1, 1],
        [0, 1, None, None, None, None, None, None, 1, 1, 0],
        [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
        [0, 1, None, None, None, None, None, None, 1, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0],
        [1, 1, None, None, None, None, None, None, 1, 1, 0],
        [0, 1, None, None, None, None, None, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        [1, 1, None, 0, None, 0, 0, 1, 1, 1, 1],
        [0, 1, None, None, 1, None, None, None, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
        [0, 0, None, None, None, None, None, None, 0, 0, 0],
        [1, 1, None, None, None, None, None, None, 1, 1, 0],
        [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1]
    ]

    test3 = [
        [0, None, 0, 0, 0, None, 1, 1, None, None, None],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, None, 0, 1, 0, None, 0, 0, None, None, None],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, None, 0, 0, 0, None, 0, 0, None, None, None],
        [0, None, 0, 0, 0, None, 0, 0, None, None, None],
        [1, None, 0, 0, 0, None, 0, 0, None, None, None],
        [0, None, 0, 0, 0, None, 0, 0, None, None, None],
        [0, None, 0, 0, 0, None, 0, 0, None, None, None],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, None, 1, 0, 0, None, 0, 0, None, None, None],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, None, 0, 0, 0, None, 0, 0, None, None, None],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, None, 0, 0, 0, None, 1, 0, None, None, None],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    ]

    test4 = [
        [None, 0, 0, 0, 0, None, None, 0, 0, None, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [None, 0, 0, 0, 0, None, None, 1, 0, None, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [None, 0, 0, 0, 0, None, None, 0, 0, None, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [None, 0, 0, 0, 0, None, None, 0, 0, None, 0, 0],
        [None, 0, 0, 0, 0, None, None, 0, 0, None, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [None, 0, 0, 0, 0, None, None, 0, 0, None, 0, 0],
        [None, 0, 1, 0, 0, None, None, 0, 0, None, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [None, 0, 1, 0, 0, None, None, 0, 0, None, 0, 0],
        [None, 0, 1, 0, 0, None, None, 0, 0, None, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    dims3, orig3, span3 = extract_dims_approx(test4)
    dims4, orig4, span4 = extract_nonb_dims_approx(test4)
    print(dims1)
    print(dims2)
    pass
    tests = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 0, 1, 1, 1, 1, 1, 0], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 0, 1, 0, 0], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 0, 0, 0, 0, 1, 0, 1, 0], 
        [1, 0, 1, 1, 1, 0, 1, 0, 0], 
        [1, 1, 0, 1, 0, 1, 0, 1, 0], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 0, 0, 0, 1, 0, 1, 0], 
        [1, 1, 1, 1, 1, 0, 1, 0, 1], 
        [1, 1, 0, 1, 0, 1, 1, 1, 0], 
        [0, 1, 0, 0, 0, 0, 0, 0, 0], 
        [1, 1, 0, 0, 0, 1, 0, 1, 0], 
        [1, 1, 0, 1, 0, 1, 0, 1, 0], 
        [1, 1, 1, 1, 1, 0, 1, 0, 1], 
        [1, 0, 1, 1, 1, 0, 1, 0, 0], 
        [1, 1, 1, 1, 1, 1, 1, 0, 1]
    ]      

    tests = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0], 
        [1, 0, 0, 0, 0, 1, 0, 1, 0], 
        [1, 1, 0, 0, 0, 1, 0, 1, 0], 
        [1, 1, 0, 1, 0, 1, 0, 1, 0], 
        [1, 1, 0, 1, 0, 1, 1, 1, 0], 
        [1, 1, 0, 1, 1, 1, 1, 1, 0], 
        [1, 0, 1, 1, 1, 0, 1, 0, 0], 
        [1, 1, 1, 1, 1, 0, 1, 0, 0], 
        [1, 1, 1, 1, 1, 0, 1, 0, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [0, 1, 0, 0, 0, 0, 0, 0, 0], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 1, 0, 0, 0, 1, 0, 1, 0], 
        [1, 1, 0, 1, 0, 1, 0, 1, 0], 
        [1, 1, 1, 1, 1, 0, 1, 0, 1], 
        [1, 0, 1, 1, 1, 0, 1, 0, 0], 
        [1, 1, 1, 1, 1, 1, 1, 0, 1]
    ]           
    tests = [
        [1,1,0,0,0,0], #a 0
        [0,0,0,0,1,1], #b 1
        [1,1,1,0,0,0], #c 2
        [1,1,0,1,1,1], #d 3
        [1,1,1,1,1,1], #f 4
        [1,1,0,1,1,1], #e 5
    ]    
    # tests = [
    #     [None,1,0,0,None,0], #a 0
    #     [None,0,None,0,1,1], #b 1
    #     [None,None,1,0,None,None], #c 2
    #     [1,1,0,1,1,1], #d 3
    #     [1,None,1,1,1,1], #f 4
    # ]

    layers = get_batch_pareto_layers2(tests, 3)
    pass
    tests2 = [
        [0,1,0,1,0,0],
        [0,0,1,1,0,0],
        [0,0,0,0,0,0],
        [1,1,0,1,0,1],
        [1,0,0,1,0,1],
        [0,0,0,1,0,1],
        [1,0,0,0,0,1]
    ]    
    layers = get_batch_pareto_layers2(tests2, 3)
    pass    
    tests1 = [
        [0,0,0]
    ]

    print("tests")
    for test_id, test in enumerate(tests):
        print(f"{test_id}:{test}")
    print("candidates")
    candidates = [[t[i] for t in tests] for i in range(len(tests[0]))]
    for cid, candidate in enumerate(candidates):
        print(f"{cid}:{candidate}")
        
    # print(extract_dims(tests))

    # tests = [
    #     [1,1,0,0,0,0], #a 0
    #     [0,0,0,0,1,1], #b 1
    #     [1,1,None,0,0,0], #c 2
    #     [1,1,0,1,1,1], #d 3
    #     [1,1,1,1,1,1], #f 4
    # ]    

    # tests = [
    #     [1,1,0,0,0,0], #a 0
    #     [0,0,0,0,1,1], #b 1
    #     [None,1,1,0,0,0], #c 2
    #     [1,1,0,1,1,1], #d 3
    #     [1,1,1,1,1,1], #f 4
    # ]        

    # print("-- mod --")
    print(extract_dims_approx(tests))

    print("res tests")
    for test_id, test in enumerate(tests):
        print(f"{test_id}:{test}")    

    pass 

    #we expect that extract_dims and extract_dims_approx return same results when matrices are not sparsed 
    n = 4
    k = 3
    fail = 0 
    succ = 0
    for i in range(2 ** (n * k)):
        test_linear = [0 for _ in range(n*k)]
        idx = i 
        j = 0
        while idx > 0:
            if (idx % 2) == 1:
                test_linear[j] = 1
            idx = idx // 2
            j += 1
        test = [[test_linear[i * k + j] for j in range(k)] for i in range(n)]
        # print("matrix --", test)
        a, _, _ = extract_dims(test)
        b, _ = extract_dims_approx(test)
        if a != b:
            print("---------------")
            print("Missmatch on matrix")
            print(test)
            print("A:")
            print(a)
            print("B:")
            print(b)
            fail += 1
        else: 
            succ += 1
    print("Failed: ", fail, "Success", succ)
    pass 

    # print(cosa_extract_archive(tests))

    

    # levels = get_batch_pareto_layers(tests, max_layers=3)
    # for level_id, test_ids in enumerate(levels):
    #     for test_id in test_ids:
    #         print(f"{level_id}:{test_id} {tests[test_id]}")
    