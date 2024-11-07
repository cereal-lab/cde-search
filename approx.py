''' Deducing missed values in the given matrix 
    Approx function signature: interactions -> approx_vector
    Interactions is square matrix with values 0,1 or None
    Approx vector returns deduced values for None position in top-down left-right order.
    It can be used to compare approx methods
'''

from typing import Optional

''' None in the interactions is treated as const_value, test fail 
    Returns number of approximations per each test in the outcomes
'''
def const_approx(interactions: list[list[Optional[int]]], *, const_value = 0) -> list[int]:
    approx_vector = []
    for test_id, test in enumerate(interactions):
        for candidate_id, outcome in enumerate(test):
            if outcome is None:
                interactions[test_id][candidate_id] = const_value
                approx_vector.append(const_value)
    return approx_vector

''' None in interactions is treated as 0 '''
def zero_approx_strategy(interactions: list[list[Optional[int]]]) -> None:
    return const_approx(interactions, const_value = 0)
    
''' None in interactions is treated as 1 '''
def one_approx_strategy(interactions: list[list[Optional[int]]]) -> None:
    return const_approx(interactions, const_value = 1)

''' Approximation with column, row or both majority vote '''
def maj_approx_strategy(interactions: list[list[Optional[int]]], kind = "column", default_value = 0) -> None:
    row_counts = [{} for _ in range(len(interactions))]
    column_counts = [{} for _ in range(len(interactions[0]))]
    for test_id in range(len(interactions)):
        for candidate_id in range(len(interactions[0])):
            outcome = interactions[test_id][candidate_id]
            if outcome is not None:
                column_counts[candidate_id][outcome] = column_counts[candidate_id].get(outcome, 0) + 1
    for candidate_id in range(len(interactions[0])):
        for test_id in range(len(interactions)):
            outcome = interactions[test_id][candidate_id]
            if outcome is not None:
                row_counts[test_id][outcome] = row_counts[test_id].get(outcome, 0) + 1                
    approx_vector = []
    for test_id, test in enumerate(interactions):
        rc = row_counts[test_id] if kind != "column" else {}
        for candidate_id, outcome in enumerate(test):
            if outcome is None:
                cc = column_counts[candidate_id] if kind != "row" else {}
                all_outcomes = set(rc.keys()).union(cc.keys())
                counts = {} 
                for outcome in all_outcomes:
                    counts[outcome] = rc.get(outcome, 0) + cc.get(outcome, 0)            
                max_outcome = default_value if len(counts) == 0 else max(all_outcomes, key = lambda x: (counts[x], -x))
                interactions[test_id][candidate_id] = max_outcome 
                approx_vector.append(max_outcome)
    return approx_vector

def maj_c_approx_strategy(interactions: list[list[Optional[int]]]) -> None:
    return maj_approx_strategy(interactions, kind = "column", default_value = 0)

def maj_r_approx_strategy(interactions: list[list[Optional[int]]]) -> None:
    return maj_approx_strategy(interactions, kind = "row", default_value = 0)

def maj_cr_approx_strategy(interactions: list[list[Optional[int]]]) -> None:
    return maj_approx_strategy(interactions, kind = "both", default_value = 0)

# a = [[1,0,1,0],[1,1,1,0],[0,1,0,1],[None,0,1,None],[None,1,0,None]]
# maj_c_approx_strategy(a)
# [[1, 0, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0]]

''' Each candidate has a group of similarly performign students, Pareto-comparable students
    We run the majority vote inside each group.
'''
def candidate_group_approx_strategy(interactions: list[list[Optional[int]]]) -> None:
    # candidates are columns in interactions
    idxs = [(i, j) for i in range(len(interactions)) for j in range(len(interactions[0])) if interactions[i][j] is None]
    candidates = [[None if interactions[i][j] is None else 1 - interactions[i][j] for i in range(len(interactions))] for j in range(len(interactions[0])) ]
    # we collect pareto relatable candidates for each candidate 
    candidate_groups = {}
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            pairs = [ (o1, o2) for o1, o2 in zip(candidates[i], candidates[j]) if o1 is not None and o2 is not None ]
            if len(pairs) > 0 and any(o1 > o2 for o1, o2 in pairs) and any(o1 < o2 for o1, o2 in pairs):
                continue 
            #pareto relatable
            candidate_groups.setdefault(i, set()).add(j)
            candidate_groups.setdefault(j, set()).add(i)
    for i, group in candidate_groups.items():
        candidate = candidates[i]
        for test_id, o in enumerate(candidate):
            if o is None:
                counts = {}
                for j in group:
                    o2 = candidates[j][test_id]
                    if o2 is not None:
                        counts[o2] = counts.get(o2, 0) + 1
                max_outcome = 0 if len(counts) == 0 else (max(counts.items(), key = lambda x: x[1])[0])
                value = 1 - max_outcome
                interactions[test_id][i] = value
    approx_vector = [interactions[i][j] for i, j in idxs]
    return approx_vector

    # a = [[1,0,1,0],[1,1,1,0],[0,1,0,1],[None,0,1,None],[None,1,0,None]]
    # candidate_group_approx_strategy(None, a)
    #[[1, 0, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]

''' Further split Pareto-domination group onto subgroups by distance and order them.
    First subgroup decides approximation values
'''
def candidate_subgroup_approx_strategy(interactions: list[list[Optional[int]]]) -> None:
    # candidates are columns in interactions
    idxs = [(i, j) for i in range(len(interactions)) for j in range(len(interactions[0])) if interactions[i][j] is None]
    candidates = [[None if interactions[i][j] is None else 1 - interactions[i][j] for i in range(len(interactions))] for j in range(len(interactions[0])) ]
    # we collect pareto relatable candidates for each candidate 
    candidate_groups = {}
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            pairs = [ (o1, o2) for o1, o2 in zip(candidates[i], candidates[j]) if o1 is not None and o2 is not None ]
            if len(pairs) > 0 and any(o1 > o2 for o1, o2 in pairs) and any(o1 < o2 for o1, o2 in pairs):
                continue 
            
            dist = sum(0 if o1 == o2 else 1 for o1, o2 in zip(candidates[i], candidates[j]) if o1 is not None and o2 is not None)

            candidate_groups.setdefault(i, {}).setdefault(dist, set()).add(j)
            candidate_groups.setdefault(j, {}).setdefault(dist, set()).add(i)
    for i, groups in candidate_groups.items():
        candidate = candidates[i]
        group = min(groups.items(), key = lambda x: x[0])[1]
        for test_id, o in enumerate(candidate):
            if o is None:
                counts = {}
                for j in group:
                    o2 = candidates[j][test_id]
                    if o2 is not None:
                        counts[o2] = counts.get(o2, 0) + 1
                max_outcome = 0 if len(counts) == 0 else (max(counts.items(), key = lambda x: x[1])[0])
                value = 1 - max_outcome
                interactions[test_id][i] = value 
    approx_vector = [interactions[i][j] for i, j in idxs]
    return approx_vector

if __name__ == "__main__":

    a = [[1,0,1,0],[1,1,1,0],[0,1,0,1],[None,0,1,None],[None,1,0,None]]
    maj_cr_approx_strategy(a)    

    from params import rnd

    n = 10
    m = 10
    k = 50
    interactions = [[0 for j in range(m)] for i in range(n)]    
    for i in range(n):
        for j in range(m):
            v = int(rnd.choice([0, 1]))
            interactions[i][j] = v
    all_idxs = [(i, j) for i in range(n) for j in range(m)]
    idxs_ids = rnd.choice(len(all_idxs), size = k, replace = False)
    idxs = [all_idxs[i] for i in idxs_ids]
    orig_vect = [interactions[i][j] for i,j in idxs]
    for i,j in idxs:
        interactions[i][j] = None 
    v1 = zero_approx_strategy(interactions)
    v1_cnt = sum(1 for o1, o2 in zip(v1, orig_vect) if o1 != o2)
    for i,j in idxs:
        interactions[i][j] = None 
    v2 = one_approx_strategy(interactions)    
    v2_cnt = sum(1 for o1, o2 in zip(v2, orig_vect) if o1 != o2)
    for i,j in idxs:
        interactions[i][j] = None 
    v3 = maj_c_approx_strategy(interactions) 
    v3_cnt = sum(1 for o1, o2 in zip(v3, orig_vect) if o1 != o2)   
    for i,j in idxs:
        interactions[i][j] = None 
    v4 = candidate_group_approx_strategy(interactions)
    v4_cnt = sum(1 for o1, o2 in zip(v4, orig_vect) if o1 != o2)
    for i,j in idxs:
        interactions[i][j] = None 
    v5 = candidate_subgroup_approx_strategy(interactions)
    v5_cnt = sum(1 for o1, o2 in zip(v5, orig_vect) if o1 != o2)
    for i,j in idxs:
        interactions[i][j] = None 
    print(orig_vect)
    print(v1, v1_cnt)
    print(v2, v2_cnt)
    print(v3, v3_cnt)
    print(v4, v4_cnt)
    print(v5, v5_cnt)
    pass