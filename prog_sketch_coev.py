''' Coevolution of programs and sketches 
    Number of populations depends on number of types of trees 
    The goal of coevolution is to find the program and best way to recombine them in a form of sketches

    Start: sketches have one extension point 
    TODO: two free var sketches 

    Note: free vars, or holes of sketches are typed, represent same tree.

    To generalize, programs are kind 0 sketches, sketches of kind N have N unique holes (num of pos in tree could be > N)
'''

import numpy as np
from gp import Node, node_copy
from nsga2 import get_pareto_front_indexes
# from utils import create_named_function
from rnd import default_rnd
from itertools import product

# import pandas as pd 

# df = pd.read_csv('sms-log.csv')
# df.columns
# df['Price']
# dff = df.groupby(['To', 'Status']).agg(Count = ('Status', 'size'), TotalPrice = ('Price', 'sum')).reset_index().sort_values(by='Count', ascending=False)
# dff2 = df.groupby(['To', 'Status']).agg(Count = ('Status', 'size'), TotalPrice = ('Price', 'sum')).reset_index().sort_values(by='TotalPrice', ascending=True)
# dff2[dff2['TotalPrice'] <= -1]
# dff2[dff2['TotalPrice'] <= -1].to_csv('sms-log-nov-p.csv', index=False)
# ddfr = dff[(dff['Count'] > 1) & (dff['Status'] != 'delivered')]
# ddfr
# ddfr.to_csv('sms-log-dec.csv', index=False)

# all_bindings = {} 

def create_binding(leaves: list[Node]):
    global all_bindings
    l = tuple(sorted(leaves))
    if l in all_bindings:
        

# class Binding():
#     def __init__(self, leaves: list[Node]):
#         self.value = None # not bound 
#         self.id = sorted(leaves) # sorted leaves define the binding identity 
#         self.leaves_map = {leaf: idx for idx, leaf in enumerate(leaves)}
#     def set(self, new_node: Node):
#         self.value = new_node
#     def get(self):
#         return self.value
#     def __iter__(self):
#         return iter(self.leaves_map.keys())
#     def __contains__(self, leaf):
#         return leaf in self.leaves_map
#     def __eq__(self, other):
#         return self.id == other.id
#     def __hash__(self):
#         return hash(self.id)
    
class Bindings():
    def __init__(self, *bindings: list[Binding]):
        self.bindings = tuple(sorted(bindings, key = lambda x: x.id))
    # def add_binding(self, leaves: list[Node]):
    #     binding = Binding(leaves)
    #     self.bindings.append(binding)
    #     return binding
    def __contains__(self, leaf):
        # if isinstance(leaf, Binding):
        #     return leaf in self.bindings
        return any(leaf in binding for binding in self.bindings)
    def __iter__(self):
        return iter(self.bindings)
    def __eq__(self, other):
        return self.bindings == other.bindings
    def __hash__(self):
        return hash(self.bindings)

# def create_binding_for_positions(leaves: list[Node], node_bindings = {}):
#     ''' Creates binders for a given leaves 
#         All provided leaves share same binder
#         Returns map of leaf to binder, the provided node_bindings (updates it)
#     '''
#     binding_ref = [None] # None binding by default
#     def binder_getter_setter(new_node = None, binding_ref = binding_ref):
#         if new_node is None:
#             return binding_ref[0]
#         else:
#             binding_ref[0] = new_node
#     for leaf in leaves:
#         node_bindings[leaf] = binder_getter_setter
#     return node_bindings

# def simple_init_kind_n_sketch(program_init, num_bindings: int):
#     ''' Initializes a program with assigned number of bindings to program leaves.
#         Note that program still defines the structure of the sketch and only leaves are bound.

#         Args:
#             program_init: function that returns a program, e.g. ramped_half_and_half
#             num_bindings: number of bindings to be created. One binding could be attached to many positions in the program 

#         Returns: tuple of 
#             program: initialized program
#             node_bindings: map of leaf node to binder function
#     '''
#     try_times = 10
#     while try_times > 0: # try to gen programs until num_bindings is satisfied
#         program = program_init()
#         program_leaves = [l for l in program.traverse(lambda a, n: n.is_leaf(), lambda a, b:True)]
#         if len(program_leaves) >= num_bindings:
#             break 
#         try_times -= 1
#     if len(program_leaves) < num_bindings:
#         num_bindings = len(program_leaves)
#     if num_bindings == 0:
#         return program, {}
#     default_rnd.shuffle(program_leaves)
#     split_points = [(0, len(program_leaves))]
#     if num_bindings > 1:
#         lengths = 1 + default_rnd.choice(len(program_leaves), size = num_bindings, replace=False)
#         indexes = [0, *np.cumsum(lengths).tolist()[:-1]]
#         split_points = list(zip(indexes, lengths))
#     node_bindings = {}
#     for start, size_n in split_points:
#         num_holes = default_rnd.randint(1, size_n + 1)
#         leaves = [leaf for leaf in program_leaves[start:start + num_holes]]        
#         create_binding_for_positions(leaves, node_bindings)
#     return program, node_bindings

def leaves_selection(sketch: Node, sketch_bindings: Bindings, num_sketches) -> list[list[Node]]:
    ''' Selects sketch leaves for binding which were not bound yet '''
    leaves = [l for l in sketch.traverse(lambda a, n: (n not in sketch_bindings) and n.is_leaf(), lambda a, n:n not in sketch_bindings)]
    if len(leaves) == 0: # all bound
        return [] # cannot create such bindings for given sketch    
    total_res = []
    for _ in range(num_sketches):
        sz = default_rnd.randint(1, len(leaves) + 1)
        selected_leaves = default_rnd.choice(leaves, size = sz, replace=False)
        total_res.append(selected_leaves)
    return total_res

def pick_node_subgroup_rand(nodes):
    ''' Randomly selects a subgroup of nodes '''
    sz = default_rnd.randint(1, len(nodes) + 1)
    selected = default_rnd.choice(nodes, size = sz, replace=False)
    return selected

def semantic_group_selection(semantics_getter, sketch: Node, sketch_bindings: Bindings, num_sketches, pick_node_subgroup = pick_node_subgroup_rand) -> list[list[Node]]:
    ''' Selects sketch positions that correspond to the same semantics '''
    semantics_map = {}
    for node in sketch.traverse(lambda a, n: n not in sketch_bindings, lambda a, n: n not in sketch_bindings):
        node_semantics = semantics_getter(node) # should return hashable, tuple 
        semantics_map.setdefault(node_semantics, []).append(node)
    if len(semantics_map) == 0: # all bound
        return []
    total_res = []
    for _ in range(num_sketches):
        node_groups = [(nodes, default_rnd.random()) for nodes in semantics_map.values()]
        biggest_group, _ = max(node_groups, key = lambda x: (len(x[0]), x[1]))
        subgroup = pick_node_subgroup(biggest_group)
        total_res.append(subgroup)
    return total_res

def generalize_sketch(tree_position_selection, sketch: Node, sketch_bindings: Bindings, num_sketches: int):
    ''' Increate number of free variables by one for the given sketch or program by selecting bound positions.

        Args:
            tree_position_selection: function that returns nodes to be bound, e.g. leaves
                Returns num_sketches of node lists which would be bound, not present already in sketch_bindings
                Examples: leaves_selection, semantic_group_selection
            sketch: generalized sketch 
            sketch_bindings: already bound positions
            num_sketches: number of sketches to be created from the given sketch

        Returns: list of tuples
            sketch: given sketch
            new_bindings: map of leaf node to binder function, should include sketch_bindings
    '''
    program_position_groups = tree_position_selection(sketch, sketch_bindings, num_sketches) #returns list of lists of Nodes
    res = {}
    for sketch_leaves in program_position_groups:
        if len(sketch_leaves) == 0:
            continue
        new_binding = Binding(sketch_leaves)
        new_bindings = Bindings(new_binding, *sketch_bindings)
        res[new_bindings] = sketch
    new_sketches = [ (sketch, bindings) for bindings, sketch in res.items()]
    return new_sketches

def concretize_sketch(sketch: Node, sketch_bindings: Bindings, selected_binding: Binding, program: Node):
    ''' Decrease number of free variables by one for the given sketch by selecting bound positions '''
    nodes_to_replace = {}
    for node in selected_binding:
        nodes_to_replace[node] = program
    new_bindings = Bindings(*(binding for binding in sketch_bindings if binding != selected_binding))
    new_sketch = node_copy(sketch, nodes_to_replace)
    return new_sketch, new_bindings    

def all_sketch_init(all_trees_init, generalize_sketch):
    programs = all_trees_init()
    program_with_bindings = [(p, {}) for p in programs]
    

## NOTE: next should be implemented through semantic_cache
# def uniq_init_up_to_kind_n_sketch(*iter_times_kind_0_init):
#     ''' Note that final population lengths could be varying. 
#         iter_times gives num of attempted tree builds
#     '''
#     p_map = {}
#     for kind_n, (iter_times, kind_0_init) in enumerate(iter_times_kind_0_init):
#         for _ in range(iter_times):
#             program = simple_init_kind_n_sketch(kind_n, kind_0_init)
#             program() #exec or partially exec
#             program_signature = [] #all intermediate signatures
#             for node in program.bottom_up_left_right():
#                 program_signature.append(tuple(node.outcomes))
#                 node_sig = tuple(program_signature)
#                 p_map[node_sig] = node
#     populations = [[] for _ in range(len(iter_times_kind_0_init))]
#     for node in p_map.items():
#         populations[len(node.holes)].append(node)
#     return populations

# def bind_sketch(kind_k_sketch, *subsketches):
#     ''' More optimal to compute target semantics without building the syntactic tree (expand_sketch)'''
#     kind_k_sketch.bind(*subsketches)

# def expand_sketch(kind_k_sketch, *subsketches):
#     ''' Usually used to build kind_(k-l) sketches, more concrete, or programs'''
#     return kind_k_sketch.bind_replace(*subsketches)

# def generalize_sketch(kind_k_sketch, min_sem_freq = 1, max_l_holes = 2, max_hole_kinds = 2):
#     ''' Possibly converts kind_k sketch into kind_(k+l) sketch. Note that max kind_(k + l) = max_hole_kinds
#         Generalization happens by analysing semantics at subtrees to figure out the points of replacement to new hole. 
#         If semantics happens many times in the sketch, it would replaced with a hole everywhere.
#         This process is greedy (max frequent semantics). If several semantics have same freq, all of them allocate new holes 
#         If each semantic is unique in the tree - noop? or should we pick random?
#         min_sem_freq - filters out semantics under consideration, max_l_holes - defines how many holes to allocate at max. 
#         Should be called after evaluation????
#     '''
#     max_l = min(max_l_holes, max_hole_kinds - len(kind_k_sketch.holes))
#     if max_l == 0:
#         return kind_k_sketch
#     semantic_stats = {}
#     semantic_filter = lambda at_depth, node: node.outcomes is not None and len(node.outcomes) > 0
#     no_break_filter = lambda at_depth, node: True
#     for _, node in kind_k_sketch.traverse(semantic_filter, no_break_filter):
#         outcome_vect = tuple(node.outcomes)
#         semantic_stats.setdefault(outcome_vect, []).append(node)
#     candidates = [(nodes, default_rnd.random()) for _, nodes in semantic_stats.items() if len(nodes) >= min_sem_freq]
#     if len(candidates) == 0:
#         return kind_k_sketch
#     l = default_rnd.randint(1, max_l + 1)
#     node_groups = sorted(candidates, key = lambda x: (len(x[0]), x[1]), reverse=True)[:l]
#     l = len(node_groups)
#     hole_nodes = build_hole_nodes(l, len(kind_k_sketch.holes))
#     copy_dict = {}
#     for hole_node, (nodes, _) in zip(hole_nodes, node_groups):
#         for node in nodes:
#             copy_dict[node] = hole_node
#     res = kind_k_sketch.copy(copy_dict)
#     return res

# def generalize_sketch(kind_k_sketch, min_sem_freq = 1, max_l_holes = 2, max_hole_kinds = 2):
#     ''' Possibly converts kind_k sketch into kind_(k+l) sketch. Note that max kind_(k + l) = max_hole_kinds
#         Generalization happens by analysing semantics at subtrees to figure out the points of replacement to new hole. 
#         If semantics happens many times in the sketch, it would replaced with a hole everywhere.
#         This process is greedy (max frequent semantics). If several semantics have same freq, all of them allocate new holes 
#         If each semantic is unique in the tree - noop? or should we pick random?
#         min_sem_freq - filters out semantics under consideration, max_l_holes - defines how many holes to allocate at max. 
#         Should be called after evaluation????
#     '''
#     max_l = min(max_l_holes, max_hole_kinds - len(kind_k_sketch.holes))
#     if max_l == 0:
#         return kind_k_sketch
#     semantic_stats = {}
#     semantic_filter = lambda at_depth, node: node.outcomes is not None and len(node.outcomes) > 0
#     no_break_filter = lambda at_depth, node: True
#     for _, node in kind_k_sketch.traverse(semantic_filter, no_break_filter):
#         outcome_vect = tuple(node.outcomes)
#         semantic_stats.setdefault(outcome_vect, []).append(node)
#     candidates = [(nodes, default_rnd.random()) for _, nodes in semantic_stats.items() if len(nodes) >= min_sem_freq]
#     if len(candidates) == 0:
#         return kind_k_sketch
#     l = default_rnd.randint(1, max_l + 1)
#     node_groups = sorted(candidates, key = lambda x: (len(x[0]), x[1]), reverse=True)[:l]
#     l = len(node_groups)
#     hole_nodes = build_hole_nodes(l, len(kind_k_sketch.holes))
#     copy_dict = {}
#     for hole_node, (nodes, _) in zip(hole_nodes, node_groups):
#         for node in nodes:
#             copy_dict[node] = hole_node
#     res = kind_k_sketch.copy(copy_dict)
#     return res

def simple_sketch_breed(programs, *sketches):
    children = [[] for _ in range(len(sketches) + 1)]
    for fn_pop in sketches:
        for fn in fn_pop:
            arg_idxs = np.choice(len(programs), size = len(fn.holes))
            args = [programs[i] for i in arg_idxs]
            new_program = expand_sketch(fn, *args)
            children[0].append(new_program)
    for p in programs:
        for i in enumerate(sketches):
            kind_n = i + 1
            sketch = generalize_sketch(p, max_l_holes = kind_n, max_hole_kinds = (len(populations) - 1))
            if sketch != p:
                children[len(sketch.holes)].append(sketch)
    res = [[*p, *c] for p, c in zip(populations, children)]
    return res

# very similar to gp_evaluate but simplified/changed
# def simple_program_interactions(int_size, int_fn, fitness_fns, programs):
#     ''' Returns 2-D tensor of programs vs tests '''
#     program_eval_mask = np.array([p.has_no_interactions() for p in programs], dtype=bool)
#     program_interactions = np.array([p.get_interactions_or_zero(int_size) for p in programs])
#     program_call_results = np.array([p() for p in programs])
#     program_interactions[program_eval_mask] = int_fn(program_call_results[program_eval_mask])
#     program_fitnesses = np.array([ p.get_fitness_or_zero(len(fitness_fns)) for p in programs])
#     program_fitnesses_T = program_fitnesses.T
#     for i, fitness_fn in enumerate(fitness_fns):
#         fitness_fn(program_fitnesses_T[i], program_interactions, eval_mask = program_eval_mask, population = programs)
#     for i, p in enumerate(programs):
#         p.interactions = program_interactions[i]
#         p.fitness = program_fitnesses[i]
#     return program_interactions

def extract_best_programs(select_programs_fn, interactions):
    ''' First extracts pareto front and then selects subset of if according to given select_programs_fn '''
    front_indices = get_pareto_front_indexes(interactions)
    selected_indexes_ids = select_programs_fn(interactions[front_indices]) #, fitnesses[all_fronts_indexes])
    selected_indexes = front_indices[selected_indexes_ids]
    return selected_indexes

def extract_best_sketches(select_sketches_fn, interactions, sketch_bindings, sketch_args):
    ''' First extracts pareto front and then selects subset of if according to given select_sketches_fn '''
    front_indices = get_pareto_front_indexes(interactions)
    selected_indexes_ids = select_sketches_fn(interactions[front_indices]) #, fitnesses[all_fronts_indexes])
    selected_indexes = front_indices[selected_indexes_ids]
    return selected_indexes

def simple_sketch_interract(extract_programs_fn, extract_sketches_fn, int_size, int_fn, fitness_fns, programs, *sketches):
    program_interactions = simple_program_interactions(int_size, int_fn, fitness_fns, programs)
    program_indexes = extract_programs_fn(program_interactions)
    selected_programs = [programs[i] for i in program_indexes] # these are arguments for sketches    
    indexes = list(range(len(selected_programs)))
    for sketch_group in sketches:
        applied_sketches = []
        sketch_call_results = []
        for sketch in sketch_group:
            for arg_indexes in product(*([indexes] * len(sketch.holes))):
                applied_sketches.append((sketch, arg_indexes))
                args = [selected_programs[i] for i in arg_indexes]
                sketch.bind(*args)
                sketch_call_results.append(sketch())
        sketch_call_results = np.array(sketch_call_results)
        sketch_interactions = int_fn(sketch_call_results) 
        sketch_indexes = extract_sketches_fn(sketch_interactions, sketch_bindings = applied_sketches, sketch_args = selected_programs)   
        #TODO: stats for sketches assuming binding of vars to semantics of best programs by fitness???

    return test_interactions

def run_prog_sketch_coevolution(max_generations, initialization, breed, interract, get_metrics, select_parents = None):
    # Create initial population
    populations = initialization()
    stats = []
    generation = 0
    while True:
        test_interactions = interract(*populations) # idea here that interract returns interactions of each population with set of tests
        # NOTE: inside interract we assume that populations communicate and actually form (N + 1)-D sparse tensor in general case
        # N - num of populations, 1 for set of tests
        new_populations = []
        is_first = True
        for popc, interactions in zip(populations, test_interactions):
            new_front_indices = get_pareto_front_indexes(interactions)
            if len(new_front_indices) == 0:
                new_populations.append(popc)
            else:
                if is_first:
                    is_first = False
                    best_front = [popc[i] for i in new_front_indices]
                    best_front_fitnesses = [ind.fitness for ind in best_front]
                    best_found, metrics = get_metrics(best_front_fitnesses, best_front)
                    stats.append(metrics)
                    if best_found or (generation >= max_generations):
                        return stats
                selected_indexes_ids = select_parents(interactions[new_front_indices]) #, fitnesses[all_fronts_indexes])
                selected_indexes = new_front_indices[selected_indexes_ids]
                new_population = [popc[i] for i in selected_indexes]
                new_populations.append(new_population)
        populations = breed(*new_populations)
        generation += 1