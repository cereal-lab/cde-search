''' Coevolution of programs and sketches 
    Number of populations depends on number of types of trees 
    The goal of coevolution is to find the program and best way to recombine them in a form of sketches

    Start: sketches have one extension point 
    TODO: two free var sketches 

    Note: free vars, or holes of sketches are typed, represent same tree.

    To generalize, programs are kind 0 sketches, sketches of kind N have N unique holes (num of pos in tree could be > N)
'''

import numpy as np
from gp import Node, ramped_half_and_half
from nsga2 import get_pareto_front_indexes
from utils import create_named_function
from rnd import default_rnd
from itertools import product

def build_holes(n: int, start: int):
    ''' Created number of uniquely distinguished holes as ERCs in the tree, Free vars replaced by other tree. ADF, not ADM '''
    hole_bindings = []
    holes_with_binders = []
    for i in range(n):
        hole_bindings.append(None)
        def h(i=i, hole_bindings = hole_bindings) -> np.ndarray:
            # hole should be bound end target tree should be evaluated to call this.
            # it means that evaluation of bound sketches should happen AFTER we evaluate arguments (ADF, not ADM!!!)
            # it is normal as we need first to check args as solutions before do recombination
            nodeOpt = hole_bindings[i] #should be ExtNode or Node, already evaluated
            if nodeOpt is None or nodeOpt.outcomes is None:
                return []
            return nodeOpt.outcomes 
        def bind_h(tree: Node, i = i, hole_bindings = hole_bindings):
            ''' Currently we assume that tree should not have other holes '''
            hole_bindings[i] = tree 
        holes_with_binders.append((create_named_function(f"H{(start + i)}", h), bind_h))
    return holes_with_binders

class ExtNode(Node):
    ''' Extends node with some subtree stats and holes '''
    def __init__(self, func, args=[], binder = None):
        self.my_binder = binder
        super().__init__(func, args)
        self.leaves = []
        self.holes = []
        if not self.is_leaf():
            for arg_i, arg in enumerate(self.args):                
                if arg.is_leaf():
                    arg_leaves = [arg]
                    if arg.my_binder is not None:
                        arg_holes = [arg]
                    else:
                        arg_holes = []
                else:
                    arg_leaves = arg.leaves
                    arg_holes = arg.holes
                if arg_i == 0:
                    self.leaves = arg_leaves
                    self.holes = arg_holes
                else:
                    self.leaves = [*self.leaves, *arg_leaves]
                    if len(arg_holes) > 0:
                        new_holes = [*self.holes]
                        for h in arg_holes:
                            if h not in new_holes:
                                new_holes.append(h)
                        self.holes = new_holes
    def is_hole(self):
        return self.my_binder is not None
    def is_hole_or_has_holes(self):
        return self.is_hole() or (len(self.holes) > 0)
    def bind(self, *trees):
        for h, t in zip(self.holes, trees):
            if t is not None:
                h.my_binder(t)
    def bind_replace(self, *trees):
        copy_dict = {}
        for h, t in zip(self.holes, trees):
            if t is not None:
                copy_dict[h] = t
        res = self.copy(copy_dict)
        return res
    def __call__(self, *args, **kwds):
        if self.outcomes is None:
            node_args = [] 
            for arg in self.args:
                arg_outcomes = arg.__call__(*args, **kwds)
                if len(arg_outcomes) == 0: #allows partial evaluation
                    node_args = None 
                    break
                else:
                    node_args.append(arg_outcomes)
            if node_args is None:
                self.outcomes = []
            else:
                new_outcomes = self.func(*node_args, *args, **kwds)
                self.outcomes = new_outcomes
        return self.outcomes    
    
def build_hole_nodes(n: int, start: int):
    if n == 0:
        return []
    hole_fns = build_holes(n, start)
    hole_nodes = [ExtNode(hole_fn, binder=binder_fn) for hole_fn, binder_fn in hole_fns]
    return hole_nodes

def ramped_half_and_half_mod(min_depth, max_depth, func_list, terminal_list, min_num_leaves = 0, node_class = ExtNode):
    ''' modification of ramped half and half with assurance that we have min_num_leaves '''
    while True: 
        tree = ramped_half_and_half(min_depth, max_depth, func_list, terminal_list, node_class = node_class)
        if len(tree.leaves) >= min_num_leaves:
            break
    return tree 

# initialization 
def simple_init_kind_n_sketch(kind_n: int, kind_0_init):
    ''' kind 0 - program, and default init, i.e. ramped_half_and_half, could be used 
        The idea of this simple init is to generate kind 0 sketch with minimum of kind_n = len(hole_nodes) leaves.
        Then, to replace randomly choosen n >= kind_n positions with holes randomly choosen from set [0, kind_n)
    '''
    hole_nodes = build_hole_nodes(kind_n, 0)
    sketch = kind_0_init(min_num_leaves = kind_n, node_class = ExtNode) #should return 
    if kind_n == 0:
        return sketch
    leaves = sketch.leaves
    default_rnd.shuffle(leaves)
    split_points = [(0, len(leaves))]
    if kind_n > 1:
        lengths = 1 + default_rnd.choice(len(leaves), size = kind_n, replace=False)
        indexes = [0, *np.cumsum(lengths).tolist()[:-1]]
        split_points = list(zip(indexes, lengths))
    copy_dict = {}
    for hole_node, (start, size_n) in zip(hole_nodes, split_points):
        num_holes = default_rnd.randint(1, size_n + 1)
        for leaf in leaves[start:start + num_holes]:
            copy_dict[leaf] = hole_node
    res = sketch.copy(copy_dict)
    return res

def uniq_init_up_to_kind_n_sketch(*iter_times_kind_0_init):
    ''' Note that final population lengths could be varying. 
        iter_times gives num of attempted tree builds
    '''
    p_map = {}
    for kind_n, (iter_times, kind_0_init) in enumerate(iter_times_kind_0_init):
        for _ in range(iter_times):
            program = simple_init_kind_n_sketch(kind_n, kind_0_init)
            program() #exec or partially exec
            program_signature = [] #all intermediate signatures
            for node in program.bottom_up_left_right():
                program_signature.append(tuple(node.outcomes))
                node_sig = tuple(program_signature)
                p_map[node_sig] = node
    populations = [[] for _ in range(len(iter_times_kind_0_init))]
    for node in p_map.items():
        populations[len(node.holes)].append(node)
    return populations

def bind_sketch(kind_k_sketch, *subsketches):
    ''' More optimal to compute target semantics without building the syntactic tree (expand_sketch)'''
    kind_k_sketch.bind(*subsketches)

def expand_sketch(kind_k_sketch, *subsketches):
    ''' Usually used to build kind_(k-l) sketches, more concrete, or programs'''
    return kind_k_sketch.bind_replace(*subsketches)

def generalize_sketch(kind_k_sketch, min_sem_freq = 1, max_l_holes = 2, max_hole_kinds = 2):
    ''' Possibly converts kind_k sketch into kind_(k+l) sketch. Note that max kind_(k + l) = max_hole_kinds
        Generalization happens by analysing semantics at subtrees to figure out the points of replacement to new hole. 
        If semantics happens many times in the sketch, it would replaced with a hole everywhere.
        This process is greedy (max frequent semantics). If several semantics have same freq, all of them allocate new holes 
        If each semantic is unique in the tree - noop? or should we pick random?
        min_sem_freq - filters out semantics under consideration, max_l_holes - defines how many holes to allocate at max. 
        Should be called after evaluation????
    '''
    max_l = min(max_l_holes, max_hole_kinds - len(kind_k_sketch.holes))
    if max_l == 0:
        return kind_k_sketch
    semantic_stats = {}
    semantic_filter = lambda at_depth, node: node.outcomes is not None and len(node.outcomes) > 0
    no_break_filter = lambda at_depth, node: True
    for _, node in kind_k_sketch.traverse(semantic_filter, no_break_filter):
        outcome_vect = tuple(node.outcomes)
        semantic_stats.setdefault(outcome_vect, []).append(node)
    candidates = [(nodes, default_rnd.random()) for _, nodes in semantic_stats.items() if len(nodes) >= min_sem_freq]
    if len(candidates) == 0:
        return kind_k_sketch
    l = default_rnd.randint(1, max_l + 1)
    node_groups = sorted(candidates, key = lambda x: (len(x[0]), x[1]), reverse=True)[:l]
    l = len(node_groups)
    hole_nodes = build_hole_nodes(l, len(kind_k_sketch.holes))
    copy_dict = {}
    for hole_node, (nodes, _) in zip(hole_nodes, node_groups):
        for node in nodes:
            copy_dict[node] = hole_node
    res = kind_k_sketch.copy(copy_dict)
    return res

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
def simple_program_interactions(int_size, int_fn, fitness_fns, programs):
    ''' Returns 2-D tensor of programs vs tests '''
    program_eval_mask = np.array([p.has_no_interactions() for p in programs], dtype=bool)
    program_interactions = np.array([p.get_interactions_or_zero(int_size) for p in programs])
    program_call_results = np.array([p() for p in programs])
    program_interactions[program_eval_mask] = int_fn(program_call_results[program_eval_mask])
    program_fitnesses = np.array([ p.get_fitness_or_zero(len(fitness_fns)) for p in programs])
    program_fitnesses_T = program_fitnesses.T
    for i, fitness_fn in enumerate(fitness_fns):
        fitness_fn(program_fitnesses_T[i], program_interactions, eval_mask = program_eval_mask, population = programs)
    for i, p in enumerate(programs):
        p.interactions = program_interactions[i]
        p.fitness = program_fitnesses[i]
    return program_interactions

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