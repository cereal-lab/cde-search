''' Module for classic genetic programming. '''
import inspect
from functools import partial
from itertools import product
from typing import Any, Optional
import utils

import numpy as np
from rnd import default_rnd
import utils

class Node():
    ''' Node class for tree representation. Immutable!! '''
    def __init__(self, func, args = []):
        self.func = func # symbol to identify and also can be called 
        self.args = args # List of Nodes 
        self.str = None
        self.depth = None
        self.nodes = None # list of all nodes for direct access
        self.signature = inspect.signature(func)
        self.return_type = self.signature.return_annotation
    def call(self, node_outcomes = {}): #*args, node_outcomes = {}, #**kwargs, node_bindings = {}, node_called: Optional[dict] = None, **kwargs):
        ''' Executes Node tree, 
            @param node_bindings - redirects execution to another subtree (actually getters!!!)
            @param node_outcomes - map, allows to collect the outputs of all nodes into the dict if provided
            @param node_executed - map, tracks loops if passed 
        '''
        # NOTE: WARN: be careful not to cache for different node_bindings!!!
        if self in node_outcomes:
            return node_outcomes[self]
        # if node_called is not None :
        #     if self in node_called: # this could happen only through binding or incorrect build of a tree
        #         raise ValueError(f"Execution loop detected. Node {str(self)} was already executed") 
        #     else:
        #         node_called[self] = True
        # if self in node_bindings and (other_self := node_bindings[self]()) is not None:
        #     return other_self.call(*args, node_bindings = node_bindings, 
        #                                     node_outcomes = node_outcomes, node_called = node_called, 
        #                                     test_ids = test_ids, **kwargs)
        node_args = []
        for arg in self.args:
            arg_outcomes = arg.call(node_outcomes = node_outcomes)            
            # arg_outcomes = arg.call(*args, node_bindings = node_bindings, 
            #                                 node_outcomes = node_outcomes, node_called = node_called, 
            #                                 test_ids = test_ids, **kwargs)
            node_args.append(arg_outcomes)
        if len(node_args) == 0: # leaf 
            # new_outcomes = self.func(*args, test_ids = test_ids, **kwargs)
            new_outcomes = self.func()
        else:
            new_outcomes = self.func(*node_args) #, *args, **kwargs)
        node_outcomes[self] = new_outcomes
        return new_outcomes

    def __str__(self):
        if self.str is None:
            if len(self.args) == 0:
                self.str = self.func.__name__
            else:
                node_args = ", ".join(arg.__str__() for arg in self.args)
                self.str = self.func.__name__ + "(" + node_args + ")"
        return self.str
    def __repr__(self):
        return self.__str__()

    def get_depth(self):
        if self.depth is None:
            if len(self.args) == 0:
                self.depth = 0
            else:
                self.depth = max(n.get_depth() for n in self.args) + 1
        return self.depth
    def is_leaf(self):
        return len(self.args) == 0
    def is_of_type(self, node):
        return self.return_type == node.return_type
    def get_nodes(self):
        if self.nodes is None:
            self.nodes = [(0, self), *((arg_depth + 1, n) for arg in self.args for arg_depth, n in arg.get_nodes())]
        return self.nodes
    def get_node(self, i):
        nodes = self.get_nodes()
        return nodes[i]

# TODO: make this comparison less strict
# def have_compatible_types(func1, func2):
#     sig1 = inspect.signature(func1)
#     sig2 = inspect.signature(func2)
    
#     params1 = [(param.annotation, param.default) for param in sig1.parameters.values()]
#     return1 = sig1.return_annotation
#     params2 = [(param.annotation, param.default) for param in sig2.parameters.values()]
#     return2 = sig2.return_annotation
    
#     return params1 == params2 and return1 == return2
def simple_node_builder(func, args):
    return Node(func, args)

def cached_node_builder(func, args, save_stats = True, *, syntax_cache, node_builder, stats):
    ''' Builds the node with node_builder, but first checks cache '''
    # NOTE: trees are always built from bottom to up, so we can use existing Node objects as key elements
    key = (func.__name__, *args)
    if key not in syntax_cache:
        new_node = node_builder(func, args)
        syntax_cache[key] = new_node
    else:
        stats["syntax_cache_hits"] = stats.get("syntax_cache_hits", 0) + 1
    return syntax_cache[key]

# import numpy as np 

# a1 = np.array([[0, 1, 1], [1, 1, 0]])
# a2 = np.array([[1, 1, 1], [1, 0, 1], [0, 0, 0]])
# np.all(a1[:, None] >= a2, axis = 2)
# np.sum(np.any(np.all(a1[:, None] <= a2, axis = 2) & np.any(a1[:, None] < a2), axis = 0))

class BreedingStats():
    def __init__(self):
        self.parent_child_relations = []

def count_good_bad_children(parents: np.ndarray, children: np.ndarray):
    domination_matrix = np.all(parents[:, None] <= parents, axis=2) & np.any(parents[:, None] < parents, axis=2)
    indexes = np.where(~np.any(domination_matrix, axis=1))[0]
    parents_front = parents[indexes]
    num_good_children = np.sum(np.any(np.all(parents_front[:, None] <= children, axis = 2) & np.any(parents_front[:, None] < children), axis = 0))
    num_bad_children = np.sum(np.any(np.all(parents_front[:, None] >= children, axis = 2), axis = 0))
    return (num_good_children, num_bad_children)


# def node_from_list(node_list, i = 0, *, node_builder):
#     cur_func = node_list[i].func
#     cur_func_params = [p for _, p in node_list[i].signature.parameters.items() if p.default is not inspect.Parameter.empty ]
#     if len(cur_func_params) == 0:
#         return i + 1, node_builder(cur_func, [])
#     else:
#         new_i = i + 1
#         new_args = []
#         for p in cur_func_params:
#             new_i, new_node = node_from_list(node_list, i = new_i, node_builder = node_builder)
#             new_args.append(new_node)
#         return new_i, node_builder(cur_func, new_args)

def replace_syntax(node: Node, replacements: dict[Node, Node], *, node_builder):
    ''' Replace all syntax appearance given by node in dict '''    
    if node in replacements:
        return replacements[node]
    new_args = [replace_syntax(arg, replacements, node_builder = node_builder) for arg in node.args]
    return node_builder(node.func, new_args)

def replace_pos_syntax(node: Node, replacements: dict[int, Node], *, node_builder):
    new_replacements = {} 
    for i, (_, n) in enumerate(node.get_nodes()):
        if i in replacements:
            new_replacements[n] = replacements[i]
    return replace_syntax(node, new_replacements, node_builder = node_builder)

def node_copy(node: Node, replacements: dict[int, Node], idx = 0, *, node_builder):
    res = None
    replaced = False
    if idx in replacements:
        res = replacements[idx]
        replacements = {} 
        replaced = True
    new_idx = idx + 1
    new_args = []
    for arg in node.args:
        new_idx, new_arg, arg_replaced = node_copy(arg, replacements, idx = new_idx, node_builder = node_builder)
        replaced = replaced or arg_replaced
        new_args.append(new_arg)    
    if res is None:
        if replaced:
            res = node_builder(node.func, new_args)
        else:
            res = node
    return new_idx, res, replaced

def replace_positions(node: Node, replacements: dict[int, Node], *, node_builder):
    _, res, _ = node_copy(node, replacements, idx = 0, node_builder = node_builder)
    return res
    
def test_based_interactions(gold_outputs: np.ndarray, program_outputs: np.ndarray):
    return (program_outputs == gold_outputs).astype(int)
    
def _compute_fitnesses(fitness_fns, interactions, population, gold_outputs, derived_objectives = [], derived_info = {}):
    fitness_list = []
    for fitness_fn in fitness_fns:
        fitness = fitness_fn(interactions, population = population, gold_outputs = gold_outputs, 
                                derived_objectives = derived_objectives, **derived_info)
        fitness_list.append(fitness) 
    fitnesses = np.array(fitness_list).T  
    return fitnesses   

def gp_eval(nodes: list[Node], int_fn = test_based_interactions, derive_objs_fn = None, save_stats = True, *, out_cache, int_cache, gold_outputs, fitness_fns, stats):
    ''' Cached node evaluator '''
    # NOTE: derived objectives does not work with cache as they are computed on per given group of nodes
    if len(nodes) == 0:
        raise ValueError("Empty population")
    node_ids_to_eval = [node_id for node_id, node in enumerate(nodes) if node not in int_cache]
    int_size = 0
    out_size = 0
    if save_stats:
        stats.setdefault("num_eval_nodes", []).append(len(nodes))
        stats.setdefault("num_active_evals", []).append(len(node_ids_to_eval))
        stats.setdefault("eval_cache_hits", []).append(len(nodes) - len(node_ids_to_eval))
    if len(node_ids_to_eval) > 0:
        nodes_to_eval = [nodes[node_id] for node_id in node_ids_to_eval]
        new_outputs = np.array([node.call(node_outcomes=out_cache) for node in nodes_to_eval ])
        new_interactions = int_fn(gold_outputs, new_outputs)
        # fit_size = new_fitnesses.shape[1]
        int_size = new_interactions.shape[1]
        out_size = new_outputs.shape[1]
        for node, new_ints in zip(nodes_to_eval, new_interactions):
            int_cache[node] = new_ints
    else:
        node = nodes[0]
        int_size = len(int_cache[node])
        out_size = len(out_cache[node])
    interactions = np.zeros((len(nodes), int_size), dtype = float)
    outputs = np.zeros((len(nodes), out_size), dtype = float)
    for node_id, node in enumerate(nodes):
        interactions[node_id] = int_cache[node]
        outputs[node_id] = out_cache[node]
    if derive_objs_fn is not None:
        derived_objectives, derived_info = derive_objs_fn(interactions)
    else:
        derived_objectives = None
        derived_info = {}        
    fitnesses = _compute_fitnesses(fitness_fns, interactions, nodes, gold_outputs, derived_objectives, derived_info)
    if derived_objectives is not None:
        return outputs, fitnesses, interactions, derived_objectives
    return outputs, fitnesses, interactions

def pick_min(selected_fitnesses):
    best_id_id, best_fitness = min([(fid, tuple(ft)) for fid, ft in enumerate(selected_fitnesses)], key = lambda x: x[1])
    return best_id_id

def tournament_selection(population: list[Any], fitnesses: np.ndarray, fitness_comp_fn = pick_min, tournament_selection_size = 7):
    ''' Select parents using tournament selection '''
    selected_ids = default_rnd.choice(len(population), tournament_selection_size, replace=True)
    selected_fitnesses = fitnesses[selected_ids]
    best_id_id = fitness_comp_fn(selected_fitnesses)
    best_id = selected_ids[best_id_id]
    # best = population[best_id]
    return best_id

def tournament_selection_scalar(population: list[Any], fitnesses: np.ndarray, fitness_index = 0, fitness_comp_fn = pick_min, tournament_selection_size = 7):
    ''' Select parents using tournament selection '''
    selected_ids = default_rnd.choice(len(population), tournament_selection_size, replace=True)
    selected_fitnesses = fitnesses[selected_ids, fitness_index]
    best_id_id = fitness_comp_fn(selected_fitnesses)
    best_id = selected_ids[best_id_id]
    return best_id

def random_selection(population: list[Any], fitnesses: np.ndarray):
    ''' Select parents using random selection '''
    rand_id = default_rnd.choice(len(population))
    # selected = population[rand_id]
    return rand_id

def grow(grow_depth = 5, grow_leaf_prob = None, *, func_list, terminal_list, node_builder):
    ''' Grow a tree with a given depth '''
    if grow_depth == 0:
        terminal_index = default_rnd.choice(len(terminal_list))
        terminal = terminal_list[terminal_index]
        return node_builder(terminal, [])
    else:
        if grow_leaf_prob is None:
            func_index = default_rnd.choice(len(func_list + terminal_list))
            func = func_list[func_index] if func_index < len(func_list) else terminal_list[func_index - len(func_list)]
        else:
            if default_rnd.rand() < grow_leaf_prob:
                terminal_index = default_rnd.choice(len(terminal_list))
                func = terminal_list[terminal_index]
            else:
                func_index = default_rnd.choice(len(func_list))
                func = func_list[func_index]
        args = []
        for _, p in inspect.signature(func).parameters.items():
            if p.default is not inspect.Parameter.empty:
                continue
            node = grow(grow_depth = grow_depth - 1, grow_leaf_prob = grow_leaf_prob, 
                        func_list = func_list, terminal_list = terminal_list, node_builder = node_builder)
            args.append(node)
        return node_builder(func, args)

def full(full_depth = 5, *, func_list, terminal_list, node_builder):
    ''' Grow a tree with a given depth '''
    if full_depth == 0:
        terminal_id = default_rnd.choice(len(terminal_list))
        terminal = terminal_list[terminal_id]
        return node_builder(terminal, [])
    else:
        func_id = default_rnd.choice(len(func_list))
        func = func_list[func_id]
        args = []
        for _, p in inspect.signature(func).parameters.items():
            if p.default is not inspect.Parameter.empty:
                continue
            node = full(full_depth=full_depth - 1, func_list = func_list, terminal_list = terminal_list, node_builder = node_builder)
            args.append(node)
        return node_builder(func, args)

def ramped_half_and_half(rhh_min_depth = 1, rhh_max_depth = 5, rhh_grow_prob = 0.5, *, func_list, terminal_list, node_builder):
    ''' Generate a population of half full and half grow trees '''
    depth = default_rnd.randint(rhh_min_depth, rhh_max_depth+1)
    if default_rnd.rand() < rhh_grow_prob:
        return grow(grow_depth = depth, func_list = func_list, terminal_list = terminal_list, node_builder = node_builder)
    else:
        return full(full_depth = depth, func_list = func_list, terminal_list = terminal_list, node_builder = node_builder)
    
def init_each(size, init_fn = ramped_half_and_half):
    return [init_fn() for _ in range(size)]    
    
def init_all(size, depth = 3, *, func_list, terminal_list, node_builder):
    ''' Generate all possible trees till given depth 
        Very expensive for large depth
    '''    
    zero_depth = []
    for terminal in terminal_list:
        zero_depth.append(node_builder(terminal, []))
        size -= 1
        if size <= 0:
            return zero_depth
    trees_by_depth = [zero_depth]
    for _ in range(1, depth + 1):
        depth_trees = []
        for func in func_list:
            args = []
            for _, p in inspect.signature(func).parameters.items():
                if p.default is not inspect.Parameter.empty:
                    continue
                args.append(trees_by_depth[-1])
            for a in product(*args):
                depth_trees.append(node_builder(func, a))
                size -= 1
                if size <= 0:
                    trees_by_depth.append(depth_trees)
                    return [t for trees in trees_by_depth for t in trees]
        trees_by_depth.append(depth_trees)
    all_trees = [t for trees in trees_by_depth for t in trees]
    return all_trees
    
def _select_node_id(in_node: Node, filter, select_node_leaf_prob = None) -> Optional[Node]:
    if select_node_leaf_prob is None: 
        places = [i for i, (at_d, n) in enumerate(in_node.get_nodes()) if filter(at_d, n) ]
        if len(places) == 0:
            return None 
        selected_idx = default_rnd.choice(len(places))
        selected = places[selected_idx]
    else:
        nonleaves = []
        leaves = []
        for i, (at_d, n) in enumerate(in_node.get_nodes()):
            if filter(at_d, n):
                if n.is_leaf():
                    leaves.append(i)
                else:
                    nonleaves.append(i)
        if len(nonleaves) == 0 and len(leaves) == 0:
            return None
        if (default_rnd.rand() < select_node_leaf_prob and len(leaves) > 0) or len(nonleaves) == 0:
            selected_idx = default_rnd.choice(len(leaves))
            selected = leaves[selected_idx]
        else:
            selected_idx = default_rnd.choice(len(nonleaves))
            selected = nonleaves[selected_idx]
    return selected

def subtree_mutation(node, select_node_leaf_prob = 0.1, tree_max_depth = 17, repl_fn = replace_positions, 
                     *, func_list, terminal_list, node_builder):
    new_node = grow(grow_depth = 5, func_list = func_list, terminal_list = terminal_list, 
                    grow_leaf_prob = None, node_builder = node_builder)
    new_node_depth = new_node.get_depth()
    # at_depth, at_node = select_node(leaf_prob, node, lambda d, n: (d > 0) and n.is_of_type(new_node), 
    #                                     lambda d, n: (d + new_node_depth) <= max_depth)
    node_id = _select_node_id(node, lambda d, n: n.is_of_type(new_node) and ((d + new_node_depth) <= tree_max_depth),
                                        select_node_leaf_prob = select_node_leaf_prob)
    if node_id is None:
        return node
    res = repl_fn(node, {node_id: new_node}, node_builder = node_builder)
    return res

def no_mutation(node):
    return node
    
def subtree_crossover(parent1: Node, parent2: Node, select_node_leaf_prob = 0.1, tree_max_depth = 17, 
                      repl_fn = replace_positions, *, node_builder):
    ''' Crossover two trees '''
    # NOTE: we can crossover root nodes
    # if parent1.get_depth() == 0 or parent2.get_depth() == 0:
    #     return parent1, parent2
    parent1, parent2 = sorted([parent1, parent2], key = lambda x: x.get_depth())
    # for _ in range(3):
    # at1_at_depth, at1 = select_node(leaf_prob, parent1, lambda d, n: (d > 0), lambda d, n: True)
    at1_id = _select_node_id(parent1, lambda d, n: True, select_node_leaf_prob=select_node_leaf_prob)
    if at1_id is None:
        return parent1, parent2
    at1_at_depth, at1 = parent1.get_node(at1_id)
    at1_depth = at1.get_depth()
    at2_id = _select_node_id(parent2, 
                        lambda d, n: n.is_of_type(at1) and at1.is_of_type(n) and ((n.get_depth() + at1_at_depth) <= tree_max_depth) and (at1_at_depth > 0 or d > 0) and ((d + at1_depth) <= tree_max_depth), 
                        select_node_leaf_prob=select_node_leaf_prob)
    # at2_depth, at2
    # at2_depth, at2 = select_node(leaf_prob, parent2, 
    #                     lambda d, n: (d > 0) and n.is_of_type(at1) and at1.is_of_type(n), 
    #                     lambda d, n: ((d + at1_depth) <= max_depth) and ((n.get_depth() + at1_at_depth) <= max_depth))
    if at2_id is None:
        # NOTE: should not be here
        # continue # try another pos
        return parent1, parent2 
        # return parent1, parent2
    at2_at_depth, at2 = parent2.get_node(at2_id)
    child1 = repl_fn(parent1, {at1_id: at2}, node_builder = node_builder)
    child2 = repl_fn(parent2, {at2_id: at1}, node_builder = node_builder)
    return child1, child2       

def subtree_breed(size, population, fitnesses,
                    breed_select_fn = tournament_selection, mutation_fn = subtree_mutation, crossover_fn = subtree_crossover,
                    mutation_rate = 0.1, crossover_rate = 0.9, *, breeding_stats: BreedingStats):
    new_population = []
    breeding_stats.parent_child_relations = []
    while len(new_population) < size:
        # Select parents for the next generation
        parent1_id = breed_select_fn(population, fitnesses)
        parent2_id = breed_select_fn(population, fitnesses)
        parent1 = population[parent1_id]
        parent2 = population[parent2_id]
        if default_rnd.rand() < mutation_rate:
            child1 = mutation_fn(parent1)
        else:
            child1 = parent1
        if default_rnd.rand() < mutation_rate:
            child2 = mutation_fn(parent2)
        else:
            child2 = parent2
        if default_rnd.rand() < crossover_rate:
            child1, child2 = crossover_fn(child1, child2)   
        breeding_stats.parent_child_relations.append(([parent1, parent2], [child1, child2]))
        new_population.extend([child1, child2])
    return new_population
    
def depth_fitness(interactions, population = [], **_):
    return [p.get_depth() for p in population]

def hamming_distance_fitness(interactions, **_):
    return np.sum(1 - interactions, axis = 1)

def ifs_fitness(interactions, **_):
    counts = (np.sum(interactions, axis = 0) * interactions).astype(float)
    counts[counts > 0] = 1.0 / counts[counts > 0]
    ifs = np.sum(counts, axis=1)
    return -ifs

# def ifs_fitness_fn(interactions, **kwargs):
#     counts = np.sum((interactions[:, None] == interactions) & (interactions == 1), axis = 0).astype(float)
#     counts[counts > 0] = 1 / counts[counts > 0]
#     ifs = np.sum(counts, axis=1)
#     return ifs


# interactions = np.array([[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]])
# ifss = []
# for i in range(interactions.shape[0]):
#     r = []
#     for t in range(interactions.shape[1]):
#         if interactions[i][t] == 1:
#             r.append(np.sum(interactions[:, t] == 1))
#         else:
#             r.append(np.inf)
#     ifss.append(r)
# ifss2 = np.array(ifss)
# np.sum(1 / ifss2, axis=1)

# ifs_fitness_fn(interactions)

# ifss.append(ifs_fitness(0, interactions, eval_idxs = [i]))

# ifs_fitness(0, np.array([[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 1]]))

def collect_additional_stats(stats, nodes: list[Node], outputs):
    syntax_counts = {}
    sem_counts = {}
    sem_repr_counts = {}
    for node_id, node in enumerate(nodes):
        # first, syntax stats - number of times each syntax was evaluated
        node_str = str(node)
        syntax_counts[node_str] = syntax_counts.get(node_str, 0) + 1
        # second, number of times semantics appears in evaluation
        node_sem = tuple(outputs[node_id])
        sem_counts[node_sem] = sem_counts.get(node_sem, 0) + 1
        # third, how many representatives this semantics has
        sem_repr_counts.setdefault(node_sem, set()).add(node_str)
    stats.setdefault('stats_nodes', []).append(len(nodes))

    syntax_dupl_rate = sum(c - 1 for c in syntax_counts.values())
    stats.setdefault('syntax_dupl', []).append(syntax_dupl_rate)

    sem_dupl_rate = sum(c - 1 for c in sem_counts.values())
    stats.setdefault('sem_dupl', []).append(sem_dupl_rate)

    sem_repr_rate = np.mean([len(c) for c in sem_repr_counts.values()])
    stats.setdefault('sem_repr_rate', []).append(round(sem_repr_rate, 2))

    num_uniq_syntaxes = len(syntax_counts)
    stats.setdefault('num_uniq_syntaxes', []).append(num_uniq_syntaxes)

    num_uniq_sems = len(sem_counts)
    stats.setdefault('num_uniq_sems', []).append(num_uniq_sems)

def analyze_population(population, outputs, fitnesses, save_stats = True, *, stats, fitness_fns, main_fitness_fn, int_cache, breeding_stats: BreedingStats, **_):
    ''' Get the best program in the population '''
    fitness_order = np.lexsort(fitnesses.T[::-1])
    best_index = fitness_order[0]
    best_fitness = fitnesses[best_index]
    best = population[best_index]
    stats['best'] = str(best)
    is_best = False 
    if main_fitness_fn is None and len(fitness_fns) > 0:
        main_fitness_fn = fitness_fns[0]
    for fitness_idx, fitness_fn in enumerate(fitness_fns):
        if fitness_fn == main_fitness_fn:
            is_best = best_fitness[fitness_idx] == 0
        stats.setdefault(fitness_fn.__name__, []).append(best_fitness[fitness_idx])
    if save_stats:
        collect_additional_stats(stats, population, outputs)
        total_good_ch = 0 
        total_bad_ch = 0
        for parents, children in breeding_stats.parent_child_relations:
            parent_ints = np.array([ int_cache[n] for n in parents ])
            child_ints = np.array([ int_cache[n] for n in children ])
            good_ch, bad_ch = count_good_bad_children(parent_ints, child_ints)
            total_good_ch += good_ch
            total_bad_ch += bad_ch
        if total_good_ch > 0 or total_bad_ch > 0:
            stats.setdefault('good_children', []).append(total_good_ch)
            stats.setdefault('bad_children', []).append(total_bad_ch)
    if is_best:
        return population[best_index]
    return None

def identity_map(population):
    return population

def syntax_dedupl_map(population):
    ''' Removes syntactic duplicates '''
    pop_set = {n: True for n in population}
    new_population = list(pop_set.keys())
    return new_population

def evol_loop(population_size, max_gens, init_fn, map_fn, breed_fn, eval_fn, analyze_pop_fn):
    ''' Classic evolution loop '''
    population = init_fn(population_size)
    gen = 0
    best_ind = None
    while gen < max_gens:
        population = map_fn(population)
        outputs, fitnesses, *_ = eval_fn(population)
        best_ind = analyze_pop_fn(population, outputs, fitnesses) 
        if best_ind is not None:
            break        
        population = breed_fn(population_size, population, fitnesses)  
        gen += 1
    
    return best_ind, gen

# NOTE: fitness function should not have any shared structure bound, but given by eval_fn method
#       it means, that there is no binding of these structures at pipeline design time
# NOTE: next is the list of shared structures (bound at design time): 
#           gold_outputs (labels or expected outcomes)
#           func_list (list of gp node functions)
#           terminal_list (list of gp terminal functions)
#           fitness_fns (list of fitness functions)
#           main_fitness_fn (main fitness function)
#           node_builder (function to build nodes)
#           syntax_cache (cache for syntax trees)
#           int_cache, out_cache (cache for evaluation results)
#           stats (dictionary for statistics)
# NOTE: binding happens by name of these parameters in funcs and after *!
# Any other parameters should be bound explicitly or defaults should be used
def koza_evolve(gold_outputs, func_list, terminal_list,
                population_size = 1000, max_gens = 100,
                fitness_fns = [hamming_distance_fitness, depth_fitness], main_fitness_fn = hamming_distance_fitness,
                init_fn = init_each, map_fn = identity_map, breed_fn = subtree_breed, 
                eval_fn = gp_eval, analyze_pop_fn = analyze_population):
    stats = {}
    syntax_cache = {}
    node_builder = partial(cached_node_builder, syntax_cache = syntax_cache, node_builder = simple_node_builder, stats = stats)
    shared_context = dict(
        gold_outputs = gold_outputs, func_list = func_list, terminal_list = terminal_list,
        fitness_fns = fitness_fns, main_fitness_fn = main_fitness_fn, node_builder = node_builder,
        syntax_cache = syntax_cache, int_cache = {}, out_cache = {}, stats = stats, breeding_stats = BreedingStats())
    evol_fns = utils.bind_fns(shared_context, init_fn, map_fn, breed_fn, eval_fn, analyze_pop_fn)
    best_ind, gen = evol_loop(population_size, max_gens, *evol_fns)
    stats["gen"] = gen
    stats["best_found"] = best_ind is not None
    return best_ind, stats

gp = koza_evolve

ifs = partial(koza_evolve, fitness_fns = [ifs_fitness, hamming_distance_fitness, depth_fitness])

gp_sim_names = [ 'gp', 'ifs' ]

if __name__ == '__main__':
    import gp_benchmarks
    game_name, (gold_outputs, func_list, terminal_list) = gp_benchmarks.get_benchmark('cmp6')
    best_prog, stats = gp(gold_outputs, func_list, terminal_list)
    print(best_prog)
    print(stats)
    pass    
