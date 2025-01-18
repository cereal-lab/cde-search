''' Module for classic genetic programming. '''
from functools import partial
import inspect
from itertools import product
from typing import Optional
from utils import write_metrics

import numpy as np
from rnd import default_rnd, seed

class Node():
    ''' Node class for tree representation '''
    def __init__(self, func, args = []):
        self.func = func # symbol to identify and also can be called 
        self.args = args # List of Nodes 
        self.str = None
        self.depth = None
        self.signature = inspect.signature(func)
        self.return_type = self.signature.return_annotation
    def call(self, *args, node_bindings = {}, node_outcomes: Optional[dict] = None, node_called: Optional[dict] = None, 
                test_ids: Optional[np.ndarray] = None, **kwargs):
        ''' Executes Node tree, 
            @param node_bindings - redirects execution to another subtree (actually getters!!!)
            @param node_outcomes - map, allows to collect the outcomes of all nodes into the dict if provided
            @param node_executed - map, tracks loops if passed 
        '''
        if node_called is not None :
            if self in node_called: # this could happen only through binding or incorrect build of a tree
                raise ValueError(f"Execution loop detected. Node {str(self)} was already executed") 
            else:
                node_called[self] = True
        if self in node_bindings and (other_self := node_bindings[self]()) is not None:
            return other_self.call(*args, node_bindings = node_bindings, 
                                            node_outcomes = node_outcomes, node_called = node_called, 
                                            test_ids = test_ids, **kwargs)
        node_args = []
        for arg in self.args:
            arg_outcomes = arg.call(*args, node_bindings = node_bindings, 
                                            node_outcomes = node_outcomes, node_called = node_called, 
                                            test_ids = test_ids, **kwargs)
            node_args.append(arg_outcomes)
        if len(node_args) == 0: # leaf 
            new_outcomes = self.func(*args, test_ids = test_ids, **kwargs)
        else:
            new_outcomes = self.func(*node_args, *args, **kwargs)
        if node_outcomes is not None:
            node_outcomes[self] = new_outcomes
        return new_outcomes

    def __str__(self):
        if self.str is None:
            node_args = ", ".join(arg.__str__() for arg in self.args)
            if len(self.args) == 0:
                self.str = self.func.__name__
            else:
                self.str = self.func.__name__ + "(" + node_args + ")"
        return self.str
    def __repr__(self):
        return self.__str__()
    def copy(self, replacements: dict = {}):
        if self in replacements:
            return replacements[self]
        else:
            new_args = [arg.copy(replacements) for arg in self.args]
            return self.__class__(self.func, new_args)

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
    def traverse(self, filter, break_filter, at_depth = 0):
        if break_filter(at_depth, self):
            if filter(at_depth, self):
                yield (at_depth, self)
            for arg in self.args:
                yield from arg.traverse(filter, break_filter, at_depth = at_depth + 1)

def tournament_selection(population, fitnesses, comp_fn = min, selection_size = 7):
    ''' Select parents using tournament selection '''
    selected = default_rnd.choice(len(population), selection_size, replace=True)
    best_index = comp_fn(selected, key=lambda i: (*fitnesses[i].tolist(),))
    best = population[best_index]
    return best

def random_selection(population, fitnesses):
    ''' Select parents using random selection '''
    return default_rnd.choice(population)

def grow(depth, leaf_prob, func_list, terminal_list, node_class = Node):
    ''' Grow a tree with a given depth '''
    if depth == 0:
        terminal_index = default_rnd.choice(len(terminal_list))
        terminal = terminal_list[terminal_index]
        return node_class(terminal)
    else:
        if leaf_prob is None:
            func_index = default_rnd.choice(len(func_list + terminal_list))
            func = func_list[func_index] if func_index < len(func_list) else terminal_list[func_index - len(func_list)]
        else:
            if default_rnd.rand() < leaf_prob:
                terminal_index = default_rnd.choice(len(terminal_list))
                func = terminal_list[terminal_index]
            else:
                func_index = default_rnd.choice(len(func_list))
                func = func_list[func_index]
        args = []
        for _, p in inspect.signature(func).parameters.items():
            if p.default is not inspect.Parameter.empty:
                continue
            node = grow(depth - 1, leaf_prob, func_list, terminal_list)
            args.append(node)
        return node_class(func, args)

def full(depth, func_list, terminal_list, node_class = Node):
    ''' Grow a tree with a given depth '''
    if depth == 0:
        terminal_index = default_rnd.choice(len(terminal_list))
        terminal = terminal_list[terminal_index]
        return node_class(terminal)
    else:
        func_index = default_rnd.choice(len(func_list))
        func = func_list[func_index]
        args = []
        for _, p in inspect.signature(func).parameters.items():
            if p.default is not inspect.Parameter.empty:
                continue
            node = full(depth - 1, func_list, terminal_list)
            args.append(node)
        return node_class(func, args)

def ramped_half_and_half(min_depth, max_depth, func_list, terminal_list, node_class = Node):
    ''' Generate a population of half full and half grow trees '''
    depth = default_rnd.randint(min_depth, max_depth+1)
    if default_rnd.rand() < 0.5:
        return grow(depth, None, func_list, terminal_list, node_class = node_class)
    else:
        return full(depth, func_list, terminal_list, node_class = node_class)
    
def init_each(init_fn, population_size):
    return [init_fn() for _ in range(population_size)]    
    
def init_all(depth, max_size, func_list, terminal_list, node_class = Node):
    ''' Generate all possible trees till given depth 
        Very expensive for large depth
    '''    
    zero_depth = []
    for terminal in terminal_list:
        zero_depth.append(node_class(terminal))
        max_size -= 1
        if max_size <= 0:
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
                depth_trees.append(node_class(func, a))
                max_size -= 1
                if max_size <= 0:
                    trees_by_depth.append(depth_trees)
                    return [t for trees in trees_by_depth for t in trees]
        trees_by_depth.append(depth_trees)
    all_trees = [t for trees in trees_by_depth for t in trees]
    return all_trees
    
def select_node(leaf_prob, in_node: Node, filter, break_filter) -> Optional[Node]:
    places = [(at_d, n) for at_d, n in in_node.traverse(filter, break_filter) ]
    if len(places) == 0:
        return None, None 
    if leaf_prob is None: 
        selected_idx = default_rnd.choice(len(places))
        selected = places[selected_idx]
    else:
        nonleaves = [(at_d, n) for (at_d, n) in places if not n.is_leaf()]
        leaves = [(at_d, n) for (at_d, n) in places if n.is_leaf()]
        if (default_rnd.rand() < leaf_prob and len(leaves) > 0) or len(nonleaves) == 0:
            selected_idx = default_rnd.choice(len(leaves))
            selected = leaves[selected_idx]
        else:
            selected_idx = default_rnd.choice(len(nonleaves))
            selected = nonleaves[selected_idx]
    return selected

node_cache = {} # global archive of trees for syntax dedup 
def syntax_dedup(node_class):
    ''' Node class modification for syntactic dedupl '''
    def new_node_class(*args):
        global node_cache
        candidate = node_class(*args)
        candidate_id = str(candidate) # we use string representation as a key
        if candidate_id in node_cache:
            return node_cache[candidate_id]
        else:
            node_cache[candidate_id] = candidate
            return candidate
    return new_node_class

def subtree_mutation(leaf_prob, max_depth, func_list, terminal_list, node, node_class = Node):
    new_node = grow(5, None, func_list, terminal_list, node_class = node_class)
    new_node_depth = new_node.get_depth()
    # at_depth, at_node = select_node(leaf_prob, node, lambda d, n: (d > 0) and n.is_of_type(new_node), 
    #                                     lambda d, n: (d + new_node_depth) <= max_depth)
    at_depth, at_node = select_node(leaf_prob, node, lambda d, n: n.is_of_type(new_node), 
                                        lambda d, n: (d + new_node_depth) <= max_depth)
    if at_node is None:
        return node
    return node.copy({at_node: new_node})

def no_mutation(node):
    return node
    
def subtree_crossover(leaf_prob, max_depth, parent1: Node, parent2: Node):
    ''' Crossover two trees '''
    # NOTE: we can crossover root nodes
    # if parent1.get_depth() == 0 or parent2.get_depth() == 0:
    #     return parent1, parent2
    parent1, parent2 = sorted([parent1, parent2], key = lambda x: x.get_depth())
    # for _ in range(3):
    # at1_at_depth, at1 = select_node(leaf_prob, parent1, lambda d, n: (d > 0), lambda d, n: True)
    at1_at_depth, at1 = select_node(leaf_prob, parent1, lambda d, n: True, lambda d, n: True)
    if at1 is None:
        return parent1, parent2
    at1_depth = at1.get_depth()
    at2_depth, at2 = select_node(leaf_prob, parent2, 
                        lambda d, n: n.is_of_type(at1) and at1.is_of_type(n) and ((n.get_depth() + at1_at_depth) <= max_depth) and (at1_at_depth > 0 or d > 0), 
                        lambda d, n: ((d + at1_depth) <= max_depth))
    # at2_depth, at2 = select_node(leaf_prob, parent2, 
    #                     lambda d, n: (d > 0) and n.is_of_type(at1) and at1.is_of_type(n), 
    #                     lambda d, n: ((d + at1_depth) <= max_depth) and ((n.get_depth() + at1_at_depth) <= max_depth))
    if at2 is None:
        # NOTE: should not be here
        # continue # try another pos
        return parent1, parent2 
        # return parent1, parent2
    child1 = parent1.copy({at1: at2})
    child2 = parent2.copy({at2: at1})
    return child1, child2       

def subtree_breed(mutation_rate, crossover_rate, 
            selection_fn, mutation_fn, crossover_fn, breed_size, population, fitnesses):
    new_population = []
    while len(new_population) < breed_size:
        # Select parents for the next generation
        parent1 = selection_fn(population, fitnesses)
        parent2 = selection_fn(population, fitnesses)
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
        new_population.extend([child1, child2])
    return new_population

# eval_cache = {}

# def get_cached_evals(int_size, fitness_size, population: list[Node]):
#     global eval_cache
#     outcomes = np.zeros((len(population), int_size), dtype = float)
#     interactions = np.zeros((len(population), int_size), dtype=bool) # np.array([p.get_interactions_or_zero(int_size) for p in population])
#     fitnesses = np.zeros_like((len(population), fitness_size))
#     eval_idxs = []
#     for i, p in enumerate(population):
#         if p in eval_cache:
#             p_outcomes, p_ints, p_fitness = eval_cache[p]
#             outcomes[i] = p_outcomes
#             interactions[i] = p_ints
#             fitnesses[i] = p_fitness
#         else:
#             eval_idxs.append(i)
#     return eval_idxs, outcomes, interactions, fitnesses

# def update_cached_evals(new_population: list[Node], new_outcomes, new_interactions, new_fitnesses):
#     global eval_cache
#     keys_to_keep = {}
#     for i, node in enumerate(new_population):
#         if node not in eval_cache:
#             eval_cache[node] = (new_outcomes[i], new_interactions[i], new_fitnesses[i])
#         keys_to_keep[node] = True    
#     eval_cache = {node: eval_cache[node] for node in keys_to_keep.keys() }

def test_based_interactions(gold_outputs: np.ndarray, program_outputs: np.ndarray):
    return (program_outputs == gold_outputs).astype(int)
    
def gp_eager_call(gold_outputs, node, test_ids: Optional[np.ndarray] = None):
    outcomes = node.call(test_ids = test_ids)
    return outcomes

gp_eval_masks = {}
gp_outcomes = {}

def gp_cached_subset_call(gold_outputs, node: Node, test_ids: Optional[np.ndarray] = None):
    ''' Cached call to GP node '''
    req_mask = np.ones(len(gold_outputs), dtype=bool)
    if test_ids is not None:
        req_mask[test_ids] = 0
        req_mask = ~req_mask
    if node in gp_eval_masks:
        done_mask = gp_eval_masks[node]
        eval_mask = req_mask & ~done_mask
        eval_test_ids = np.where(eval_mask)[0]
        if len(eval_test_ids) > 0:
            eval_outcomes = node.call(test_ids = eval_test_ids)
            gp_outcomes[node][eval_test_ids] = eval_outcomes
            done_mask[eval_test_ids] = 1        
    else:
        if test_ids is None:
            eval_outcomes = node.call()
        else:
            eval_outcomes = node.call(test_ids = test_ids)
        gp_outcomes[node] = np.zeros(len(gold_outputs))
        gp_outcomes[node][req_mask] = eval_outcomes
        gp_eval_masks[node] = req_mask
    return gp_outcomes[node][req_mask]

def gp_cached_call(gold_outputs, node: Node):
    ''' Cached call to GP node '''
    if node not in gp_outcomes:
        gp_outcomes[node] = node.call()
    return gp_outcomes[node]
    
def gp_evaluate(gold_outputs, population: list[Node], gp_call = gp_cached_call, int_fn = test_based_interactions, 
                    fitness_fns = [], derive_objectives = None):
    outcomes = np.array([gp_call(gold_outputs, p) for p in population])
    interactions = int_fn(gold_outputs, outcomes)
    if derive_objectives is not None:
        derived_objectives, derived_info = derive_objectives(interactions)
    else:
        derived_objectives = None
        derived_info = {}
    fitnesses = []
    for fitness_fn in fitness_fns:
        fitness = fitness_fn(interactions, population = population, derived_objectives = derived_objectives, **derived_info)
        fitnesses.append(fitness)
    fitnesses_all = np.array(fitnesses).T
    return (fitnesses_all, interactions, outcomes, derived_objectives)

def run_gp(max_generations, initialization_fn, breed, evaluate, get_metrics):
    ''' Koza style GP game schema '''
    population = initialization_fn()
    stats = [] # stats per generation
    generation = 0
    while True:
        fitnesses, *_ = evaluate(population)
        best_found, metrics = get_metrics(fitnesses, population)
        stats.append(metrics)
        if best_found or (generation >= max_generations):
            break        
        generation += 1
        population = breed(population, fitnesses)  
    
    return stats
    

def depth_fitness(interactions, *, population = [], **kwargs):
    return [p.get_depth() for p in population]

def hamming_distance_fitness(interactions, **kwargs):
    return np.sum(1 - interactions, axis = 1)

def ifs_fitness(interactions, **kwargs):
    counts = np.sum((interactions[:, None] == interactions) & (interactions == 1), axis = 0).astype(float)
    counts[counts > 0] = 1 / counts[counts > 0]
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


def get_metrics(main_fitness_index: int, fitnesses, population):
    ''' Get the best program in the population '''
    fitness_order = np.lexsort(fitnesses.T[::-1])
    best_index = fitness_order[0]
    best_fitness = fitnesses[best_index]
    best = population[best_index]
    is_best = best_fitness[main_fitness_index] == 0
    return is_best, (best, *best_fitness)

def run_gp_experiment(sim_name, game_name, gold_outputs, func_list, terminal_list,
                            evaluate_fn = gp_evaluate,
                            init_fn = partial(init_each, partial(ramped_half_and_half, 1, 5)),
                            selection_fn = partial(tournament_selection, selection_size = 7),
                            mutation_fn = partial(subtree_mutation, 0.1, 17),
                            crossover_fn = partial(subtree_crossover, 0.1, 17),
                            breed_fn = partial(subtree_breed, mutation_rate = 0.1, crossover_rate = 0.9),
                            get_metrics_fn = partial(get_metrics, 0),
                            fitness_fns = [hamming_distance_fitness, depth_fitness],
                            record_fitness_ids = [0, 1],
                            metrics_file = "data/metrics/objs.jsonlist",
                            max_generations = 100,
                            population_size = 1000,
                            num_runs = 30):
    initialization = partial(init_fn, func_list, terminal_list, population_size)
    mutation_fn = partial(mutation_fn, func_list, terminal_list)
    breed = partial(breed_fn, selection_fn = selection_fn, mutation_fn = mutation_fn, crossover_fn = crossover_fn, breed_size = population_size)
    evaluate = partial(evaluate_fn, gold_outputs, fitness_fns = fitness_fns)
    for run_id in range(num_runs):
        stats = run_gp(max_generations, initialization, breed, evaluate, get_metrics_fn)
        best_inds, *fitness_metrics = zip(*stats)
        metrics = dict(game = game_name, sim = sim_name, seed = seed, run_id = run_id, best_ind = str(best_inds[-1]), best_ind_depth = best_inds[-1].get_depth())
        for i, metric in enumerate(fitness_metrics):
            if i in record_fitness_ids:
                i_i = record_fitness_ids.index(i)
                metrics["fitness" + str(i_i)] = metric
        write_metrics(metrics, metrics_file)
        pass

gp = partial(run_gp_experiment, "gp")

ifs = partial(run_gp_experiment, "ifs", 
              fitness_fns = [ifs_fitness, hamming_distance_fitness, depth_fitness], 
              get_metrics_fn = partial(get_metrics, 1),
              record_fitness_ids = [1, 2, 0])

ifs0 = partial(run_gp_experiment, "ifs0", 
              fitness_fns = [ifs_fitness], get_metrics_fn = partial(get_metrics, 1),
              record_fitness_ids = [1, 2, 0])


gp_sim_names = [ 'gp', 'ifs0', 'ifs' ]