''' Module for classic genetic programming. '''
import inspect
from typing import Optional

import numpy as np
from rnd import default_rnd

class Node():
    ''' Node class for tree representation '''
    def __init__(self, func, args = []):
        self.func = func # symbol to identify and also can be called 
        self.args = args # List of Nodes 
        self.outcomes = None
        self.interactions = None
        self.int_fn = None
        self.fitness = None # distance to desired objectives
        self.fitness_func = None
        self.str = None
        self.depth = None
        self.signature = inspect.signature(func)
        self.return_type = self.signature.return_annotation
    def __call__(self, *args, **kwds):
        if self.outcomes is None:
            node_args = [arg.__call__(*args, **kwds) for arg in self.args]
            new_outcomes = self.func(*node_args, *args, **kwds)
            self.outcomes = new_outcomes
        return self.outcomes
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
            return Node(self.func, new_args)
    def get_interactions(self, int_fn):
        if self.int_fn != int_fn:
            self.interactions = int_fn(self)
            self.int_fn = int_fn        
        return self.interactions
    def get_interactions_or_zero(self, int_size: int):
        if self.interactions is None:
            return np.zeros(int_size)
        return self.interactions
    def has_no_interactions(self):
        return self.interactions is None
    def get_fitness(self, fitness_fn, i, interactions, **kwargs):
        if self.fitness_func != fitness_fn:
            self.fitness_func = fitness_fn
            self.fitness = None
        self.fitness = fitness_fn(i, interactions, prev_fitness = self.fitness, **kwargs)
        return self.fitness
    def get_fitness_or_zero(self, fitness_size: int):
        if self.fitness is None:
            return np.zeros(fitness_size)
        return self.fitness
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
    
# a = np.array([1,2,3])
# b = np.array([4,5,6])
# np.full((3,), 1)

# class ERC(Node):
#     ''' Class for terminal nodes '''
#     def __call__(self, *args, **kwds):
#         return kwds[self.func]
#     def __str__(self):
#         return self.func

def tournament_selection(tournament_selection_size, population):
    ''' Select parents using tournament selection '''
    selected = default_rnd.choice(len(population), tournament_selection_size, replace=True)
    best_index = min(selected, key=lambda i: (*population[i].fitness.tolist(),))
    best = population[best_index]
    return best

# TODO: make this comparison less strict
# def have_compatible_types(func1, func2):
#     sig1 = inspect.signature(func1)
#     sig2 = inspect.signature(func2)
    
#     params1 = [(param.annotation, param.default) for param in sig1.parameters.values()]
#     return1 = sig1.return_annotation
#     params2 = [(param.annotation, param.default) for param in sig2.parameters.values()]
#     return2 = sig2.return_annotation
    
#     return params1 == params2 and return1 == return2

# def have_compatible_return_types(func1, func2):
#     sig1 = inspect.signature(func1)
#     sig2 = inspect.signature(func2)
    
#     return1 = sig1.return_annotation
#     return2 = sig2.return_annotation
    
#     return return1 == return2

# def f(x: float) -> float:
#     return x + 42

# def g(y: float, z:float = 10) -> float:
#     return y + 43

# have_compatible_types(f, g)
# ft = inspect.signature(f).parameters
# [(param.annotation, param.default) for param in inspect.signature(g).parameters.values()]
# gt = inspect.signature(g).parameters


def grow(depth, leaf_prob, func_list, terminal_list):
    ''' Grow a tree with a given depth '''
    if depth == 0:
        terminal_index = default_rnd.choice(len(terminal_list))
        terminal = terminal_list[terminal_index]
        return Node(terminal)
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
        return Node(func, args)

def full(depth, func_list, terminal_list):
    ''' Grow a tree with a given depth '''
    if depth == 0:
        terminal_index = default_rnd.choice(len(terminal_list))
        terminal = terminal_list[terminal_index]
        return Node(terminal)
    else:
        func_index = default_rnd.choice(len(func_list))
        func = func_list[func_index]
        args = []
        for _, p in inspect.signature(func).parameters.items():
            if p.default is not inspect.Parameter.empty:
                continue
            node = full(depth - 1, func_list, terminal_list)
            args.append(node)
        return Node(func, args)

def ramped_half_and_half(min_depth, max_depth, func_list, terminal_list):
    ''' Generate a population of half full and half grow trees '''
    depth = default_rnd.randint(min_depth, max_depth+1)
    if default_rnd.rand() < 0.5:
        return grow(depth, None, func_list, terminal_list)
    else:
        return full(depth, func_list, terminal_list)

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

def subtree_mutation(leaf_prob, max_depth, func_list, terminal_list, node):
    new_node = grow(5, None, func_list, terminal_list)
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
            selection, mutation, crossover, breed_size, population):
    new_population = []
    while len(new_population) < breed_size:
        # Select parents for the next generation
        parent1 = selection(population)
        parent2 = selection(population)
        if default_rnd.rand() < mutation_rate:
            child1 = mutation(parent1)
        else:
            child1 = parent1
        if default_rnd.rand() < mutation_rate:
            child2 = mutation(parent2)        
        else:
            child2 = parent2
        if default_rnd.rand() < crossover_rate:
            child1, child2 = crossover(child1, child2)   
        new_population.extend([child1, child2])
    return new_population

# def run_koza(population_size, max_generations, initialization, breed, int_fn, fitness_fn, get_metrics):
#     ''' Koza style GP game schema '''
#     population = [initialization() for _ in range(population_size)]
#     stats = [] # stats per generation
#     for generation in range(max_generations):
#         interactions = np.stack([p.get_interactions(int_fn) for p in population], axis = 0)
#         fitnesses = [p.get_fitness(fitness_fn, i, interactions) for i, p in enumerate(population)] # to compute p.fitness in individuals
#         best_found, metrics = get_metrics(population)
#         stats.append(metrics)
#         if best_found:
#             break        
#         population = breed(population)

#     if not best_found:
#         interactions = np.stack([p.get_interactions(int_fn) for p in population], axis = 0)
#         fitnesses = [p.get_fitness(fitness_fn, i, interactions) for i, p in enumerate(population)] # to compute p.fitness in individuals
#         _, metrics = get_metrics(population)
#         stats.append(metrics)    
    
#     return stats

# np.array([np.array([1,2,3]), np.array([4,5,6])])

def gp_evaluate(int_size, int_fn, fitness_fns, population, *, derive_objectives = None):
    eval_mask = np.array([p.has_no_interactions() for p in population], dtype=bool)
    interactions = np.array([p.get_interactions_or_zero(int_size) for p in population])
    call_results = np.array([p() for p in population])
    interactions[eval_mask] = int_fn(call_results[eval_mask])
    if derive_objectives is not None:
        derived_objectives, derived_info = derive_objectives(interactions)
    else:
        derived_objectives = None
        derived_info = {}
    fitnesses = np.array([ p.get_fitness_or_zero(len(fitness_fns)) for p in population])
    fitnesses_T = fitnesses.T
    for i, fitness_fn in enumerate(fitness_fns):
        fitness_fn(fitnesses_T[i], interactions, eval_mask = eval_mask, population = population, derived_objectives = derived_objectives, **derived_info)
    # interactions = np.stack([p.get_interactions(int_fn) for p in population], axis = 0)
    # fitnesses = [p.get_fitness(fitness_fn, i, interactions) for i, p in enumerate(population)] # to compute p.fitness in individuals
    for i, p in enumerate(population):
        p.interactions = interactions[i]
        p.fitness = fitnesses[i]  
    return (fitnesses, interactions, call_results, derived_objectives)

def run_koza(population_size, max_generations, initialization, breed, evaluate, get_metrics):
    ''' Koza style GP game schema '''
    population = [initialization() for _ in range(population_size)]
    stats = [] # stats per generation
    generation = 0
    while True:
        fitnesses, *_ = evaluate(population)
        best_found, metrics = get_metrics(fitnesses, population)
        stats.append(metrics)
        if best_found or (generation >= max_generations):
            break        
        generation += 1
        population = breed(population_size, population)  
    
    return stats