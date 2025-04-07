''' Module for classic genetic programming. '''
from dataclasses import dataclass, field
import inspect
from functools import partial
from itertools import chain, product
from typing import Any, Callable, Optional

import torch
import utils

import numpy as np
from rnd import default_rnd
import utils

class Node():
    ''' Node class for tree representation. Immutable!! '''
    def __init__(self, func_meta: utils.AnnotatedFunc, args = []):
        self.func = func_meta.func # symbol to identify and also can be called 
        self.func_meta = func_meta
        self.args = args # List of Nodes 
        self.str = None
        self.depth = None
        self.nodes = None # list of all nodes for direct access
        self.signature = inspect.signature(self.func)
        self.return_type = self.signature.return_annotation

    def __str__(self):
        if self.str is None:
            if len(self.args) == 0:
                self.str = self.func_meta.name()
            else:
                node_args = ", ".join(arg.__str__() for arg in self.args)
                self.str = self.func_meta.name() + "(" + node_args + ")"
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

Outputs = list[Any] | np.ndarray | torch.Tensor
Vectors = np.ndarray | torch.Tensor

@dataclass 
class RuntimeContext:
    free_vars: dict[str, Any] = field(default_factory=dict) #bindings of free variables
    gold_outputs: Outputs = field(default_factory=list) #gold outputs
    stats: dict[str, Any] = field(default_factory=dict)
    int_cache: dict[Node, np.ndarray] = field(default_factory=dict)
    out_cache: dict[Node, Outputs] = field(default_factory=dict)
    counts_cache: dict[Node, dict[str, int]] = field(default_factory=dict)
    syntax_cache: dict[(Callable, list[Node]), Node] = field(default_factory=dict)
    parent_child_relations: list[tuple[list[Node], list[Node]]] = field(default_factory=list)
    counts_constraints: Optional[dict[str, int]] = None
    fitness_fns: list[Callable] = field(default_factory=list)
    main_fitness_fn: Optional[Callable] = None
    select_fitness_ids: Optional[list[int]] = None
    func_list: list[utils.AnnotatedFunc] = field(default_factory=list)
    terminal_list: list[utils.AnnotatedFunc] = field(default_factory=list)
    node_builder: Callable[[utils.AnnotatedFunc, list[Node]], Node] = Node
    # tree_contexts: dict[Node, dict[str, Any]] = field(default_factory=dict)
    breeding_stats: dict[str, dict[Any, int]] = field(default_factory=dict)
    anneal_context: dict[str, Any] = field(default_factory=dict) # depending on operator - stores annealing context
    # aging_type: str = '' # none, syntax, semantics, syntax_w_position, semantics_w_position

    # elitism tracks num_elites of fitnesses and bins nodes from node cache
    # elite_fitnesses: Optional[np.ndarray] = None
    # elite_bins: Optional[list[Node]] = None #bin i has fitness elite_fitnesses[i]
    # num_elites: int = 0

    def update(self, **kwargs):        
        for k, v in kwargs.items():
            assert hasattr(self, k), f'Runtime context does not have property {k}'
            setattr(self, k, v)

def call(tree: Node, free_vars: dict[str, Outputs], node_outcomes: Optional[dict[Node, Outputs]] = None) -> Outputs: #*args, node_outcomes = {}, #**kwargs, node_bindings = {}, node_called: Optional[dict] = None, **kwargs):
    ''' Executes Node tree, 
        @param free_vars - free var bindings
        @param node_outcomes - map, allows to collect the outputs of all nodes into the dict if provided
        @param node_bindings - redirects execution to another subtree (actually getters!!!)
        @param node_executed - map, tracks loops if passed 
    '''
    # NOTE: WARN: be careful not to cache for different node_bindings!!!
    if node_outcomes is not None and tree in node_outcomes:
        return node_outcomes[tree]
    node_args = []
    for arg in tree.args:
        arg_outcomes = call(arg, free_vars, node_outcomes = node_outcomes)            
        node_args.append(arg_outcomes)
    if len(node_args) == 0: # leaf 
        # new_outcomes = self.func.func(*args, test_ids = test_ids, **kwargs)
        new_outcomes = tree.func(free_vars = free_vars)
    else:
        new_outcomes = tree.func(*node_args, free_vars = free_vars) #, *args, **kwargs)
    if node_outcomes is not None:
        node_outcomes[tree] = new_outcomes
    return new_outcomes

def get_func_counts(node: Node, counts_constraints: Optional[dict[str, int]], counts_cache: dict[Node, dict[str, int]]):
    if counts_constraints is None:
        return {}
    if node not in counts_cache:
        res = {}
        for arg in node.args:
            arg_counts = get_func_counts(arg, counts_constraints, counts_cache)
            for k, v in arg_counts.items():
                res[k] = res.get(k, 0) + v
        if node.func_meta.category in counts_constraints:
            res[node.func_meta.category] = res.get(node.func_meta.category, 0) + 1
        counts_cache[node] = res
    return counts_cache[node]

def are_counts_constraints_satisfied(node: Node, counts_constraints: Optional[dict[str, int]], counts_cache: dict[Node, dict[str, int]]):
    if counts_constraints is None:
        return True 
    node_counts = get_func_counts(node, counts_constraints, counts_cache)
    return all(v >= node_counts.get(k, 0) for k, v in counts_constraints.items())

def are_counts_constraints_satisfied_together(node1: Node, node2: Node, counts_constraints: Optional[dict[str, int]], counts_cache: dict[Node, dict[str, int]]):
    if counts_constraints is None:
        return True 
    node_counts1 = get_func_counts(node1, counts_constraints, counts_cache)
    node_counts2 = get_func_counts(node2, counts_constraints, counts_cache)
    return all(v >= (node_counts1.get(k, 0) + node_counts2.get(k, 0)) for k, v in counts_constraints.items())

def default_node_builder(func: utils.AnnotatedFunc, args, *, syntax_cache, stats):
    ''' Builds the node with node_builder, but first checks cache '''
    # NOTE: trees are always built from bottom to up, so we can use existing Node objects as key elements
    key = (func.func, *args)
    if key not in syntax_cache:
        new_node = Node(func, args)
        syntax_cache[key] = new_node
    elif stats is not None:
        stats["syntax_cache_hits"] = stats.get("syntax_cache_hits", 0) + 1
    return syntax_cache[key]

# def cached_node_builder_init(stats):
#     syntax_cache = {}
#     return dict(node_builder = partial(default_node_builder, syntax_cache = syntax_cache, stats = stats), syntax_cache = syntax_cache)

# import numpy as np 

# a1 = np.array([[0, 1, 1], [1, 1, 0]])
# a2 = np.array([[1, 1, 1], [1, 0, 1], [0, 0, 0]])
# np.all(a1[:, None] >= a2, axis = 2)
# np.sum(np.any(np.all(a1[:, None] <= a2, axis = 2) & np.any(a1[:, None] < a2), axis = 0))

def count_good_bad_children(parents: np.ndarray, children: np.ndarray):
    parent_wins = np.sum(parents, axis = 1)
    children_wins = np.sum(children, axis = 1)
    comparison_result = parent_wins[:, np.newaxis] < children_wins[np.newaxis, :]
    num_best_children = np.sum(np.all(comparison_result, axis = 0))
    num_good_children = np.sum(np.any(comparison_result, axis = 0))

    domination_matrix = np.all(parents[:, None] <= parents, axis=2) & np.any(parents[:, None] < parents, axis=2)
    indexes = np.where(~np.any(domination_matrix, axis=1))[0]
    parents_front = parents[indexes]
    parent_vs_child_domination_of_child = np.all(parents_front[:, None] <= children, axis = 2) & np.any(parents_front[:, None] < children, axis = 2)
    num_dom_best_children = np.sum(np.all(parent_vs_child_domination_of_child, axis = 0))
    any_domination_of_child = np.any(parent_vs_child_domination_of_child, axis = 0)
    num_dom_good_children = np.sum(any_domination_of_child)
    num_bad_children = np.sum(np.any(np.all(parents_front[:, None] >= children, axis = 2), axis = 0))
    # num_dominated_parents = np.sum(np.any(parent_vs_child_domination_of_child, axis = 1))

    return (num_best_children, num_good_children, num_dom_best_children, num_dom_good_children, num_bad_children)

# parents = np.array([[0,0,1,1,0,1], [1,0,0,1,0,0], [0,1,0,0,0,0]])
# children = np.array([[1,0,1,1,0,1], [0,0,0,0,0,0], [0,1,0,0,0,0], [1,1,1,1,1,1]])
# count_good_bad_children(parents, children)

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
    return node_builder(node.func_meta, new_args)

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
            res = node_builder(node.func_meta, new_args)
        else:
            res = node
    return new_idx, res, replaced

def replace_positions(node: Node, replacements: dict[int, Node], *, node_builder):
    _, res, _ = node_copy(node, replacements, idx = 0, node_builder = node_builder)
    return res
    
def test_based_interactions(gold_outputs: np.ndarray, program_outputs: np.ndarray) -> np.ndarray:
    return (program_outputs == gold_outputs).astype(int)

def dist_test_based_interactions(gold_outputs: torch.Tensor, program_outputs: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.abs(gold_outputs - program_outputs))
    # Also use explonent and RELU - exponent goes to 0 faster

def np_fitness_prep(fitness_list):
    return np.array(fitness_list).T

def torch_fitness_prep(fitness_list):
    return torch.stack(fitness_list).t()
    
def _compute_fitnesses(fitness_fns, interactions, outputs, population, gold_outputs, derived_objectives = [], derived_info = {}, fitness_prep = np_fitness_prep):
    fitness_list = []
    for fitness_fn in fitness_fns:
        fitness = fitness_fn(interactions, outputs, population = population, gold_outputs = gold_outputs, 
                                derived_objectives = derived_objectives, **derived_info)
        fitness_list.append(fitness) 
    fitnesses = fitness_prep(fitness_list)
    return fitnesses   

def default_eval_node(node: Node, free_vars, gold_outputs, out_cache: dict[Node, Any]) -> np.ndarray:
    outputs = call(node, free_vars, node_outcomes=out_cache)
    # if torch.is_tensor(outputs):
    #     outputs = outputs.detach().numpy()
    return outputs

def np_output_prep(output_list):
    return np.array(output_list)

def torch_output_prep(output_list):
    return torch.stack(output_list)

def gp_eval(nodes: list[Node], int_fn = test_based_interactions, derive_objs_fn = None, save_stats = True, 
            eval_node = default_eval_node, output_prep = np_output_prep, fitness_prep = np_fitness_prep, *, runtime_context: RuntimeContext):
    ''' Cached node evaluator '''
    # NOTE: derived objectives does not work with cache as they are computed on per given group of nodes
    if len(nodes) == 0:
        raise ValueError("Empty population")
    node_ids_to_eval = [node_id for node_id, node in enumerate(nodes) if node not in runtime_context.int_cache]
    int_size = 0
    out_size = 0
    if save_stats:
        runtime_context.stats.setdefault("num_eval_nodes", []).append(len(nodes))
        runtime_context.stats.setdefault("num_active_evals", []).append(len(node_ids_to_eval))
        runtime_context.stats.setdefault("eval_cache_hits", []).append(len(nodes) - len(node_ids_to_eval))
    if len(node_ids_to_eval) > 0:
        nodes_to_eval = [nodes[node_id] for node_id in node_ids_to_eval]
        new_outputs = []
        for node in nodes_to_eval:
            node_outputs = eval_node(node, runtime_context.free_vars, runtime_context.gold_outputs, runtime_context.out_cache)
            new_outputs.append(node_outputs)
        new_outputs = output_prep(new_outputs)
        new_interactions = int_fn(runtime_context.gold_outputs, new_outputs)
        # fit_size = new_fitnesses.shape[1]
        int_size = new_interactions.shape[1]
        out_size = new_outputs.shape[1]
        for node, new_ints in zip(nodes_to_eval, new_interactions):
            if torch.is_tensor(new_ints):
                new_ints = new_ints.numpy()
            runtime_context.int_cache[node] = new_ints
    else:
        node = nodes[0]
        int_size = len(runtime_context.int_cache[node])
        out_size = len(runtime_context.out_cache[node])
    interactions = np.zeros((len(nodes), int_size), dtype = float)
    outputs = np.zeros((len(nodes), out_size), dtype = float)
    for node_id, node in enumerate(nodes):
        interactions[node_id] = runtime_context.int_cache[node]
        outputs[node_id] = runtime_context.out_cache[node]
    if derive_objs_fn is not None:
        derived_objectives, derived_info = derive_objs_fn(interactions)
    else:
        derived_objectives = None
        derived_info = {}        
    fitnesses = _compute_fitnesses(runtime_context.fitness_fns, interactions, outputs, 
                                   nodes, runtime_context.gold_outputs, derived_objectives, derived_info, fitness_prep = fitness_prep)
    if derived_objectives is not None:
        return outputs, fitnesses, interactions, derived_objectives
    return outputs, fitnesses, interactions

def pick_min(selected_fitnesses: np.ndarray):
    best_id_id, best_fitness = min([(fid, (ft if len(selected_fitnesses.shape) == 1 else tuple(ft))) for fid, ft in enumerate(selected_fitnesses)], key = lambda x: x[1])
    return best_id_id

def tournament_selection(population: list[Any], fitnesses: np.ndarray, interactions: np.ndarray, fitness_comp_fn = pick_min, tournament_selection_size = 7, *, runtime_context: RuntimeContext):
    ''' Select parents using tournament selection '''
    selected_ids = default_rnd.choice(len(population), tournament_selection_size, replace=True)
    selected_fitnesses = fitnesses[selected_ids]
    best_id_id = fitness_comp_fn(selected_fitnesses)
    best_id = selected_ids[best_id_id]
    # best = population[best_id]
    return best_id

def lexicase_selection(population: list[Any], fitnesses: np.ndarray, interactions: np.ndarray, *, runtime_context: RuntimeContext):
    """ Based on Lee Spector's team: Solving Uncompromising Problems With Lexicase Selection """
    test_ids = np.arange(interactions.shape[1]) # direction is hardcoded 0 - bad, 1 - good
    default_rnd.shuffle(test_ids)
    candidate_ids = np.arange(len(population))
    for test_id in test_ids:
        test_max_outcome = np.max(interactions[candidate_ids, test_id])
        candidate_id_ids = np.where(interactions[candidate_ids, test_id] == test_max_outcome)[0]
        candidate_ids = candidate_ids[candidate_id_ids]
        if len(candidate_ids) == 1:
            break
    if len(candidate_ids) == 1:
        return candidate_ids[0]
    best_id = default_rnd.choice(candidate_ids)
    return best_id


# def tournament_selection_with_syntax_againg(population: list[Any], fitnesses: np.ndarray, fitness_comp_fn = pick_min, tournament_selection_size = 7, *, runtime_context: RuntimeContext):
#     ''' Select parents using tournament selection '''
#     breed_counts = np.array([runtime_context.aging_stats.get(p, 0) for p in population], dtype=float)
#     breed_weights = 1. / (1. + breed_counts)
#     select_proba = breed_weights / np.sum(breed_weights)
#     selected_ids = default_rnd.choice(len(population), tournament_selection_size, replace=True, p = select_proba)
#     selected_fitnesses = fitnesses[selected_ids]
#     best_id_id = fitness_comp_fn(selected_fitnesses)
#     best_id = selected_ids[best_id_id]
#     # best = population[best_id]
#     return best_id

# def tournament_selection_with_semantic_againg(population: list[Any], fitnesses: np.ndarray, fitness_comp_fn = pick_min, tournament_selection_size = 7, *, runtime_context: RuntimeContext):
#     ''' Select parents using tournament selection '''
#     breed_counts = np.array([runtime_context.aging_stats.get(runtime_context.int_cache.get(p, None), 0) for p in population], dtype=float)
#     breed_weights = 1. / (1. + breed_counts)
#     select_proba = breed_weights / np.sum(breed_weights)
#     selected_ids = default_rnd.choice(len(population), tournament_selection_size, replace=True, p = select_proba)
#     selected_fitnesses = fitnesses[selected_ids]
#     best_id_id = fitness_comp_fn(selected_fitnesses)
#     best_id = selected_ids[best_id_id]
#     # best = population[best_id]
#     return best_id

def random_selection(population: list[Any], fitnesses: np.ndarray, interactions, *, runtime_context: RuntimeContext):
    ''' Select parents using random selection '''
    rand_id = default_rnd.choice(len(population))
    # selected = population[rand_id]
    return rand_id

def grow(grow_depth = 5, grow_leaf_prob: Optional[float] = None, *, 
         counts_constraints: Optional[dict[str, int]] = None, 
         func_list: list[utils.AnnotatedFunc], terminal_list: list[utils.AnnotatedFunc], 
         node_builder: Callable[[utils.AnnotatedFunc, list[Node]], Node]) -> Optional[Node]:
    ''' Grow a tree with a given depth '''
    if counts_constraints is None:
        allowed_funcs = func_list
        allowed_terminals = terminal_list
    else:
        allowed_funcs = [f for f in func_list if (f.category not in counts_constraints) or (counts_constraints[f.category] > 0)]
        allowed_terminals = [t for t in terminal_list if (t.category not in counts_constraints) or (counts_constraints[t.category] > 0)]
    args = []
    if (grow_depth == 0) or len(allowed_funcs) == 0:
        if len(allowed_terminals) == 0:
            return None 
        terminal_index = default_rnd.choice(len(allowed_terminals))
        func_builder = allowed_terminals[terminal_index]
        func = func_builder()
        if counts_constraints is not None and func_builder.category in counts_constraints:
            counts_constraints[func_builder.category] -= 1        
    else:
        disallowed_symbols = set() #already attempted with None results
        while True: # backtrack in case if counts_constraints were noto satisfied - attempting other symbol - or return none
            allowed_funcs = [f for f in allowed_funcs if f.category not in disallowed_symbols]
            allowed_terminals = [t for t in allowed_terminals if t.category not in disallowed_symbols]
            if len(allowed_funcs) == 0 and len(allowed_terminals) == 0:
                return None
            if grow_leaf_prob is None:
                func_index = default_rnd.choice(len(allowed_funcs) + len(allowed_terminals))
                func_builder = allowed_funcs[func_index] if func_index < len(allowed_funcs) else allowed_terminals[func_index - len(allowed_funcs)]
            elif (len(allowed_terminals) > 0) and (default_rnd.rand() < grow_leaf_prob):
                terminal_index = default_rnd.choice(len(allowed_terminals))
                func_builder = allowed_terminals[terminal_index]
            else:
                func_index = default_rnd.choice(len(allowed_funcs))
                func_builder = allowed_funcs[func_index]
            new_counts_constraints = None if counts_constraints is None else dict(counts_constraints)
            if new_counts_constraints is not None and func_builder.category in new_counts_constraints:
                new_counts_constraints[func_builder.category] -= 1            
            func = func_builder()
            for _, p in inspect.signature(func.func).parameters.items():
                if p.default is not inspect.Parameter.empty or p.kind == p.KEYWORD_ONLY or p.kind == p.VAR_KEYWORD:
                    continue
                node = grow(grow_depth = grow_depth - 1, grow_leaf_prob = grow_leaf_prob, counts_constraints = new_counts_constraints,
                            func_list = func_list, terminal_list = terminal_list, node_builder = node_builder)
                args.append(node)
                if node is None:
                    break
            if len(args) > 0 and args[-1] is None:
                disallowed_symbols.add(func_builder.category)
                continue 
            if counts_constraints is not None:
                counts_constraints.update(new_counts_constraints)
            break
    return node_builder(func, args)

def full(full_depth = 5, *, 
         counts_constraints: Optional[dict[str, int]] = None, 
         func_list: list[utils.AnnotatedFunc], terminal_list: list[utils.AnnotatedFunc], 
         node_builder: Callable[[utils.AnnotatedFunc, list[Node]], Node]) -> Optional[Node]:
    ''' Grow a tree with a given depth '''
    if counts_constraints is None:
        allowed_funcs = func_list
        allowed_terminals = terminal_list
    else:
        allowed_funcs = [f for f in func_list if (f.category not in counts_constraints) or (counts_constraints[f.category] > 0)]
        allowed_terminals = [t for t in terminal_list if (t.category not in counts_constraints) or (counts_constraints[t.category] > 0)]    
    args = []
    if full_depth == 0 or len(allowed_funcs) == 0:
        if len(allowed_terminals) == 0:
            return None         
        terminal_id = default_rnd.choice(len(allowed_terminals))
        func_builder = terminal_list[terminal_id]
        func = func_builder()
        if counts_constraints is not None and func_builder.category in counts_constraints:
            counts_constraints[func_builder.category] -= 1          
    else:
        disallowed_symbols = set() #already attempted with None results
        while True: # backtrack in case if counts_constraints were noto satisfied - attempting other symbol - or return none
            allowed_funcs = [f for f in allowed_funcs if f.category not in disallowed_symbols]
            if len(allowed_funcs) == 0:
                return None
            func_id = default_rnd.choice(len(allowed_funcs))
            func_builder = func_list[func_id]
            new_counts_constraints = None if counts_constraints is None else dict(counts_constraints)
            if new_counts_constraints is not None and func_builder.category in new_counts_constraints:
                new_counts_constraints[func_builder.category] -= 1
            func = func_builder()
            for _, p in inspect.signature(func.func).parameters.items():
                if p.default is not inspect.Parameter.empty or p.kind == p.KEYWORD_ONLY or p.kind == p.VAR_KEYWORD:
                    continue
                node = full(full_depth=full_depth - 1, counts_constraints = new_counts_constraints, func_list = func_list, terminal_list = terminal_list, node_builder = node_builder)
                args.append(node)
                if node is None:
                    break 
            if len(args) > 0 and args[-1] is None:
                disallowed_symbols.add(func)
                continue  
            if counts_constraints is not None:
                counts_constraints.update(new_counts_constraints)       
            break  
    return node_builder(func, args)

def ramped_half_and_half(rhh_min_depth = 1, rhh_max_depth = 5, rhh_grow_prob = 0.5, *, 
         counts_constraints: Optional[dict[str, int]] = None, 
         func_list: list[utils.AnnotatedFunc], terminal_list: list[utils.AnnotatedFunc], 
         node_builder: Callable[[utils.AnnotatedFunc, list[Node]], Node]) -> Optional[Node]:                         
    ''' Generate a population of half full and half grow trees '''
    depth = default_rnd.randint(rhh_min_depth, rhh_max_depth+1)
    if default_rnd.rand() < rhh_grow_prob:
        return grow(grow_depth = depth, counts_constraints = counts_constraints, func_list = func_list, terminal_list = terminal_list, node_builder = node_builder)
    else:
        return full(full_depth = depth, counts_constraints = counts_constraints, func_list = func_list, terminal_list = terminal_list, node_builder = node_builder)
    
def init_each(size: int, init_fn = ramped_half_and_half, *, runtime_context: RuntimeContext):
    res = []
    for _ in range(size):
        node = init_fn(counts_constraints = (None if runtime_context.counts_constraints is None else dict(runtime_context.counts_constraints)), 
                    func_list = runtime_context.func_list, 
                    terminal_list = runtime_context.terminal_list,
                    node_builder = runtime_context.node_builder)
        if node is not None:
            res.append(node)
    return res

def init_all(size: int, depth = 3, *, runtime_context: RuntimeContext):
    ''' Generate all possible trees till given depth 
        Very expensive for large depth
    '''    
    # counts_constraints = None, counts_cache, func_list, terminal_list, node_builder
    zero_depth = []
    for terminal_builder in runtime_context.terminal_list:
        node = runtime_context.node_builder(terminal_builder(), [])
        if are_counts_constraints_satisfied(node, runtime_context.counts_constraints, runtime_context.counts_cache):
            zero_depth.append(node)
            size -= 1
            if size <= 0:
                return [x for x, _ in zero_depth]
    trees_by_depth = [zero_depth]
    for _ in range(1, depth + 1):
        depth_trees = []
        for func_builder in runtime_context.func_list:
            args = []
            func = func_builder()
            for _, p in inspect.signature(func.func).parameters.items():
                if p.default is not inspect.Parameter.empty or p.kind == p.KEYWORD_ONLY or p.kind == p.VAR_KEYWORD:
                    continue
                args.append(trees_by_depth[-1])
            for a in product(*args):
                node = runtime_context.node_builder(func, a)
                if are_counts_constraints_satisfied(node, runtime_context.counts_constraints, runtime_context.counts_cache):
                    depth_trees.append(node)
                    size -= 1
                    if size <= 0:
                        trees_by_depth.append(depth_trees)
                        return [t for trees in trees_by_depth for t, _ in trees]
        trees_by_depth.append(depth_trees)
    all_trees = [t for trees in trees_by_depth for t, _ in trees]
    return all_trees
    
def _select_node_id(in_node: Node, filter, select_node_leaf_prob = None) -> Optional[Node]:
    if select_node_leaf_prob is None: 
        places = [(n, i, at_d) for i, (at_d, n) in enumerate(in_node.get_nodes()) if filter(at_d, n) ]
        if len(places) == 0:
            return None, None, None
        selected_idx = default_rnd.choice(len(places))
        selected = places[selected_idx]
    else:
        nonleaves = []
        leaves = []
        for i, (at_d, n) in enumerate(in_node.get_nodes()):
            if filter(at_d, n):
                if n.is_leaf():
                    leaves.append((n, i, at_d))
                else:
                    nonleaves.append((n, i, at_d))
        if len(nonleaves) == 0 and len(leaves) == 0:
            return None, None, None
        if (default_rnd.rand() < select_node_leaf_prob and len(leaves) > 0) or len(nonleaves) == 0:
            selected_idx = default_rnd.choice(len(leaves))
            selected = leaves[selected_idx]
        else:
            selected_idx = default_rnd.choice(len(nonleaves))
            selected = nonleaves[selected_idx]
    return selected

# NOTE: first we do select and then gen muation tree
# TODO: later add to grow and full type constraints on return type
# IDEA: dropout in GP, frozen tree positions which cannot be mutated or crossovered - for later
def subtree_mutation(node, select_node_leaf_prob = 0.1, tree_max_depth = 17, repl_fn = replace_positions, *, runtime_context: RuntimeContext):
    position, position_id, position_depth = _select_node_id(node, lambda d, n: True, select_node_leaf_prob = select_node_leaf_prob)
    if position is None:
        return node    
    position_func_counts = get_func_counts(position, runtime_context.counts_constraints, runtime_context.counts_cache)
    grow_depth = min(5, tree_max_depth - position_depth)
    if runtime_context.counts_constraints is None:
        grow_counts_constraints = None
    else:
        grow_counts_constraints = {}
        for k, v in runtime_context.counts_constraints.items():
            grow_counts_constraints[k] = v - position_func_counts.get(k, 0)
    new_node = grow(grow_depth = grow_depth, func_list = runtime_context.func_list, terminal_list = runtime_context.terminal_list, 
                    counts_constraints = grow_counts_constraints, grow_leaf_prob = None, node_builder = runtime_context.node_builder)
    # new_node_depth = new_node.get_depth()
    # at_depth, at_node = select_node(leaf_prob, node, lambda d, n: (d > 0) and n.is_of_type(new_node), 
    #                                     lambda d, n: (d + new_node_depth) <= max_depth)
    if new_node is None:
        return node
    res = repl_fn(node, {position_id: new_node}, node_builder = runtime_context.node_builder)
    return res

def no_mutation(node):
    return node
        
def subtree_crossover(parent1: Node, parent2: Node, select_node_leaf_prob = 0.1, tree_max_depth = 17, 
                      repl_fn = replace_positions, *, runtime_context: RuntimeContext):
    ''' Crossover two trees '''
    # NOTE: we can crossover root nodes
    # if parent1.get_depth() == 0 or parent2.get_depth() == 0:
    #     return parent1, parent2
    parent1, parent2 = sorted([parent1, parent2], key = lambda x: x.get_depth())
    # for _ in range(3):
    # at1_at_depth, at1 = select_node(leaf_prob, parent1, lambda d, n: (d > 0), lambda d, n: True)
    at1, at1_id, at1_at_depth = _select_node_id(parent1, lambda d, n: True, select_node_leaf_prob=select_node_leaf_prob)
    if at1_id is None:
        return parent1, parent2
    # at1_at_depth, at1 = parent1.get_node(at1_id)
    at1_depth = at1.get_depth()
    at2, at2_id, at2_at_depth = _select_node_id(parent2, 
                        lambda d, n: n.is_of_type(at1) and at1.is_of_type(n) and ((n.get_depth() + at1_at_depth) <= tree_max_depth) and (at1_at_depth > 0 or d > 0) and ((d + at1_depth) <= tree_max_depth) \
                                            and are_counts_constraints_satisfied_together(n, at1, runtime_context.counts_constraints, runtime_context.counts_cache), 
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
    child1 = repl_fn(parent1, {at1_id: at2}, node_builder = runtime_context.node_builder)
    child2 = repl_fn(parent2, {at2_id: at1}, node_builder = runtime_context.node_builder)
    return child1, child2       

def subtree_breed(size, population, fitnesses, interactions,
                    breed_select_fn = tournament_selection, mutation_fn = subtree_mutation, crossover_fn = subtree_crossover,
                    mutation_rate = 0.1, crossover_rate = 0.9, *, runtime_context: RuntimeContext):
    new_population = []
    runtime_context.parent_child_relations = []
    if runtime_context.select_fitness_ids is not None and fitnesses is not None:
        fitnesses = fitnesses[:, runtime_context.select_fitness_ids]
    collect_parents = ("syntax" in runtime_context.breeding_stats) or ("semantics" in runtime_context.breeding_stats)
    all_parents = []
    while len(new_population) < size:
        # Select parents for the next generation
        parent1_id = breed_select_fn(population, fitnesses, interactions, runtime_context = runtime_context)
        parent2_id = breed_select_fn(population, fitnesses, interactions, runtime_context = runtime_context)
        parent1 = population[parent1_id]
        parent2 = population[parent2_id]
        if default_rnd.rand() < mutation_rate:
            child1 = mutation_fn(parent1, runtime_context = runtime_context)
        else:
            child1 = parent1
        if default_rnd.rand() < mutation_rate:
            child2 = mutation_fn(parent2, runtime_context = runtime_context)
        else:
            child2 = parent2
        if default_rnd.rand() < crossover_rate:
            child1, child2 = crossover_fn(child1, child2, runtime_context = runtime_context)   
        runtime_context.parent_child_relations.append(([parent1, parent2], [child1, child2]))
        if collect_parents:
            all_parents.extend((parent1, parent2))
        new_population.extend([child1, child2])
    for parent in all_parents:
        if 'syntax' in runtime_context.breeding_stats:
            runtime_context.breeding_stats['syntax'][parent] = runtime_context.breeding_stats['syntax'].get(parent, 0) + 1 
        if 'semantics' in runtime_context.breeding_stats:
            parent_ints = tuple(runtime_context.int_cache[parent])
            runtime_context.breeding_stats['semantics'][parent_ints] = runtime_context.breeding_stats['semantics'].get(parent_ints, 0) + 1 
    return new_population
    
def depth_fitness(interactions, outputs, population = [], **_):
    return [p.get_depth() for p in population]

def hamming_distance_fitness(interactions, outputs, **_):
    return np.sum(1 - interactions, axis = 1)

from torch.nn.functional import mse_loss
def mse_fitness(interaction, outputs, *, gold_outputs, **_):
    losses = []
    for output in outputs:
        loss = mse_loss(torch.tensor(output), gold_outputs)
        losses.append(loss.item())
    res =  np.array(losses)
    res[np.isnan(res)] = np.inf
    return res

def ifs_fitness(interactions, outputs, **_):
    counts = (np.sum(interactions, axis = 0) * interactions).astype(float)
    counts[counts > 0] = 1.0 / counts[counts > 0]
    ifs = np.sum(counts, axis=1)
    return -ifs

# TODO: age fitness: group of fitness functions 
# aging could be simulated differently 
# most interesting case for us is aging of semantics with number of attempted breedings in coevol
# default aging could consider only syntactic tree and number of  generations it exists in lexicographic tournament selection

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

def collect_additional_stats(stats: dict[str, Any], nodes: list[Node], outputs):
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

def exact_best(fitness_value):
    return fitness_value == 0

def approx_best(fitness_value, epsilon = 1e-6):
    return fitness_value < epsilon

def analyze_population(population, outputs, fitnesses, save_stats = True, best_cond = exact_best, *, runtime_context: RuntimeContext, **_):
    ''' Get the best program in the population '''
    stats = runtime_context.stats
    fitness_order = np.lexsort(fitnesses.T[::-1])
    best_index = fitness_order[0]
    best_fitness = fitnesses[best_index]
    best = population[best_index]
    stats['best'] = str(best)
    is_best = False 
    if (runtime_context.main_fitness_fn is None) and (len(runtime_context.fitness_fns) > 0):
        main_fitness_fn = runtime_context.fitness_fns[0]
    else:
        main_fitness_fn = runtime_context.main_fitness_fn
    for fitness_idx, fitness_fn in enumerate(runtime_context.fitness_fns):
        if fitness_fn == main_fitness_fn:
            is_best = best_cond(best_fitness[fitness_idx])
        stats.setdefault(fitness_fn.__name__, []).append(best_fitness[fitness_idx])
    if save_stats:
        collect_additional_stats(stats, population, outputs)
        total_best_ch = 0
        total_good_ch = 0 
        total_best_dom_ch = 0
        total_good_dom_ch = 0
        total_bad_ch = 0
        for parents, children in runtime_context.parent_child_relations:
            parent_ints = np.array([ runtime_context.int_cache[n] for n in parents ])
            child_ints = np.array([ runtime_context.int_cache[n] for n in children ])
            best_ch, good_ch, best_dom_ch, good_dom_ch, bad_ch = count_good_bad_children(parent_ints, child_ints)
            total_best_ch += best_ch
            total_good_ch += good_ch
            total_best_dom_ch += best_dom_ch
            total_good_dom_ch += good_dom_ch
            total_bad_ch += bad_ch
        # if total_good_ch > 0 or total_bad_ch > 0:
        stats.setdefault('best_children', []).append(total_best_ch)
        stats.setdefault('good_children', []).append(total_good_ch)
        stats.setdefault('best_dom_children', []).append(total_best_dom_ch)
        stats.setdefault('good_dom_children', []).append(total_good_dom_ch)
        stats.setdefault('bad_children', []).append(total_bad_ch)
    if is_best:
        return population[best_index]
    return None

def identity_map(population, **_):
    return population

def syntax_dedupl_map(population, **_):
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
        outputs, fitnesses, interactions, *_ = eval_fn(population)
        best_ind = analyze_pop_fn(population, outputs, fitnesses)
        if best_ind is not None:
            break        
        population = breed_fn(population_size, population, fitnesses, interactions)  
        gen += 1
    
    return best_ind, gen

def create_runtime_context(fitness_fns, main_fitness_fn = None, select_fitness_ids = None, context_class = RuntimeContext, **kwargs):
    runtime_context = context_class(fitness_fns=fitness_fns, main_fitness_fn = main_fitness_fn, select_fitness_ids = select_fitness_ids, **kwargs)
    node_builder = partial(default_node_builder, syntax_cache = runtime_context.syntax_cache, stats = runtime_context.stats)
    runtime_context.update(node_builder = node_builder)
    return runtime_context



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
def koza_evolve(problem_init, *,
                population_size = 1000, max_gens = 100,
                fitness_fns = [hamming_distance_fitness, depth_fitness], main_fitness_fn = hamming_distance_fitness,
                select_fitness_ids = None,
                init_fn = init_each, map_fn = identity_map, breed_fn = subtree_breed, 
                eval_fn = gp_eval, analyze_pop_fn = analyze_population): 
    runtime_context = create_runtime_context(fitness_fns, main_fitness_fn, select_fitness_ids)
    problem_init(runtime_context = runtime_context)
    evo_funcs = [init_fn, map_fn, breed_fn, eval_fn, analyze_pop_fn]
    evo_funcs_bound = [partial(fn, runtime_context = runtime_context) for fn in evo_funcs]
    best_ind, gen = evol_loop(population_size, max_gens, *evo_funcs_bound)
    runtime_context.stats["gen"] = gen 
    runtime_context.stats["best_found"] = best_ind is not None
    return best_ind, runtime_context.stats

# Remove Parsimony pressure
# gp = koza_evolve
gp = partial(koza_evolve, select_fitness_ids = [0])
lexicase = partial(koza_evolve, select_fitness_ids = [0], breed_fn = partial(subtree_breed, breed_select_fn = lexicase_selection))

# ifs = partial(koza_evolve, fitness_fns = [ifs_fitness, hamming_distance_fitness, depth_fitness])
ifs = partial(koza_evolve, fitness_fns = [ifs_fitness, hamming_distance_fitness, depth_fitness], 
                select_fitness_ids = [0, 1])

# gp_a = partial(koza_evolve, fitness_fns = [mse_fitness, depth_fitness], main_fitness_fn = mse_fitness,
#                             eval_fn = partial(gp_eval, int_fn = dist_test_based_interactions, 
#                                               output_prep = torch_output_prep),
#                             analyze_pop_fn = partial(analyze_population, best_cond = approx_best))

gp_sim_names = [ 'gp', 'ifs', 'lexicase' ]

if __name__ == '__main__':
    import gp_benchmarks
    problem_builder = gp_benchmarks.get_benchmark('cmp8')
    best_prog, stats = lexicase(problem_builder)
    print(best_prog)
    print(stats)
    pass    
