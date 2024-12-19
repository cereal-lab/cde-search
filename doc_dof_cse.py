''' Running of DOC, DOF and CSE on benchmarks from bool and alg0 '''

import numpy as np

from derivedObj import matrix_factorization, xmean_cluster
from nsga2 import run_nsga2, run_front_coverage
from utils import write_metrics
from functools import partial

seed = 117
seed2 = 119

num_runs = 1

from domain_alg0 import build_vars, disc, f_a1, f_a2, f_a3, f_a4, f_a5, malcev
from gp import gp_evaluate, ramped_half_and_half, run_koza, subtree_breed, subtree_mutation, subtree_crossover, tournament_selection
from domain_bool import cmp, maj, mul, par, default_funcs, build_lines

# included setups: GP - classic Koza
#   initialization - ramped half-and-half
#   mutation: subtree replace mutation with 0.1 probability
#   crossover: subtree crossover with 0.9 probability
#   fitness_fn: distance to ideal L1 or L2 norm 

archive_size = 10
max_generations = 100
population_size = 1000

# included setups: NSGA2
#   initialization - ramped half-and-half
#   breed: same as above for GP
#   fitness_fn: is multiobjective and is defined by either clustering or factoriation from interaction semantics

# included setups: CSE - TODO: CSE based evolutionary algo vs NSGA2 

bool_benchmark_problems = [
    ("cmp6", partial(cmp, 6)), ("cmp8", partial(cmp, 8)), ("maj6", partial(maj, 6)), ("mux6", partial(mul, 2)), ("par5", partial(par, 5))
]

def build_bool_bench_problem(problem): 
    inputs, outputs = problem()
    return outputs, default_funcs, build_lines(inputs)

bool_benchmark = [ (name, partial(build_bool_bench_problem, problem)) for name, problem in bool_benchmark_problems]

alg0_benchmark_problems = [ disc, malcev ]

def build_alg0_bench_problem(problem, f_a): 
    inputs, outputs = problem()
    return outputs, [f_a], build_vars(inputs)

alg0_benchmark = [ (problem.__name__ + str(i + 1), partial(build_alg0_bench_problem, problem, f_a)) for problem in alg0_benchmark_problems for i, f_a in enumerate([f_a1, f_a2, f_a3, f_a4, f_a5])]

benchmark = bool_benchmark + alg0_benchmark

def depth_fitness(prev_fitness, interactions, *, population = [], **kwargs):
    for i in range(len(population)):
        prev_fitness[i] = population[i].get_depth()

def hamming_distance_fitness(prev_fitness, interactions, *, eval_mask = None, **kwargs):
    if eval_mask is None:
        eval_mask = np.ones(interactions.shape[0], dtype=bool)
    prev_fitness[eval_mask] = np.sum(1 - interactions[eval_mask], axis = 1)

def ifs_fitness(prev_fitness, interactions, **kwargs):
    counts = np.sum((interactions[:, None] == interactions) & (interactions == 1), axis = 0).astype(float)
    counts[counts > 0] = 1 / counts[counts > 0]
    ifs = np.sum(counts, axis=1)
    prev_fitness[:] = -ifs

# ifs_fitness(0, np.array([[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 1]]))

def hypervolume_fitness(prev_fitness, interactions, *, derived_objectives = [], **kwargs):
    prev_fitness[:] = -np.prod(derived_objectives, axis = 1)

def weighted_hypervolume_fitness(prev_fitness, interactions, *, derived_objectives = [], test_clusters = [], **kwargs):
    weights = np.array([np.sum(interactions[:, tc] == 1, axis=1) for tc in test_clusters]).T
    weighted_objs = weights * derived_objectives
    prev_fitness[:] = np.prod(weighted_objs, axis = 1)

# a = np.array([ 1, 0, 1])
# aa = np.array([[ 1, 1, 1], [0, 0, 1], [1, 0, 1]])    
# np.sum((a == aa)[:, a == 1], axis=0)

def program_test_interactions(outputs, program_outputs):
    return (program_outputs == outputs).astype(int)

def get_metrics(main_fitness_index: int, fitnesses, population):
    ''' Get the best program in the population '''
    fitness_order = np.lexsort(fitnesses.T[::-1])
    best_index = fitness_order[0]
    best_fitness = fitnesses[best_index]
    best = population[best_index]
    is_best = best_fitness[main_fitness_index] == 0
    return is_best, (best, *best_fitness)

# np.lexsort(np.array([np.array([1, 2, 3]), np.array([3, 2, 1])]).T[::-1])

def run_gp_on_benchmarks(sim_name, fitness_fns, get_metrics, idxs = [], metrics_file = "data/metrics/objs.jsonlist"):
    if len(idxs) == 0:
        idxs = range(len(benchmark))
    for idx in idxs:
        name, problem = benchmark[idx]
        outputs, func_list, terminal_list = problem()
        rnd = np.random.RandomState(seed)
        int_fn = partial(program_test_interactions, outputs)
        initialization = partial(ramped_half_and_half, rnd, 1, 5, func_list, terminal_list)
        selection = partial(tournament_selection, rnd, 7)
        mutation = partial(subtree_mutation, rnd, 0.1, 17, func_list, terminal_list)
        crossover = partial(subtree_crossover, rnd, 0.1, 17)
        breed = partial(subtree_breed, rnd, 0.1, 0.9, selection, mutation, crossover, population_size)
        evaluate = partial(gp_evaluate, len(outputs), int_fn, fitness_fns)
        for run_id in range(num_runs):
            stats = run_koza(population_size, max_generations, initialization, breed, evaluate, get_metrics)
            best_inds, *fitness_metrics = zip(*stats)
            metrics = dict(game = name, sim = sim_name, seed = seed, seed2 = seed2, run_id = run_id, best_ind = str(best_inds[-1]), best_ind_depth = best_inds[-1].get_depth())
            for i, metric in enumerate(fitness_metrics):
                metrics["fitness" + str(i)] = metric
            write_metrics(metrics, metrics_file)
            pass

# np.stack([np.array([1, 2, 3]), np.array([3, 2, 1])], axis=0)

def full_objectives(interactions):
    return interactions, dict()

def doc_objectives(interactions):
    test_clusters, centers = xmean_cluster(interactions.T.tolist(), 4)
    derived_objectives = np.array(centers).T
    return derived_objectives, dict(test_clusters = test_clusters)

rnd2 = np.random.RandomState(seed2)
def rand_objectives(interactions):
    rand_ints = rnd2.random(interactions.shape)
    res = doc_objectives(rand_ints)
    return res

def dof_w_objectives(k, alpha, interactions):
    if alpha < 1: 
        num_columns_to_take = int(alpha * interactions.shape[1])
        random_column_indexes = rnd2.choice(interactions.shape[1], num_columns_to_take, replace = False)
        interactions = interactions[:, random_column_indexes]
    W, _, _ = matrix_factorization(interactions.tolist(), k)
    return W, {}

def dof_wh_objectives(k, alpha, interactions):
    if alpha < 1: 
        num_columns_to_take = int(alpha * interactions.shape[1])
        random_column_indexes = rnd2.choice(interactions.shape[1], num_columns_to_take, replace = False)
        interactions = interactions[:, random_column_indexes]    
    W, H, _ = matrix_factorization(interactions.tolist(), k)
    objs = np.sum(W[:, :, None] * H, axis = -1)
    return objs, {}

def run_do_on_benchmarks(sim_name, fitness_fns, get_metrics, derive_objectives, idxs = [], metrics_file = "data/metrics/objs.jsonlist"):
    if len(idxs) == 0:
        idxs = range(len(benchmark))
    for idx in idxs:
        name, problem = benchmark[idx]
        outputs, func_list, terminal_list = problem()
        rnd = np.random.RandomState(seed)
        int_fn = partial(program_test_interactions, outputs)
        initialization = partial(ramped_half_and_half, rnd, 1, 5, func_list, terminal_list)
        selection = partial(tournament_selection, rnd, 7)
        mutation = partial(subtree_mutation, rnd, 0.1, 17, func_list, terminal_list)
        crossover = partial(subtree_crossover, rnd, 0.1, 17)
        breed = partial(subtree_breed, rnd, 0.1, 0.9, selection, mutation, crossover, population_size)
        evaluate = partial(gp_evaluate, len(outputs), int_fn, fitness_fns)
        for run_id in range(num_runs):
            stats = run_nsga2(archive_size, population_size, max_generations, initialization, breed, derive_objectives, evaluate, get_metrics)
            best_inds, *fitness_metrics = zip(*stats)
            metrics = dict(game = name, sim = sim_name, seed = seed, seed2 = seed2, run_id = run_id, best_ind = best_inds[-1])
            for i, metric in enumerate(fitness_metrics):
                metrics["fitness" + str(i)] = metric
            write_metrics(metrics, metrics_file)

def select_full_test_coverage(test_selection, selection_size, interactions):
    ''' There are different ways to have full test coverage    
    Hypothesis 1. selects hardest test and then best program that solves it 
    Hypothesis 2. selects easiest test and then best program that solves it
    Hypothesis 3. selects random test and then best program that solves it'''
    ints = np.copy(interactions)    
    # non_zero_test_ids  = np.where(test_stats != 0)[0]
    selected = []
    while len(selected) < selection_size:
        test_stats = np.sum(ints, axis = 0) #sum accross columns to see how many programs pass each test    
        non_zero_test_ids = np.where(test_stats != 0)[0]
        if len(non_zero_test_ids) == 0:
            break 
        non_zero_tests = test_stats[non_zero_test_ids]
        test_with_min_programs_id_id = test_selection(non_zero_tests)
        test_with_min_programs_id = non_zero_test_ids[test_with_min_programs_id_id]
        program_candidate_ids = np.where(ints[:, test_with_min_programs_id] != 0)[0]
        program_stats = np.sum(ints[program_candidate_ids], axis = 1)
        # non_zero_program_ids = np.where(ints[:, test_with_min_programs_id] == 1)[0]
        best_program_id_id = np.argmax(program_stats)
        best_program_id = program_candidate_ids[best_program_id_id] #the program that passes the most tests including selected rare test
        selected.append(best_program_id)
        ints[best_program_id] = 0
        ints[:, test_with_min_programs_id] = 0
    return np.array(selected)

def select_rand_coverage(rnd, selection_size, interactions): 
    ''' Randomly samples programs from interactions '''
    program_ids = rnd.choice(interactions.shape[0], selection_size, replace = False)
    return program_ids

def select_pair_coverage(rnd, select_kind, selection_size, interactions):
    ''' Picks two tests to be solved in parent programs combination 
        There could be many ways to pick a pair, we have some hypothesis but need to check different pair selections
        Hypothesis 1: tests that are easily solvable is easy to recombine
        Hypothesis 2: solving first pairs of most difficult tests gives better convergence speed 
        Hypothesis 3: random pair selection is good enough.
        NOTE: We need to measure average time to recombine targeted test pair         
    '''
    ints = np.copy(interactions)
    selected = []
    while len(selected) < selection_size:
        test_stats = np.sum(ints, axis = 0)
        non_zero_test_ids = np.where(test_stats != 0)[0]
        if len(non_zero_test_ids) == 0:
            break 
        if len(non_zero_test_ids) == 1:
            new_program_selection_ids = np.where(ints[:, non_zero_test_ids[0]] != 0)[0]
            new_program_selection = rnd.choice(new_program_selection_ids, min(len(new_program_selection_ids), selection_size - len(selected)), replace = False)   
            selected.extend(new_program_selection)
            break 
        non_zero_tests = test_stats[non_zero_test_ids]
        test_sorting = np.argsort(non_zero_tests)
        if select_kind == 0:
            # pick easiest 2 tests
            test1, test2 = non_zero_test_ids[test_sorting[-2:]].tolist()
            to_select 
        # test_with_min_programs_id_id = test_selection(non_zero_tests)
        
    return np.array(selected)

# import numpy as np

# a = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1]])
# t = np.sum(a, axis = 0)
    


def run_coverage_on_benchmarks(sim_name, fitness_fns, get_metrics, idxs = [], metrics_file = "data/metrics/objs.jsonlist"):
    if len(idxs) == 0:
        idxs = range(len(benchmark))
    for idx in idxs:
        name, problem = benchmark[idx]
        outputs, func_list, terminal_list = problem()
        rnd = np.random.RandomState(seed)
        int_fn = partial(program_test_interactions, outputs)    
        initialization = partial(ramped_half_and_half, rnd, 1, 5, func_list, terminal_list)
        selection = partial(tournament_selection, rnd, 7)
        mutation = partial(subtree_mutation, rnd, 0.1, 17, func_list, terminal_list)
        crossover = partial(subtree_crossover, rnd, 0.1, 17)
        breed = partial(subtree_breed, rnd, 0.1, 0.9, selection, mutation, crossover)
        evaluate = partial(gp_evaluate, len(outputs), int_fn, fitness_fns)
        select_parents = partial(select_full_test_coverage, archive_size)
        for run_id in range(num_runs):
            stats = run_front_coverage(archive_size, population_size, max_generations, initialization, breed, select_parents, evaluate, get_metrics)
            best_inds, *fitness_metrics = zip(*stats)
            metrics = dict(game = name, sim = sim_name, seed = seed, seed2 = seed2, run_id = run_id, best_ind = best_inds[-1])
            for i, metric in enumerate(fitness_metrics):
                metrics["fitness" + str(i)] = metric
            write_metrics(metrics, metrics_file)

gp = partial(run_gp_on_benchmarks, "gp", [hamming_distance_fitness, depth_fitness], partial(get_metrics, 0))
ifs = partial(run_gp_on_benchmarks, "ifs", [ifs_fitness, hamming_distance_fitness, depth_fitness], partial(get_metrics, 1))
rand = partial(run_do_on_benchmarks, "rand", [hamming_distance_fitness, depth_fitness], partial(get_metrics, 0), rand_objectives)
nsga = partial(run_do_on_benchmarks, "nsga", [hamming_distance_fitness, depth_fitness], partial(get_metrics, 0), full_objectives)
doc = partial(run_do_on_benchmarks, "doc", [hamming_distance_fitness, depth_fitness], partial(get_metrics, 0), doc_objectives)
doc_p = partial(run_do_on_benchmarks, "doc_p", [hypervolume_fitness, hamming_distance_fitness, depth_fitness], partial(get_metrics, 1), doc_objectives)
doc_d = partial(run_do_on_benchmarks, "doc_d", [weighted_hypervolume_fitness, hamming_distance_fitness, depth_fitness], partial(get_metrics, 1), doc_objectives)
dof_w_2 = partial(run_do_on_benchmarks, "doc_w_2", [hamming_distance_fitness, depth_fitness], partial(get_metrics, 0), partial(dof_w_objectives, 2, 1))
dof_w_3 = partial(run_do_on_benchmarks, "doc_w_3", [hamming_distance_fitness, depth_fitness], partial(get_metrics, 0), partial(dof_w_objectives, 3, 1))
dof_wh_2 = partial(run_do_on_benchmarks, "doc_wh_2", [hamming_distance_fitness, depth_fitness], partial(get_metrics, 0), partial(dof_wh_objectives, 2, 1))
dof_wh_3 = partial(run_do_on_benchmarks, "doc_wh_3", [hamming_distance_fitness, depth_fitness], partial(get_metrics, 0), partial(dof_wh_objectives, 3, 1))
dof_w_2_80 = partial(run_do_on_benchmarks, "doc_w_2_80", [hamming_distance_fitness, depth_fitness], partial(get_metrics, 0), partial(dof_w_objectives, 2, 0.8))
dof_w_3_80 = partial(run_do_on_benchmarks, "doc_w_3_80", [hamming_distance_fitness, depth_fitness], partial(get_metrics, 0), partial(dof_w_objectives, 3, 0.8))
dof_wh_2_80 = partial(run_do_on_benchmarks, "doc_wh_2_80", [hamming_distance_fitness, depth_fitness], partial(get_metrics, 0), partial(dof_wh_objectives, 2, 0.8))
dof_wh_3_80 = partial(run_do_on_benchmarks, "doc_wh_3_80", [hamming_distance_fitness, depth_fitness], partial(get_metrics, 0), partial(dof_wh_objectives, 3, 0.8))
# cse = partial(run_cse_on_benchmarks, "cse", [hamming_distance_fitness, depth_fitness], partial(get_metrics, 0))       

if __name__ == "__main__":
    print("testing evo runs")
    # gp(idxs=[0])
    cse(idxs=[0])
    pass