''' Running of DOC, DOF and CSE on benchmarks from bool and alg0 '''

import numpy as np

from de import run_cse
from derivedObj import matrix_factorization, xmean_cluster
from nsga2 import run_nsga2
from utils import write_metrics
from functools import partial

seed = 117
seed2 = 119

num_runs = 1

from domain_alg0 import build_vars, disc, f_a1, f_a2, f_a3, f_a4, f_a5, malcev
from gp import Node, ramped_half_and_half, run_koza, subtree_breed, subtree_mutation, subtree_crossover, tournament_selection
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

def hamming_distance_fitness(i, interactions, *, prev_fitness = None, **kwargs):
    if prev_fitness is not None:
        return prev_fitness
    return (np.sum(1 - interactions[i]), )

def ifs_fitness(i, interactions, *, prev_fitness = None, **kwargs):
    program_interactions = interactions[i]
    programs_solving_same_tests = (program_interactions == interactions)[:, program_interactions == 1]
    num_programs_solving_same_tests = np.sum(programs_solving_same_tests, axis = 0)
    ifs = np.sum(1 / num_programs_solving_same_tests)
    if prev_fitness is not None:
        hamming = prev_fitness[-1]
    else:
        hamming = hamming_distance_fitness(i, interactions)[0]
    return (-ifs, hamming)

# ifs_fitness(0, np.array([[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 1]]))

def hypervolume_fitness(i, interactions, *, derived_objectives = [], prev_fitness = None, **kwargs):
    if prev_fitness is not None:
        hamming = prev_fitness[-1]
    else:
        hamming = hamming_distance_fitness(i, interactions)[0]
    return (np.prod(derived_objectives[i]), hamming)

def weighted_hypervolume_fitness(i, interactions, *, derived_objectives = [], test_clusters = [], prev_fitness = None, **kwargs):
    weights = np.array([len(tc) for tc in test_clusters])
    weighted_objs = weights * derived_objectives[i]
    if prev_fitness is not None:
        hamming = prev_fitness[-1]
    else:
        hamming = hamming_distance_fitness(i, interactions)[0]
    return (np.prod(weighted_objs), hamming)

# a = np.array([ 1, 0, 1])
# aa = np.array([[ 1, 1, 1], [0, 0, 1], [1, 0, 1]])    
# np.sum((a == aa)[:, a == 1], axis=0)

def program_test_interactions(outputs, program):
    return (program() == outputs).astype(int)

def get_metrics(population):
    ''' Get the best program in the population '''
    best_index = min(range(len(population)), key=lambda i: (*population[i].fitness, population[i].get_depth()))
    best = population[best_index]
    best_fitness = best.fitness
    is_best = best_fitness[-1] == 0
    return is_best, (best, *best_fitness)

def run_gp_on_benchmarks(sim_name, fitness_fn, idxs = [], metrics_file = "data/metrics/objs.jsonlist"):
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
        for run_id in range(num_runs):
            stats = run_koza(population_size, max_generations, initialization, breed, int_fn, fitness_fn, get_metrics)
            best_inds, *fitness_metrics = zip(*stats)
            metrics = dict(game = name, sim = sim_name, seed = seed, seed2 = seed2, run_id = run_id, best_ind = str(best_inds[-1]), best_ind_depth = best_inds[-1].get_depth())
            for i, metric in enumerate(fitness_metrics):
                metrics["fitness" + str(i)] = metric
            write_metrics(metrics, metrics_file)
            pass

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

# def get_nsga_metrics(pareto_front, pareto_front_derived_objs):
#     metrics = get_metrics(pareto_front)
#     return metrics
        
def run_do_on_benchmarks(sim_name, fitness_fn, derive_objectives, idxs = [], metrics_file = "data/metrics/objs.jsonlist"):
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
        metrics = []
        for run_id in range(num_runs):
            stats = run_nsga2(archive_size, population_size, max_generations, initialization, breed, derive_objectives, int_fn, fitness_fn, get_metrics)
            best_inds, *fitness_metrics = zip(*stats)
            metrics = dict(game = name, sim = sim_name, seed = seed, seed2 = seed2, run_id = run_id, best_ind = best_inds[-1])
            for i, metric in enumerate(fitness_metrics):
                metrics["fitness" + str(i)] = metric
            write_metrics(metrics, metrics_file)

def run_cse_on_benchmarks(sim_name, fitness_fn, idxs = [], metrics_file = "data/metrics/objs.jsonlist"):
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
        metrics = []
        for run_id in range(num_runs):
            stats = run_cse(archive_size, population_size, max_generations, initialization, breed, int_fn, fitness_fn, get_metrics)
            best_inds, *fitness_metrics = zip(*stats)
            metrics = dict(game = name, sim = sim_name, seed = seed, seed2 = seed2, run_id = run_id, best_ind = best_inds[-1])
            for i, metric in enumerate(fitness_metrics):
                metrics["fitness" + str(i)] = metric
            write_metrics(metrics, metrics_file)

gp = partial(run_gp_on_benchmarks, "gp", hamming_distance_fitness)
ifs = partial(run_gp_on_benchmarks, "ifs", ifs_fitness)
rand = partial(run_do_on_benchmarks, "rand", hamming_distance_fitness, rand_objectives)
doc = partial(run_do_on_benchmarks, "doc", hamming_distance_fitness, doc_objectives)
doc_p = partial(run_do_on_benchmarks, "doc_p", hypervolume_fitness, doc_objectives)
doc_d = partial(run_do_on_benchmarks, "doc_d", weighted_hypervolume_fitness, doc_objectives)
dof_w_2 = partial(run_do_on_benchmarks, "doc_w_2", hamming_distance_fitness, partial(dof_w_objectives, 2, 1))
dof_w_3 = partial(run_do_on_benchmarks, "doc_w_3", hamming_distance_fitness, partial(dof_w_objectives, 3, 1))
dof_wh_2 = partial(run_do_on_benchmarks, "doc_wh_2", hamming_distance_fitness, partial(dof_wh_objectives, 2, 1))
dof_wh_3 = partial(run_do_on_benchmarks, "doc_wh_3", hamming_distance_fitness, partial(dof_wh_objectives, 3, 1))
dof_w_2_80 = partial(run_do_on_benchmarks, "doc_w_2_80", hamming_distance_fitness, partial(dof_w_objectives, 2, 0.8))
dof_w_3_80 = partial(run_do_on_benchmarks, "doc_w_3_80", hamming_distance_fitness, partial(dof_w_objectives, 3, 0.8))
dof_wh_2_80 = partial(run_do_on_benchmarks, "doc_wh_2_80", hamming_distance_fitness, partial(dof_wh_objectives, 2, 0.8))
dof_wh_3_80 = partial(run_do_on_benchmarks, "doc_wh_3_80", hamming_distance_fitness, partial(dof_wh_objectives, 3, 0.8))
cse = partial(run_cse_on_benchmarks, "cse", hamming_distance_fitness)       

if __name__ == "__main__":
    print("testing evo runs")
    # gp(idxs=[0])
    ifs(idxs=[0])
    pass