


from functools import partial

from domain_alg0 import *
from domain_bool import *


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

all_benchmarks = bool_benchmark + alg0_benchmark

benchmark_map = {name: i for i, (name, _) in enumerate(all_benchmarks) }

def get_benchmark(name_or_id: str | int):
    if isinstance(name_or_id, str):
        game_name, bm_builder = all_benchmarks[benchmark_map[name_or_id]]
    else: 
        game_name, bm_builder = all_benchmarks[name_or_id]
    return game_name, bm_builder()
