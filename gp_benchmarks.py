

from domain_bool import benchmark as bool_benchmark
from domain_alg0 import benchmark as alg0_benchmark
from domain_alg_torch import benchmark as alg_benchmark

def get_benchmark(name):
    problem_builder = bool_benchmark.get(name, None) or alg0_benchmark.get(name, None) or alg_benchmark.get(name, None)
    return problem_builder

def all_discrete_benchmarks():
    return {**bool_benchmark, **alg0_benchmark}