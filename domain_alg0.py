''' Simple algebraic domain with values in {0, 1, 2}
    See papers "Genetic Programming for Finite Algebras" of Lee Spector 
    and "Automatic Derivation of Search Objectives for Test-Based Genetic Programming" of Krawiec
'''

from functools import partial
import numpy as np

from gp import RuntimeContext
from utils import create_free_vars, create_simple_func_builder, new_func
import utils

def f_a1(x: np.ndarray, y: np.ndarray, **_) -> np.ndarray:
    outputs = np.array([[2, 1, 2], [1, 0, 0], [0, 0, 1]])
    return outputs[x, y]

def f_a2(x: np.ndarray, y: np.ndarray, **_) -> np.ndarray:
    outputs = np.array([[2, 0, 2], [1, 0, 2], [1, 2, 1]])
    return outputs[x, y]

def f_a3(x: np.ndarray, y: np.ndarray, **_) -> np.ndarray:
    outputs = np.array([[1, 0, 1], [1, 2, 0], [0, 0, 0]])
    return outputs[x, y]

def f_a4(x: np.ndarray, y: np.ndarray, **_) -> np.ndarray:
    outputs = np.array([[1, 0, 1], [0, 2, 0], [0, 1, 0]])
    return outputs[x, y]

def f_a5(x: np.ndarray, y: np.ndarray, **_) -> np.ndarray:
    outputs = np.array([[1, 0, 2], [1, 2, 0], [0, 1, 0]])
    return outputs[x, y]    

from itertools import product

def disc():
    x = y = z = [0, 1, 2]
    inputs = np.array(list(product(x, y, z)))
    inputs_T = inputs.T
    outputs = np.where(inputs_T[0] != inputs_T[1], inputs_T[0], inputs_T[2])
    return inputs_T, outputs

# disc()

def malcev():
    inputs = []
    outputs = []
    for x, y, z in product(*([[0, 1, 2]] * 3)):
        if x == y: 
            inputs.append([x, y, z])
            outputs.append(z)
        elif y == z:
            inputs.append([x, y, z])
            outputs.append(x)
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    return inputs.T, outputs

def alg0_problem_init(problem_fn, func_list, *, runtime_context: RuntimeContext):
    inputs, outputs = problem_fn()
    terminal_list, free_vars = create_free_vars(inputs, prefix = "x")
    func_list = [create_simple_func_builder(fn) for fn in func_list]
    counts_constraints = None
    runtime_context.update(gold_outputs = outputs, free_vars = free_vars, 
                           func_list = func_list, terminal_list = terminal_list, 
                           counts_constraints = counts_constraints)

disc1 = partial(alg0_problem_init, disc, [f_a1])
disc2 = partial(alg0_problem_init, disc, [f_a2])
disc3 = partial(alg0_problem_init, disc, [f_a3])
disc4 = partial(alg0_problem_init, disc, [f_a4])
disc5 = partial(alg0_problem_init, disc, [f_a5])

malcev1 = partial(alg0_problem_init, malcev, [f_a1])
malcev2 = partial(alg0_problem_init, malcev, [f_a2])
malcev3 = partial(alg0_problem_init, malcev, [f_a3])
malcev4 = partial(alg0_problem_init, malcev, [f_a4])
malcev5 = partial(alg0_problem_init, malcev, [f_a5])

benchmark = {
    'disc1': disc1,
    'disc2': disc2,
    'disc3': disc3,
    'disc4': disc4,
    'disc5': disc5,
    'malcev1': malcev1,
    'malcev2': malcev2,
    'malcev3': malcev3,
    'malcev4': malcev4,
    'malcev5': malcev5
}