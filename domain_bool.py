''' Symbolic regression on boolean domain. Here we define major functions and test cases '''

from functools import partial
import numpy as np

from gp import RuntimeContext
from utils import create_free_vars, create_simple_func_builder, new_func
import utils

def f_and(a: np.ndarray, b: np.ndarray, **_) -> np.ndarray:
    ''' Logical AND '''
    return a & b

def f_or(a: np.ndarray, b: np.ndarray, **_) -> np.ndarray:
    ''' Logical OR '''
    return a | b

def f_not(a: np.ndarray, **_) -> np.ndarray:
    ''' Logical NOT '''
    return ~a

def f_xor(a: np.ndarray, b: np.ndarray, **_) -> np.ndarray:
    ''' Logical XOR '''
    return a ^ b

def f_nand(a: np.ndarray, b: np.ndarray, **_) -> np.ndarray:
    ''' Logical NAND '''
    return ~(a & b)

def f_nor(a: np.ndarray, b: np.ndarray, **_) -> np.ndarray:
    ''' Logical NOR '''
    return ~(a | b)

def t_true(**_):
    return True

def t_false(**_):
    return False    

# benchmarks 
def cmp(size: int):
    half_size = size // 2
    true_table_inputs_list = []
    true_table_outputs_list = []
    for i in range(2 ** size):
        upper_bits = i >> half_size
        lower_bits = i & ((1 << half_size) - 1)
        outcome = lower_bits < upper_bits
        true_table_outputs_list.append(outcome)
        true_table_inputs_list.append([(i >> j) & 1 for j in range(size)])
    true_table_inputs = np.array(true_table_inputs_list, dtype=bool)
    true_table_outputs = np.array(true_table_outputs_list)
    return true_table_inputs.T, true_table_outputs
        
# cmp(6)        

def maj(size: int):
    true_table_inputs_list = []
    true_table_outputs_list = []
    for i in range(2 ** size):
        bits = [(i >> j) & 1 for j in range(size)]
        outcome = sum(bits) > size // 2
        true_table_outputs_list.append(outcome)
        true_table_inputs_list.append(bits)
    true_table_inputs = np.array(true_table_inputs_list, dtype=bool)
    true_table_outputs = np.array(true_table_outputs_list)
    return true_table_inputs.T, true_table_outputs

from itertools import product

def mux(address_size: int):
    ''' size is the size of address '''
    true_table_inputs_list = []
    true_table_outputs_list = []
    
    data_bits_size = 2 ** address_size
    for i in range(data_bits_size):
        address_bits = [((i >> j) & 1) == 1 for j in range(address_size)]
        for data_bits in product(*([[True, False]] * data_bits_size)):
            data_bits = list(data_bits)
            bits = address_bits + data_bits
            true_table_inputs_list.append(bits)
            true_table_outputs_list.append(data_bits[i])
    true_table_inputs = np.array(true_table_inputs_list, dtype=bool)
    true_table_outputs = np.array(true_table_outputs_list)
    return true_table_inputs.T, true_table_outputs

# mul(2)

def par(size: int):
    ''' Parity '''
    true_table_inputs_list = []
    true_table_outputs_list = []
    for i in range(2 ** size):
        bits = [(i >> j) & 1 for j in range(size)]
        outcome = sum(bits) % 2 == 1
        true_table_outputs_list.append(outcome)
        true_table_inputs_list.append(bits)
    true_table_inputs = np.array(true_table_inputs_list, dtype=bool)
    true_table_outputs = np.array(true_table_outputs_list)
    return true_table_inputs.T, true_table_outputs

def bool_problem_init(problem_fn, size, funcs = [f_and, f_or, f_nand, f_nor], *, runtime_context: RuntimeContext):
    inputs, outputs = problem_fn(size)    
    terminal_list, free_vars = create_free_vars(inputs, prefix = "x")
    func_list = [create_simple_func_builder(fn) for fn in funcs]
    counts_constraints = None 
    runtime_context.update(gold_outputs = outputs, free_vars = free_vars, 
                           func_list = func_list, terminal_list = terminal_list, 
                           counts_constraints = counts_constraints)

cmp6 = partial(bool_problem_init, cmp, 6)
cmp8 = partial(bool_problem_init, cmp, 8)
maj6 = partial(bool_problem_init, maj, 6)
mux6 = partial(bool_problem_init, mux, 6)
par5 = partial(bool_problem_init, par, 5)

benchmark = {
    "cmp6": cmp6,
    "cmp8": cmp8,
    "maj6": maj6,
    "mux6": mux6,
    "par5": par5
}
