''' Symbolic regression on boolean domain. Here we define major functions and test cases '''

import numpy as np

from utils import create_named_function

def f_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ''' Logical AND '''
    return a & b

def f_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ''' Logical OR '''
    return a | b

def f_not(a: np.ndarray) -> np.ndarray:
    ''' Logical NOT '''
    return ~a

def f_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ''' Logical XOR '''
    return a ^ b

def f_nand(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ''' Logical NAND '''
    return ~(a & b)

def f_nor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ''' Logical NOR '''
    return ~(a | b)

# ERC - bus line is indexes 
def build_lines(true_table_inputs: np.ndarray):
    lines = []
    true_table_inputs_T = true_table_inputs.T
    for i in range(len(true_table_inputs_T)):
        def x(i=i, true_table_inputs_T = true_table_inputs_T) -> np.ndarray:
            return true_table_inputs_T[i]
        lines.append(create_named_function(f"x{i}", x))
    return lines

def t_true():
    return True

def t_false():
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
    return true_table_inputs, true_table_outputs
        
# cmp(8)        

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
    return true_table_inputs, true_table_outputs

from itertools import product

def mul(address_size: int):
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
    return true_table_inputs, true_table_outputs

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
    return true_table_inputs, true_table_outputs

default_funcs = [f_and, f_or, f_nand, f_nor]
