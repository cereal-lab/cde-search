''' Simple algebraic domain with values in {0, 1, 2}
    See papers "Genetic Programming for Finite Algebras" of Lee Spector 
    and "Automatic Derivation of Search Objectives for Test-Based Genetic Programming" of Krawiec
'''

from typing import Optional
import numpy as np

from utils import create_named_function

def f_a1(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    outputs = np.array([[2, 1, 2], [1, 0, 0], [0, 0, 1]])
    return outputs[x, y]

def f_a2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    outputs = np.array([[2, 0, 2], [1, 0, 2], [1, 2, 1]])
    return outputs[x, y]

def f_a3(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    outputs = np.array([[1, 0, 1], [1, 2, 0], [0, 0, 0]])
    return outputs[x, y]

def f_a4(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    outputs = np.array([[1, 0, 1], [0, 2, 0], [0, 1, 0]])
    return outputs[x, y]

def f_a5(x: np.ndarray, y: np.ndarray) -> np.ndarray:
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

def build_vars(inputs: np.ndarray):
    res = []
    for i, name in enumerate(['x', 'y', 'z']):
        def v(i=i, inputs = inputs, test_ids: Optional[np.ndarray] = None):
            if test_ids is not None:
                return inputs[i][test_ids]
            return inputs[i]
        res.append(create_named_function(name, v))
    return res 