''' Continuous algebraic domain '''

from itertools import product
from typing import Optional
import numpy as np
from rnd import default_rnd

def f_add(x: np.ndarray, y: np.ndarray, **_) -> np.ndarray:
    return x + y

def f_sub(x: np.ndarray, y: np.ndarray, **_) -> np.ndarray:
    return x - y

def f_mul(x: np.ndarray, y: np.ndarray, **_) -> np.ndarray:
    return x * y

# NOTE: we do not apply sanitization - resuls could have inf or nan - should be handled  
def f_div(x: np.ndarray, y: np.ndarray, **_) -> np.ndarray:
    return x / y

def f_neg(x: np.ndarray, **_) -> np.ndarray:
    return -x

# NOTE: we do not apply sanitization - resuls could have inf or nan - should be handled  
def f_inv(x: np.ndarray, **_) -> np.ndarray:
    return 1 / x

def f_cos(x: np.ndarray, **_) -> np.ndarray:
    return np.cos(x)

def f_sin(x: np.ndarray, **_) -> np.ndarray:
    return np.sin(x)

def f_square(x: np.ndarray, **_) -> np.ndarray:
    return x ** 2
    
def f_cube(x: np.ndarray, **_) -> np.ndarray:
    return x ** 3

def f_exp(x: np.ndarray, **_) -> np.ndarray:
    return np.exp(x)

# NOTE: we do not apply sanitization - resuls could have inf or nan - should be handled  
def f_log(x: np.ndarray, **_) -> np.ndarray:
    return np.log(x)

# additional less used 
def f_neg_exp(x: np.ndarray, **_) -> np.ndarray:
    return np.exp(-x)

# NOTE: we do not apply sanitization 
def f_sqrt(x: np.ndarray, **_) -> np.ndarray:
    return np.sqrt(x)

def f_tanh(x: np.ndarray, **_) -> np.ndarray:
    return np.tanh(x)

def f_tan(x: np.ndarray, **_) -> np.ndarray:
    return np.tan(x)

# benchmarks

def koza_1(x:np.ndarray) -> np.ndarray:
    return x*x*x*x + x*x*x + x*x + x

def koza_2(x:np.ndarray) -> np.ndarray:
    return x*x*x*x*x - 2.0*x*x*x + x

def koza_3(x:np.ndarray) -> np.ndarray:
    return x*x*x*x*x*x - 2.0*x*x*x*x + x*x

def nguyen_1(x:np.ndarray) -> np.ndarray:
    return x*x*x + x*x + x

def nguyen_2(x:np.ndarray) -> np.ndarray:
    return x*x*x*x + x*x*x + x*x + x

def nguyen_3(x:np.ndarray) -> np.ndarray:
    return x*x*x*x*x + x*x*x*x + x*x*x + x*x + x

def nguyen_4(x:np.ndarray) -> np.ndarray:
    return x*x*x*x*x*x + x*x*x*x*x + x*x*x*x + x*x*x + x*x + x

def nguyen_5(x:np.ndarray) -> np.ndarray:
    return np.sin(x*x) * np.cos(x) - 1.0

def nguyen_6(x:np.ndarray) -> np.ndarray:
    return np.sin(x) + np.sin(x + x*x)

def nguyen_7(x:np.ndarray) -> np.ndarray:
    return np.log(x + 1.0) + np.log(x*x + 1.0)

def nguyen_8(x:np.ndarray) -> np.ndarray:
    return np.sqrt(x)

def nguyen_9(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return np.sin(x) + np.sin(y * y)

def nguyen_10(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return 2.0 * np.sin(x) + np.cos(y)

def pagie_1(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + 1.0 / (x * x * x * x)) + 1.0 / (1.0 + 1.0 / (y * y * y * y))

def pagie_2(x:np.ndarray, y:np.ndarray, z:np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + 1.0 / ( x * x * x * x)) + 1.0 / (1.0 + 1.0 / (y * y * y * y)) + 1.0 / (1.0 + 1.0 / (z * z * z * z))

def korns_1(*xs: list[np.ndarray]) -> np.ndarray:
    return 1.57 + (24.3 * xs[3])

def korns_2(*xs: list[np.ndarray]) -> np.ndarray:
    return 0.23 + (14.2 * ((xs[3]+ xs[1])/(3.0 * xs[4])))

def korns_3(*xs: list[np.ndarray]) -> np.ndarray:
    return -5.41 + (4.9 * (((xs[3] - xs[0]) + (xs[1]/xs[4])) / (3 * xs[4])))

def korns_4(*xs: list[np.ndarray]) -> np.ndarray:
    return -2.3 + (0.13 * np.sin(xs[2]))

def korns_5(*xs: list[np.ndarray]) -> np.ndarray:
    return 3.0 + (2.13 * np.log(xs[4]))

def korns_6(*xs: list[np.ndarray]) -> np.ndarray:
    return 1.3 + (0.13 * np.sqrt(xs[0]))

def korns_7(*xs: list[np.ndarray]) -> np.ndarray:
    return 213.80940889 - (213.80940889 * np.exp(-0.54723748542 * xs[0]))

def korns_8(*xs: list[np.ndarray]) -> np.ndarray:
    return 6.87 + (11.0 * np.sqrt(7.23 * xs[0] * xs[3] * xs[4]))

def korns_9(*xs: list[np.ndarray]) -> np.ndarray:
    return np.sqrt(xs[0]) / np.log(xs[1]) * np.exp(xs[2]) / (xs[3] * xs[3])

def korns_10(*xs: list[np.ndarray]) -> np.ndarray:
    return 0.81 + (24.3 * (((2.0 * xs[1]) + (3.0 * (xs[2] * xs[2]))) / ((4.0 * (xs[3]*xs[3]*xs[3])) + (5.0 * (xs[4]*xs[4]*xs[4]*xs[4])))))

def korns_11(*xs: list[np.ndarray]) -> np.ndarray:
    return 6.87 + (11.0 * np.cos(7.23 * xs[0]*xs[0]*xs[0]))

def korns_12(*xs: list[np.ndarray]) -> np.ndarray:
    return 2.0 - (2.1 * (np.cos(9.8 * xs[0]) * np.sin(1.3 * xs[4])))

def korns_13(*xs: list[np.ndarray]) -> np.ndarray:
    return 32.0 - (3.0 * ((np.tan(xs[0]) / np.tan(xs[1])) * (np.tan(xs[2])/np.tan(xs[3]))))

def korns_14(*xs: list[np.ndarray]) -> np.ndarray:
    return 22.0 - (4.2 * ((np.cos(xs[0]) - np.tan(xs[1]))*(np.tanh(xs[2])/np.sin(xs[3]))))

def korns_15(*xs: list[np.ndarray]) -> np.ndarray:
    return 12.0 - (6.0 * ((np.tan(xs[0])/np.exp(xs[1])) * (np.log(xs[2]) - np.tan(xs[3]))))

def keijzer_1(x:np.ndarray) -> np.ndarray:
    return 0.3 * x * np.sin(2.0 * np.pi * x)

# NOTE: keijzer_2 == keijzer_3 == keijzer_1

def keijzer_4(x:np.ndarray) -> np.ndarray:
    return x*x*x * np.exp(-x)*np.cos(x)*np.sin(x)* (np.sin(x)*np.sin(x)*np.cos(x) - 1)

def keijzer_5(x:np.ndarray, y:np.ndarray, z:np.ndarray) -> np.ndarray:
    return (30.0 * x * z) / ((x - 10.0) * y * y)

def keijzer_6(x:np.ndarray) -> np.ndarray:
    fl = np.array([np.sum(1.0 / np.arange(1, np.floor(xi) + 1)) for xi in x])
    return fl

def keijzer_7(x:np.ndarray) -> np.ndarray:
    return np.log(x)

def keijzer_8(x:np.ndarray) -> np.ndarray:
    return np.sqrt(x)

def keijzer_9(x:np.ndarray) -> np.ndarray:
    return np.arcsinh(x)

def keijzer_10(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return np.power(x, y)

def keijzer_11(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return x * y + np.sin((x - 1.0) * (y - 1.0))

def keijzer_12(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return x*x*x*x - x*x*x + y*y/2.0 - y

def keijzer_13(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return 6.0 * np.sin(x) * np.cos(y)

def keijzer_14(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return 8.0 / (2.0 + x*x + y*y)

def keijzer_15(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return x*x*x / 5.0 + y*y*y/2.0 - y - x

def vladislavleva_1(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return np.exp(-(x-1)*(x-1)) / (1.2 + (y - 2.5)*(y-2.5))

def vladislavleva_2(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return np.exp(-x)*x*x*x*np.cos(x)*np.sin(x)*(np.cos(x)*np.sin(x)*np.sin(x) - 1)

def vladislavleva_3(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return np.exp(-x)*x*x*x*np.cos(x)*np.sin(x)*(np.cos(x)*np.sin(x)*np.sin(x) - 1) * (y - 5)

def vladislavleva_4(*xs: list[np.ndarray]) -> np.ndarray:
    return 10.0 / (5.0 + np.sum((xs - 3.0) ** 2, axis=0))

def vladislavleva_5(x:np.ndarray, y:np.ndarray, z:np.ndarray) -> np.ndarray:
    return (30.0 * (x - 1.0) * (z - 1.0)) / (y * y * (x - 10.0))

def vladislavleva_6(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return 6.0 * np.sin(x) * np.cos(y)

def vladislavleva_7(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return (x - 3.0) * (y - 3.0) + 2 * np.sin((x - 4.0) * (y - 4.0))

def vladislavleva_8(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return ((x - 3.0) * (x - 3.0) * (x - 3.0) * (x - 3.0) + (y - 3.0) * (y - 3.0) * (y - 3.0) - (y - 3.0)) / ((y - 2.0) * (y - 2.0) * (y - 2.0) * (y - 2.0) + 10.0)

# benchmark meta data (num of vars, ranges and samplings)

# class Benchmark():
#     def __init__(self, gold_fn, free_var_ranges: np.ndarray, train_sampling, test_sampling):
#         self.fn = gold_fn
#         self.free_var_ranges = free_var_ranges
#         self.train_sampling = train_sampling
#         self.test_sampling = test_sampling

def rand_sampling(num_samples, free_var_ranges: np.ndarray, gold_fn):
    mins = np.array([mi for mi, _ in free_var_ranges])
    maxs = np.array([ma for _, ma in free_var_ranges])
    dist = maxs - mins
    inputs = [x for x in mins[:, np.newaxis] + dist[:, torch.newaxis] * default_rnd.rand(len(free_var_ranges), num_samples)]
    outputs = gold_fn(*inputs)
    return inputs, outputs

# rand_sampling(100, [[0.0, 1.0], [0.0, 1.0]], lambda x, y: torch.sin(x) + torch.cos(y))

def interval_samling(step, free_var_ranges: list[list[int]], gold_fn, deltas: Optional[list[int]] = None, rand_deltas = False):
    mins = np.array([mi for mi, _ in free_var_ranges])
    maxs = np.array([ma for _, ma in free_var_ranges])
    if type(step) == list:
        assert len(step) == len(free_var_ranges)
    else:
        step = [step] * len(free_var_ranges)
    step = np.array(step)
    if deltas is not None:
        deltas = np.array(deltas)
    if rand_deltas:
        if deltas is None:
            deltas = step * default_rnd.rand(free_var_ranges.shape[0])
        else:
            deltas = deltas * default_rnd.rand(free_var_ranges.shape[0])
    if deltas is not None:
        # deltas = torch.zeros_like(mins)
        mins += deltas
    
    mesh = list(product(*(np.arange(mi.item(), ma.item(), s.item()).tolist() for mi, ma, s in zip(mins, maxs, step))))
    inputs = [x for x in np.array(mesh).T]
    outputs = gold_fn(*inputs)
    return inputs, outputs

# interval_samling(0.1, np.array([[0.0, 1.0], [0.0, 1.0]]), lambda x, y: torch.sin(x) + torch.cos(y), deltas=[0.05, 0.01])

# https://en.wikipedia.org/wiki/Chebyshev_nodes
def chebyshev_sampling(num_samples, free_var_ranges: np.ndarray, gold_fn, rand_deltas = False):
    mins = free_var_ranges[:, 0]
    maxs = free_var_ranges[:, 1]
    dist = maxs - mins
    indexes = np.tile(np.arange(0, num_samples), (free_var_ranges.shape[0], 1))
    if rand_deltas:
        deltas = default_rnd.rand(free_var_ranges.shape[0])
    else:
        deltas = np.zeros(free_var_ranges.shape[0], dtype=free_var_ranges.dtype)
    indexes = indexes + deltas[:, np.newaxis]
    index_vs = np.cos((2.0 * indexes - 1) / (2.0 * num_samples) * np.pi)
    inputs = (maxs[:, np.newaxis] + mins[:, np.newaxis]) / 2 + dist[:, np.newaxis] / 2 * index_vs
    # inputs = np.array([0.5 * (mi + ma) + 0.5 * dist * np.cos((2 * i + 1) * np.pi / (2 * num_samples)) for i, mi, ma in zip(range(num_samples), mins, maxs)])
    outputs = gold_fn(*inputs)
    return inputs, outputs

# chebyshev_sampling(10, np.array([[0, 1], [0, 1]]), lambda x, y: np.sin(x) + np.cos(y), rand_deltas=True)

# values: (gold_fn, train set settings, test set settings (None if test set == train set))
benchmarks = {
    "koza_1":           (koza_1,            ([[-1.0, 1.0]], 20, rand_sampling), # train set 
                                            None),                              # test set, if None - train set isi suggested
    "koza_2":           (koza_2,            ([[-1.0, 1.0]], 20, rand_sampling),
                                            None),
    "koza_3":           (koza_3,            ([[-1.0, 1.0]], 20, rand_sampling),
                                            None),
    "nguyen_1":         (nguyen_1,          ([[-1.0, 1.0]], 20, rand_sampling),
                                            None),
    "nguyen_2":         (nguyen_2,          ([[-1.0, 1.0]], 20, rand_sampling),
                                            None),
    "nguyen_3":         (nguyen_3,          ([[-1.0, 1.0]], 20, rand_sampling),
                                            None),
    "nguyen_4":         (nguyen_4,          ([[-1.0, 1.0]], 20, rand_sampling),
                                            None),
    "nguyen_5":         (nguyen_5,          ([[-1.0, 1.0]], 20, rand_sampling),
                                            None),
    "nguyen_6":         (nguyen_6,          ([[-1.0, 1.0]], 20, rand_sampling),
                                            None),
    "nguyen_7":         (nguyen_7,          ([[0.0, 2.0]], 20, rand_sampling),
                                            None),
    "nguyen_8":         (nguyen_8,          ([[0.0, 4.0]], 20, rand_sampling),
                                            None),
    "nguyen_9":         (nguyen_9,          ([[0.0, 1.0], [0.0, 1.0]], 100, rand_sampling),
                                            None),
    "nguyen_10":        (nguyen_10,         ([[0.0, 1.0], [0.0, 1.0]], 100, rand_sampling),
                                            None),
    "pagie_1":          (pagie_1,           ([[-5.0, 5.0], [-5.0, 5.0]], [0.4, 0.4], interval_samling),
                                            None),
    "pagie_2":          (pagie_2,           ([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]], [0.4, 0.4, 0.4], interval_samling),
                                            None),
    "korns_1":          (korns_1,           ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling),
                                            ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling)),
    "korns_2":          (korns_2,           ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling),
                                            ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling)),
    "korns_3":          (korns_3,           ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling),
                                            ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling)),
    "korns_4":          (korns_4,           ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling),
                                            ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling)),
    "korns_5":          (korns_5,           ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling),
                                            ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling)),
    "korns_6":          (korns_6,           ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling),
                                            ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling)),
    "korns_7":          (korns_7,           ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling),
                                            ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling)),
    "korns_8":          (korns_8,           ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling),
                                            ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling)),
    "korns_9":          (korns_9,           ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling),
                                            ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling)),
    "korns_10":         (korns_10,          ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling),
                                            ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling)),
    "korns_11":         (korns_11,          ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling),
                                            ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling)),
    "korns_12":         (korns_12,          ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling),
                                            ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling)),
    "korns_13":         (korns_13,          ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling),
                                            ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling)),
    "korns_14":         (korns_14,          ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling),
                                            ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling)),
    "korns_15":         (korns_15,          ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling),
                                            ([[-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0], [-50.0, 50.0]], 10000, rand_sampling)),
    "keijzer_1":        (keijzer_1,         ([[-1.0, 1.0]], 0.1, interval_samling),
                                            ([[-1.0, 1.0]], 0.001, interval_samling)),
    "keijzer_2":        (keijzer_1,         ([[-2.0, 2.0]], 0.1, interval_samling),
                                            ([[-2.0, 2.0]], 0.001, interval_samling)),
    "keijzer_3":        (keijzer_1,         ([[-3.0, 3.0]], 0.1, interval_samling),
                                            ([[-3.0, 3.0]], 0.001, interval_samling)),
    "keijzer_4":        (keijzer_4,         ([[0.0, 10.0]], 0.05, interval_samling),
                                            ([[0.05, 10.05]], 0.05, interval_samling)),
    "keijzer_5":        (keijzer_5,         ([[-1.0, 1.0],[1.0,2.0],[-1.0,1.0]], 1000, rand_sampling),
                                            ([[-1.0, 1.0],[1.0,2.0],[-1.0,1.0]], 10000, rand_sampling)),
    "keijzer_6":        (keijzer_6,         ([[1.0, 50.0]], 1.0, interval_samling),
                                            ([[1.0, 120.0]], 1.0, interval_samling)),
    "keijzer_7":        (keijzer_7,         ([[1.0, 100.0]], 1.0, interval_samling),
                                            ([[1.0, 100.0]], 0.1, interval_samling)),
    "keijzer_8":        (keijzer_8,         ([[0.0, 100.0]], 1.0, interval_samling),
                                            ([[0.0, 100.0]], 0.1, interval_samling)),
    "keijzer_9":        (keijzer_9,         ([[0.0, 100.0]], 1.0, interval_samling),
                                            ([[0.0, 100.0]], 0.1, interval_samling)),
    "keijzer_10":       (keijzer_10,        ([[0.0, 1.0], [0.0, 1.0]], 100, rand_sampling),
                                            ([[0.0, 1.0], [0.0, 1.0]], [0.01, 0.01], interval_samling)),
    "keijzer_11":       (keijzer_11,        ([[-3.0, 3.0], [-3.0, 3.0]], 20, rand_sampling),
                                            ([[-3.0, 3.0], [-3.0, 3.0]], [0.01, 0.01], interval_samling)),
    "keijzer_12":       (keijzer_12,        ([[-3.0, 3.0], [-3.0, 3.0]], 20, rand_sampling),
                                            ([[-3.0, 3.0], [-3.0, 3.0]], [0.01, 0.01], interval_samling)),
    "keijzer_13":       (keijzer_13,        ([[-3.0, 3.0], [-3.0, 3.0]], 20, rand_sampling),
                                            ([[-3.0, 3.0], [-3.0, 3.0]], [0.01, 0.01], interval_samling)),
    "keijzer_14":       (keijzer_14,        ([[-3.0, 3.0], [-3.0, 3.0]], 20, rand_sampling),
                                            ([[-3.0, 3.0], [-3.0, 3.0]], [0.01, 0.01], interval_samling)),
    "keijzer_15":       (keijzer_15,        ([[-3.0, 3.0], [-3.0, 3.0]], 20, rand_sampling),
                                            ([[-3.0, 3.0], [-3.0, 3.0]], [0.01, 0.01], interval_samling)),
    "vladislavleva_1":  (vladislavleva_1,   ([[0.3, 4.0], [0.3, 4.0]], 100, rand_sampling),
                                            ([[-0.2, 4.2], [-0.2, 4.2]], [0.1,0.1], interval_samling)),
    "vladislavleva_2":  (vladislavleva_2,   ([[0.05, 10]], 0.1, interval_samling),
                                            ([[-0.5, 10.5]], 0.05, interval_samling)),
    "vladislavleva_3":  (vladislavleva_3,   ([[0.05, 10], [0.05, 10.05]], [0.1, 2.0], interval_samling),
                                            ([[-0.5, 10.5], [-0.5, 10.5]], [0.05, 0.5], interval_samling)),
    "vladislavleva_4":  (vladislavleva_4,   ([[0.05, 6.05], [0.05, 6.05], [0.05, 6.05], [0.05, 6.05], [0.05, 6.05]], 1024, rand_sampling),
                                            ([[-0.25, 6.35], [-0.25, 6.35], [-0.25, 6.35], [-0.25, 6.35], [-0.25, 6.35]], 5000, rand_sampling)),
    "vladislavleva_5":  (vladislavleva_5,   ([[0.05, 2.0], [1.0, 2.0], [0.05, 2.0]], 300, rand_sampling),
                                            ([[-0.05, 2.1], [0.95, 2.05], [-0.05, 2.1]], [0.15, 0.15, 0.1], interval_samling)),
    "vladislavleva_6":  (vladislavleva_6,   ([[0.1, 5.9], [0.1, 5.9]], 30, rand_sampling),
                                            ([[-0.05, 6.05], [-0.05, 6.05]], [0.02, 0.02], interval_samling)),
    "vladislavleva_7":  (vladislavleva_7,   ([[0.05, 6.05], [0.05, 6.05]], 300, rand_sampling),
                                            ([[-0.25, 6.35], [-0.25, 6.35]], 1000, rand_sampling)),
    "vladislavleva_8":  (vladislavleva_8,   ([[0.05, 6.05], [0.05, 6.05]], 50, rand_sampling),
                                            ([[-0.25, 6.35], [-0.25, 6.35]], [0.2, 0.2], interval_samling)),
}        

# TODO: drawing of funcs in ranges  + use rnd state 