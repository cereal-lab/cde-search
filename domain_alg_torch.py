''' Continuous algebraic domain '''

from functools import partial
from itertools import product
from typing import Optional
import torch

from gp import RuntimeContext
import utils


def f_add(x: torch.Tensor, y: torch.Tensor, **_) -> torch.Tensor:
    return x + y

def f_sub(x: torch.Tensor, y: torch.Tensor, **_) -> torch.Tensor:
    return x - y

def f_mul(x: torch.Tensor, y: torch.Tensor, **_) -> torch.Tensor:
    return x * y

# NOTE: we do not apply sanitization - resuls could have inf or nan - should be handled  
def f_div(x: torch.Tensor, y: torch.Tensor, **_) -> torch.Tensor:
    return x / y

def f_neg(x: torch.Tensor, **_) -> torch.Tensor:
    return -x

# NOTE: we do not apply sanitization - resuls could have inf or nan - should be handled  
def f_inv(x: torch.Tensor, **_) -> torch.Tensor:
    return 1 / x

def f_cos(x: torch.Tensor, **_) -> torch.Tensor:
    return torch.cos(x)

def f_sin(x: torch.Tensor, **_) -> torch.Tensor:
    return torch.sin(x)

def f_square(x: torch.Tensor, **_) -> torch.Tensor:
    return x ** 2
    
def f_cube(x: torch.Tensor, **_) -> torch.Tensor:
    return x ** 3

def f_exp(x: torch.Tensor, **_) -> torch.Tensor:
    return torch.exp(x)

# NOTE: we do not apply sanitization - resuls could have inf or nan - should be handled  
def f_log(x: torch.Tensor, **_) -> torch.Tensor:
    return torch.log(x)

# additional less used 
def f_neg_exp(x: torch.Tensor, **_) -> torch.Tensor:
    return torch.exp(-x)

# NOTE: we do not apply sanitization 
def f_sqrt(x: torch.Tensor, **_) -> torch.Tensor:
    return torch.sqrt(x)

def f_tanh(x: torch.Tensor, **_) -> torch.Tensor:
    return torch.tanh(x)

def f_tan(x: torch.Tensor, **_) -> torch.Tensor:
    return torch.tan(x)

# benchmarks

def koza_1(x:torch.Tensor) -> torch.Tensor:
    return x*x*x*x + x*x*x + x*x + x

def koza_2(x:torch.Tensor) -> torch.Tensor:
    return x*x*x*x*x - 2.0*x*x*x + x

def koza_3(x:torch.Tensor) -> torch.Tensor:
    return x*x*x*x*x*x - 2.0*x*x*x*x + x*x

def nguyen_1(x:torch.Tensor) -> torch.Tensor:
    return x*x*x + x*x + x

def nguyen_2(x:torch.Tensor) -> torch.Tensor:
    return x*x*x*x + x*x*x + x*x + x

def nguyen_3(x:torch.Tensor) -> torch.Tensor:
    return x*x*x*x*x + x*x*x*x + x*x*x + x*x + x

def nguyen_4(x:torch.Tensor) -> torch.Tensor:
    return x*x*x*x*x*x + x*x*x*x*x + x*x*x*x + x*x*x + x*x + x

def nguyen_5(x:torch.Tensor) -> torch.Tensor:
    return torch.sin(x*x) * torch.cos(x) - 1.0

def nguyen_6(x:torch.Tensor) -> torch.Tensor:
    return torch.sin(x) + torch.sin(x + x*x)

def nguyen_7(x:torch.Tensor) -> torch.Tensor:
    return torch.log(x + 1.0) + torch.log(x*x + 1.0)

def nguyen_8(x:torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x)

def nguyen_9(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return torch.sin(x) + torch.sin(y * y)

def nguyen_10(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.sin(x) + torch.cos(y)

def pagie_1(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + 1.0 / (x * x * x * x)) + 1.0 / (1.0 + 1.0 / (y * y * y * y))

def pagie_2(x:torch.Tensor, y:torch.Tensor, z:torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + 1.0 / ( x * x * x * x)) + 1.0 / (1.0 + 1.0 / (y * y * y * y)) + 1.0 / (1.0 + 1.0 / (z * z * z * z))

def korns_1(*xs: list[torch.Tensor]) -> torch.Tensor:
    return 1.57 + (24.3 * xs[3])

def korns_2(*xs: list[torch.Tensor]) -> torch.Tensor:
    return 0.23 + (14.2 * ((xs[3]+ xs[1])/(3.0 * xs[4])))

def korns_3(*xs: list[torch.Tensor]) -> torch.Tensor:
    return -5.41 + (4.9 * (((xs[3] - xs[0]) + (xs[1]/xs[4])) / (3 * xs[4])))

def korns_4(*xs: list[torch.Tensor]) -> torch.Tensor:
    return -2.3 + (0.13 * torch.sin(xs[2]))

def korns_5(*xs: list[torch.Tensor]) -> torch.Tensor:
    return 3.0 + (2.13 * torch.log(xs[4]))

def korns_6(*xs: list[torch.Tensor]) -> torch.Tensor:
    return 1.3 + (0.13 * torch.sqrt(xs[0]))

def korns_7(*xs: list[torch.Tensor]) -> torch.Tensor:
    return 213.80940889 - (213.80940889 * torch.exp(-0.54723748542 * xs[0]))

def korns_8(*xs: list[torch.Tensor]) -> torch.Tensor:
    return 6.87 + (11.0 * torch.sqrt(7.23 * xs[0] * xs[3] * xs[4]))

def korns_9(*xs: list[torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(xs[0]) / torch.log(xs[1]) * torch.exp(xs[2]) / (xs[3] * xs[3])

def korns_10(*xs: list[torch.Tensor]) -> torch.Tensor:
    return 0.81 + (24.3 * (((2.0 * xs[1]) + (3.0 * (xs[2] * xs[2]))) / ((4.0 * (xs[3]*xs[3]*xs[3])) + (5.0 * (xs[4]*xs[4]*xs[4]*xs[4])))))

def korns_11(*xs: list[torch.Tensor]) -> torch.Tensor:
    return 6.87 + (11.0 * torch.cos(7.23 * xs[0]*xs[0]*xs[0]))

def korns_12(*xs: list[torch.Tensor]) -> torch.Tensor:
    return 2.0 - (2.1 * (torch.cos(9.8 * xs[0]) * torch.sin(1.3 * xs[4])))

def korns_13(*xs: list[torch.Tensor]) -> torch.Tensor:
    return 32.0 - (3.0 * ((torch.tan(xs[0]) / torch.tan(xs[1])) * (torch.tan(xs[2])/torch.tan(xs[3]))))

def korns_14(*xs: list[torch.Tensor]) -> torch.Tensor:
    return 22.0 - (4.2 * ((torch.cos(xs[0]) - torch.tan(xs[1]))*(torch.tanh(xs[2])/torch.sin(xs[3]))))

def korns_15(*xs: list[torch.Tensor]) -> torch.Tensor:
    return 12.0 - (6.0 * ((torch.tan(xs[0])/torch.exp(xs[1])) * (torch.log(xs[2]) - torch.tan(xs[3]))))

def keijzer_1(x:torch.Tensor) -> torch.Tensor:
    return 0.3 * x * torch.sin(2.0 * torch.pi * x)

# NOTE: keijzer_2 == keijzer_3 == keijzer_1

def keijzer_4(x:torch.Tensor) -> torch.Tensor:
    return x*x*x * torch.exp(-x)*torch.cos(x)*torch.sin(x)* (torch.sin(x)*torch.sin(x)*torch.cos(x) - 1)

def keijzer_5(x:torch.Tensor, y:torch.Tensor, z:torch.Tensor) -> torch.Tensor:
    return (30.0 * x * z) / ((x - 10.0) * y * y)

def keijzer_6(x:torch.Tensor) -> torch.Tensor:
    fl = torch.stack([torch.sum(1.0 / torch.arange(1, torch.floor(xi) + 1)) for xi in x])
    return fl

def keijzer_7(x:torch.Tensor) -> torch.Tensor:
    return torch.log(x)

def keijzer_8(x:torch.Tensor) -> torch.Tensor:
    return torch.sqrt(x)

def keijzer_9(x:torch.Tensor) -> torch.Tensor:
    return torch.arcsinh(x)

def keijzer_10(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return torch.float_power(x, y)

def keijzer_11(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return x * y + torch.sin((x - 1.0) * (y - 1.0))

def keijzer_12(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return x*x*x*x - x*x*x + y*y/2.0 - y

def keijzer_13(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return 6.0 * torch.sin(x) * torch.cos(y)

def keijzer_14(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return 8.0 / (2.0 + x*x + y*y)

def keijzer_15(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return x*x*x / 5.0 + y*y*y/2.0 - y - x

def vladislavleva_1(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return torch.exp(-(x-1)*(x-1)) / (1.2 + (y - 2.5)*(y-2.5))

def vladislavleva_2(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return torch.exp(-x)*x*x*x*torch.cos(x)*torch.sin(x)*(torch.cos(x)*torch.sin(x)*torch.sin(x) - 1)

def vladislavleva_3(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return torch.exp(-x)*x*x*x*torch.cos(x)*torch.sin(x)*(torch.cos(x)*torch.sin(x)*torch.sin(x) - 1) * (y - 5)

def vladislavleva_4(*xs: list[torch.Tensor]) -> torch.Tensor:
    return 10.0 / (5.0 + torch.sum((xs - 3.0) ** 2, axis=0))

def vladislavleva_5(x:torch.Tensor, y:torch.Tensor, z:torch.Tensor) -> torch.Tensor:
    return (30.0 * (x - 1.0) * (z - 1.0)) / (y * y * (x - 10.0))

def vladislavleva_6(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return 6.0 * torch.sin(x) * torch.cos(y)

def vladislavleva_7(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return (x - 3.0) * (y - 3.0) + 2 * torch.sin((x - 4.0) * (y - 4.0))

def vladislavleva_8(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return ((x - 3.0) * (x - 3.0) * (x - 3.0) * (x - 3.0) + (y - 3.0) * (y - 3.0) * (y - 3.0) - (y - 3.0)) / ((y - 2.0) * (y - 2.0) * (y - 2.0) * (y - 2.0) + 10.0)

# benchmark meta data (num of vars, ranges and samplings)

# class Benchmark():
#     def __init__(self, gold_fn, free_var_ranges: torch.Tensor, train_sampling, test_sampling):
#         self.fn = gold_fn
#         self.free_var_ranges = free_var_ranges
#         self.train_sampling = train_sampling
#         self.test_sampling = test_sampling

def rand_sampling(num_samples, free_var_ranges: torch.Tensor, gold_fn):
    mins = torch.tensor([mi for mi, _ in free_var_ranges])
    maxs = torch.tensor([ma for _, ma in free_var_ranges])
    dist = maxs - mins
    inputs = [x for x in mins[:, torch.newaxis] + dist[:, torch.newaxis] * torch.rand(len(free_var_ranges), num_samples)]
    outputs = gold_fn(*inputs)
    return inputs, outputs

# rand_sampling(100, [[0.0, 1.0], [0.0, 1.0]], lambda x, y: torch.sin(x) + torch.cos(y))

def interval_samling(step, free_var_ranges: list[list[int]], gold_fn, deltas: Optional[list[int]] = None, rand_deltas = False):
    mins = torch.tensor([mi for mi, _ in free_var_ranges])
    maxs = torch.tensor([ma for _, ma in free_var_ranges])
    if type(step) == list:
        assert len(step) == len(free_var_ranges)
    else:
        step = [step] * len(free_var_ranges)
    step = torch.tensor(step)
    if deltas is not None:
        deltas = torch.tensor(deltas)
    if rand_deltas:
        if deltas is None:
            deltas = step * torch.rand(free_var_ranges.shape[0])
        else:
            deltas = deltas * torch.rand(free_var_ranges.shape[0])
    if deltas is not None:
        # deltas = torch.zeros_like(mins)
        mins += deltas
    
    mesh = list(product(*(torch.arange(mi.item(), ma.item(), s.item()).tolist() for mi, ma, s in zip(mins, maxs, step))))
    inputs = [x for x in torch.tensor(mesh).t()]
    outputs = gold_fn(*inputs)
    return inputs, outputs

interval_samling(0.1, [[0.0, 1.0], [0.0, 1.0]], lambda x, y: torch.sin(x) + torch.cos(y), deltas=[0.05, 0.01])

# https://en.wikipedia.org/wiki/Chebyshev_nodes
def chebyshev_sampling(num_samples, free_var_ranges: torch.Tensor, gold_fn, rand_deltas = False):
    mins = free_var_ranges[:, 0]
    maxs = free_var_ranges[:, 1]
    dist = maxs - mins
    indexes = torch.tile(torch.arange(0, num_samples), (free_var_ranges.shape[0], 1))
    if rand_deltas:
        deltas = torch.rand(free_var_ranges.shape[0])
    else:
        deltas = torch.zeros(free_var_ranges.shape[0], dtype=free_var_ranges.dtype)
    indexes = indexes + deltas[:, torch.newaxis]
    index_vs = torch.cos((2.0 * indexes - 1) / (2.0 * num_samples) * torch.pi)
    inputs = (maxs[:, torch.newaxis] + mins[:, torch.newaxis]) / 2 + dist[:, torch.newaxis] / 2 * index_vs
    # inputs = torch.tensor([0.5 * (mi + ma) + 0.5 * dist * torch.cos((2 * i + 1) * torch.pi / (2 * num_samples)) for i, mi, ma in zip(range(num_samples), mins, maxs)])
    outputs = gold_fn(*inputs)
    return inputs, outputs

# chebyshev_sampling(10, torch.tensor([[0, 1], [0, 1]]), lambda x, y: torch.sin(x) + torch.cos(y), rand_deltas=True)

def rand_const(min_v = 0.0, max_v = 1.0, size = 1):
    v = min_v + (max_v - min_v) * torch.rand(size)
    return v

# values: (gold_fn, train set settings, test set settings (None if test set == train set))
benchmark_data = {
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

def t_one_builder(*, ones_tensor):
    def t_one(*, free_vars):
        return ones_tensor
    res = utils.AnnotatedFunc(t_one, (lambda : "1"), "t_one")
    return res 

def t_zero_builder(*, ones_tensor):
    zeros = torch.zeros_like(ones_tensor)
    def t_zero(*, free_vars):
        return zeros
    res = utils.AnnotatedFunc(t_zero, (lambda : "0"), "t_zero")
    return res

def t_unknown_builder(*, const_init_fn = rand_const, ones_tensor):
    c = const_init_fn(size = len(ones_tensor))
    def t_unknown(c=c, free_vars = {}):
        return c
    res = utils.AnnotatedFunc(t_unknown, (lambda : "?"), category="t_unknown", context=c)
    return res

def t_const_builder(*, const_init_fn = rand_const, ones_tensor):    
    c = const_init_fn()
    def t_const(c=c, free_vars = {}):
        return c * ones_tensor
    res = utils.AnnotatedFunc(t_const, (lambda c=c: str(c.item())), category="t_const", context=c)
    return res

# Default const --> allocated once on const node build --> we should have func_builder function 
# Trained const --> allocated once on const node build but optimized on eval! 

def alg_problem_init(gold_fn, train_set_build, symbol_list, num_consts = 10, const_range = None, *, runtime_context: RuntimeContext):
    train_ranges, train_num_samples, train_set_builder = train_set_build
    inputs, outputs = train_set_builder(train_num_samples, train_ranges, gold_fn)
    if const_range is None:
        min_v = min(torch.min(outputs).item(), *[torch.min(i).item() for i in inputs])
        max_v = max(torch.max(outputs).item(), *[torch.max(i).item() for i in inputs])
    else:
        min_v, max_v = const_range
    rand_in_range = partial(rand_const, min_v = min_v, max_v = max_v)
    t_const_builder_bound = partial(t_const_builder, const_init_fn = rand_in_range, ones_tensor = torch.ones_like(outputs))
    terminal_list, free_vars = utils.create_free_vars(inputs, prefix = "x")
    terminal_list.append(utils.AnnotatedFunc(t_const_builder_bound, t_const_builder.__name__, "t_const"))
    if num_consts is None:
        counts_constraints = None 
    else:
        counts_constraints = {"t_const": num_consts}
    func_list = [utils.create_simple_func_builder(fn) for fn in symbol_list]
    runtime_context.update(gold_outputs = outputs, free_vars = free_vars, 
                           func_list = func_list, terminal_list = terminal_list, 
                           counts_constraints = counts_constraints)

# IDEA: generalize further to allow dynamic list of symbols - from reduced to reach set of symbols in search.
func_set1 = [f_add, f_sub, f_mul, f_div]
func_set2 = [f_add, f_mul, f_neg, f_inv, f_sin, f_cos, f_exp, f_log]
benchmark = {k:partial(alg_problem_init, gold_fn, train_set_build, func_set2) \
              for k, (gold_fn, train_set_build, _) in benchmark_data.items()}


def optimize_term(epsilon = 1e-4, max_steps = 50, lr=0.1):    

    best_cs = None
    best_loss = None
    prev_loss = None

    optimizer = None 

    def closure_builder(optimizer):
        optimizer.zero_grad()        
        outputs = gp_call(test_term, all_free_vars)
        loss = mse_loss(outputs, gold_outputs)
        loss.backward()
        return loss    

    for step_id in range(max_steps):

        if optimizer is None:
            for c in cs:
                c.requires_grad = False   
                c[:] = min_range + (max_range - min_range) * torch.rand_like(c)
                c.requires_grad = True

            optimizer = optim.LBFGS([*cs], lr=0.1)

            closure = partial(closure_builder, optimizer)
            prev_loss = None
                
        loss = optimizer.step(closure)
        if loss < epsilon:
            break

        if prev_loss is not None and abs(prev_loss - loss) < 10 * epsilon:
            optimizer = None # restart

            # test_ids = torch.randperm(len(gold_outputs))
            
            # for batch_ids in test_ids.view(-1, batch_size): # NOTE: len(test_ids) % batch_size == 0

            #     all_free_vars = {k:v[batch_ids] for k, v in free_vars.items()}
            #     all_free_vars.update(const_free_vars)

            #     outputs = gp_call(test_term, all_free_vars)
                
            #     loss = mse_loss(outputs, gold_outputs[batch_ids])

            

            #     loss.backward()
            #     # print(f"Step loss={loss.item()}, cs={cs}, grads={[c.grad for c in cs]}")

            #     optimizer.step()    
            #     scheduler.step()        

            # print(f"Epoch {epoch_id} loss={loss.item()}, cs={cs}")     
        elif best_cs is None or loss.item() < best_loss:
            best_cs = [c.item() for c in cs]
            best_loss = loss.item()
        prev_loss = loss
        print(f"Step id {step_id} loss={loss.item()}, cs={[c.item() for c in cs]}")


# def mse_loss(outputs: torch.Tensor, gold_outputs: torch.Tensor) -> torch.Tensor:    
#     return torch.mean((outputs - gold_outputs) ** 2)

# TODO: drawing of funcs in ranges  + use rnd state 

# memetic algo ideas:
# 1. optimize shape by consts - gard based optimizers 
# 2. optimize free var selelction for shape - int based optimizers
# 3. optimize subtree selection??? - int based optimizers 
# 4. symbolic constraints - using z3 optimizer?? 
# 5. semantic operators - search inverse semantics with gred based optimizers (without inversion)
# 6. idea of scaling???? - grad opt


if __name__ == "__main__":
    #Test of grads
    from gp import default_node_builder, build_free_vars
    from gp import call as gp_call
    import torch.optim as optim
    from torch.nn.functional import mse_loss
    nd = lambda f, *args: default_node_builder(f, args)

    gold_fn = lambda x: torch.sin(3.1* x) + torch.cos(-1.3 * x)
    # gold_fn = lambda x: 121 * x * x

    inputs, gold_outputs = interval_samling(0.01, [[0.009, 3.0]], gold_fn)

    min_range = min(torch.min(gold_outputs).item(), *[torch.min(i).item() for i in inputs])
    max_range = max(torch.max(gold_outputs).item(), *[torch.max(i).item() for i in inputs])

    (t_x, ), free_vars = build_free_vars(inputs, prefix="x")

    cs = [torch.rand(1) for _ in range(4)]
    (t_c1, t_c2, t_c3, t_c4), const_free_vars = build_free_vars(cs, prefix="c")             

    # optimizer = optim.AdamW([*cs], lr=1)
    # optimizer = optim.LBFGS([*cs], lr=0.01)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)    

    test_term = nd(f_add, nd(f_mul, nd(t_c4), nd(t_x)), nd(f_add, nd(t_c3), nd(f_add, nd(f_sin, nd(f_mul, nd(t_c1), nd(t_x))), nd(f_cos, nd(f_mul, nd(t_c2), nd(t_x))))))
    # test_term = nd(f_mul, nd(t_c1), nd(f_mul, nd(t_x), nd(t_x)))

    all_free_vars = {**free_vars, **const_free_vars}


    # batch_size = 100
    epsilon = 1e-4

    best_cs = None
    best_loss = None
    prev_loss = None

    optimizer = None 

    def closure_builder(optimizer):
        optimizer.zero_grad()        
        outputs = gp_call(test_term, all_free_vars)
        loss = mse_loss(outputs, gold_outputs)
        loss.backward()
        return loss    

    for step_id in range(50):

        if optimizer is None:
            for c in cs:
                c.requires_grad = False   
                c[:] = min_range + (max_range - min_range) * torch.rand_like(c)
                c.requires_grad = True

            optimizer = optim.LBFGS([*cs], lr=0.1)

            closure = partial(closure_builder, optimizer)
            prev_loss = None
                
        loss = optimizer.step(closure)
        if loss < epsilon:
            break

        if prev_loss is not None and abs(prev_loss - loss) < 10 * epsilon:
            optimizer = None # restart

            # test_ids = torch.randperm(len(gold_outputs))
            
            # for batch_ids in test_ids.view(-1, batch_size): # NOTE: len(test_ids) % batch_size == 0

            #     all_free_vars = {k:v[batch_ids] for k, v in free_vars.items()}
            #     all_free_vars.update(const_free_vars)

            #     outputs = gp_call(test_term, all_free_vars)
                
            #     loss = mse_loss(outputs, gold_outputs[batch_ids])

            

            #     loss.backward()
            #     # print(f"Step loss={loss.item()}, cs={cs}, grads={[c.grad for c in cs]}")

            #     optimizer.step()    
            #     scheduler.step()        

            # print(f"Epoch {epoch_id} loss={loss.item()}, cs={cs}")     
        elif best_cs is None or loss.item() < best_loss:
            best_cs = [c.item() for c in cs]
            best_loss = loss.item()
        prev_loss = loss
        print(f"Step id {step_id} loss={loss.item()}, cs={[c.item() for c in cs]}")
    pass