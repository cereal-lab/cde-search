
import fcntl
from functools import partial
import inspect
import json
import types
from typing import Optional

import numpy as np

def f(x, y, z, *, d = 42):
    return x + y + z + d

f1 = partial(f, z = 100, d = 43)
f2 = partial(f1, z = 101)
inspect.signature(f1)
inspect.signature(f)

def create_named_function(name, func):
    ''' Create a function with a specific __name__ at runtime. '''
    func.__name__ = name
    return func
    # new_func = types.FunctionType(func.__code__, func.__globals__, name, func.__defaults__, func.__closure__)
    # new_func.__dict__.update(func.__dict__)
    # return new_func

def bind_fn(kwargs, fn):
    ''' For a given callable fn, goes through its parameters and applies bindings from kwargs to parameters after *.
        If any of the parameters of fn is also callable, recursively applies bindings to it. '''
    signature = inspect.signature(fn)
    star_bindings = {param.name: kwargs[param.name] for param in signature.parameters.values() 
                if param.kind == param.KEYWORD_ONLY and param.default is inspect.Parameter.empty and param.name in kwargs}
    for param in signature.parameters.values():
        if param.default is not inspect.Parameter.empty and callable(param.default): 
            new_default = bind_fn(kwargs, param.default)
            if new_default != param.default:
                star_bindings[param.name] = new_default
    if isinstance(fn, partial):
        new_fn = partial(fn.func, **{**fn.keywords, **star_bindings})
        # name = fn.func.__name__
    else:
        new_fn = partial(fn, **star_bindings)
        # name = fn.__name__
    # new_fn.__name__ = name
    return new_fn

def bind_fns(kwargs, *fns):
    ''' For a given list of callables, applies bind_all_from to all of them. '''
    return [bind_fn(kwargs, fn) for fn in fns]

def bind_flat(fn, **kwargs):
    if isinstance(fn, partial):
        new_fn = partial(fn.func, **fn.keywords, **kwargs)
        name = fn.func.__name__
    else:
        new_fn = partial(fn, **kwargs)
        name = fn.__name__
    new_fn.__name__ = name
    return new_fn

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)

def write_metrics(metrics, metric_file):
    with open(metric_file, "a") as f:                
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(metrics, cls=NpEncoder) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)    

# def unspecified(*args, **kwargs):
#     raise ValueError("Please specify correct function")

# def bind_all_from(fn, kwargs, *, used_keys: Optional[dict] = None):
#     ''' All parameters after * are considered to be modified - bound according to kwargs or if callabled are bound_from recursively '''
#     signature = inspect.signature(fn)
#     allowed_set = {param.name: param.default for param in signature.parameters.values() if param.kind == param.KEYWORD_ONLY }
#     if len(allowed_set) == 0:
#         return fn
#     new_bindings = {}
#     for param_name, param_default in allowed_set.items():
#         if param_name in kwargs:
#             new_bindings[param_name] = kwargs[param_name]
#             if used_keys is not None:
#                 used_keys.setdefault(param_name, []).append(fn.__name__)
#         elif param_default is not inspect.Parameter.empty and callable(param_default):
#             new_param_default = bind_all_from(param_default, kwargs, used_keys = used_keys)
#             new_bindings[param_name] = new_param_default
#     if len(new_bindings) == 0:
#         return fn
#     new_fn = bind(fn, **new_bindings)
#     return new_fn

# def order_fns(fn_dict: dict):
#     fn_deps = {}
#     for fn_name, fn in fn_dict.items():
#         signature = inspect.signature(fn)
#         fn_deps[fn_name] = set(param.name for param in signature.parameters.values() if param.kind == param.KEYWORD_ONLY and param.name in fn_dict)
#     fns_ordered = []
#     fn_deps_ordered = sorted(fn_deps.items(), key = lambda x: len(x[1]))
#     while len(fn_deps_ordered) > 0:
#         fn_lst = []
#         while len(fn_deps_ordered[0][1]) == 0:
#             fn_name, _ = fn_deps_ordered.pop()
#             fn_lst.append(fn_name)
#         if len(fn_lst) == 0:
#             raise ValueError(f"Cannot arrange func dependencies: {fn_deps_ordered}")
#         fns_ordered.extend([(name, fn[name]) for name in fn_lst])
#         for fn_name, deps in fn_deps_ordered:
#             deps.difference_update(fn_lst)
#         fn_deps_ordered.sort(key = lambda x: len(x[1]))
#     return fns_ordered    