
import fcntl
import json
import types

import numpy as np

def create_named_function(name, func):
    ''' Create a function with a specific __name__ at runtime. '''
    func.__name__ = name
    return func
    # new_func = types.FunctionType(func.__code__, func.__globals__, name, func.__defaults__, func.__closure__)
    # new_func.__dict__.update(func.__dict__)
    # return new_func

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