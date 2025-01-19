import os
import numpy as np

seed = int(os.environ.get('CSE_RAND_SEED', 109))
''' NOTE: each benchmark should be run from scratch in new new process to reproduce same rand sequence '''
default_rnd = np.random.RandomState(seed)