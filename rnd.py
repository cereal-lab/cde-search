import os
import numpy as np
import torch 

seed = int(os.environ.get('GP_RAND_SEED', 109))
''' NOTE: each benchmark should be run from scratch in new new process to reproduce same rand sequence '''
default_rnd = np.random.RandomState(seed)

torch.manual_seed(seed)
gpu_seed = seed + 1
torch.cuda.manual_seed(gpu_seed)
torch.cuda.manual_seed_all(gpu_seed)
