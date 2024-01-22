param_times = 30
param_steps = 100
param_min_num = 0 
param_max_num = 300
param_reg_min_num = 100
param_reg_max_num = 110
param_seed = 19

import numpy as np
rnd = np.random.RandomState(param_seed)

PARAM_INTS = "game_ints"
PARAM_UNIQ_INTS = "game_uniq_ints"
PARAM_MAX_INTS = "game_max_ints"

PARAM_UNIQ_INDS = "game_uniq_inds"
PARAM_MAX_INDS = "game_max_inds"

# game specific goal metric
PARAM_GAME_GOAL = "game_goal"
PARAM_GAME_GOAL_MOMENT = "game_goal_moment"
#changes of goal with game step
PARAM_GAME_GOAL_STORY = "game_goal_story"