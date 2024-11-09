param_seed1 = 19
param_seed2 = 23
param_steps = 100
param_min_num = 0 
param_max_num = 100
# param_reg_min_num = 30
# param_reg_max_num = 49
param_num_intransitive_regions = 3
param_selection_size = 10
param_batch_size = 10 #for OneTimeSequence population
param_draw_dynamics = 0
param_space_dir = "data/num-game-spaces"

import numpy as np
rnd = np.random.RandomState(param_seed1)
# we use second instance for OneSequencePopulation to preserve same candidate order between all configs
# otherwise, some Algos could use rnd more or less than others, changing results for OneSequencePopulation
seq_rnd = np.random.RandomState(param_seed2)

PARAM_INTS = "made_ints"
PARAM_UNIQ_INTS = "uniq_ints"
PARAM_MAX_INTS = "game_max_ints"

PARAM_UNIQ_INDS = "uniq_inds"
PARAM_MAX_INDS = "max_inds"
PARAM_IND_CHANGES_STORY = "ind_changes_story"

# game specific goal metric
PARAM_GAME_GOAL = "game_goal"
PARAM_GAME_GOAL_MOMENT = "game_goal_moment"
#changes of goal with game step
PARAM_GAME_GOAL_STORY = "game_goal_story"