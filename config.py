''' Defines different configuration of interest for research 
    Each config provides the explanation of why we would like to test the config and what we plan to collect 
    Finall CONFIG array is used to collect all configs and used in CLI entrypoint
    TODO: add here your config of interest 
'''

# Number games 
from itertools import product
from typing import Any
from cde import CDESpace
from games import PCHC, PPHC, CDESpaceGame, CandidateTestInteractions, CompareOnOneGame, FocusingGame, InteractionGame, IntransitiveRegionGame, GreaterThanGame, RandSampling

from population import CosaBasedSample, HCPopulation, OneTimeSequential, DECARanksSample, UneqGroupsSample, InteractionFeatureOrder, ParetoRankBasedSample
from params import *

def get_args(kwargs: dict[str, Any], prefix):
    glob_params = {k:v for k, v in kwargs.items() if not k.startswith(prefix)}
    specific_params = {k[len(prefix):]:v for k, v in kwargs.items() if k.startswith(prefix)}
    params = {**glob_params, **specific_params}
    return params

# def PCHC_SIM(**kwargs):
#     def b(game: InteractionGame, **kwargs2):
#         kwargsAll = {**kwargs, **kwargs2}
#         return PCHC(game, HCPopulation(game.get_all_candidates(), **kwargsAll))
#     return b

def RAND_SIM(**kwargs):
    def b(game: InteractionGame, **kwargs2):
        kwargsAll = {**kwargs, **kwargs2}
        return RandSampling(game, **kwargsAll)
    return b 

def PPHC_SIM(**kwargs):
    def b(game: InteractionGame, **kwargs2):
        kwargsAll = {**kwargs, **kwargs2}
        candidates = OneTimeSequential(game.get_all_candidates(), **get_args(kwargsAll, "cand_"))
        tests = HCPopulation(game.get_all_tests(), **get_args(kwargsAll, "test_"))
        sim = PPHC(game, candidates, tests)
        return sim
    return b

def SS_SIM(**kwargs):
    def b(game: InteractionGame, **kwargs2):
        kwargsAll = {**kwargs, **kwargs2}
        candidates = OneTimeSequential(game.get_all_candidates(), **get_args(kwargsAll, "cand_"))
        tests = InteractionFeatureOrder(game.get_all_tests(), **get_args(kwargsAll, "test_"))
        sim = CandidateTestInteractions(game, candidates, tests)
        return sim    
    return b

def DS_SIM(**kwargs):
    def b(game: InteractionGame, **kwargs2):
        kwargsAll = {**kwargs, **kwargs2}
        candidates = OneTimeSequential(game.get_all_candidates(), **get_args(kwargsAll, "cand_"))
        tests = DECARanksSample(game.get_all_tests(), **get_args(kwargsAll, "test_"))
        sim = CandidateTestInteractions(game, candidates, tests)
        return sim 
    return b   

def UG_SIM(**kwargs):
    def b(game: InteractionGame, **kwargs2):
        kwargsAll = {**kwargs, **kwargs2}
        candidates = OneTimeSequential(game.get_all_candidates(), **get_args(kwargsAll, "cand_"))
        tests = UneqGroupsSample(game.get_all_tests(), **get_args(kwargsAll, "test_"))
        sim = CandidateTestInteractions(game, candidates, tests)
        return sim 
    return b   

def PR_SIM(**kwargs):
    def b(game: InteractionGame, **kwargs2):
        kwargsAll = {**kwargs, **kwargs2}
        candidates = OneTimeSequential(game.get_all_candidates(), **get_args(kwargsAll, "cand_"))
        tests = ParetoRankBasedSample(game.get_all_tests(), **get_args(kwargsAll, "test_"))
        sim = CandidateTestInteractions(game, candidates, tests)
        return sim 
    return b   

def CB_SIM(**kwargs):
    def b(game: InteractionGame, **kwargs2):
        kwargsAll = {**kwargs, **kwargs2}
        candidates = OneTimeSequential(game.get_all_candidates(), **get_args(kwargsAll, "cand_"))
        tests = CosaBasedSample(game.get_all_tests(), **get_args(kwargsAll, "test_"))
        sim = CandidateTestInteractions(game, candidates, tests)
        return sim 
    return b 

NUMBER_GAMES = {game.__name__: game for game in [ GreaterThanGame, FocusingGame, IntransitiveRegionGame, CompareOnOneGame ] }

def get_spanned_space():
    space = CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 1)
    for i in range(10):
        space = space.with_spanned_point([(i - 1, -1), (i, -1)], 1)
    return space

def build_space_game(space_builder):
    def b(**kwargs):
        space = space_builder()
        return CDESpaceGame(space, **kwargs)
    return b

GAMES = {
    **NUMBER_GAMES,
    # RQ0: No skew of points on axes, no spanned, no non-informative, no duplicates, strong independance
    #     Space with 10 axes and 5 points per axis. Num cands = 50, num_tests = 50. 
    #     Max ints = 2500, expected ints = 50 (cands) * 10 (axes) = 500
    "ideal": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 1)),
    
    # RQ1: Under skew of points on axes, how does D change for different algos? 
    #         Unbalanced representation of objectives/misconceptions by tests    
    #     Max ints = 2500, expected ints = 50 (cands) * 10 (axes) = 500
    "skew-p-1": build_space_game(lambda: CDESpace([6, 6, 6, 6, 6, 4, 4, 4, 4, 4]).with_test_distribution(0, 1).with_candidate_distribution(0, 1)),
    "skew-p-2": build_space_game(lambda: CDESpace([7, 7, 7, 7, 7, 3, 3, 3, 3, 3]).with_test_distribution(0, 1).with_candidate_distribution(0, 1)),
    "skew-p-3": build_space_game(lambda: CDESpace([8, 8, 8, 8, 8, 2, 2, 2, 2, 2]).with_test_distribution(0, 1).with_candidate_distribution(0, 1)),
    "skew-p-4": build_space_game(lambda: CDESpace([9, 9, 9, 9, 9, 1, 1, 1, 1, 1]).with_test_distribution(0, 1).with_candidate_distribution(0, 1)),
    
    # RQ2: Under skew of tests on points, how does ARR change for different algos (extreme to nonInfo)? Noise of test triviality and similarity
    "trivial-1": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(1, 1).with_candidate_distribution(0, 1)), #with 1 non-info
    "trivial-5": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(5, 1).with_candidate_distribution(0, 1)), #with 2 non-info
    "trivial-10": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(10, 1).with_candidate_distribution(0, 1)), #with 3 non-info
    "trivial-15": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(15, 1).with_candidate_distribution(0, 1)), #with 4 non-info
    "trivial-20": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(20, 1).with_candidate_distribution(0, 1)), #with 5 non-info
    "trivial-25": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(25, 1).with_candidate_distribution(0, 1)), #with 10 non-info

    "skew-t-1": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, [6,1,1,1,1]).with_candidate_distribution(0, 1)), #duplicates
    "skew-t-2": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, [1,6,1,1,1]).with_candidate_distribution(0, 1)),
    "skew-t-3": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, [1,1,6,1,1]).with_candidate_distribution(0, 1)),
    "skew-t-4": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, [1,1,1,6,1]).with_candidate_distribution(0, 1)),
    "skew-t-5": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, [1,1,1,1,6]).with_candidate_distribution(0, 1)),

    "skew-c-1": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, [6,1,1,1,1])), #duplicates
    "skew-c-2": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, [1,6,1,1,1])),
    "skew-c-3": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, [1,1,6,1,1])),
    "skew-c-4": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, [1,1,1,6,1])),
    "skew-c-5": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, [1,1,1,1,6])),
    
    # RQ3: Under presense of spanned points how R, D, ARR (ARR*) changes? Noise of test complexity
    "span-all-ends-1": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 1)\
                              .with_spanned_point([(i, -1) for i in range(10)], 1)),
    "span-all-ends-5": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 1)\
                              .with_spanned_point([(i, -1) for i in range(10)], 5)),
    "span-one-pair-1": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 1)\
                              .with_spanned_point([(0, -1), (1, -1)], 1)),
    "span-all-pairs-1": build_space_game(get_spanned_space),

    #RQ4: Under presense of duplicated tests how Dup, D, ARR (ARR*) changes? Noise of test similarity
    "dupl-t-2": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 2).with_candidate_distribution(0, 1)),
    "dupl-t-3": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 3).with_candidate_distribution(0, 1)),
    "dupl-t-4": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 4).with_candidate_distribution(0, 1)),
    "dupl-t-5": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 5).with_candidate_distribution(0, 1)),
    "dupl-c-2": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 2)),
    "dupl-c-3": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 3)),
    "dupl-c-4": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 4)),
    "dupl-c-5": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 5)),

    #RQ5: Under increase number of objectives per candidate, how algo behavior change? Noise of weak independance of axes
    # space = CDESpace([5] * 10).with_axes_dependency(0, -1, 1, -1)
    # space = CDESpace([5] * 10).with_axes_dependency(0, -1, 1, -1).with_axes_dependency(2, -1, 3, -1)
    "dependant-all-1": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 1)\
                              .with_axes_dependency(0, -1, 1, -1).with_axes_dependency(2, -1, 3, -1).with_axes_dependency(4, -1, 5, -1) \
                              .with_axes_dependency(6, -1, 7, -1).with_axes_dependency(8, -1, 9, -1)),
    "dependant-all-2": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 1)\
                              .with_axes_dependency(0, -1, 1, -1).with_axes_dependency(2, -1, 3, -1).with_axes_dependency(4, -1, 5, -1) \
                              .with_axes_dependency(6, -1, 7, -1).with_axes_dependency(8, -1, 9, -1) \
                              .with_axes_dependency(0, -2, 1, -2).with_axes_dependency(2, -2, 3, -2).with_axes_dependency(4, -2, 5, -2) \
                              .with_axes_dependency(6, -2, 7, -2).with_axes_dependency(8, -2, 9, -2))
}

SIM = {
    "rand-10": RAND_SIM(),
    # "pchc-pmo-w": PCHC_SIM(mutation = "plus_minus_one", init="zero_init", ind_range=(param_min_num, param_max_num), selection = "num_wins", init_range=3), 
    # "pchc-r-w": PCHC_SIM(mutation = "resample", selection = "num_wins", init="zero_init", init_range=3),
    "pphc-pmo-i": PPHC_SIM(mutation = "plus_minus_one", ind_range=(param_min_num, param_max_num), test_selection="informativeness_select"), # init="zero_init", init_range=3), 
    "pphc-pmo-p": PPHC_SIM(mutation = "plus_minus_one", ind_range=(param_min_num, param_max_num), test_selection="pareto_select"), # init="zero_init", init_range=3),
    "pphc-r-i": PPHC_SIM(mutation = "resample", test_selection="informativeness_select"), # init="zero_init", init_range=3), 
    "pphc-r-p": PPHC_SIM(mutation = "resample", test_selection="pareto_select"), # init="zero_init", init_range=3),
    "s-0_nd": SS_SIM(strategy=["nond", "dom", "kn"]),
    # "s-0_ndXdm": SS_SIM(strategy=["nd", "kn"]),
    # "s-0_dm": SS_SIM(strategy=["dom", "kn"]),
    # "s-0_d": SS_SIM(strategy=["d", "kn"]),
    # "s-0_sXd": SS_SIM(strategy=["sd", "kn"]),
    # "s-0_Zs": SS_SIM(strategy=["-s", "kn"]),
    # "s-0_k": SS_SIM(strategy=["kn"]),

    # "s-0_dp_kn": SS_SIM(strategy=["dup","kn"]),
    # "s-0_dp_d": SS_SIM(strategy=["dup", "d", "kn"]),
    # "s-0_dp_sXd": SS_SIM(strategy=["dup", "sd", "kn"]),
    # "s-0_dp_kn": SS_SIM(strategy=["d", "kn"]),
    # "s-0_dp_Zs": SS_SIM(strategy=["dup", "-s", "kn"]),
    # "s-0_dp_dm": SS_SIM(strategy=["dup", "dom", "kn"]),
    # "s-0_dp_nd": SS_SIM(strategy=["dup", "nond", "kn"]),
    # "s-0_dp_ndXdm": SS_SIM(strategy=["dup", "nd", "kn"]),

    # "ds-100-001": DS_SIM(directed_explore_prop = 0.5, obj_bonus = 100, span_penalty = 0.01), 
    "ds-100-001": DS_SIM(obj_bonus = 100, span_penalty = 0.01), 
    "ug": UG_SIM(),
    "pr-10-2": PR_SIM(rank_bonus = 10, max_layers = 2),
    "cb-100": PR_SIM(archive_bonus = 100)
    # "aco-05-10-1-1": ACO_SIM(pheromone_decay = 0.5, pheromone_inc = 10, dom_bonus = 1, span_penalty = 1)      
}

# SPACE_SIM = {    

#     # "pphc-pmo-i": PPHC_SPACE_SIM(mutation = "plus_minus_one", test_selection="informativeness_select"), 
#     # "pphc-pmo-p": PPHC_SPACE_SIM(mutation = "plus_minus_one", test_selection="pareto_select"),        
#     # "pchc-r-i": PPHC_SPACE_SIM(mutation = "resample", test_selection="informativeness_select"), 
#     # "pchc-r-p": PPHC_SPACE_SIM(mutation = "resample", test_selection="pareto_select"),
    
#     "s-0_nd": SS_SPACE_SIM(strategy=["nond", "dom", "kn"]),
#     # "s-0_ndXdm": SS_SPACE_SIM(strategy=["nd", "kn"]),
#     # "s-0_dm": SS_SPACE_SIM(strategy=["dom", "kn"]),
#     # "s-0_d": SS_SPACE_SIM(strategy=["d", "kn"]),
#     # "s-0_sXd": SS_SPACE_SIM(strategy=["sd", "kn"]),
#     # "s-0_Zs": SS_SPACE_SIM(strategy=["-s", "kn"]),
#     # "s-0_k": SS_SPACE_SIM(strategy=["kn"]),

#     # "s-0_dp_kn": SS_SPACE_SIM(strategy=["dup","kn"]),
#     # "s-0_dp_d": SS_SPACE_SIM(strategy=["dup", "d", "kn"]),
#     # "s-0_dp_sXd": SS_SPACE_SIM(strategy=["dup", "sd", "kn"]),
#     # "s-0_dp_kn": SS_SPACE_SIM(strategy=["d", "kn"]),
#     # "s-0_dp_Zs": SS_SPACE_SIM(strategy=["dup", "-s", "kn"]),
#     # "s-0_dp_dm": SS_SPACE_SIM(strategy=["dup", "dom", "kn"]),
#     # "s-0_dp_nd": SS_SPACE_SIM(strategy=["dup", "nond", "kn"]),
#     # "s-0_dp_ndXdm": SS_SPACE_SIM(strategy=["dup", "nd", "kn"]),

#     "ds-100-001": DS_SPACE_SIM(obj_bonus = 100, span_penalty = 0.01), 
#     # "aco-05-10-1-1": ACO_SPACE_SIM(pheromone_decay = 0.5, pheromone_inc = 10, dom_bonus = 1, span_penalty = 1)      
# }


from tabulate import tabulate

def print_config():
    # printing stats of all present configs
    print(f"{len(GAMES)} games and {len(SIM)} simulations")
    rows = [[i, game_name, sim_name] for i, (game_name, sim_name) in product(GAMES.keys(), SIM.keys())]
    print(tabulate(rows, headers=["ID", "game", "simulation"], tablefmt="github"))

if __name__ == '__main__':
    print_config()        