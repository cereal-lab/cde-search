''' Defines different configuration of interest for research 
    Each config provides the explanation of why we would like to test the config and what we plan to collect 
    Finall CONFIG array is used to collect all configs and used in CLI entrypoint
    TODO: add here your config of interest 
'''

# Number games 
from itertools import product
from typing import Any
from cde import CDESpace
from games import CDESpaceGame, CompareOnOneGame, FocusingGame, InteractionGame, IntransitiveRegionGame, GreaterThanGame, NumberGame, run_game

from population import DEScores, HillClimbing, OneTimeSequential, ParetoLayersSelection, DESelection, RandSelection
from params import *

def get_args(kwargs: dict[str, Any], prefix):
    glob_params = {k:v for k, v in kwargs.items() if not k.startswith(prefix)}
    specific_params = {k[len(prefix):]:v for k, v in kwargs.items() if k.startswith(prefix)}
    params = {**glob_params, **specific_params}
    return params

def simulate(algo, *, cand_algo = OneTimeSequential, **kwargs):
    def start(game: InteractionGame, **kwargs2) -> dict:
        kwargsAll = {**kwargs, **kwargs2}
        candidates = cand_algo(game.get_all_candidates(), **get_args(kwargsAll, "cand_"))
        size = param_num_game_selection_size if isinstance(game, NumberGame) else param_selection_size
        if isinstance(game, IntransitiveRegionGame):
            size = (param_num_intransitive_regions * param_num_intransitive_regions) - 2
        tests = algo(game.get_all_tests(), size = size, **get_args(kwargsAll, "test_"))
        return run_game(game, candidates, tests, **get_args(kwargsAll, "sim_"))
    return start  

NUMBER_GAMES = {game.__name__: game for game in [ GreaterThanGame, FocusingGame, IntransitiveRegionGame, CompareOnOneGame ] }

def get_2ax_spanned_space(num_spanned):
    space = CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 1)
    from itertools import combinations
    cnt = 0 
    for ax_1, ax_2 in combinations(range(10), 2):
        space = space.with_spanned_point([(ax_1, -1), (ax_2, -1)], 1)
        cnt += 1
        if cnt == num_spanned:
            break
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
    # "ideal-2": build_space_game(lambda: CDESpace([5] * 2).with_test_distribution(0, 1).with_candidate_distribution(0, 1)),
    # "ideal-4": build_space_game(lambda: CDESpace([5] * 4).with_test_distribution(0, 1).with_candidate_distribution(0, 1)),
    # "ideal-6": build_space_game(lambda: CDESpace([5] * 6).with_test_distribution(0, 1).with_candidate_distribution(0, 1)),
    # "ideal-8": build_space_game(lambda: CDESpace([5] * 8).with_test_distribution(0, 1).with_candidate_distribution(0, 1)),
    "ideal": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 1)),
    
    # RQ1: Under skew of points on axes, how does D change for different algos? 
    #         Unbalanced representation of objectives/misconceptions by tests    
    #     Max ints = 2500, expected ints = 50 (cands) * 10 (axes) = 500
    "skew-p-1": build_space_game(lambda: CDESpace([6, 6, 6, 6, 6, 4, 4, 4, 4, 4]).with_test_distribution(0, 1).with_candidate_distribution(0, 1)),
    "skew-p-2": build_space_game(lambda: CDESpace([7, 7, 7, 7, 7, 3, 3, 3, 3, 3]).with_test_distribution(0, 1).with_candidate_distribution(0, 1)),
    "skew-p-3": build_space_game(lambda: CDESpace([8, 8, 8, 8, 8, 2, 2, 2, 2, 2]).with_test_distribution(0, 1).with_candidate_distribution(0, 1)),
    "skew-p-4": build_space_game(lambda: CDESpace([9, 9, 9, 9, 9, 1, 1, 1, 1, 1]).with_test_distribution(0, 1).with_candidate_distribution(0, 1)),
    
    # RQ2: Under skew of tests on points, how does ARR change for different algos (extreme to nonInfo)? Noise of test triviality and similarity
    # "trivial-1": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(1, 1).with_candidate_distribution(0, 1)), #with 1 non-info
    # "trivial-5": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(5, 1).with_candidate_distribution(0, 1)), #with 2 non-info
    "trivial-5": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(5, 1).with_candidate_distribution(0, 1)), #with 3 non-info
    # "trivial-15": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(15, 1).with_candidate_distribution(0, 1)), #with 4 non-info
    "trivial-25": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(25, 1).with_candidate_distribution(0, 1)), #with 5 non-info
    # "trivial-25": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(25, 1).with_candidate_distribution(0, 1)), #with 10 non-info
    "trivial-50": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(50, 1).with_candidate_distribution(0, 1)),
    
    "skew-t-1": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, [10,1,1,1,1]).with_candidate_distribution(0, 1)), #duplicates
    "skew-t-2": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, [1,10,1,1,1]).with_candidate_distribution(0, 1)),
    "skew-t-3": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, [1,1,10,1,1]).with_candidate_distribution(0, 1)),
    "skew-t-4": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, [1,1,1,10,1]).with_candidate_distribution(0, 1)),
    "skew-t-5": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, [1,1,1,1,10]).with_candidate_distribution(0, 1)),

    "skew-c-1": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, [10,1,1,1,1])), #duplicates
    "skew-c-2": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, [1,10,1,1,1])),
    "skew-c-3": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, [1,1,10,1,1])),
    "skew-c-4": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, [1,1,1,10,1])),
    "skew-c-5": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, [1,1,1,1,10])),
    
    # RQ3: Under presense of spanned points how R, D, ARR (ARR*) changes? Noise of test complexity
    "span-all-ends-1": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 1)\
                              .with_spanned_point([(i, -1) for i in range(10)], 1)),
    "span-all-ends-5": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 1)\
                              .with_spanned_point([(i, -1) for i in range(10)], 5)),
    "span-all-ends-10": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 1)\
                              .with_spanned_point([(i, -1) for i in range(10)], 10)),
    "span-all-ends-20": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 1)\
                              .with_spanned_point([(i, -1) for i in range(10)], 20)),
    # "span-one-pair-1": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 1)\
    #                           .with_spanned_point([(0, -1), (1, -1)], 1)),
    "span-pairs-1": build_space_game(lambda: get_2ax_spanned_space(1)),
    "span-pairs-5": build_space_game(lambda: get_2ax_spanned_space(5)),
    "span-pairs-10": build_space_game(lambda: get_2ax_spanned_space(10)),
    "span-pairs-20": build_space_game(lambda: get_2ax_spanned_space(20)),

    #RQ4: Under presense of duplicated tests how Dup, D, ARR (ARR*) changes? Noise of test similarity
    # "dupl-t-2": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 2).with_candidate_distribution(0, 1)),
    # "dupl-t-3": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 3).with_candidate_distribution(0, 1)),
    # "dupl-t-4": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 4).with_candidate_distribution(0, 1)),
    "dupl-t-5": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 5).with_candidate_distribution(0, 1)),    
    "dupl-t-10": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 10).with_candidate_distribution(0, 1)),
    "dupl-t-50": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 50).with_candidate_distribution(0, 1)),
    # "dupl-c-2": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 2)),
    # "dupl-c-3": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 3)),
    # "dupl-c-4": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 4)),
    "dupl-c-5": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 5)),
    "dupl-c-10": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 10)),
    "dupl-c-50": build_space_game(lambda: CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 50)),

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
    "rand":     simulate(RandSelection),
    "hc-r-i":   simulate(HillClimbing, test_mutation_strategy = "resample", test_selection_strategy="informativeness_select"),
    "hc-r-p":   simulate(HillClimbing, test_mutation_strategy = "resample", test_selection_strategy="pareto_select"),
    "de-l":     simulate(DESelection, test_cand_sel_strategy = "local_cand_sel_strategy"), 
    "de-d-0":   simulate(DESelection, test_cand_sel_strategy = "discr_cand_sel_strategy", test_approx_strategy="zero_approx_strategy"), 
    "de-d-1":   simulate(DESelection, test_cand_sel_strategy = "discr_cand_sel_strategy", test_approx_strategy="one_approx_strategy"), 
    "de-d-m":   simulate(DESelection, test_cand_sel_strategy = "discr_cand_sel_strategy", test_approx_strategy="maj_c_approx_strategy"), 
    "de-d-g":   simulate(DESelection, test_cand_sel_strategy = "discr_cand_sel_strategy", test_approx_strategy="candidate_group_approx_strategy"), 
    "de-d-d":   simulate(DESelection, test_cand_sel_strategy = "discr_cand_sel_strategy", test_approx_strategy="deca_approx_strategy", test_spanned_memory = 100), 
    "de-d-d-0": simulate(DESelection, test_cand_sel_strategy = "discr_cand_sel_strategy", test_approx_strategy="deca_approx_strategy", test_spanned_memory = 0), 
    "de-d-d-1": simulate(DESelection, test_cand_sel_strategy = "discr_cand_sel_strategy", test_approx_strategy="deca_approx_strategy", test_spanned_memory = 1), 
    "de-d-d-5": simulate(DESelection, test_cand_sel_strategy = "discr_cand_sel_strategy", test_approx_strategy="deca_approx_strategy", test_spanned_memory = 5), 
    "des-mea":  simulate(DEScores, test_score_strategy="mean_score_strategy"),
    "des-med":  simulate(DEScores, test_score_strategy="median_score_strategy"),
    "des-mod":  simulate(DEScores, test_score_strategy="mode_score_strategy"),
    "pl-l-0":     simulate(ParetoLayersSelection, test_cand_sel_strategy = "local_cand_sel_strategy", test_discard_spanned = 0),
    "pl-l-1":     simulate(ParetoLayersSelection, test_cand_sel_strategy = "local_cand_sel_strategy", test_discard_spanned = 1),
    "pl-d-0":     simulate(ParetoLayersSelection, test_cand_sel_strategy = "discr_cand_sel_strategy", test_discard_spanned = 0),
    "pl-d-1":     simulate(ParetoLayersSelection, test_cand_sel_strategy = "discr_cand_sel_strategy", test_discard_spanned = 1),
}

from tabulate import tabulate

def print_config():
    # printing stats of all present configs
    print(f"{len(GAMES)} games and {len(SIM)} simulations")
    rows = [[i, game_name, sim_name] for i, (game_name, sim_name) in product(GAMES.keys(), SIM.keys())]
    print(tabulate(rows, headers=["ID", "game", "simulation"], tablefmt="github"))

if __name__ == '__main__':
    # print_config()  
    cnt = 0
    with open("lst.txt", "w") as f:
        for sim_name in SIM.keys():
            for game_name in GAMES.keys():
                # game_name = "IntransitiveRegionGame"
                cnt += 1
                print(f"'{sim_name}:{game_name}'", file = f)
    print('Number of configs:', cnt)