''' Defines different configuration of interest for research 
    Each config provides the explanation of why we would like to test the config and what we plan to collect 
    Finall CONFIG array is used to collect all configs and used in CLI entrypoint
    TODO: add here your config of interest 
'''

# Number games 
from typing import Any
from cde import CDESpace
from games import PCHC, PPHC, CDESpaceGame, CandidateTestInteractions, CompareOnOneGame, FocusingGame, InteractionGame, IntransitiveGame, RandSampling

from population import ACOPopulation, HCPopulation, OneTimeSequential, ParetoGraphSample, SamplingStrategySample
from params import *

def set_name(sim, kwargs):
    if "name" in kwargs:
        sim.name = kwargs["name"]
    return sim

def get_args(kwargs: dict[str, Any], prefix):
    glob_params = {k:v for k, v in kwargs.items() if not k.startswith(prefix)}
    specific_params = {k[len(prefix):]:v for k, v in kwargs.items() if k.startswith(prefix)}
    params = {**glob_params, **specific_params}
    return params

def PCHC_SIM(popsize = 100, **kwargs):
    def b(game: InteractionGame):
        return set_name(PCHC(game, param_steps, HCPopulation(game.get_all_candidates(), popsize, **kwargs)), kwargs)
    return b

def PPHC_SIM(popsize = 50, **kwargs):
    def b(game: InteractionGame):
        candidates = HCPopulation(game.get_all_candidates(), popsize, **get_args(kwargs, "cand_"))
        tests = HCPopulation(game.get_all_tests(), popsize, **get_args(kwargs, "test_"))
        sim = set_name(PPHC(game, param_steps, candidates, tests), kwargs)
        return sim
    return b

def SS_SIM(popsize = 50, **kwargs):
    def b(game: InteractionGame):
        candidates = SamplingStrategySample(game.get_all_candidates(), popsize, **get_args(kwargs, "cand_"))
        tests = SamplingStrategySample(game.get_all_tests(), popsize, **get_args(kwargs, "test_"))
        sim = set_name(CandidateTestInteractions(game, param_steps, candidates, tests), kwargs)
        return sim    
    return b

def PGS_SIM(popsize = 50, **kwargs):
    def b(game: InteractionGame):
        candidates = ParetoGraphSample(game.get_all_candidates(), popsize, **get_args(kwargs, "cand_"))
        tests = ParetoGraphSample(game.get_all_tests(), popsize, **get_args(kwargs, "test_"))
        sim = set_name(CandidateTestInteractions(game, param_steps, candidates, tests), kwargs)
        return sim 
    return b

def ACO_SIM(popsize = 50, **kwargs):
    def b(game: InteractionGame):
        candidates = ACOPopulation(game.get_all_candidates(), popsize, **get_args(kwargs, "cand_"))
        tests = ACOPopulation(game.get_all_tests(), popsize, **get_args(kwargs, "test_"))
        sim = set_name(CandidateTestInteractions(game, param_steps, candidates, tests), kwargs)
        return sim
    return b

GAMES = [ 
    *[g(param_min_num, param_max_num) for g in [IntransitiveGame, FocusingGame, CompareOnOneGame] ] 
]

#TODO: add another simulations
GAME_SIM = [ 
        PCHC_SIM(name="pchc-pmo-w", mutation = "plus_minus_one", selection = "num_wins"), 
        PCHC_SIM(name="pchc-r-w", mutation = "resample", selection = "num_wins"),
        PPHC_SIM(name="pphc-pmo-p-i", mutation = "plus_minus_one", cand_selection="pareto_select", test_selection="informativeness_select"), 
        PPHC_SIM(name="pphc-pmo-p-p", mutation = "plus_minus_one", cand_selection="pareto_select", test_selection="pareto_select"),        
        PPHC_SIM(name="pchc-r-p-i", mutation = "resample", cand_selection="pareto_select", test_selection="informativeness_select"), 
        PPHC_SIM(name="pchc-r-p-p", mutation = "resample", cand_selection="pareto_select", test_selection="pareto_select"),
        SS_SIM(name="s-0_nd", strategy=[{"t":0, "keys":[["nond", "kn"], ["nond", "kn"], ["nond", "kn"]]}]),
        SS_SIM(name="s-0_ndXdm", strategy=[{"t":0, "keys":[["nd", "kn"], ["nd", "kn"], ["nd", "kn"]]}]),
        SS_SIM(name="s-0_dm", strategy=[{"t":0, "keys":[["dom", "kn"], ["dom", "kn"], ["dom", "kn"]]}]),
        SS_SIM(name="s-0_d", strategy=[{"t":0, "keys":[["d", "kn"], ["d", "kn"], ["d", "kn"]]}]),
        SS_SIM(name="s-0_sXd", strategy=[{"t":0, "keys":[["sd", "kn"], ["sd", "kn"], ["sd", "kn"]]}]),
        SS_SIM(name="s-0_Zs", strategy=[{"t":0, "keys":[["-s", "kn"], ["-s", "kn"], ["-s", "kn"]]}]),
        SS_SIM(name="s-0_k", strategy=[{"t":0, "keys":[["kn"], ["kn"], ["kn"]]}]),

        SS_SIM(name="s-0_dp_kn", strategy=[{"t":0, "keys":[["dup","kn"], ["dup","kn"], ["dup","kn"]]}]),
        SS_SIM(name="s-0_dp_d", strategy=[{"t":0, "keys":[["dup", "d", "kn"], ["dup", "d", "kn"], ["dup", "d", "kn"]]}]),
        SS_SIM(name="s-0_dp_sXd", strategy=[{"t":0, "keys":[["dup", "sd", "kn"], ["dup", "sd", "kn"], ["dup", "sd", "kn"]]}]),
        SS_SIM(name="s-0_dp_kn", strategy=[{"t":0, "keys":[["d", "kn"], ["d", "kn"], ["d", "kn"]]}]),
        SS_SIM(name="s-0_dp_Zs", strategy=[{"t":0, "keys":[["dup", "-s", "kn"], ["dup", "-s", "kn"], ["dup", "-s", "kn"]]}]),
        SS_SIM(name="s-0_dp_dm", strategy=[{"t":0, "keys":[["dup", "dom", "kn"], ["dup", "dom", "kn"], ["dup", "dom", "kn"]]}]),
        SS_SIM(name="s-0_dp_nd", strategy=[{"t":0, "keys":[["dup", "nond", "kn"], ["dup", "nond", "kn"], ["dup", "nond", "kn"]]}]),
        SS_SIM(name="s-0_dp_ndXdm", strategy=[{"t":0, "keys":[["dup", "nd", "kn"], ["dup", "nd", "kn"], ["dup", "nd", "kn"]]}]),

        PGS_SIM(name="pg-1-50-80", rank_penalty = 1, min_exploitation_chance = 0.5, max_exploitation_chance = 0.8), 
        PGS_SIM(name="pg-2-50-80", rank_penalty = 2, min_exploitation_chance = 0.5, max_exploitation_chance = 0.8), 
        PGS_SIM(name="pg-2-80-90", rank_penalty = 2, min_exploitation_chance = 0.8, max_exploitation_chance = 0.9),

        ACO_SIM(name="aco-50-25", pheromone_decay = 0.5, dom_bonus = 1, span_penalty = 0.25),
        ACO_SIM(name="aco-80-25", pheromone_decay = 0.8, dom_bonus = 1, span_penalty = 0.25),
        ACO_SIM(name="aco-80-50", pheromone_decay = 0.8, dom_bonus = 1, span_penalty = 0.50)        
]

def RAND_SPACE_SIM(popsize = 10, **kwargs):
    def b(game: CDESpaceGame):
        sim = set_name(RandSampling(game, popsize, popsize), kwargs)
        return sim
    return b    

def PPHC_SPACE_SIM(num_cands = 2, popsize = 10, **kwargs):
    def b(game: CDESpaceGame):
        candidates = OneTimeSequential(game.get_all_candidates(), num_cands, **get_args(kwargs, "cand_"))
        max_steps = len(candidates.get_all_inds()) / num_cands
        tests = HCPopulation(game.get_all_tests(), popsize, **get_args(kwargs, "test_"))
        sim = set_name(PPHC(game, max_steps, candidates, tests), kwargs)
        return sim
    return b

def SS_SPACE_SIM(num_cands = 2, popsize = 10, **kwargs):
    def b(game: CDESpaceGame):
        candidates = OneTimeSequential(game.get_all_candidates(), num_cands, **get_args(kwargs, "cand_"))
        max_steps = len(candidates.get_all_inds()) / num_cands
        tests = SamplingStrategySample(game.get_all_tests(), popsize, **get_args(kwargs, "test_"))
        sim = set_name(CandidateTestInteractions(game, max_steps, candidates, tests), kwargs)
        return sim    
    return b

def PGS_SPACE_SIM(num_cands = 2, popsize = 10, **kwargs):
    def b(game: CDESpaceGame):
        candidates = OneTimeSequential(game.get_all_candidates(), num_cands, **get_args(kwargs, "cand_"))
        max_steps = len(candidates.get_all_inds()) / num_cands
        tests = ParetoGraphSample(game.get_all_tests(), popsize, **get_args(kwargs, "test_"))
        sim = set_name(CandidateTestInteractions(game, max_steps, candidates, tests), kwargs)
        return sim 
    return b

def ACO_SPACE_SIM(num_cands = 2, popsize = 10, **kwargs):
    def b(game: InteractionGame):
        candidates = OneTimeSequential(game.get_all_candidates(), num_cands, **get_args(kwargs, "cand_"))
        max_steps = len(candidates.get_all_inds()) / num_cands
        tests = ACOPopulation(game.get_all_tests(), popsize, **get_args(kwargs, "test_"))
        sim = set_name(CandidateTestInteractions(game, max_steps, candidates, tests), kwargs)
        return sim
    return b

def load_space_games():
    space_games = []
    with open(param_spaces, "r") as f:        
        for line in f.readlines():
            if not line.startswith("//"):
                try:
                    space = CDESpace.from_json(line)
                    space_games.append(CDESpaceGame(space))
                except:
                    pass 
    return space_games

SPACE_SIM = [

    RAND_SPACE_SIM(name="rand-10"),

    PPHC_SPACE_SIM(name="pphc-pmo-i", mutation = "plus_minus_one", test_selection="informativeness_select"), 
    PPHC_SPACE_SIM(name="pphc-pmo-p", mutation = "plus_minus_one", test_selection="pareto_select"),        
    PPHC_SPACE_SIM(name="pchc-r-i", mutation = "resample", test_selection="informativeness_select"), 
    PPHC_SPACE_SIM(name="pchc-r-p", mutation = "resample", test_selection="pareto_select"),
    
    SS_SPACE_SIM(name="s-0_nd", strategy=[{"t":0, "keys":[["nond", "kn"], ["nond", "kn"], ["nond", "kn"]]}]),
    SS_SPACE_SIM(name="s-0_ndXdm", strategy=[{"t":0, "keys":[["nd", "kn"], ["nd", "kn"], ["nd", "kn"]]}]),
    SS_SPACE_SIM(name="s-0_dm", strategy=[{"t":0, "keys":[["dom", "kn"], ["dom", "kn"], ["dom", "kn"]]}]),
    SS_SPACE_SIM(name="s-0_d", strategy=[{"t":0, "keys":[["d", "kn"], ["d", "kn"], ["d", "kn"]]}]),
    SS_SPACE_SIM(name="s-0_sXd", strategy=[{"t":0, "keys":[["sd", "kn"], ["sd", "kn"], ["sd", "kn"]]}]),
    SS_SPACE_SIM(name="s-0_Zs", strategy=[{"t":0, "keys":[["-s", "kn"], ["-s", "kn"], ["-s", "kn"]]}]),
    SS_SPACE_SIM(name="s-0_k", strategy=[{"t":0, "keys":[["kn"], ["kn"], ["kn"]]}]),

    SS_SPACE_SIM(name="s-0_dp_kn", strategy=[{"t":0, "keys":[["dup","kn"], ["dup","kn"], ["dup","kn"]]}]),
    SS_SPACE_SIM(name="s-0_dp_d", strategy=[{"t":0, "keys":[["dup", "d", "kn"], ["dup", "d", "kn"], ["dup", "d", "kn"]]}]),
    SS_SPACE_SIM(name="s-0_dp_sXd", strategy=[{"t":0, "keys":[["dup", "sd", "kn"], ["dup", "sd", "kn"], ["dup", "sd", "kn"]]}]),
    SS_SPACE_SIM(name="s-0_dp_kn", strategy=[{"t":0, "keys":[["d", "kn"], ["d", "kn"], ["d", "kn"]]}]),
    SS_SPACE_SIM(name="s-0_dp_Zs", strategy=[{"t":0, "keys":[["dup", "-s", "kn"], ["dup", "-s", "kn"], ["dup", "-s", "kn"]]}]),
    SS_SPACE_SIM(name="s-0_dp_dm", strategy=[{"t":0, "keys":[["dup", "dom", "kn"], ["dup", "dom", "kn"], ["dup", "dom", "kn"]]}]),
    SS_SPACE_SIM(name="s-0_dp_nd", strategy=[{"t":0, "keys":[["dup", "nond", "kn"], ["dup", "nond", "kn"], ["dup", "nond", "kn"]]}]),
    SS_SPACE_SIM(name="s-0_dp_ndXdm", strategy=[{"t":0, "keys":[["dup", "nd", "kn"], ["dup", "nd", "kn"], ["dup", "nd", "kn"]]}]),

    PGS_SPACE_SIM(name="pg-1-50-80", rank_penalty = 1, min_exploitation_chance = 0.5, max_exploitation_chance = 0.8), 
    PGS_SPACE_SIM(name="pg-2-50-80", rank_penalty = 2, min_exploitation_chance = 0.5, max_exploitation_chance = 0.8), 
    PGS_SPACE_SIM(name="pg-2-80-90", rank_penalty = 2, min_exploitation_chance = 0.8, max_exploitation_chance = 0.9),

    ACO_SPACE_SIM(name="aco-50-25", pheromone_decay = 0.5, dom_bonus = 1, span_penalty = 0.25),
    ACO_SPACE_SIM(name="aco-80-25", pheromone_decay = 0.8, dom_bonus = 1, span_penalty = 0.25),
    ACO_SPACE_SIM(name="aco-80-50", pheromone_decay = 0.8, dom_bonus = 1, span_penalty = 0.50)        

]

GAMES_CONFIG = [ sim(game) for sim in GAME_SIM for game in GAMES ]

SPACE_CONFIG = [ sim(space) for sim in SPACE_SIM for space in load_space_games() ]

CONFIG = [ *GAMES_CONFIG, *SPACE_CONFIG ]

if __name__ == '__main__':
    print(f"{len(CONFIG)} configs present. Games: {len(GAMES_CONFIG)}, Spaces: {len(SPACE_CONFIG)}")