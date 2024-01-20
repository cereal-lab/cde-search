''' Defines different configuration of interest for research 
    Each config provides the explanation of why we would like to test the config and what we plan to collect 
    Finall CONFIG array is used to collect all configs and used in CLI entrypoint
    TODO: add here your config of interest 
'''

# Number games 
from typing import Any
from games import PCHC, PPHC, CandidateTestInteractions, CompareOnOneGame, FocusingGame, InteractionGame, IntransitiveGame

from population import HCPopulation, ParetoGraphSample, SamplingStrategySample
from params import *

def PCHC_SIM(popsize = 100, **kwargs):
    def b(game: InteractionGame):
        return PCHC(game, param_steps, HCPopulation(game.get_all_candidates(), popsize, **kwargs))
    return b

def get_args(kwargs: dict[str, Any], prefix):
    glob_params = {k:v for k, v in kwargs.items() if not k.startswith(prefix)}
    specific_params = {k[len(prefix):]:v for k, v in kwargs.items() if k.startswith(prefix)}
    params = {**glob_params, **specific_params}
    return params

def PPHC_SIM(popsize = 50, **kwargs):
    def b(game: InteractionGame):
        candidates = HCPopulation(game.get_all_candidates(), popsize, **get_args(kwargs, "cand_"))
        tests = HCPopulation(game.get_all_tests(), popsize, **get_args(kwargs, "test_"))
        sim = PPHC(game, param_steps, candidates, tests)
        return sim
    return b

def SS_SIM(popsize = 50, **kwargs):
    def b(game: InteractionGame):
        candidates = SamplingStrategySample(game.get_all_candidates(), popsize, **get_args(kwargs, "cand_"))
        tests = SamplingStrategySample(game.get_all_tests(), popsize, **get_args(kwargs, "test_"))
        sim = CandidateTestInteractions(game, param_steps, candidates, tests)
        return sim    
    return b

def PGS_SIM(popsize = 50, **kwargs):
    def b(game: InteractionGame):
        candidates = ParetoGraphSample(game.get_all_candidates(), popsize, **get_args(kwargs, "cand_"))
        tests = ParetoGraphSample(game.get_all_tests(), popsize, **get_args(kwargs, "test_"))
        sim = CandidateTestInteractions(game, param_steps, candidates, tests)
        return sim 
    return b

GAMES = [ 
    *[g(param_min_num, param_max_num) for g in [IntransitiveGame, FocusingGame, CompareOnOneGame] ] 
]

#TODO: add another simulations
SIM = [ PCHC_SIM(mutation = "plus_minus_one", selection = "num_wins"), 
        PPHC_SIM(mutation = "plus_minus_one", cand_selection="pareto_select", test_selection="informativeness_select"), 
        SS_SIM(), 
        PGS_SIM() 
]

CONFIG = [ sim(game) for sim in SIM for game in GAMES ]

if __name__ == '__main__':
    print(f"{len(CONFIG)} configs present")