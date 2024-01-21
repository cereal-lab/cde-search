''' Implementation of algorithms for game-like simulation 
    This type of simulation assumes co-evolution of candidates and tests and evaluation of populations on each other.     
    Games could be conducted with rule or CDESpace specified interactions.
    We consider different implementations of Number Games as rules.
    Different algos specify different simulatitons of evaluatiton
'''

from abc import ABC, abstractmethod
from itertools import product
from typing import Any
from population import Population
from params import rnd

import numpy as np

from cde import CDESpace
    
class InteractionGame(ABC):
    ''' Base class that encapsulates representation of candidates/tests and their interactions 
        Candidates and tests are implemented as finite sets! 
    '''
    @abstractmethod
    def get_all_candidates(self) -> list[Any]:
        pass 

    def get_all_tests(self) -> list[Any]: 
        ''' by default assumes that tests are also candidates. Override if not'''
        return self.get_all_candidates()

    @abstractmethod
    def interact(self, candidate, test) -> int:
        ''' Should return 1 when candidate success the test (test is not better than candidate)! '''
        pass 

class GameSimulation(ABC):
    ''' Abstract game of candidates and tests where interactions happen by given CDESpace or rule '''
    def __init__(self, game: InteractionGame) -> None:
        ''' 
            @param rule: function which defines interaction between (candidate, test) -> 2
        '''
        self.game = game
        self.name = type(self).__class__.__name__
        self.game_metrics = {}

    def play(self):
        ''' Executes game, final state defines found candidates and tests '''
        pass 

    def interact_groups_into(self, group1, group2, ints, allow_self_interacts = True):
        for ind1 in group1:
            for ind2 in group2:
                if (allow_self_interacts or ind1 != ind2) and (ind1 not in ints or ind2 not in ints[ind1]):
                    ints.setdefault(ind1, {})[ind2] = self.game.interact(ind1, ind2)  

    def interact_groups_into_rev(self, group1, group2, ints, allow_self_interacts = True):
        for ind1 in group1:
            for ind2 in group2:
                if (allow_self_interacts or ind1 != ind2) and (ind1 not in ints or ind2 not in ints[ind1]):
                    ints.setdefault(ind1, {})[ind2] = 1 - self.game.interact(ind2, ind1)      

    def transpose_ints(self, ints, into_ints):
        for c, t_ints in ints.items():
            for t, outcome in t_ints.items():
                into_ints.setdefault(t, {})[c] = 1 - outcome        

    @abstractmethod
    def get_candidates(self) -> list[Any]:
        ''' Gets candidates from game simulation state '''
        pass
    
    def get_tests(self) -> list[Any]:
        ''' Gets tests from game simulation state. Assumes that tests and candidates are same entity '''
        return self.get_candidates()

class StepGameSimulation(GameSimulation):
    ''' Defines step based game with group of interactions on each step '''
    def __init__(self, game: InteractionGame, max_steps) -> None:
        super().__init__(game)
        self.max_steps = max_steps
        self.step = 0

    @abstractmethod
    def init_sim(self) -> None:
        ''' creates initial game simulation state '''
        pass 
    @abstractmethod 
    def interact(self):
        ''' interactions on one step of the game with state update '''
        pass 
    def play(self):
        self.init_sim()
        self.step = 0
        while self.step < self.max_steps:
            self.interact()
            self.step += 1

class PCHC(StepGameSimulation):
    ''' Defines PCHC game skeleton for one population of tests and candidates
        The same nature of candidates and tests are assumed
    '''
    def __init__(self, game: InteractionGame, max_steps, population: Population) -> None:
        super().__init__(game, max_steps)
        self.population = population

    def init_sim(self) -> None:
        self.population.init_inds()

    def interact(self):
        ''' computes interactions of ind on other inds in population (no-children)'''
        parents = self.population.get_inds(only_parents=True)
        children = self.population.get_inds(only_children=True)
        ints = {}
        self.interact_groups_into(parents, parents, ints, allow_self_interacts=False)
        if children is not parents:
            self.interact_groups_into(children, parents, ints, allow_self_interacts=False)
        self.population.update(ints)
    
    def get_candidates(self) -> list[Any]:
        return self.population.get_inds(only_parents=True)

class PPHC(StepGameSimulation):
    ''' Implements P-PHC approach for number game with co-evolution of two populations of candidates and tests '''
    def __init__(self, game: InteractionGame, max_steps, candidates: Population, tests: Population) -> None:
        super().__init__(game, max_steps)
        self.candidates = candidates
        self.tests = tests

    def init_sim(self) -> None:
        ''' creates populations of candidates and tests '''
        self.candidates.init_inds()
        self.tests.init_inds()

    def interact(self):
        ''' plays one step interaction between populations of candidates and tests and their children '''
        # 1 - * for the fact that 1 represents that candidate success on the test
        candidate_parents = self.candidates.get_inds(only_parents = True)
        candidate_children = self.candidates.get_inds(only_children = True)
        test_parents = self.tests.get_inds(only_parents=True)
        test_children = self.tests.get_inds(only_children=True)

        candidate_ints = {}
        self.interact_groups_into(candidate_parents, test_parents, candidate_ints)
        test_ints = {}
        self.transpose_ints(candidate_ints, test_ints)

        if candidate_children is not candidate_parents:
            self.interact_groups_into(candidate_children, test_parents, candidate_ints)
        
        if test_children is not test_parents:
            self.interact_groups_into_rev(test_children, candidate_parents, test_ints)

        self.candidates.update(candidate_ints)
        self.tests.update(test_ints)

    def get_candidates(self) -> list[Any]:
        return self.candidates.get_inds(only_parents=True)

    def get_tests(self) -> list[Any]:
        return self.tests.get_inds(only_parents=True)

class RandSampling(GameSimulation):
    ''' Samples the individuals from game as provide it as result '''
    def __init__(self, game: InteractionGame, cand_sample_size, test_sample_size) -> None:
        super().__init__(game)
        self.cand_sample_size = cand_sample_size
        self.test_sample_size = test_sample_size

    def get_candidates(self) -> list[Any]:
        return rnd.choice(self.game.get_all_candidates(), size = self.cand_sample_size, replace=False)
    
    def get_tests(self) -> list[Any]:
        return rnd.choice(self.game.get_all_tests(), size = self.test_sample_size, replace=False)    

class CandidateTestInteractions(StepGameSimulation):
    ''' Base class for different sampling approaches based on interaction matrix '''
    def __init__(self, game: InteractionGame, max_steps, candidates: Population, tests: Population) -> None:
        super().__init__(game, max_steps)
        self.candidates = candidates
        self.tests = tests
        self.candidates_first = True

    def init_sim(self) -> None:
        self.candidates.init_inds()
        self.tests.init_inds()

    def interact(self):
        if self.candidates_first:
            candidates = self.candidates.get_inds() 
            tests = self.tests.get_inds(for_group = candidates)
        else:
            tests = self.tests.get_inds()  
            candidates = self.candidates.get_inds(for_group = tests)
        self.candidates_first = not self.candidates_first           
        candidate_ints = {}
        self.interact_groups_into(candidates, tests, candidate_ints)
        test_ints = {}
        self.transpose_ints(candidate_ints, test_ints)        
        self.candidates.update(candidate_ints)
        self.tests.update(test_ints)

    def get_candidates(self) -> list[Any]:
        return self.candidates.get_inds()

    def get_tests(self) -> list[Any]:
        return self.tests.get_inds()
    
# set of games 
class NumberGame(InteractionGame):
    ''' Base class for number games with default representation of n-dim Nat tuples 
        Init is random in given nat range per dim 
        Change is +-1 random per dimension
    '''
    def __init__(self, min_num, max_num) -> None:
        nums = list(range(min_num, max_num + 1))
        self.all_numbers = list(product(nums, nums))

    def get_all_candidates(self) -> Any:
        return self.all_numbers    
    
class IntransitiveGame(NumberGame):
    ''' The IG as it was stated in Bucci article '''
    def interact(self, candidate, test) -> int:
        ''' 1 - candidate is better thah test, 0 - test fails candidate '''
        abs_diffs = [abs(x - y) for x, y in zip(candidate, test)]
        res = 1 if (abs_diffs[0] > abs_diffs[1] and candidate[1] > test[1]) or (abs_diffs[1] > abs_diffs[0] and candidate[0] > test[0]) else 0
        return res

class FocusingGame(NumberGame):
    ''' The FG as it was stated in Bucci article '''
    def interact(self, candidate, test) -> int:
        ''' 1 - candidate is better thah test, 0 - test fails candidate '''
        res = 1 if (test[0] > test[1] and candidate[0] > test[0]) or (test[1] > test[0] and candidate[1] > test[1]) else 0
        return res
    
class CompareOnOneGame(NumberGame):
    ''' Game as it was stated in Golam article '''
    def interact(self, candidate, test) -> int:
        ''' 1 - candidate is better thah test, 0 - test fails candidate '''
        max_pos = np.argmax(test)
        res = 1 if candidate[max_pos] >= test[max_pos] else 0
        return res
    
# game that is based on CDESpace 
class CDESpaceGame(InteractionGame):
    ''' Loads given CDE space and provide interactions based on it '''
    def __init__(self, space: CDESpace) -> None:
        self.space = space 
        self.candidates = sorted(space.get_candidates())
        self.tests = sorted(space.get_tests())
        self.all_fails = space.get_candidate_fails()

    def get_all_candidates(self) -> Any:
        return self.candidates
    
    def get_all_tests(self) -> Any:
        return self.tests
    
    def interact(self, candidate, test) -> int:
        ''' 1 - candidate wins, 0 - test wins '''
        return 0 if test in self.all_fails.get(candidate, set()) else 1