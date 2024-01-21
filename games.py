''' Implementation of algorithms for game-like simulation 
    This type of simulation assumes co-evolution of candidates and tests and evaluation of populations on each other.     
    Games could be conducted with rule or CDESpace specified interactions.
    We consider different implementations of Number Games as rules.
    Different algos specify different simulatitons of evaluatiton
'''

from abc import ABC, abstractmethod
from itertools import product
from typing import Any
from deca import extract_dims
from population import Population
from params import PARAM_GAME_GOAL, PARAM_GAME_GOAL_MOMENT, PARAM_GAME_GOAL_STORY, PARAM_INTS, PARAM_MAX_INDS, PARAM_MAX_INTS, PARAM_UNIQ_INDS, PARAM_UNIQ_INTS, rnd
from tabulate import tabulate, SEPARATING_LINE

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

    def is_symmetric(self):
        ''' if True - forall c, t. interact(c, t) == 1 - interact(t, c) '''
        return False 
    
    def update_game_metrics(self, inds, metrics, prefix = ""):
        ''' defines in metrics how close we are to metrics '''
        pass    

class GameSimulation(ABC):
    ''' Abstract game of candidates and tests where interactions happen by given CDESpace or rule '''
    def __init__(self, game: InteractionGame) -> None:
        ''' 
            @param rule: function which defines interaction between (candidate, test) -> 2
        '''
        self.game = game
        self.name = type(self).__class__.__name__
        all_cands = game.get_all_candidates()
        all_tests = game.get_all_tests()
        self.game_metrics = {PARAM_MAX_INTS: len(all_cands) * len(all_tests), PARAM_UNIQ_INTS: 0, PARAM_INTS: 0}

    def play(self):
        ''' Executes game, final state defines found candidates and tests '''
        pass 

    def interact_groups_into(self, group1, group2, ints, allow_self_interacts = True):
        for ind1 in group1:
            for ind2 in group2:
                self.game_metrics[PARAM_INTS] += 1
                if (allow_self_interacts or ind1 != ind2) and (ind1 not in ints or ind2 not in ints[ind1]):
                    ints.setdefault(ind1, {})[ind2] = self.game.interact(ind1, ind2)            
                    self.game_metrics[PARAM_UNIQ_INTS] += 1

    def interact_groups_into_rev(self, group1, group2, ints, allow_self_interacts = True):
        for ind1 in group1:
            for ind2 in group2:
                self.game_metrics[PARAM_INTS] += 1
                if (allow_self_interacts or ind1 != ind2) and (ind1 not in ints or ind2 not in ints[ind1]):
                    ints.setdefault(ind1, {})[ind2] = 1 - self.game.interact(ind2, ind1)
                    self.game_metrics[PARAM_UNIQ_INTS] += 1

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

    def end_sim(self) -> None:
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
        self.end_sim()          

class PCHC(StepGameSimulation):
    ''' Defines PCHC game skeleton for one population of tests and candidates
        The same nature of candidates and tests are assumed
    '''
    def __init__(self, game: InteractionGame, max_steps, population: Population) -> None:
        super().__init__(game, max_steps)
        self.population = population
        self.game_metrics["cand"] = population.pop_metrics
        self.seen_inds = set()

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
        self.game.update_game_metrics(parents, self.game_metrics, prefix = "prop_")

    def end_sim(self) -> None:
        parents = self.population.get_inds(only_parents=True)
        self.game.update_game_metrics(parents, self.game_metrics, prefix = "prop_")
    
    def get_candidates(self) -> list[Any]:
        return self.population.get_inds(only_parents=True)

class PPHC(StepGameSimulation):
    ''' Implements P-PHC approach for number game with co-evolution of two populations of candidates and tests '''
    def __init__(self, game: InteractionGame, max_steps, candidates: Population, tests: Population) -> None:
        super().__init__(game, max_steps)
        self.candidates = candidates
        self.tests = tests
        self.game_metrics["cand"] = candidates.pop_metrics
        self.game_metrics["test"] = tests.pop_metrics

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
        if self.game.is_symmetric():
            self.transpose_ints(candidate_ints, test_ints)
        else:
            self.interact_groups_into(test_parents, candidate_parents, test_ints)

        if candidate_children is not candidate_parents:
            self.interact_groups_into(candidate_children, test_parents, candidate_ints)
        
        if test_children is not test_parents:
            self.interact_groups_into_rev(test_children, candidate_parents, test_ints)

        self.candidates.update(candidate_ints)
        self.tests.update(test_ints)
        self.game.update_game_metrics(candidate_parents, self.game_metrics, prefix = "cand_")
        self.game.update_game_metrics(test_parents, self.game_metrics, prefix = "test_")

    def end_sim(self) -> None:
        candidate_parents = self.candidates.get_inds(only_parents = True)
        test_parents = self.tests.get_inds(only_parents=True)
        self.game.update_game_metrics(candidate_parents, self.game_metrics, prefix = "cand_")
        self.game.update_game_metrics(test_parents, self.game_metrics, prefix = "test_")        

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
        self.game_metrics["cand"] = {PARAM_MAX_INDS: cand_sample_size, PARAM_UNIQ_INDS: cand_sample_size}
        self.game_metrics["test"] = {PARAM_MAX_INDS: test_sample_size, PARAM_UNIQ_INDS: test_sample_size}

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
        self.game_metrics["cand"] = candidates.pop_metrics
        self.game_metrics["test"] = tests.pop_metrics        

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
        if self.game.is_symmetric():
            self.transpose_ints(candidate_ints, test_ints)        
        else:
            self.interact_groups_into(tests, candidates, test_ints)
        self.candidates.update(candidate_ints)
        self.tests.update(test_ints)
        self.game.update_game_metrics(candidates, self.game_metrics, prefix = "cand_")
        self.game.update_game_metrics(tests, self.game_metrics, prefix = "test_")        

    def end_sim(self) -> None:
        candidates = self.candidates.get_inds() 
        tests = self.tests.get_inds()
        self.game.update_game_metrics(candidates, self.game_metrics, prefix = "cand_")
        self.game.update_game_metrics(tests, self.game_metrics, prefix = "test_")

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
    def __init__(self, min_num, max_num, **kwargs) -> None:
        nums = list(range(min_num, max_num + 1))
        self.all_numbers = list(product(nums, nums))
        self.min_num = min_num
        self.max_num = max_num

    def get_all_candidates(self) -> Any:
        return self.all_numbers    
    
    def get_interaction_matrix(self):
        ints = [[self.interact(num1, num2) for num2 in self.all_numbers] for num1 in self.all_numbers ]
        return ints     

    
class IntransitiveGame(NumberGame):
    ''' The IG as it was stated in Bucci article '''
    def interact(self, candidate, test) -> int:
        ''' 1 - candidate is better thah test, 0 - test fails candidate '''
        abs_diffs = [abs(x - y) for x, y in zip(candidate, test)]
        res = 1 if (abs_diffs[0] > abs_diffs[1] and candidate[1] > test[1]) or (abs_diffs[1] > abs_diffs[0] and candidate[0] > test[0]) else 0
        return res
    
    def update_game_metrics(self, numbers, metrics, prefix = ""):
        ''' the goal of this game is to check how diverse the numbers are 
            check ints_ of this game - almost each number is uniq dim/underlying objective
            returns in metrics the percent of underlying objectives
        '''
        uniq_nums = set(numbers)
        delta = 0
        if (self.max_num, self.max_num) in uniq_nums and (self.max_num - 1, self.max_num - 1) in uniq_nums:
            delta -= 1 
        if (self.min_num, self.min_num) in uniq_nums:
            delta -= 1 
        goal = (len(uniq_nums) + delta) / len(numbers)
        metrics[prefix + PARAM_GAME_GOAL] = goal
        story = metrics.setdefault(prefix + PARAM_GAME_GOAL_STORY, [])
        story.append(goal)
        if goal > 0.9:
            metrics[prefix + PARAM_GAME_GOAL_MOMENT] = len(story)
    
class IntransitiveRegionGame(IntransitiveGame):
    ''' As IG but only applies IG rule in small region, subject for search
        All other points repond with 0 - no-information
    '''
    def __init__(self, min_num, max_num, reg_min_num = 0, reg_max_num = 1, **kwargs) -> None:
        super().__init__(min_num, max_num)
        self.reg_min_num = reg_min_num
        self.reg_max_num = reg_max_num

    def is_in_range(self, n):
        res = self.reg_min_num <= n[0] <= self.reg_max_num and self.reg_min_num <= n[1] <= self.reg_max_num
        return res 
    
    def interact(self, candidate, test) -> int:
        ''' 1 - candidate is better thah test, 0 - test fails candidate '''
        if self.is_in_range(candidate) and self.is_in_range(test):
            return super().interact(candidate, test)
        else:
            return 0
        
    def update_game_metrics(self, numbers, metrics, prefix = ""):
        ''' the goal of this game is to check how diverse the numbers are 
            check ints_ of this game - almost each number is uniq dim/underlying objective
        '''
        uniq_nums = set([n for n in numbers if self.is_in_range(n)])
        delta = 0
        if (self.reg_max_num, self.reg_max_num) in uniq_nums and (self.reg_max_num - 1, self.reg_max_num - 1) in uniq_nums:
            delta -= 1 
        if (self.reg_min_num, self.reg_min_num) in uniq_nums:
            delta -= 1 
        goal = (len(uniq_nums) + delta) / len(numbers)
        metrics[prefix + PARAM_GAME_GOAL] = goal
        story = metrics.setdefault(prefix + PARAM_GAME_GOAL_STORY, [])
        story.append(goal)
        if goal > 0.9:
            metrics[prefix + PARAM_GAME_GOAL_MOMENT] = len(story)

class FocusingGame(NumberGame):
    ''' The FG as it was stated in Bucci article '''
    def interact(self, candidate, test) -> int:
        ''' 1 - candidate is better thah test, 0 - test fails candidate '''
        res = 1 if (test[0] > test[1] and candidate[0] > test[0]) or (test[1] > test[0] and candidate[1] > test[1]) else 0
        return res
    
    def update_game_metrics(self, numbers, metrics, prefix = ""):
        ''' the goal of this game is to focus on two axes (0, max_dim) and (max_dim, 0)
            Check ints_ file of this game to understand thee structure of underelying objectives
            We compute average distance to either of two objectives. But two objectives should be discovered
        '''
        goals = [(0, self.max_num), (self.max_num, 0)]
        goal_dists = [[sum(abs(a - b) for a, b in zip(g, n)) for g in goals] for n in numbers]
        number_goals = [np.argmin(n) for n in goal_dists]
        numbers_by_goals = [[], []]
        for goal_id, dists in zip(number_goals, goal_dists):
            numbers_by_goals[goal_id].append(dists[goal_id])
        goal_scores = [min(gd, default=self.max_num) for gd in numbers_by_goals]
        score = sum(goal_scores) / len(goal_scores)
        metrics[prefix + "game_goal_1"] = goal_scores[0]
        metrics[prefix + "game_goal_2"] = goal_scores[1]
        metrics.setdefault(prefix + "game_goal_1_story", []).append(goal_scores[0])
        metrics.setdefault(prefix + "game_goal_2_story", []).append(goal_scores[1])
        metrics[prefix + PARAM_GAME_GOAL] = score #smaller is better
        metrics.setdefault(prefix + PARAM_GAME_GOAL_STORY, []).append(score)
    
class CompareOnOneGame(NumberGame):
    ''' Game as it was stated in Golam article '''
    def interact(self, candidate, test) -> int:
        ''' 1 - candidate is better than test, 0 - test fails candidate '''
        max_pos = np.argmax(test)
        res = 1 if candidate[max_pos] >= test[max_pos] else 0
        return res
    
    def update_game_metrics(self, numbers, metrics, prefix = ""):
        ''' the goal of this game is to focus on two axes (0, max_dim) and (max_dim, 0)
            Check ints_ file of this game to understand thee structure of underelying objectives
            We compute average distance to either of two objectives. But two objectives should be discovered
        '''
        goals = [(0, self.max_num), (self.max_num, 0)]
        goal_dists = [[sum(abs(a - b) for a, b in zip(g, n)) for g in goals] for n in numbers]
        number_goals = [np.argmin(n) for n in goal_dists]
        numbers_by_goals = [[], []]
        for goal_id, dists in zip(number_goals, goal_dists):
            numbers_by_goals[goal_id].append(dists[goal_id])
        goal_scores = [min(gd, default=self.max_num) for gd in numbers_by_goals]
        score = sum(goal_scores) / len(goal_scores)
        metrics[prefix + "game_goal_1"] = goal_scores[0]
        metrics[prefix + "game_goal_2"] = goal_scores[1]
        metrics.setdefault(prefix + "game_goal_1_story", []).append(goal_scores[0])
        metrics.setdefault(prefix + "game_goal_2_story", []).append(goal_scores[1])
        metrics[prefix + PARAM_GAME_GOAL] = score #smaller is better
        metrics.setdefault(prefix + PARAM_GAME_GOAL_STORY, []).append(score)   
    
# game that is based on CDESpace 
class CDESpaceGame(InteractionGame):
    ''' Loads given CDE space and provide interactions based on it '''
    def __init__(self, space: CDESpace, **kwargs) -> None:
        self.space = space 
        self.candidates = sorted(space.get_candidates())
        self.tests = sorted(space.get_tests())
        self.all_fails = space.get_candidate_fails()

    def get_all_candidates(self) -> Any:
        return self.candidates
    
    def get_all_tests(self) -> Any:
        return self.tests
    
    def is_symmetric(self):
        return True
    
    def interact(self, candidate, test) -> int:
        ''' 1 - candidate wins, 0 - test wins '''
        return 0 if test in self.all_fails.get(candidate, set()) else 1
    
if __name__ == '__main__':
    # game = IntransitiveRegionGame(1, 4, 0, 5)
    game = IntransitiveGame(0, 3)
    ints = game.get_interaction_matrix()
    dims, origin, spanned, duplicates = extract_dims(ints)
    dim_nums = [[game.all_numbers[i] for i in dim] for dim in dims]
    origin_nums = [game.all_numbers[i] for i in origin]
    spanned_nums = [game.all_numbers[i] for i in spanned]
    duplicates_nums = [game.all_numbers[i] for i in duplicates]
    # for num, line in zip(game.all_numbers, ints):
    #     print(f"{num}: {line}")

    max_dim = max([len(dim) for dim in dim_nums])
            
    rows = []    
    for dim_id, dim in enumerate(sorted(dims, key = lambda dim:game.all_numbers[dim[-1]])):
        # print(f"Dim {dim_id}")
        for i in dim:
            # print(f"{game.all_numbers[i]}: {ints[i]}")
            row = []
            num = game.all_numbers[i]
            row.append(dim_id)
            row.append(f"{num[0]},{num[1]}")
            row.extend(["" if o == 0 else 1 for o in ints[i]])
            rows.append(row)    
        rows.append(SEPARATING_LINE)
    if len(origin) > 0:        
        for num, i in zip(origin_nums, origin):
            rows.append(["ORIG", f"{num[0]},{num[1]}", *["" if o == 0 else 1 for o in ints[i]]])
        rows.append(SEPARATING_LINE)
    if len(spanned) > 0:        
        for num, i in zip(spanned_nums, spanned):
            rows.append(["SPAN", f"{num[0]},{num[1]}", *["" if o == 0 else 1 for o in ints[i]]])        
        rows.append(SEPARATING_LINE)
    if len(duplicates) > 0:        
        for num, i in zip(duplicates_nums, duplicates):
            rows.append(["DUPL", f"{num[0]},{num[1]}", *["" if o == 0 else 1 for o in ints[i]]])        
        rows.append(SEPARATING_LINE)
    int_rows = [["", f"{n[0]},{n[1]}", *["" if o == 0 else 1 for o in ind_ints]] for i, ind_ints in enumerate(ints) for n in [game.all_numbers[i]]]
    int_headers = ["   ", "num", *[f"{n[0]},{n[1]}" for n in game.all_numbers]]
    with open('ints_' + game.__class__.__name__ + ".txt", "w") as f:    
        print(f"Interactions of {game.__class__.__name__}", file=f)
        print(tabulate(int_rows, headers=int_headers, tablefmt = "simple", numalign="center", stralign="center"), file=f)
        print("\n", file=f)
        print(f"Dims: {len(dim_nums)}, points={max_dim} Origin: {len(origin_nums)} Spanned: {len(spanned_nums)} Duplicates: {len(duplicates_nums)}", file=f)
        print(tabulate(rows, headers=["dim", "num", *[f"{n[0]},{n[1]}" for n in game.all_numbers]], tablefmt='simple', numalign="center", stralign="center"), file=f)
    # print(f"{game.all_numbers}")

    # print(f"Spanned")
    # for i in spanned:
    #     print(f"{game.all_numbers[i]}: {ints[i]}")
