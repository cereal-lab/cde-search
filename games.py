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
from population import OneTimeSequential, Population
from params import PARAM_GAME_GOAL, PARAM_GAME_GOAL_MOMENT, PARAM_GAME_GOAL_STORY,\
        PARAM_MAX_INDS, PARAM_MAX_INTS, PARAM_UNIQ_INDS, rnd, param_reg_min_num, param_reg_max_num,\
        param_min_num, param_max_num
from tabulate import tabulate, SEPARATING_LINE

import numpy as np

from cde import CDESpace
from viz import draw_populations
    
class InteractionGame(ABC):
    ''' Base class that encapsulates representation of candidates/tests and their interactions 
        Candidates and tests are implemented as finite sets! 
    '''
    def __init__(self) -> None:
        self.game_params = {"class": self.__class__.__name__}

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
    
    def update_game_metrics(self, inds, metrics, is_final = False):
        ''' defines in metrics how close we are to metrics '''
        pass    

    def get_game_params() -> dict:
        return {}

class GameSimulation(ABC):
    ''' Abstract game of candidates and tests where interactions happen by given CDESpace or rule '''
    def __init__(self, game: InteractionGame) -> None:
        ''' 
            @param rule: function which defines interaction between (candidate, test) -> 2
        '''
        self.game = game
        all_cands = game.get_all_candidates()
        all_tests = game.get_all_tests()
        self.game_metrics = {PARAM_MAX_INTS: len(all_cands) * len(all_tests)}
        self.sim_params = {"class": self.__class__.__name__}    

    def play(self):
        ''' Executes game, final state defines found candidates and tests '''
        pass 

    @abstractmethod
    def get_candidates(self) -> list[Any]:
        ''' Gets candidates from game simulation state '''
        pass
    
    def get_tests(self) -> list[Any]:
        ''' Gets tests from game simulation state. Assumes that tests and candidates are same entity '''
        return self.get_candidates()

class StepGameSimulation(GameSimulation):
    ''' Defines step based game with group of interactions on each step '''
    def __init__(self, game: InteractionGame, max_steps, draw_dynamics = False) -> None:
        super().__init__(game)
        self.max_steps = max_steps
        self.step = 0
        self.sim_params["param_max_steps"] = max_steps
        self.draw_dynamics = draw_dynamics

    @abstractmethod
    def init_sim(self) -> None:
        ''' creates initial game simulation state '''
        pass 

    def end_sim(self) -> None:
        pass 

    def draw(self, parents1, children1, parents2, children2, seen_inds, logs):
        if self.draw_dynamics: #at this point we asume number game            
            name = f"{self.step}"
            name = name if len(name) >= 4 else ("0" * (4 - len(name)) + name)
            draw_populations(parents1, children1, parents2, children2,
                             xrange=(self.game.min_num, self.game.max_num), yrange=(self.game.min_num, self.game.max_num),
                             name = f"step-{name}", title = f"{self.__class__.__name__} on {self.game.__class__.__name__}",
                             prev = seen_inds, legend=logs)

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
    def __init__(self, game: InteractionGame, max_steps, population: Population, **kwargs) -> None:
        super().__init__(game, max_steps, **kwargs)
        self.population = population        
        self.sim_params["pop_params"] = population.pop_params
        self.game_metrics["cand"] = self.population.pop_metrics        

    def init_sim(self) -> None:        
        self.population.init_inds()

    def interact(self):
        ''' computes interactions of ind on other inds in population (no-children)'''
        parents = self.population.get_inds(only_parents=True)
        self.game.update_game_metrics(parents, self.population.pop_metrics, is_final=False)
        children = self.population.get_inds(only_children=True)    
        all_inds = [*parents, *children]
        inds = set(all_inds)
        ints = {}
        for candidate in inds:
            for test in inds:                
                if candidate != test:
                    ints.setdefault(candidate, {})[test] = self.game.interact(candidate, test)
        all_ints = {ind:[(ind1, ind_ints[ind1]) for ind1 in all_inds if ind1 != ind] for ind, ind_ints in ints.items()}
        self.population.update(all_ints)
        logs = []
        if self.draw_dynamics:
            logs = [f"C {p} vs {c}" for p, c in list(zip(parents, children))[:10]]
        self.draw(parents, children, [], [], self.population.seen_inds, logs)

    def end_sim(self) -> None:
        parents = self.population.get_inds(only_parents=True)
        self.game.update_game_metrics(parents, self.population.pop_metrics, is_final=True)
    
    def get_candidates(self) -> list[Any]:
        return self.population.get_inds(only_parents=True)

class TwoPopulationSimulation(StepGameSimulation):
    ''' Adds common methods to implementations with self.candidates and self.tests populations '''

    def __init__(self, game: InteractionGame, max_steps, candidates: Population, tests: Population, **kwargs) -> None:
        super().__init__(game, max_steps, **kwargs)
        self.candidates = candidates
        self.tests = tests
        self.game_metrics["cand"] = self.candidates.pop_metrics
        self.game_metrics["test"] = self.tests.pop_metrics
        self.sim_params["cand_params"] = self.candidates.pop_params
        self.sim_params["test_params"] = self.tests.pop_params     
        
    def init_sim(self) -> None:
        ''' creates populations of candidates and tests '''
        self.candidates.init_inds()
        self.tests.init_inds()

    def update_game_goal_metrics(self, candidates, tests, is_final = False):
        if type(self.candidates) is not OneTimeSequential:
            self.game.update_game_metrics(candidates, self.candidates.pop_metrics, is_final=is_final)
        if type(self.tests) is not OneTimeSequential:
            self.game.update_game_metrics(tests, self.tests.pop_metrics, is_final=is_final)

class PPHC(TwoPopulationSimulation):
    ''' Implements P-PHC approach for number game with co-evolution of two populations of candidates and tests '''

    def interact(self):
        ''' plays one step interaction between populations of candidates and tests and their children '''
        # 1 - * for the fact that 1 represents that candidate success on the test
        candidate_parents = self.candidates.get_inds(only_parents = True)
        candidate_children = self.candidates.get_inds(only_children = True)
        test_parents = self.tests.get_inds(only_parents=True)
        test_children = self.tests.get_inds(only_children=True)
        self.update_game_goal_metrics(candidate_parents, test_parents, is_final=False)
        all_candidates = [*candidate_parents, *candidate_children]
        candidates = set(all_candidates)
        all_tests = [*test_parents, *test_children]
        tests = set(all_tests)
        logs = []
        if self.draw_dynamics: #at this point we asume number game            
            logs1 = [f"C {p} vs {c}" for p, c in list(zip(candidate_parents, candidate_children))[:5]]
            logs2 = [f"T {p} vs {c}" for p, c in list(zip(test_parents, test_children))[:5]]
            logs = [*logs1, *logs2]
        self.draw(candidate_parents, candidate_children, test_parents, test_children, set.union(self.candidates.seen_inds, self.tests.seen_inds), logs)
        candidate_ints = {}
        test_ints = {}

        for cand in candidates: 
            candidate_ints[cand] = {test: self.game.interact(cand, test) for test in tests}

        if self.game.is_symmetric():
            for test in tests:
                test_ints[test] = {cand: 1 - candidate_ints[cand][test] for cand in candidates}
        else:
            for test in tests:
                test_ints[test] = {cand: self.game.interact(test, cand) for cand in candidates}

        all_candidate_ints = {cand:[(test, cand_ints[test]) for test in all_tests] for cand, cand_ints in candidate_ints.items()} 
        all_test_ints = {test:[(cand, t_ints[cand]) for cand in all_candidates] for test, t_ints in test_ints.items()} 

        self.candidates.update(all_candidate_ints)
        self.tests.update(all_test_ints)        

    def end_sim(self) -> None:
        candidate_parents = self.candidates.get_inds(only_parents = True)
        test_parents = self.tests.get_inds(only_parents=True)
        self.update_game_goal_metrics(candidate_parents, test_parents, is_final=True)

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
        self.sim_params["cand_params"] = {"size": cand_sample_size}
        self.sim_params["test_params"] = {"size": test_sample_size}

    def get_candidates(self) -> list[Any]:
        return rnd.choice(self.game.get_all_candidates(), size = self.cand_sample_size, replace=False)
    
    def get_tests(self) -> list[Any]:
        return rnd.choice(self.game.get_all_tests(), size = self.test_sample_size, replace=False)    

class CandidateTestInteractions(TwoPopulationSimulation):
    ''' Base class for different sampling approaches based on interaction matrix '''
    def init_sim(self) -> None:
        super().init_sim()        
        self.candidates_first = True

    def interact(self):
        if self.candidates_first:
            candidates = self.candidates.get_inds() 
            tests = self.tests.get_inds(for_group = candidates)
        else:
            tests = self.tests.get_inds()  
            candidates = self.candidates.get_inds(for_group = tests)

        logs = []
        if self.draw_dynamics: #at this point we asume number game            
            logs1 = [f"C {c}" for c in self.candidates.keys[:10]]
            logs2 = [f"T {t}" for t in self.tests.keys[:10]]
            logs = [*logs1, *logs2]
        self.draw(candidates, [], tests, [], set.union(self.candidates.seen_inds, self.tests.seen_inds), logs)

        self.candidates_first = not self.candidates_first           
        candidate_ints = {}
        test_ints = {}
        uniq_candidates = set(candidates)
        uniq_tests = set(tests)
        for cand in uniq_candidates:
            candidate_ints[cand] = {test:self.game.interact(cand, test) for test in uniq_tests}
        if self.game.is_symmetric():
            for test in uniq_tests:
                test_ints[test] = {cand:1 - candidate_ints[cand][test] for cand in uniq_candidates}
        else:
            for test in uniq_tests:
                test_ints[test] = {cand:self.game.interact(test, cand) for cand in uniq_candidates}
        all_candidate_ints = {cand:[(test, cand_ints[test]) for test in tests] for cand, cand_ints in candidate_ints.items()} 
        all_test_ints = {test:[(cand, t_ints[cand]) for cand in candidates] for test, t_ints in test_ints.items()} 
        self.candidates.update(all_candidate_ints)
        self.tests.update(all_test_ints)
        self.update_game_goal_metrics(candidates, tests, is_final=False)

    def end_sim(self) -> None:
        candidates = self.candidates.get_inds() 
        tests = self.tests.get_inds()
        self.update_game_goal_metrics(candidates, tests, is_final=True)

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
    def __init__(self, min_num = param_min_num, max_num = param_max_num, **kwargs) -> None:
        super().__init__()
        nums = list(range(min_num, max_num + 1))
        self.all_numbers = list(product(nums, nums))
        self.min_num = min_num
        self.max_num = max_num
        self.game_params["min_num"] = min_num
        self.game_params["max_num"] = max_num
        for k,v in kwargs.items():
            self.game_params[k] = v

    def get_all_candidates(self) -> Any:
        return self.all_numbers    
    
    def get_interaction_matrix(self):
        ints = [[self.interact(num1, num2) for num2 in self.all_numbers] for num1 in self.all_numbers ]
        return ints     

class GreaterThanGame(NumberGame):
    ''' testing game to see how everything works '''
    def interact(self, candidate, test) -> int:
        ''' 1 - candidate is better thah test, 0 - test fails candidate '''
        return 1 if candidate[0] > test[0] else 0
    
    def update_game_metrics(self, numbers, metrics, is_final = False):
        ''' the goal of this game is to check how diverse the numbers are 
            check ints_ of this game - almost each number is uniq dim/underlying objective
            returns in metrics the percent of underlying objectives
        '''
        goal = sum(abs(self.max_num - n[0]) for n in numbers) / len(numbers)
        metrics[PARAM_GAME_GOAL] = goal        
        story = metrics.setdefault(PARAM_GAME_GOAL_STORY, [])
        story.append(goal)
        if is_final:
            metrics["sample"] = numbers
        
class IntransitiveGame(NumberGame):
    ''' The IG as it was stated in Bucci article '''
    def interact(self, candidate, test) -> int:
        ''' 1 - candidate is better thah test, 0 - test fails candidate '''
        abs_diffs = [abs(x - y) for x, y in zip(candidate, test)]
        res = 1 if (abs_diffs[0] > abs_diffs[1] and candidate[1] > test[1]) or (abs_diffs[1] > abs_diffs[0] and candidate[0] > test[0]) else 0
        return res
    
    def is_in_range(self, n):
        return True
    
    def update_game_metrics(self, numbers, metrics, is_final = False):
        ''' the goal of this game is to check how diverse the numbers are 
            check ints_ of this game - almost each number is uniq dim/underlying objective
            returns in metrics the percent of underlying objectives
        '''
        uniq_nums = set([n for n in numbers if self.is_in_range(n)])
        delta = 0
        if (self.max_num, self.max_num) in uniq_nums and (self.max_num - 1, self.max_num - 1) in uniq_nums:
            delta -= 1 
        if (self.min_num, self.min_num) in uniq_nums:
            delta -= 1 
        goal = (len(uniq_nums) + delta) / len(numbers)
        metrics[PARAM_GAME_GOAL] = goal        
        # dim1 = 0 if len(uniq_nums) == 0 else round(sum(c[0] for c in uniq_nums) / len(uniq_nums))
        # dim2 = 0 if len(uniq_nums) == 0 else round(sum(c[1] for c in uniq_nums) / len(uniq_nums))
        # metrics["game_goal_1"] = dim1        
        # metrics["game_goal_2"] = dim2
        story = metrics.setdefault(PARAM_GAME_GOAL_STORY, [])
        story.append(goal)
        # metrics.setdefault("game_goal_1_story", []).append(dim1)
        # metrics.setdefault("game_goal_2_story", []).append(dim2)
        if goal > 0.9:
            metrics[PARAM_GAME_GOAL_MOMENT] = len(story)
        if is_final:
            metrics["sample"] = numbers
    
class OrigIntransitiveGame(IntransitiveGame):
    def interact(self, candidate, test) -> int:
        abs_diffs = [abs(x - y) for x, y in zip(candidate, test)]
        idx = 0 if abs_diffs[0] < abs_diffs[1] else 1
        res = 1 if candidate[idx] > test[idx] else 0
        return res

class IntransitiveRegionGame(IntransitiveGame):
    ''' As IG but only applies IG rule in small region, subject for search
        All other points repond with 0 - no-information
    '''
    def __init__(self, reg_min_num = param_reg_min_num, reg_max_num = param_reg_max_num, **kwargs) -> None:
        super().__init__(**{"reg_min_num":reg_min_num, "reg_max_num":reg_max_num, **kwargs})
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

class TwoGoalsMixin:
    def update_game_metrics(self, numbers, metrics, is_final = False):
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
        metrics["game_goal_1"] = goal_scores[0]
        metrics["game_goal_2"] = goal_scores[1]
        metrics.setdefault("game_goal_1_story", []).append(goal_scores[0])
        metrics.setdefault("game_goal_2_story", []).append(goal_scores[1])
        metrics[PARAM_GAME_GOAL] = score #smaller is better
        metrics.setdefault(PARAM_GAME_GOAL_STORY, []).append(score)
        if is_final:
            metrics["sample"] = numbers

class FocusingGame(TwoGoalsMixin, NumberGame):
    ''' The FG as it was stated in Bucci article '''
    def interact(self, candidate, test) -> int:
        ''' 1 - candidate is better thah test, 0 - test fails candidate '''
        res = 1 if (test[0] > test[1] and candidate[0] > test[0]) or (test[1] > test[0] and candidate[1] > test[1]) else 0
        return res
    
class CompareOnOneGame(NumberGame, TwoGoalsMixin):
    ''' Game as it was stated in Golam article '''
    def interact(self, candidate, test) -> int:
        ''' 1 - candidate is better than test, 0 - test fails candidate '''
        max_pos = np.argmax(test)
        res = 1 if candidate[max_pos] >= test[max_pos] else 0
        return res
    
# game that is based on CDESpace 
class CDESpaceGame(InteractionGame):
    ''' Loads given CDE space and provide interactions based on it '''
    def __init__(self, space: CDESpace, **kwargs) -> None:
        self.space = space 
        self.candidates = sorted(space.get_candidates())
        self.tests = sorted(space.get_tests())
        self.all_fails = space.get_candidate_fails()
        self.game_params["space"] = space.to_dict()            

    def get_all_candidates(self) -> Any:
        return self.candidates
    
    def get_all_tests(self) -> Any:
        return self.tests
    
    def is_symmetric(self):
        return True
    
    def interact(self, candidate, test) -> int:
        ''' 1 - candidate wins, 0 - test wins '''
        return 0 if test in self.all_fails.get(candidate, set()) else 1
    
    def update_game_metrics(self, tests, metrics: dict, is_final = False):
        ''' Update game metrics with space metrics. '''
        if is_final:
            DC = self.space.dimension_coverage(tests)
            ARR, ARRA = self.space.avg_rank_of_repr(tests)
            Dup = self.space.duplication(tests)
            R = self.space.redundancy(tests)
            nonI = self.space.noninformative(tests)
            metric_data = { "DC": DC, "ARR": ARR, "ARRA": ARRA, "Dup": Dup, "R": R, "nonI": nonI, "sample": tests}
            for k, v in metric_data.items():
                metrics[k] = v
    
if __name__ == '__main__':
    ''' This entry is used currently only to figure out the CDE space of different number games 
        TODO: probably move to separate function 
    '''
    # game = IntransitiveRegionGame(1, 4, 0, 5)
    game = OrigIntransitiveGame(0, 3)
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
    int_rows = [[sum(ind_ints), f"{n[0]},{n[1]}", *["" if o == 0 else 1 for o in ind_ints]] for i, ind_ints in enumerate(ints) for n in [game.all_numbers[i]]]
    int_headers = ["win", "num", *[f"{n[0]},{n[1]}" for n in game.all_numbers]]
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
