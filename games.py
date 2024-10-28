''' Implementation of algorithms for game-like simulation 
    This type of simulation assumes co-evolution of candidates and tests and evaluation of populations on each other.     
    Games could be conducted with rule or CDESpace specified interactions.
    We consider different implementations of Number Games as rules.
    Different algos specify different simulatitons of evaluatiton
'''

from abc import ABC, abstractmethod
from itertools import product
from typing import Any
from de import extract_dims
from params import PARAM_GAME_GOAL, PARAM_GAME_GOAL_MOMENT, PARAM_GAME_GOAL_STORY,\
        PARAM_MAX_INTS, num_intransitive_regions, \
        param_min_num, param_max_num, param_draw_dynamics, param_steps
from tabulate import tabulate, SEPARATING_LINE

import numpy as np

from cde import CDESpace
from viz import draw_populations

from population import OneTimeSequential, Selection
    
class InteractionGame(ABC):
    ''' Base class that encapsulates representation of candidates/tests and their interactions 
        Candidates and tests are implemented as finite sets! 
    '''
    def __init__(self, **kwargs) -> None:
        self.game_params = {"class": self.__class__.__name__}

    @abstractmethod
    def interact(self, candidate, test, **args) -> int:
        ''' Should return 1 when candidate succeeds on the test (test is not better than candidate)! '''
        pass     

    @abstractmethod
    def update_game_metrics(self, inds, metrics, is_final = False):
        ''' Update game metrics with game-specific metrics. '''
        pass

    @abstractmethod
    def get_all_candidates(self):
        pass 

    def get_all_tests(self):
        return self.get_all_candidates()

# set of games 
class NumberGame(InteractionGame):
    ''' Base class for number games with default representation of n-dim Nat tuples 
        Init is random in given nat range per dim 
        Change is +-1 random per dimension
    '''
    def __init__(self, min_num = param_min_num, max_num = param_max_num, **kwargs) -> None:
        super().__init__(**kwargs)
        nums = list(range(min_num, max_num + 1))
        self.all_numbers = list(product(nums, nums))
        self.min_num = min_num
        self.max_num = max_num
        self.game_params["min_num"] = min_num
        self.game_params["max_num"] = max_num
    
    def get_interaction_matrix(self):
        ''' returns rows as test outcomes:
            test1: [cand1_outcome, cand2_outcome, ...] 
            test2: [cand1_outcome, cand2_outcome, ...]
            ... 
            NOTE: Test wins when candidate loses
        '''
        ints = [[1 - self.interact(candidate, test) for candidate in self.all_numbers] for test in self.all_numbers ]
        return ints  

    def get_all_candidates(self):
        return self.all_numbers  

class GreaterThanGame(NumberGame):
    ''' testing game to see how everything works '''
    def interact(self, candidate, test, **args) -> int:
        ''' 1 - candidate is better thah test, 0 - test fails candidate '''
        return 1 if candidate[0] > test[0] else 0
    
    def update_game_metrics(self, numbers, metrics, is_final = False):
        ''' the goal of this game is to check how diverse the numbers are 
            check ints_ of this game - almost each number is uniq dim/underlying objective
            returns in metrics the percent of underlying objectives
        '''
        if len(numbers) == 0:
            return
        goal = sum(abs(self.max_num - n[0]) for n in numbers) / len(numbers)
        metrics[PARAM_GAME_GOAL] = goal        
        story = metrics.setdefault(PARAM_GAME_GOAL_STORY, [])
        story.append(goal)
        if is_final:
            metrics["sample"] = numbers
        
class IntransitiveGame(NumberGame):
    ''' The IG as it was stated in Bucci article '''
    def interact(self, candidate, test, **args) -> int:
        ''' 1 - candidate is better thah test, 0 - test fails candidate '''
        abs_diffs = [abs(x - y) for x, y in zip(candidate, test)]
        res = 1 if (abs_diffs[0] > abs_diffs[1] and candidate[1] > test[1]) or (abs_diffs[1] > abs_diffs[0] and candidate[0] > test[0]) else 0
        return res
    
    def region_id(self, n):
        return n
    
    def update_game_metrics(self, numbers, metrics, is_final = False):
        ''' the goal of this game is to check how diverse the numbers are 
            check ints_ of this game - almost each number is uniq dim/underlying objective
            returns in metrics the percent of underlying objectives
        '''
        if len(numbers) == 0:
            return        
        uniq_nums = set([self.region_id(n) for n in numbers])
        delta = 0
        # if (self.max_num, self.max_num) in uniq_nums and (self.max_num - 1, self.max_num - 1) in uniq_nums:
        #     delta -= 1 
        # if (self.min_num, self.min_num) in uniq_nums:
        #     delta -= 1 
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
    def interact(self, candidate, test, **args) -> int:
        abs_diffs = [abs(x - y) for x, y in zip(candidate, test)]
        idx = 0 if abs_diffs[0] < abs_diffs[1] else 1
        res = 1 if candidate[idx] > test[idx] else 0
        return res

class IntransitiveRegionGame(IntransitiveGame):
    ''' As IG but only applies IG rule in small region, subject for search
        All other points repond with 0 - no-information
    '''
    def __init__(self, num_intransitive_regions = num_intransitive_regions, **kwargs) -> None:
        super().__init__(**{"num_intransitive_regions":num_intransitive_regions, **kwargs})
        self.num_intransitive_regions = num_intransitive_regions
        self.region_size = (self.max_num - self.min_num) // num_intransitive_regions

    def region_id(self, n):
        n1 = n[0] // self.region_size
        n2 = n[1] // self.region_size
        return (n1, n2)
    
    def interact(self, candidate, test, **args) -> int:
        ''' 1 - candidate is better than test, 0 - test fails candidate '''
        r1 = self.region_id(candidate)
        r2 = self.region_id(test)        
        return super().interact(r1, r2)

class FocusingGame(NumberGame):
    ''' The FG as it was stated in Bucci article '''
    def interact(self, candidate, test, **args) -> int:
        ''' 1 - candidate is better than test, 0 - test fails candidate '''
        # res = 1 if (test[0] > test[1] and candidate[0] > test[0]) or (test[1] > test[0] and candidate[1] > test[1]) else 0
        # if candidate[0] > candidate[1]:
        #     res = 1 if candidate[0] > test[0] else 0
        # else:
        #     res = 1 if candidate[1] > test[1] else 0
        res = 1 if (candidate[0] > candidate[1] and candidate[0] > test[0]) or (candidate[1] > candidate[0] and candidate[1] > test[1]) else 0
        return res
    
    def update_game_metrics(self, numbers, metrics, is_final = False):
        ''' the goal of this game is to focus on two axes (0, max_dim) and (max_dim, 0)
            Check ints_ file of this game to understand thee structure of underelying objectives
            We compute average distance to either of two objectives. But two objectives should be discovered
        '''
        if len(numbers) == 0:
            return        
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
    
class CompareOnOneGame(NumberGame):
    ''' Game as it was stated in Golam article '''
    def interact(self, candidate, test, **args) -> int:
        ''' 1 - candidate is better than test, 0 - test fails candidate '''
        max_pos = np.argmax(test)
        res = 1 if candidate[max_pos] >= test[max_pos] else 0
        return res
    
    def update_game_metrics(self, numbers, metrics, is_final = False):
        ''' the goal of this game is to focus on two axes (0, max_dim) and (max_dim, 0)
            Check ints_ file of this game to understand thee structure of underelying objectives
            We compute average distance to either of two objectives. But two objectives should be discovered
        '''
        if len(numbers) == 0:
            return        
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
    
class CDESpaceGame(InteractionGame):
    ''' Loads given CDE space and provide interactions based on it '''
    def __init__(self, space: CDESpace, **kwargs) -> None:
        super().__init__(**kwargs)
        self.space = space 
        self.candidates = sorted(space.get_candidates())
        self.tests = sorted(space.get_tests())
        self.all_fails = space.get_candidate_fails()
        self.game_params["space"] = space.to_dict()            

    def interact(self, candidate, test, **args) -> int:
        ''' 1 - candidate wins, 0 - test wins '''
        return 0 if test in self.all_fails.get(candidate, set()) else 1
    
    def update_game_metrics(self, tests, metrics: dict, is_final = False):
        ''' Update game metrics with space metrics. '''
        if len(tests) == 0:
            return        
        if is_final:
            DC = self.space.dimension_coverage(tests)
            ARR, ARRA = self.space.avg_rank_of_repr(tests)
            Dup = self.space.duplication(tests)
            R = self.space.redundancy(tests)
            nonI = self.space.noninformative(tests)
            metric_data = { "DC": DC, "ARR": ARR, "ARRA": ARRA, "Dup": Dup, "R": R, "nonI": nonI, "sample": tests}
            for k, v in metric_data.items():
                metrics[k] = v

    def get_all_candidates(self):
        return self.candidates
    
    def get_all_tests(self):
        return self.tests

def step_game(step: int, num_steps: int, candidates_pop: Selection, tests_pop: Selection, game: InteractionGame) -> None:
    candidates = candidates_pop.get_selection() 
    tests = tests_pop.get_selection()
    if candidates_pop is not OneTimeSequential:
        game.update_game_metrics(candidates_pop.get_best(), candidates_pop.sel_metrics, is_final = step == num_steps)
    if tests_pop is not OneTimeSequential:
        game.update_game_metrics(tests_pop.get_best(), tests_pop.sel_metrics, is_final = step == num_steps)    

    if step == num_steps:
        return False
        
    candidate_ints = {}
    test_ints = {}
    uniq_candidates = set(candidates)
    uniq_tests = set(tests)
    for cand in uniq_candidates:
        candidate_ints[cand] = {test:game.interact(cand, test, step = step, num_steps = num_steps) for test in uniq_tests}
    for test in uniq_tests: # NOTE: test wins when candidate fails
        test_ints[test] = {cand:1 - candidate_ints[cand][test] for cand in uniq_candidates}
    all_candidate_ints = {cand:{test: cand_ints[test] for test in tests} for cand, cand_ints in candidate_ints.items()} 
    all_test_ints = {test:{cand: t_ints[cand] for cand in candidates} for test, t_ints in test_ints.items()} 
    candidates_pop.update(all_candidate_ints, uniq_tests)
    tests_pop.update(all_test_ints, uniq_candidates)
    return True
 
def run_game(game: InteractionGame, candidates: Selection, tests: Selection, *, num_steps: int = param_steps, draw_dynamics = param_draw_dynamics, **kwargs) -> None:
    num_steps = int(num_steps)
    draw_dynamics = int(param_draw_dynamics) == 1
    candidates.init_selection()
    tests.init_selection()
    candidates.sel_metrics[PARAM_MAX_INTS] = tests.sel_metrics[PARAM_MAX_INTS] = len(candidates.get_pool()) * len(tests.get_pool())
    step = 0
    should_continue = True 
    while should_continue:
        should_continue = step_game(step, num_steps, candidates, tests, game)
        if draw_dynamics and isinstance(game, NumberGame):
            cand_points = candidates.get_for_drawing(role = "cand")
            test_points = tests.get_for_drawing(role = "test")
            if len(test_points) > 0:
                point_groups = [*cand_points, *test_points]
                pop_name = tests.__class__.__name__
                name = f"{step}"
                name = name if len(name) >= 4 else ("0" * (4 - len(name)) + name)
                draw_populations(point_groups,
                                    xrange=(game.min_num, game.max_num), yrange=(game.min_num, game.max_num),
                                    name = f"step-{name}", title = f"Step {name}, {pop_name} on {game.__class__.__name__}")
        step += 1
    return dict(params = dict(game = dict(num_steps = num_steps, draw_dynamics=draw_dynamics, **game.game_params), 
                                cand = candidates.sel_params, test = tests.sel_params), 
                    metrics = dict(cand = candidates.sel_metrics, test = tests.sel_metrics))
        
if __name__ == '__main__':
    ''' This entry is used currently only to figure out the CDE space of different number games 
        TODO: probably move to separate function 
    '''
    # game = IntransitiveRegionGame(1, 4, 0, 5)
    # game = OrigIntransitiveGame(0, 3)
    game = IntransitiveRegionGame(num_intransitive_regions=2, min_num=0, max_num = 4)
    ints = game.get_interaction_matrix()
    dims, origin, spanned, duplicates = extract_dims(ints)
    dim_nums = [[game.all_numbers[test_ids[0]] for test_ids in dim] for dim in dims]
    origin_nums = [game.all_numbers[i] for i in origin]
    spanned_nums = [game.all_numbers[i] for i in spanned.keys()]
    duplicates_nums = [game.all_numbers[i] for i in duplicates]
    # for num, line in zip(game.all_numbers, ints):
    #     print(f"{num}: {line}")

    max_dim = max([len(dim) for dim in dim_nums])
            
    rows = []    
    for dim_id, dim in enumerate(sorted(dims, key = lambda dim:game.all_numbers[dim[-1][0]])):
        # print(f"Dim {dim_id}")
        for test_ids in dim:
            # print(f"{game.all_numbers[i]}: {ints[i]}")
            i = test_ids[0]
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
        for num, i in zip(spanned_nums, list(spanned.keys())):
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
