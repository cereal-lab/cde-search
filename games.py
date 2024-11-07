''' Implementation of algorithms for game-like simulation 
    This type of simulation assumes co-evolution of candidates and tests and evaluation of populations on each other.     
    Games could be conducted with rule or CDESpace specified interactions.
    We consider different implementations of Number Games as rules.
    Different algos specify different simulatitons of evaluatiton
'''

from abc import ABC, abstractmethod
from itertools import product
import os
from typing import Any
from de import extract_dims_fix
from params import PARAM_GAME_GOAL, PARAM_GAME_GOAL_MOMENT, PARAM_GAME_GOAL_STORY,\
        PARAM_MAX_INTS, param_num_intransitive_regions, \
        param_min_num, param_max_num, param_draw_dynamics, param_steps
from tabulate import tabulate, SEPARATING_LINE

import numpy as np

from cde import CDESpace
from viz import draw_populations

import json

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
    def get_all_candidates(self):
        pass 

    def get_all_tests(self):
        return self.get_all_candidates()
    
    def get_extracted_dimensions(self):
        return [], {}, set()

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
        self.space = None
    
    def get_interaction_matrix(self):
        ''' returns rows as test outcomes:
            test1: [cand1_outcome, cand2_outcome, ...] 
            test2: [cand1_outcome, cand2_outcome, ...]
            ... 
            NOTE: Test wins when candidate loses, therefore 1 - 
            We are interested in finding best tests 
        '''
        ints = [[ 1 - self.interact(candidate, test) for candidate in self.all_numbers] for test in self.all_numbers ]
        return ints  

    def get_all_candidates(self):
        return list(self.all_numbers)
    
    def save_space(self, dir = "."):
        dims, o, sp = extract_dims_fix(self.get_interaction_matrix())
        spanned = [[tid, list(coords)] for tid, coords in sp.items()]
        json_space = json.dumps(dict(axes = dims, origin = o, spanned = spanned))
        with open(os.path.join(dir, self.__class__.__name__ + "-space.json"), "w") as f:
            f.write(json_space)

    def load_space(self, dir = "."):
        if self.space is not None:
            return self.space
        with open(os.path.join(dir, self.__class__.__name__ + "-space.json"), "r") as f:
            json_space = json.loads(f.read())
        axes = [ [set([self.all_numbers[i] for i in point]) for point in dim ] for dim in json_space["axes"] ]
        origin = set([self.all_numbers[i] for i in json_space["origin"]])
        spanned = {self.all_numbers[i]: tuple(sp_dims) for i, sp_dims in json_space["spanned"]}
        self.space = (axes, origin, spanned)
        return self.space
    
    def get_extracted_dimensions(self):        
        return self.load_space()

class GreaterThanGame(NumberGame):
    def interact(self, candidate, test, **args) -> int:
        ''' 1 - candidate is better than test, 0 - test fails the candidate, test is better '''
        return 1 if candidate[0] > test[0] else 0
                
class IntransitiveGame(NumberGame):
    ''' The IG as it was stated in Bucci article '''
    def interact(self, candidate, test, **args) -> int:
        abs_diffs = [abs(x - y) for x, y in zip(candidate, test)]
        idx = 0 if abs_diffs[0] < abs_diffs[1] else 1
        res = 1 if candidate[idx] > test[idx] else 0
        return res

class IntransitiveRegionGame(IntransitiveGame):
    ''' As IG but only applies IG rule in small region, subject for search
        All other points repond with 0 - no-information
    '''
    def __init__(self, *, num_intransitive_regions = param_num_intransitive_regions, **kwargs) -> None:
        super().__init__(**{"num_intransitive_regions":num_intransitive_regions, **kwargs})
        self.num_intransitive_regions = num_intransitive_regions
        self.region_size = ((self.max_num - self.min_num) // num_intransitive_regions) + 1

    def remap(self, n):
        n1 = n[0] // self.region_size
        n2 = n[1] // self.region_size
        return (n1, n2)
    
    def interact(self, candidate, test, **args) -> int:
        ''' 1 - candidate is better than test, 0 - test fails candidate '''
        c = self.remap(candidate)
        t = self.remap(test)        
        return super().interact(c, t)

class FocusingGame(NumberGame):
    ''' The FG as it was stated in Bucci (not Golam's) article '''
    def interact(self, candidate, test, **args) -> int:
        ''' 1 - candidate is better than test, 0 - test fails candidate '''        
        # next is according to Golam article
        # if test[0] > test[1]:
        #     res = 1 if candidate[0] > test[0] else 0
        # else:
        #     res = 1 if candidate[1] > test[1] else 0
        # note: next is according to Bucci article
        res = 1 if (test[0] > test[1] and candidate[0] > test[0]) or (test[1] > test[0] and candidate[1] > test[1]) else 0
        return res
    
class CompareOnOneGame(NumberGame):
    ''' Game as it was stated in Golam article '''
    def interact(self, candidate, test, **args) -> int:
        ''' 1 - candidate is better than test, 0 - test fails candidate '''
        max_pos = np.argmax(test)
        res = 1 if candidate[max_pos] >= test[max_pos] else 0
        return res
      
class CDESpaceGame(InteractionGame):
    ''' Loads given CDE space and provide interactions based on it '''
    def __init__(self, space: CDESpace, **kwargs) -> None:
        super().__init__(**kwargs)
        self.space = space 
        self.candidates = sorted(space.get_candidates())
        self.tests = sorted(space.get_tests())
        self.all_fails = space.get_candidate_fails()
        self.game_params["space"] = space.to_dict()      
        self.space_plain = None      

    def interact(self, candidate, test, **args) -> int:
        ''' 1 - candidate wins, 0 - test wins '''
        return 0 if test in self.all_fails.get(candidate, set()) else 1

    def get_all_candidates(self):
        return list(self.candidates)
    
    def get_all_tests(self):
        return list(self.tests)
    
    def get_extracted_dimensions(self):
        if self.space_plain is None:
            axes = [[set(point.tests) for point in axis] for axis in self.space.axes]
            origin = set(self.space.origin.tests)
            spanned = {test: pos for position, point in self.space.spanned.items() for pos in [{axis_id: pos_id for axis_id, pos_id in position}] for test in point.tests}
            self.space_plain = (axes, origin, spanned)
        return self.space_plain

def step_game(step: int, num_steps: int, sel1: Selection, sel2: Selection, game: InteractionGame) -> None:
    selection1 = sel1.get_selection() 
    selection2 = sel2.get_selection()
    sel1.collect_metrics(*game.get_extracted_dimensions(), is_final = step == num_steps)
    sel2.collect_metrics(*game.get_extracted_dimensions(), is_final = step == num_steps)

    if step == num_steps:
        return False
        
    set1 = set(selection1)
    set2 = set(selection2)

    # here we test set2 on set1 (set1 plays role of tests) and we adjust set of tests, sel1
    # test win when candidate loses, therefore 1 -, because interact is not symmetric 
    # and we pick best tests on update 
    if sel1 is not OneTimeSequential:
        ints = {test: {cand:1 - game.interact(cand, test, step = step, num_steps = num_steps) for cand in set2} for test in set1}
        sel1.update(ints, set2)

    # here we test set1 on set2 (set2 plays role of tests) and we adjust set of tests, sel2
    if sel2 is not OneTimeSequential:
        ints = {test: {cand:1 - game.interact(cand, test, step = step, num_steps = num_steps) for cand in set1} for test in set2}
        sel2.update(ints, set1)        

    return True
 
def run_game(game: InteractionGame, sel1: Selection, sel2: Selection, *, num_steps: int = param_steps, draw_dynamics = param_draw_dynamics, **kwargs) -> None:
    num_steps = int(num_steps)
    draw_dynamics = int(param_draw_dynamics) == 1
    sel1.init_selection()
    sel2.init_selection()
    sel1.sel_metrics[PARAM_MAX_INTS] = sel2.sel_metrics[PARAM_MAX_INTS] = len(sel1.get_pool()) * len(sel2.get_pool())
    step = 0
    should_continue = True 
    matrix = None
    while should_continue:
        should_continue = step_game(step, num_steps, sel1, sel2, game)
        if draw_dynamics and isinstance(game, NumberGame):
            cand_points = sel1.get_for_drawing(role = "cand")
            test_points = sel2.get_for_drawing(role = "test")
            if len(test_points) > 0:
                if matrix is None:
                    axes, _, _ = game.space
                    matrix = [[0 for _ in range(game.max_num + 1)] for _ in range(game.max_num + 1)]
                    for ax in axes:
                        for point_id, point in enumerate(ax):
                            c = (point_id + 1) / len(ax)
                            for test in point:
                                matrix[test[1]][test[0]] = c
                # dots = [ {"xy": point, "class": dict(c=(c, c, c), s=2)} ]]
                point_groups = [*cand_points, *test_points]
                pop_name = sel2.__class__.__name__
                name = f"{step}"
                name = name if len(name) >= 4 else ("0" * (4 - len(name)) + name)
                draw_populations(point_groups,
                                    xrange=(game.min_num, game.max_num), yrange=(game.min_num, game.max_num),
                                    name = f"step-{name}", title = f"Step {name}, {pop_name} on {game.__class__.__name__}",
                                    matrix = matrix)
        step += 1
    return dict(params = dict(game = dict(num_steps = num_steps, draw_dynamics=draw_dynamics, **game.game_params), 
                                sel = sel2.sel_params), 
                    metrics = dict(sel = sel2.sel_metrics))
        
if __name__ == '__main__':
    ''' This entry is used currently only to figure out the CDE space of different number games 
        TODO: probably move to separate function 
    '''
    # game = IntransitiveRegionGame(1, 4, 0, 5)
    # game = OrigIntransitiveGame(0, 3)
    # game = IntransitiveRegionGame(num_intransitive_regions=2, min_num=0, max_num = 4)
    game = CompareOnOneGame()
    game.save_space()
    # game = FocusingGame(0, 5)
    ints = game.get_interaction_matrix()
    dims, origin, spanned = extract_dims_fix(ints)
    dim_nums = [[[game.all_numbers[i] for i in test_ids] for test_ids in dim] for dim in dims]
    origin_nums = [game.all_numbers[i] for i in origin]
    spanned_nums = [game.all_numbers[i] for i in spanned.keys()]
    # duplicates_nums = [game.all_numbers[i] for i in duplicates]
    # for num, line in zip(game.all_numbers, ints):
    #     print(f"{num}: {line}")

    max_dim = max([len(dim) for dim in dim_nums])
            
    rows = []    
    for dim_id, dim in enumerate(sorted(dims, key = lambda dim:game.all_numbers[dim[-1][0]])):
        # print(f"Dim {dim_id}")
        for point_id, test_ids in enumerate(dim):
            # print(f"{game.all_numbers[i]}: {ints[i]}")
            for i in test_ids:
                row = []
                num = game.all_numbers[i]
                row.append(dim_id)
                row.append(point_id)
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
        print(tabulate(rows, headers=["dim", "pos", "num", *[f"{n[0]},{n[1]}" for n in game.all_numbers]], tablefmt='simple', numalign="center", stralign="center"), file=f)
    # print(f"{game.all_numbers}")

    # print(f"Spanned")
    # for i in spanned:
    #     print(f"{game.all_numbers[i]}: {ints[i]}")
