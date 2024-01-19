#TODO: classes for algo 

from abc import ABC, abstractmethod

import numpy as np

from cde import CDESpace

class TestSelector(ABC):
    def init_tests(self, tests):
        self.tests = list(tests)
    @abstractmethod
    def get_tests(self, candidate_id: int, num_tests: int) -> list[int]:
        pass
    @abstractmethod
    def provide_interactions(self, candidate_id: int, interactions:dict[int, bool]):
        pass

class RandTestSelector(TestSelector):
    def __init__(self, rnd: np.random.RandomState, **kwargs) -> None:
        super().__init__()
        self.rnd = rnd
    def get_tests(self, candidate_id: int, num_tests: int) -> set[int]:
        selected = [int(el) for el in self.rnd.choice(self.tests, size = min(len(self.tests), num_tests), replace = False)]
        return selected
    def provide_interactions(self, candidate_id: int, interactions:dict[int, bool]):
        pass

#     When parent-child are nondominant:
#         1) due to presence of candidates which fail them 
#            then child becomes new parent (added to population), old parent is also in the population 
#         2) due to fact that they are same 
#             then child is discarded and other child is created for a parent
#     When parent or child dominates - they replace the position in the population
#     Mutation with Tabu list - each test has a set of tests with which it was already evaluated 
#             mutation will pick random test which is not in tabu list 
#     Dominated individuals are ended up in tabu list 
#     Informative duplicates are also less preferable but not in tabu 
#     --
#     When new child is added into population, the least performant parent is removed
#     Least performant is the parent the most trivial test of population without non-dominant children detected
# '''
    
#TODO PPHC and SamplingStrategies
#             

class PCHCSearch(TestSelector):
    ''' Implements PCHC algorithm - population based co-evolutionary hillclimber
        According to: Bucci. Focusing versus Intransitivity Geometrical Aspects of Co-evolution
        https://link.springer.com/chapter/10.1007/3-540-45105-6_32

        Mutation +- 1 from current test 
        Parent-Child are compared based on aggregation of all known currently common interactions
    '''
    def __init__(self, rnd: np.random.RandomState, popsize = 1, **kwargs) -> None:
        super().__init__()
        self.rnd = rnd
    def get_tests(self, candidate_id: int, num_tests: int) -> set[int]:
        selected = [int(el) for el in self.rnd.choice(self.tests, size = min(len(self.tests), num_tests), replace = False)]
        return selected
    def provide_interactions(self, candidate_id: int, interactions:dict[int, bool]):
        pass    


class EvoSearch(TestSelector):
    ''' Selection adjusted by population-based evolutionary process
        Representation: individual is one test 
        Initialization: choice without replacement of pop_size tests from all tests 
        Evaluation: num_interactions individuals of population is given to candidate and evaluated according to CDESpace 
                    after pareto_n=2 candidates, subset of population changes (mutation) (non-generational)
        Mutation is based on Pareto relation of selected tests from population.
        Note that each individual/test has associated concluded axis id     

        TODO: check NSGA and NSGA 2 - seems to be similar to their ideas    
        test_axes maintain current deduced picture of the space 
        Originally all tests are at origin - treated as trivial
        When 1 is encountered in the interactions, test is removed from the space and replaced at new position
        Tests are sampled according to their weights which are computed from current space position and number of times test was shown 
        (Alternative): pick best tests without probabilistic choice
    '''
    def __init__(self, rnd: np.random.RandomState, pareto_n = 2, 
                    selection_strategy = "pick_prob", axis_pos_weight = 3, **kwargs) -> None:
        super().__init__()
        self.rnd = rnd
        self.pareto_n = pareto_n
        self.axis_pos_weight = axis_pos_weight
        self.selection_strategy = selection_strategy
    def weight_to_scalar(self, axis_pos, num_shown):
        ''' origin  - noninformative, least favorable, axis_pos == () 
            axis    - most favorable depending on position on axis, 
            spanned - better than origin but worse than axis 
        '''
        if len(axis_pos) == 0: #origin - noninformative, least favorable
            return -num_shown
        elif len(axis_pos) == 1: #axis - most favorable depending on position on axis
            pos = axis_pos[0][1] #take the position on axis 
            return self.axis_pos_weight * (pos + 1) - num_shown        
        else: #spanned point - better than origin but worse than axis 
            return axis_pos * 1000000 - num_shown
    def calc_test_weights(self):
        return {t:(, -len(self.test_evals[t])) for t, space_pos in self.test_axes.items()}
    def pick_best(self, num_tests: int):
        ''' Picks best tests according to weights '''
        test_weights = self.calc_test_weights()
        sorted_tests = sorted(list(test_weights.keys()), reverse=True, key = lambda x: test_weights[x])
        selected = sorted_tests[:num_tests]
        return selected
    def pick_prob(self, num_tests: int):
        ''' Picks tests from deduced space according to their weights  '''
        test_weights = self.calc_test_weights()
        total_weight = sum(test_weights.values())
        tests, test_chances = zip(*[(t, w / total_weight) for t, w in test_weights.items()])
        selected = [int(t) for t in self.rnd.choice(tests, size = min(len(tests), num_tests), replace=False, p = test_chances)]
        return selected 
    def pick_tests(self, num_tests: int) -> list[int]:
        ''' Sets current_test_group for next pareto_n candidates ''' 
        if self.test_group_shown_count == self.pareto_n:
            selected = getattr(self, self.selection_strategy)(num_tests)
            self.current_test_group = {t:{} for t in selected}
            self.test_group_shown_count = 0
        else:
            self.test_group_shown_count += 1
        return list(self.current_test_group.keys())
    def init_tests(self, tests):
        super().init_tests(tests)
        self.axes = {(): set()} #() - origin, int - axis, tuples - spanned 
        self.test_axes = {t: () for t in self.tests}
        self.test_evals = {t: {} for t in self.tests} #all responses for each test
        self.test_group_shown_count = 0
        self.current_test_group = {}
    def get_tests(self, candidate_id: int, num_tests: int) -> set[int]:        
        return self.pick_tests(num_tests)
    def provide_interactions(self, candidate_id: int, interactions:dict[int, bool]):
        for test, result in interactions.items():
            if result:
                self.cfs[test].add(candidate_id)
        pareto_group_id = self.active_candidates[candidate_id]
        del self.active_candidates[candidate_id]
        pareto_group = self.pareto_groups[pareto_group_id]
        pareto_group_candidates = pareto_group["candidates"]
        pareto_group_candidates.discard(candidate_id)
        if len(pareto_group_candidates) == 0: #start Pareto comparison 
            del self.pareto_groups[pareto_group_id]
            tests = pareto_group["tests"]
            non_dominant_tests = set()
            same_tests = set()
            for i in range(len(tests)):
                for j in range(i + 1, len(tests)):
                    t1 = tests[i]
                    t2 = tests[j]
                    if len(self.cfs[t1] - self.cfs[t2]) > 0 and len(self.cfs[t2] - self.cfs[t1]) > 0:
                        non_dominant_tests.update([t1,t2])        