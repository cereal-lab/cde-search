#TODO: classes for algo 

from abc import ABC, abstractmethod

import numpy as np

from cde import CDESpace

class TestSelector(ABC):
    @abstractmethod
    def init_space(self, space: CDESpace):
        pass     
    @abstractmethod
    def get_tests(self, candidate_id: int, num_tests: int) -> set[int]:
        pass
    @abstractmethod
    def provide_interactions(self, candidate_id: int, interactions:dict[int, bool]):
        pass

class RandTestSelector(TestSelector):
    def __init__(self, rnd: np.random.RandomState, **kwargs) -> None:
        super().__init__()
        self.rnd = rnd 
    def init_space(self, space: CDESpace):
        self.tests = list(space.get_tests())
    def get_tests(self, candidate_id: int, num_tests: int) -> set[int]:
        selected = self.rnd.choice(self.tests, size = num_tests, replace = False)
        return set(selected)
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

class EvoSearch(TestSelector):
    ''' Selection adjusted by population-based evolutionary process
        Representation: individual is one test 
        Initialization: choice without replacement of pop_size tests from all tests 
        Evaluation: num_interactions individuals of population is given to candidate and evaluated according to CDESpace 
                    after pareto_n=2 candidates, subset of population changes (mutation) (non-generational)
        Mutation is based on Pareto relation of selected tests from population.
        Note that each individual/test has associated concluded axis id         
    '''
    def __init__(self, rnd: np.random.RandomState, pop_size = 1, pareto_n = 2, 
                    mutation_strategy = "mutate_random", **kwargs) -> None:
        super().__init__()
        self.rnd = rnd 
        self.pop_size = pop_size
        self.pareto_n = pareto_n
        self.mutation_strategy = mutation_strategy
    def mutate_random(self, parent: list[int]):
        ''' Tries to create random child that differs from parent in at least one gene '''
        children = set()
        num_tries = 0
        while num_tries < 5 * self.child_n:
            child = self.init_genotype()
            if child != parent and child not in children:
                children.add(child)
            num_tries += 1
        return list(children)
    def mutate(self, parent: list[int]) -> list[list[int]]: 
        return getattr(self, self.mutation_strategy)(parent)   
    def init_space(self, space: CDESpace):
        self.tests = list(space.get_tests())
        self.inds = [int(t) for t in self.rnd.choice(self.tests, size = min(len(self.tests), self.pop_size), replace = False)]
        self.ind_axes = {}
        self.axes = {None: []} #None - origin, int - axis, tuples - spanned 
        self.next_axis_id = 0 # not allocated yet axis 
        self.evaluations = {t:{} for t in self.tests} #test evaluations
    def get_tests(self, candidate_id: int, num_tests: int) -> set[int]:
        self.inds[:num_tests]
    def provide_interactions(self, candidate_id: int, interactions:dict[int, bool]):
        pass    