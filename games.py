''' Implementation of algorithms for game-like simulation 
    This type of simulation assumes co-evolution of candidates and tests and evaluation of populations on each other.     
    Games could be conducted with rule or CDESpace specified interactions.
    We consider different implementations of Number Games as rules.
    Different algos specify different simulatitons of evaluatiton
'''

from abc import ABC, abstractmethod
from math import sqrt
from typing import Any
from matplotlib.pylab import RandomState, RandomState as RandomState

import numpy as np

from cde import CDEPoint, CDESpace

class InteractionGame(ABC):
    ''' Base class that encapsulates representation of candidates/tests and their interactions 
        Candidates and tests are implemented as finite sets! 
    '''
    def __init__(self, rnd: np.random.RandomState) -> None:
        super().__init__()
        self.rnd = rnd

    @abstractmethod
    def get_candidates(self) -> list[Any]:
        pass 

    def get_tests(self) -> list[Any]: 
        ''' by default assumes that tests are also candidates. Override if not'''
        return self.get_candidates()

    @abstractmethod
    def interact(self, candidate, test) -> int:
        ''' Should return 1 when candidate fails the test (test is better than candidate)! '''
        pass 

class GameSimulation(ABC):
    ''' Abstract game of candidates and tests where interactions happen by given CDESpace or rule '''
    def __init__(self, game: InteractionGame, rnd: np.random.RandomState) -> None:
        ''' 
            @param rule: function which defines interaction between (candidate, test) -> 2
            @param rnd: random number generator for experiment reproducibility 
        '''
        self.game = game
        self.rnd = rnd 

    def play(self):
        ''' Executes game, final state defines found candidates and tests '''
        pass 

    def get_candidates(self) -> list[Any]:
        ''' Gets candidates from game simulation state '''
        return []
    
    def get_tests(self) -> list[Any]:
        ''' Gets tests from game simulation state. Assumes that tests and candidates are same entity '''
        return self.get_candidates()

class StepGameSimulation(GameSimulation):
    ''' Defines step based game with group of interactions on each step '''
    def __init__(self, game: InteractionGame, rnd: np.random.RandomState, max_steps = 0) -> None:
        super().__init__(game, rnd)
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

class HC(StepGameSimulation):
    ''' Base class for hill climbers. Defines different mutation strategies on finite sets '''
    def __init__(self, game: InteractionGame, rnd: RandomState, 
                    candidate_mutation_strategy = "plus_minus_one", test_mutation_strategy = "plus_minus_one", **kwargs) -> None:
        super().__init__(game, rnd, **kwargs)
        self.candidate_mutation_strategy = getattr(self, candidate_mutation_strategy)
        self.test_mutation_strategy = getattr(self, test_mutation_strategy)
        self.all_candidates = game.get_candidates()
        self.all_tests = game.get_tests()

    def plus_minus_one(self, parent_index: int, pool: list[int]):
        # parent_index = next(i for i, ind in enumerate(pool) if ind == parent)
        child_index = parent_index + self.rnd.choice([-1,1])
        if child_index < 0: 
            child_index = len(pool) - 1
        elif child_index >= len(pool):
            child_index = 0
        return child_index
    
    def resample(self, parent_index: int, pool: list[int]):
        for _ in range(10):
            child_index = self.rnd.randint(0, len(pool))
            if child_index != parent_index:
                return child_index 
        return parent_index   
    
    def get_candidate(self, index:int):
        return self.all_candidates[index]
    
    def get_test(self, index:int):
        return self.all_tests[index]


class PCHC(HC):
    ''' Defines PCHC game skeleton for merged population of tests and candidates
        The same nature of candidates and tests are assumed
    '''
    def __init__(self, game: InteractionGame, rnd: np.random.RandomState, popsize = 1, **kwargs) -> None:
        super().__init__(game, rnd, **kwargs)
        self.popsize = popsize

    def produce_all_children(self):   
        self.children_index = [self.candidate_mutation_strategy(parent_index, self.all_candidates) for parent_index in self.population_index] 

    def init_sim(self) -> None:
        ''' creates parent and one child per parent as in PCHC'''
        self.population_index = self.rnd.choice(len(self.all_candidates), size = self.popsize, replace=False)
        self.produce_all_children()

    def interact(self):
        ''' computes interactions of ind on other inds in population (no-children)'''
        parent_interactions = [[self.game.interact(self.get_candidate(t), self.get_candidate(c)) 
                                    for t in self.population_index if t != c] 
                                for c in self.population_index]
        child_interactions = [[self.game.interact(self.get_candidate(t), self.get_candidate(c)) 
                                    for t in self.population_index if t != c]
                                for c in self.children_index]
        # how many other candidates are failed by parent and child
        parent_sums = [sum(scores) for scores in parent_interactions]
        child_sums = [sum(scores) for scores in child_interactions]
        # next line is the selection based on aggregation 
        self.population_index = [self.children_index[i] if child_sum > parent_sum else self.population_index[i]
                                    for i, (parent_sum, child_sum) in enumerate(zip(parent_sums, child_sums))]
        self.produce_all_children()
    
    def get_candidates(self) -> list[Any]:
        return [self.get_candidate(c) for c in self.population_index]        

class PPHC(HC):
    ''' Implements P-PHC approach for number game with co-evolution of two populations of candidates and tests '''
    def __init__(self, game: InteractionGame, rnd: np.random.RandomState, 
                    candidate_selection = "pareto_select", test_selection = "informativeness_select",
                    candidate_popsize = 1, test_popsize = 1, **kwargs) -> None:
        super().__init__(game, rnd, **kwargs)
        self.candidate_popsize = candidate_popsize
        self.test_popsize = test_popsize 
        self.candidate_selection = getattr(self, candidate_selection)
        self.test_selection = getattr(self, test_selection)

    def produce_all_children(self):
        self.candidate_children_index = [self.candidate_mutation_strategy(parent_index, self.all_candidates) for parent_index in self.candidate_index] 
        self.test_children_index = [self.test_mutation_strategy(parent_index, self.all_tests) for parent_index in self.test_index]

    def init_sim(self) -> None:
        ''' creates populations of candidates and tests '''
        self.candidate_index = self.rnd.choice(len(self.all_candidates), size = self.candidate_popsize, replace=False)
        self.test_index = self.rnd.choice(len(self.all_tests), size = self.test_popsize, replace=False)
        self.produce_all_children()

    def pareto_select(self, parent_interactions, child_interactions, parents, children):
        selected = [ children[i] if all(cint >= pint for cint, pint in zip(child_ints, parent_ints)) and 
                                    any(cint > pint for cint, pint in zip(child_ints, parent_ints)) else 
                        parents[i]
                        for i, (parent_ints, child_ints) in enumerate(zip(parent_interactions, child_interactions))]
        return selected
    
    def informativeness_select(self, parent_interactions, child_interactions, parents, children):
        ''' selection based on ninformativeness score aggregation '''
        def informativeness_score(ints: list[int]):
            ''' counts pairs of interactions of same outcome '''
            return len(1 for i in range(len(ints)) for j in range(i+1, len(ints)) if ints[i] == ints[j])
        parent_scores = [informativeness_score(tints) for tints in parent_interactions]
        child_scores = [informativeness_score(tints) for tints in child_interactions]
        selected = [ children[i] if child_score > parent_score else parents[i]
                            for i, (parent_score, child_score) in enumerate(zip(parent_scores, child_scores))]  
        return selected

    def interact(self):
        ''' plays one step interaction between populations of candidates and tests and their children '''
        # 1 - * for the fact that 1 represents that candidate success on the test
        candidate_parent_interactions = [[1 - self.game.interact(self.get_candidate(c), self.get_test(t)) for t in self.test_index] 
                                            for c in self.candidate_index]
        # we transpose the prev matrix, with test successes now
        test_parent_interactions = [[1 - candidate_parent_interactions[j][i] for j in range(len(self.candidate_index))] 
                                        for i in range(len(self.test_index))]
        # successes of candidate children
        candidate_child_interactions = [[1 - self.game.interact(self.get_candidate(c), self.get_test(t)) for t in self.test_index] 
                                            for c in self.candidate_children_index]
        # successes of tests children
        test_child_interactions = [[self.game.interact(self.get_candidate(c), self.get_test(t)) for c in self.candidate_index] 
                                        for t in self.test_children_index]
        # update of candidates and tests 
        # candidates are updated with Pareto principle 
        self.candidate_index = self.candidate_selection(candidate_parent_interactions, candidate_child_interactions, self.candidate_index, self.candidate_children_index)
        self.test_index = self.test_selection(test_parent_interactions, test_child_interactions, self.test_index, self.test_children_index)
        #produce new children
        self.produce_all_children()

    def get_candidates(self) -> list[Any]:
        return [self.get_candidate(c) for c in self.candidate_index]

    def get_tests(self) -> list[Any]:
        return [self.get_candidate(c) for c in self.test_index]

class ZeroSimulation(GameSimulation):
    ''' Samples the individuals from game as provide it as result '''
    def __init__(self, game: InteractionGame, rnd: np.random.RandomState, sample_size = 1, **kwargs) -> None:
        super().__init__(game, rnd)
        self.sample_size = sample_size

    def get_candidates(self) -> list[Any]:
        return self.rnd.choice(self.game.get_candidates(), size = self.sample_size, replace=False)
    
    def get_tests(self) -> list[Any]:
        return self.rnd.choice(self.game.get_tests(), size = self.sample_size, replace=False)
    
class SamplingStrategies(StepGameSimulation):
    ''' Implements sampling of pools based on features of built at moment interaction matrix '''
    def __init__(self, game: InteractionGame, rnd: RandomState, max_steps=0, 
                    candidate_sample_size = 10, candidate_strategy = None, 
                    test_sample_size = 10, test_strategy = None, 
                    epsilon = None, softmax = None, tau = 1, **kwargs) -> None:
        super().__init__(game, rnd, max_steps)
        self.candidate_interactions = {} #dict candidate: {test: outcome}
        self.test_interactions = {} #3dict test: {candidate: outcome}
        self.total_interaction_count = 0
        self.candidate_sample_size = candidate_sample_size
        self.candidate_strategy = candidate_strategy
        self.test_sample_size = test_sample_size
        self.test_strategy = test_strategy
        self.epsilon = epsilon
        self.softmax = softmax
        self.tau = tau
        self.all_candidates = game.get_candidates()
        self.all_tests = game.get_tests()

    def calc_knowledge_score(self, ind, selected_inds, interactions):
        ''' Computes features of interactions about how well ind was alreeady explored '''
        ind_interactions = interactions.get(ind, {})
        knowledge_force = 1 if self.total_interaction_count == 0 else (1 - len(ind_interactions) / self.total_interaction_count)
        scores = {}
        # metrics of interactions based on knowledge about ind and already selected inds
        scores["kn"] = knowledge_force    
        scores["rel-kn"] = -(sum(len([oid for oid in selected_ind_ints.keys() if oid in ind_interactions]) 
                                        for selected_ind in selected_inds 
                                        for selected_ind_ints in [interactions.get(selected_ind, {})]))
        scores["kn-n"] = len(ind_interactions)
        return scores

    def calc_domination_score(self, ind, selected_inds, interactions):
        ''' Computes metrics of individual about how well ind dominates others 
            ind - individual (candidate/test) for which we need to compute score 
            selected_inds - already sampled inds 
            interactions - set of interactions for all inds of ind kind
        '''
        ind_interactions = interactions.get(ind, {})
        other_ids = set(ind_interactions.keys()) #other ids represent ids of other kind (for candidate it is test and vise versa)
        def build_common_interactions(pool_inds):
            ''' computes common interactions with other inds of the pool '''
            common_interactions = {ind1: interactions
                for ind1 in pool_inds if ind1 != ind
                for ind1_interactions in [interactions.get(ind1, {})]
                for interactions in [{common_other_id: (ind_interactions[common_other_id], ind1_interactions[common_other_id]) 
                    for common_other_id in set.intersection(other_ids, set(ind1_interactions.keys()))}]
                if len(interactions) > 0} #non-empty interactions set
            return common_interactions
        common_interactions = build_common_interactions(interactions.keys())
        non_dominated = [ind1 for ind1, ints in common_interactions.items()
                                if len(ints) >= 2 and
                                    any(ind_outcome > ind1_outcome for _, (ind_outcome, ind1_outcome) in ints.items()) and 
                                    any(ind_outcome < ind1_outcome for _, (ind_outcome, ind1_outcome) in ints.items())]
        dominated = [ind1 for ind1, ints in common_interactions.items()
                            if all(ind_outcome >= ind1_outcome for _, (ind_outcome, ind1_outcome) in ints.items()) and 
                               any(ind_outcome > ind1_outcome for _, (ind_outcome, ind1_outcome) in ints.items())]        

        # the idea that whenever current ind of qX fails other el, the already selected ind qY also fails the other el ==> cur ind is duplicate
        common_interactions_with_selected = build_common_interactions(selected_inds)
        duplicate_of_selected = any(ind1 for ind1, ints in common_interactions_with_selected.items()
                                    for ind_cfs in [[(ind_outcome, ind1_outcome) for _, (ind_outcome, ind1_outcome) in ints.items() if ind_outcome == 1]] 
                                    if len(ind_cfs) >= 1 and all(ind_outcome == ind1_outcome for (ind_outcome, ind1_outcome) in ind_cfs))

        # selected_dominated = [did1 for did1 in selected_inds if did1 in dominated]

        # spanned_of_selected = any((did1, did2) for did1 in selected_dominated
        #                 for did2 in selected_dominated if did2 != did1 
        #                 for did1_interactions in [self.inverted_interactions.get(int(did1), {})]
        #                 for did2_interactions in [self.inverted_interactions.get(int(did2), {})]
        #                 for interactions in [{sid: (did1_interactions[sid], did2_interactions[sid]) 
        #                     for sid in set.intersection(set(did1_interactions.keys()), set(did2_interactions.keys()))}]
        #                 if len(interactions) >= 2 and
        #                     any(did1_outcome > did2_outcome for _, (did1_outcome, did2_outcome) in interactions.items()) and 
        #                     any(did2_outcome > did1_outcome for _, (did1_outcome, did2_outcome) in interactions.items()))

        #NOTE: spanned_of_selected vs spanned: spanned is peramnent while spanned_of_selected is only temporary block

        non_domination_score = len(non_dominated) / max(len(interactions), 1)
        domination_score = len(dominated) / max(len(interactions), 1)
        scores = {}
        scores["nond"] = non_domination_score
        # scores["nondb"] = 1 if non_domination_score > 0 else 0
        scores["nd"] = sqrt(non_domination_score * domination_score)
        scores["dom"] = domination_score
        scores["dup"] = -1 if duplicate_of_selected else 1 #prefer non-duplicants
        # scores["sp"] = -1 if spanned_of_selected else 1 #prefer nonspanned
        return scores
    
    def calc_complexity_score(self, ind, selected_inds, interactions):
        ind_interactions = interactions.get(ind, {})
        cfs = [s for s, r in ind_interactions.items() if r == 1]
        css = [s for s, r in ind_interactions.items() if r == 0]
        simplicity_score = 0 if len(ind_interactions) == 0 else (1 - len(cfs) / len(ind_interactions))
        difficulty_score = 0 if len(ind_interactions) == 0 else (1 - len(css) / len(ind_interactions))
        scores = {}
        # scores["simple"] = simplicity_score
        # scores["difficult"] = difficulty_score
        scores["sd"] = sqrt(simplicity_score * difficulty_score)
        scores["d"] = len(cfs)
        scores["-d"] = -scores["d"]
        scores["s"] = len(css)
        scores["-s"] = -scores["s"]
        return scores
    
    def build_selector(self, keys):
        ''' function that builds selector of features based on keys '''
        return lambda scores: tuple(scores[k] for k in keys)
        
    def sample_inds(self, pool, sample_strategy, sample_size, step, interactions):
        default_strategy = [{"t": 0, "keys": [["rel-kn", "kn", "nond", "sd", "dom"]] * sample_size }]
        sample_strategy = default_strategy if sample_strategy is None or len(sample_strategy) == 0 else sample_strategy
        sample_strategy = sorted(sample_strategy, key = lambda x: x["t"])
        if sample_strategy[0]["t"] != 0:
            sample_strategy = [*default_strategy, *sample_strategy]        
        curr_key_spec = next(spec for spec in reversed(sample_strategy) if step >= spec["t"])["keys"]
        selected = set()
        for i in range(sample_size):
            i_key_spec = curr_key_spec[i] if i < len(curr_key_spec) else curr_key_spec[-1]
            if (self.epsilon is not None) and (self.rnd.rand() < self.epsilon): #apply default epsilon strategy - ordered exploration
                i_key_spec = ["kn", "nond", "d"]
            elif self.softmax is not None:
                softmax_idx = [i for i, k in enumerate(i_key_spec) if k == self.softmax][0]
                i_key_spec = i_key_spec[:softmax_idx]
            key_selector = self.build_selector(i_key_spec)
            ind_scores = [ {"ind": ind, "scores": scores, "key": key_selector(scores) }
                                    for ind in pool if ind not in selected
                                    for scores in [{**self.calc_knowledge_score(ind, selected, interactions), 
                                                    **self.calc_domination_score(ind, selected, interactions), 
                                                    **self.calc_complexity_score(ind, selected, interactions)}]]
            sorted_inds = sorted(ind_scores, key=lambda x: x["key"], reverse=True)
            if len(sorted_inds) == 0:
                break
            best_ind_key = sorted_inds[0]["key"]
            best_inds = [] 
            i = 0 
            while best_ind_key == sorted_inds[i]["key"]:
                best_inds.append(sorted_inds[i])
                i += 1
            if self.softmax is not None: 
                softmax_sum = 0
                from math import e
                for ind in best_inds:
                    fitness = ind['scores'][self.softmax] #first criterion
                    ind['softmax'] = e ** (fitness / self.tau)
                    softmax_sum += ind['softmax']
                for ind in best_inds:
                    ind['softmax'] = ind['softmax'] / softmax_sum
                selected_one = self.rnd.choice(best_inds, p = [d['softmax'] for d in best_inds])
            else:                 
                selected_one = self.rnd.choice(best_inds)
            selected_ind = selected_one["ind"]
            # print(f"t={self.t} {i}/{self.n} d={selected_did} from alts {len(best_candidates)} {selected_candidate} kspec: {curr_key_spec[i]}")
            # self.block_similar(selected_did, blocked_dids)
            # print(f"\t selected {qid}: {selected_did}. {selected_dids}")
            selected.add(selected_ind)
        return list(selected)

    def init_sim(self) -> None:
        self.candidate_interactions = {}
        self.test_interactions = {}

    def interact(self):
        candidates = self.get_candidates()
        tests = self.get_tests()
        candidate_success = [[1 - self.game.interact(c, t) for t in tests] for c in candidates]
        #transpose prev matrix and negate it 
        test_success = [[1 - candidate_success[j][i] for j in range(len(candidates))] for i in range(len(tests))]
        for c, ints in zip(candidates, candidate_success):
            c_ints = self.candidate_interactions.setdefault(c, {})
            for t, outcome in zip(tests, ints):
                c_ints[t] = outcome 
        for t, ints in zip(tests, test_success):
            t_ints = self.test_interactions.setdefault(t, {})
            for c, outcome in zip(candidates, ints):
                t_ints[c] = outcome

    def get_candidates(self) -> list[Any]:
        return self.sample_inds(self.all_candidates, self.candidate_strategy, self.candidate_sample_size, self.step, self.candidate_interactions)
    
    def get_tests(self) -> list[Any]:
        return self.sample_inds(self.all_tests, self.test_strategy, self.test_sample_size, self.step, self.test_interactions)
    
class DynamicSpaceBuild(StepGameSimulation):
    ''' Implements the idea of gradual CDE space building and sampling '''
    def __init__(self, game: InteractionGame, rnd: RandomState, 
                    w_on_axis = 5, w_spanned  = 3, 
                    candidate_sample_size = 10, test_sample_size = 10,
                    **kwargs) -> None:
        super().__init__(game, rnd, **kwargs)
        self.all_candidates = game.get_candidates()
        self.all_tests = game.get_tests()
        self.w_on_axis = w_on_axis
        self.w_spanned = w_spanned
        self.candidate_sample_size = candidate_sample_size 
        self.test_sample_size = test_sample_size

    def init_sim(self) -> None:
        self.candidate_space_origin = CDEPoint(tests = set(self.all_candidates), candidates = set(self.all_tests))
        self.candidate_space_axes = [] #no axes at start, axis list of CDEPoint list
        self.candidate_space_spanned = [] #list of (ax_id, point_id, CDEPoint)
        self.candidate_shown = {} #number of times candidate was selected for interractions
        self.test_space_origin = CDEPoint(candidates = set(self.all_candidates), tests = set(self.all_tests))
        self.test_space_axes = []
        self.test_space_spanned = []
        self.test_shown = {} #number of times test was selected for interractions

    def sample_space(self, origin: CDEPoint, spanned, axes: list[list[CDEPoint]], shown: dict, sample_size: int):        
        origin_weights = {ind: -shown.get(ind, 0) for ind in origin.tests}
        axis_weights = {ind: (point_id + 1) * self.w_on_axis - shown.get(ind, 0) 
                            for axis in axes for point_id, point in enumerate(axis) 
                            for ind in point.tests}
        spanned_weights = {ind: self.w_spanned - shown.get(ind, 0)
                            for coords, point in spanned
                            for ind in point.tests }
        ind_weights = {**origin_weights, **axis_weights, **spanned_weights}
        inds, weights = zip(*ind_weights.items())
        sum_w = sum(weights) + 1
        probs = [w / sum_w for w in weights]
        selected = self.rnd.choice(inds, size = sample_size, replace=False, p=probs)
        for ind in selected:
            shown[ind] = shown.get(ind, 0) + 1
        return selected
    
    def get_candidates(self) -> list[Any]:
        return self.sample_space(self.candidate_space_origin, self.candidate_space_spanned, self.candidate_space_axes, self.candidate_shown, self.candidate_sample_size)

    def get_tests(self) -> list[Any]:
        return self.sample_space(self.test_space_origin, self.test_space_spanned, self.test_space_axes, self.test_shown, self.test_sample_size)
    
    def update_space(self, origin: CDEPoint, spanned, axes: list[list[CDEPoint]], interactions:dict[Any, dict[Any, int]]):
        ''' Moves inds in the corresponding space '''
        inds_cfs = {t:cfs for t, ints in interactions.items() for cfs in [set(c for c, o in ints.items() if o == 1)] if len(cfs) > 0}
        # TODO
        pass

    def interact(self):
        # sampling  
        candidates = self.get_candidates()
        tests = self.get_tests()
        candidate_success = {c:{t:1 - self.game.interact(c, t) for t in tests} for c in candidates}
        #transpose prev matrix and negate it 
        test_success = {tests[i]:{candidates[j]:1 - candidate_success[j][i] for j in range(len(candidates))} for i in range(len(tests))}
        self.update_space(self.candidate_space_origin, self.candidate_space_spanned, self.candidate_space_axes, candidate_success)
        self.update_space(self.test_space_origin, self.test_space_spanned, self.test_space_axes, test_success)

# set of games 

class NumberGame(InteractionGame):
    ''' Base class for number games with default representation of n-dim Nat tuples 
        Init is random in given nat range per dim 
        Change is +-1 random per dimension
    '''
    def __init__(self, rnd: np.random.RandomState, dims = 2, min_num = 0, max_num = 500, **kwargs) -> None:
        super().__init__(rnd)
        self.min_num = min_num
        self.max_num = max_num
        self.dims = dims
        super().__init__()

    def init_candidate(self) -> Any:
        return tuple(self.rnd.randint(self.min_num, self.max_num) for _ in range(self.dims))
    
    def change_candidate(self, parent: Any) -> Any:
        ''' classic schema +-1 for each dim based on random decision 
            Parent is n-dim tuple created with init
        '''
        def adjust(v: int):
            if v == self.min_num:
                return v + 1 
            elif v == self.max_num - 1:
                return v - 1 
            return v + self.rnd.choice([-1, 1])
        return tuple(adjust(d) for d in parent)
    
class IntransitiveGame(NumberGame):
    ''' The IG as it was stated in Bucci article '''
    def interact(self, candidate, test) -> int:
        ''' 0 - candidate is better thah test, 1 - test fails candidate '''
        abs_diffs = [abs(x - y) for x, y in zip(candidate, test)]
        res = 0 if (abs_diffs[0] > abs_diffs[1] and candidate[1] > test[1]) or (abs_diffs[1] > abs_diffs[0] and candidate[0] > test[0]) else 1
        return res

class FocusingGame(NumberGame):
    ''' The FG as it was stated in Bucci article '''
    def interact(self, candidate, test) -> int:
        ''' 0 - candidate is better thah test, 1 - test fails candidate '''
        res = 0 if (test[0] > test[1] and candidate[0] > test[0]) or (test[1] > test[0] and candidate[1] > test[1]) else 1
        return res
    
class CompareOnOneGame(NumberGame):
    ''' Game as it was stated in Golam article '''
    def interact(self, candidate, test) -> int:
        ''' 0 - candidate is better thah test, 1 - test fails candidate '''
        max_pos = np.argmax(test)
        res = 0 if candidate[max_pos] >= test[max_pos] else 1
        return res
    
# game that is based on CDESpace 
class CDESpaceGame(InteractionGame):
    ''' Loads given CDE space from the system and provide interactions based on it 
        Candidates and individuals are Nat numbers
        Init is random from correspondign pools 
        Change is mutation, adjustable by mutation strategy.
        Supported mutations: +-1 at rand, rand resample
    '''
    def __init__(self, rnd: RandomState, space: CDESpace, 
                    candidate_mutation_strategy = "plus_minus_one", test_mutation_strategy = "plus_minus_one", **kwargs) -> None:
        super().__init__(rnd)
        self.space = space 
        self.candidates = sorted(space.get_candidates())
        self.tests = sorted(space.get_tests())
        self.candidate_mutation_strategy = getattr(self, candidate_mutation_strategy)
        self.test_mutation_strategy = getattr(self, test_mutation_strategy)
        self.all_fails = space.get_candidate_fails()

    def init_candidate(self) -> Any:
        return self.rnd.choice(self.candidates)
    
    def init_test(self) -> Any:
        return self.rnd.choice(self.tests)

    def change_candidate(self, parent: Any) -> Any:
        return self.candidate_mutation_strategy(parent, self.candidates)
    
    def change_test(self, parent: Any) -> Any:
        return self.test_mutation_strategy(parent, self.tests)    
    
    def interact(self, candidate, test) -> int:
        return 1 if test in self.all_fails.get(candidate, set()) else 0