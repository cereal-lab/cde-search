''' Contains classes that define different populations and the way they change 
    See base class Population
'''

from abc import abstractmethod
from collections import deque
from math import sqrt
from typing import Any
from deca import extract_dims
from params import PARAM_IND_CHANGES_STORY, PARAM_INTS, PARAM_UNIQ_INTS, rnd, PARAM_UNIQ_INDS, PARAM_MAX_INDS, param_steps, param_popsize

class Population:
    ''' Base class for population of candidates/tests which defines different ways of breeding 
        and going to next generation
    '''            
    def __init__(self, all_inds: list[Any], popsize, **kwargs) -> None:
        self.size = popsize 
        self.population = [] #first call of get_inds should initialize
        self.interactions = {}
        self.all_inds = all_inds #pool of all possible individuals
        self.pop_metrics = {PARAM_UNIQ_INDS: 0, PARAM_MAX_INDS: 0}
        self.seen_inds = set()
        self.pop_params = {"size": popsize, "class":self.__class__.__name__, **kwargs}

    def get_inds(self, **filters) -> list[Any]:
        return self.population
    
    def get_all_inds(self) -> list[Any]:
        return self.all_inds

    def init_inds(self) -> None:
        self.interactions = {}
        self.pop_metrics[PARAM_UNIQ_INDS] = 0
        self.pop_metrics[PARAM_MAX_INDS] = 0
        self.seen_inds = set()

    def merge_interactions(self, interactions: dict[Any, list[tuple[Any, int]]]) -> None:
        num_uniq_ints = 0
        for ind, ints in interactions.items():                        
            ind_ints = self.interactions.setdefault(ind, {})
            self.pop_metrics[PARAM_INTS] = self.pop_metrics.get(PARAM_INTS, 0) + len(ints)
            for t, outcome in ints:
                if t not in ind_ints:
                    num_uniq_ints += 1
                ind_ints[t] = outcome
        self.pop_metrics[PARAM_UNIQ_INTS] = self.pop_metrics.get(PARAM_UNIQ_INTS, 0) + num_uniq_ints

    def update(self, interactions: dict[Any, list[tuple[Any, int]]]) -> None:
        self.merge_interactions(interactions)

class HCPopulation(Population):
    ''' Population that evolves with parent-child hill climbing '''
    def __init__(self, all_inds: list[Any], popsize = param_popsize,
                    mutation = "plus_minus_one", 
                    selection = "pareto_select", 
                    init = "rand_init", init_range = None,
                    ind_range = None, **kwargs) -> None:
        super().__init__(all_inds, popsize, **{"mutation":mutation, "selection":selection, **kwargs})
        self.mutation_strategy = getattr(self, mutation)
        self.selection_strategy = getattr(self, selection)
        self.init_strategy = getattr(self, init)
        self.ind_range = (min(all_inds), max(all_inds)) if ind_range is None else ind_range
        self.init_range = init_range
    
    def plus_minus_one(self, ind: tuple[int,int]):
        # Mutation for number games inds (i, j) -> (i+-1, j+-1)    
        # def wrap_value(v: int):
        #     if self.ind_range is None:
        #         return v
        #     if v < self.ind_range[0]: 
        #         v = self.ind_range[1]
        #     elif v > self.ind_range[1]:
        #         v = self.ind_range[0]
        #     return v
        
        # mutated = tuple(wrap_value(c + rnd.choice([-1,1])) for c in ind)        
        # return mutated
        # while True: 
        mutated = tuple(c + rnd.choice([-1,1]) for c in ind)            
        # if mutated[0] < self.ind_range[0] or mutated[0] > self.ind_range[1] or\
        #     mutated[1] < self.ind_range[0] or mutated[1] > self.ind_range[1]:
        #     continue
        return mutated

    def resample(self, ind: Any):
        ''' Picks random other ind from pool '''
        for _ in range(10):
            index = rnd.randint(0, len(self.all_inds))
            mutated = self.all_inds[index]
            if mutated != ind:
                return mutated
        return ind 
    
    def pareto_select(self, i, all_ints):
        ''' Version of Pareto selection 
            If parent and child are same by performance - pick child for progress 
            Then, consider domination and preserve parent if non-dominant
        '''
        common_ints = [(co, po) for co, po in all_ints if co is not None and po is not None]
        # if all(co == po for co, po in common_ints):
        #     return self.children[i] #prefer progress on of performance change
        #prefer domination 
        ind = self.children[i] if len(common_ints) > 1 and \
                            all(co >= po for co, po in common_ints) and \
                            any(co > po for co, po in common_ints) else self.population[i]
        # todo: what to do on non-dominance? Currently parent is picked - maybe count wins? 
        return ind
    
    def informativeness_select(self, i, all_ints):
        ''' selection based on informativeness score aggregation '''
        def informativeness_score(ints: list[int]):
            ''' counts pairs of interactions of same outcome '''
            # return sum(1 for i in range(len(ints)) for j in range(i+1, len(ints)) if ints[i] == ints[j])
            return sum(1 for s1 in ints for s2 in ints if s1 == s2)
        common_ints = [(co, po) for co, po in all_ints if co is not None and po is not None]
        parent_informativeness = informativeness_score([o for _, o in common_ints])
        child_informativeness = informativeness_score([o for o, _ in common_ints])
        ind = self.children[i] if child_informativeness > parent_informativeness else self.population[i] 
        return ind    
    
    def num_wins(self, i, all_ints):
        parent_score = sum([o for _, o in all_ints if o is not None])
        child_score = sum([o for o, _ in all_ints if o is not None])
        ind = self.children[i] if child_score > parent_score else self.population[i] 
        return ind
    
    def rand_init(self):
        parent_indexes = rnd.choice(len(self.all_inds), size = self.size, replace=False)
        self.population = [self.all_inds[i] for i in parent_indexes]

    def zero_init(self):
        self.population = [tuple(0 for _ in self.all_inds[0]) if type(self.all_inds[0]) is tuple else 0 for _ in range(self.size)]

    def range_init(self):
        self.population = [tuple(rnd.choice(self.init_range) for _ in self.all_inds[0]) if type(self.all_inds[0]) is tuple else rnd.choice(self.init_range) for _ in range(self.size)]

    def init_inds(self) -> None:
        super().init_inds()
        self.init_strategy()
        self.children = [self.mutation_strategy(parent) for parent in self.population]

        self.pop_metrics[PARAM_MAX_INDS] += 2 * self.size
        self.seen_inds.update(self.population, self.children)
        self.pop_metrics[PARAM_UNIQ_INDS] = len(self.seen_inds)          

    def get_inds(self, *, only_parents = False, only_children = False, **filters) -> list[Any]:        
        return [*([] if only_children else self.population), *([] if only_parents else self.children)]

    def update(self, interactions: dict[Any, list[tuple[Any, int]]]) -> None:
        super().update(interactions)
        parent_ints = [interactions.get(p, []) for p in self.population]
        child_ints = [interactions.get(c, []) for c in self.children]   
        def get_counts(ints: list[tuple[Any, int]]):
            res = {} 
            for k, _ in ints: 
                res[k] = res.get(k, 0) + 1
            return res 
        def get_union_outcomes(parent_ints: list[tuple[Any, int]], child_ints: list[tuple[Any, int]]):
            parent_dict = {k:v for k, v in parent_ints}
            child_dict = {k:v for k, v in child_ints}
            parent_counts = get_counts(parent_ints)
            child_counts = get_counts(child_ints)
            key_set = set(*parent_dict.keys(), *child_dict.keys())
            int_dict = {k: (child_counts.get(k, 0), child_dict.get(k, None), parent_counts.get(k, 0), parent_dict.get(k, None)) for k in key_set}
            child_vect = []
            parent_vect = []
            for child_count, child_outcome, parent_count, parent_outcome in int_dict.values():
                n = max(child_count, parent_count)
                child_vect.extend([child_outcome] * child_count)
                child_vect.extend([None] * (n - child_count))
                parent_vect.extend([parent_outcome] * parent_count)
                parent_vect.extend([None] * (n - parent_count))
            return list(zip(child_vect, parent_vect))
        all_ints = [get_union_outcomes(p_ints, c_ints) for p_ints, c_ints in zip(parent_ints, child_ints)]
        
        selected = [ self.selection_strategy(i, ints) for i, ints in enumerate(all_ints)]
        num_changes = sum(1 for s, p in zip(selected, self.population) if s != p)
        self.pop_metrics.setdefault(PARAM_IND_CHANGES_STORY, []).append(num_changes)
        self.population = selected 
        self.children = [self.mutation_strategy(parent) for parent in self.population]
                
        self.pop_metrics[PARAM_MAX_INDS] += 2 * self.size
        self.seen_inds.update(self.population, self.children)
        self.pop_metrics[PARAM_UNIQ_INDS] = len(self.seen_inds)    

class Sample(Population):
    ''' Population is a sample from interaction matrix '''
    def __init__(self, all_inds: list[Any], size:int, **kwargs) -> None:
        super().__init__(all_inds, size, **kwargs)        
        self.t = 0
        self.keys = []

    @abstractmethod
    def sample_inds(self) -> list[Any]:
        pass

    def init_inds(self) -> None:
        super().init_inds()        
        self.population = None #self.sample_inds()
        self.for_group = None
        self.t = 0

    def get_same_interactions_percent(self):
        ''' We would like to avoid same interactions for the following steps
            This method computes the percent of given group in self.for_group with already present interactions
            It is up to concrete population how to handle same interactions (current simulations assume uselessness of such interactions)
        '''
        if len(self.for_group) == 0:
            return {}
        scores = {ind: existing / len(self.for_group)
                    for ind, ind_ints in self.interactions.items()
                    for existing in [sum(1 for t in self.for_group if t in ind_ints)]
                    if existing > 0}
        return scores 

    def get_inds(self, *, for_group = [], **filters) -> list[Any]:
        if self.population is not None and for_group == self.for_group:
            return self.population
        self.for_group = for_group #adjusts sampling by avoiding repetitions of interactions - smaller chance to interruct again
        self.population = self.sample_inds()
        self.pop_metrics[PARAM_MAX_INDS] += self.size
        self.seen_inds.update(self.population)
        self.pop_metrics[PARAM_UNIQ_INDS] = len(self.seen_inds)
        return self.population
        
class InteractionFeatureOrder(Sample):
    ''' Sampling of search space based on grouping by interactions features and ordering of the groups '''
    def __init__(self, all_inds: list[Any], size: int, dedupl = False, anneal_max_time = param_steps, 
                    strategy = None, epsilon = None, softmax = None, tau = 1,
                    **kwargs) -> None:
        super().__init__(all_inds, size, **{"dedupl": dedupl, "anneal_max_time":anneal_max_time, "strategy":strategy, "epsilon": epsilon, "softmax": softmax, "tau": tau, **kwargs})        
        self.epsilon = epsilon
        self.softmax = softmax
        self.tau = tau
        self.dedupl = dedupl
        self.anneal_max_time = anneal_max_time
        self.features = {} #contains set of features per interacted individual 
        default_strategy = [{"t": 0, "keys": [["kn-rel", "kn", "nond", "sd", "dom"]] }]
        strategy = default_strategy if strategy is None else strategy
        if type(strategy) is list:
            if len(strategy) == 0:
                strategy = default_strategy
            elif type(strategy[0]) is str: 
                strategy = [{"t": 0, "keys": [strategy] }]
            elif type(strategy[0]) is list: 
                strategy = [{"t": 0, "keys": strategy }]
        else:
            strategy = [strategy]
        strategy = sorted(strategy, key = lambda x: x["t"])
        if strategy[0]["t"] != 0:
            strategy = [*default_strategy, *strategy]          
        self.strategy = list(reversed(strategy))
        # this is alt strategy when for_group is specified
        # third = self.size // 3
        # self.exploration = [{"t": 0, "keys": [["-one", "kn"]] * third + [["one", "kn"]] * third + self.strategy[-1]["keys"] }]
        # twothrids = 2 * self.size // 3
        # self.exploration = [{"t": 0, "keys": [["-one", "kn"]] * twothrids + self.strategy[-1]["keys"] }]
        # self.strategy = [{"t": 0, "keys": self.strategy[-1]["keys"] * twothrids + [["-one", "kn"]] }]


    def init_inds(self) -> None:
        super().init_inds()
        self.features = {}        

    def build_common_interactions(self, ind: Any, ind_ints: dict[Any, int], pool_inds):
        ''' computes common interactions with other inds of the pool '''
        common_interactions = {ind1: interactions
            for ind1 in pool_inds if ind1 != ind
            for ind1_ints in [self.interactions.get(ind1, {})]
            for interactions in [{common_test_id: (ind_ints[common_test_id], ind1_ints[common_test_id]) 
                for common_test_id in set.intersection(set(ind_ints.keys()), set(ind1_ints.keys()))}]
            if len(interactions) > 0} #non-empty interactions set
        return common_interactions     

    def update(self, interactions: dict[Any, list[tuple[Any, int]]]) -> None:
        ''' Also updates individual features from interactions '''
        super().update(interactions)        
        other_inds = set(interactions.keys())
        for ind, cur_ind_ints in interactions.items():
            ind_features = self.features.setdefault(ind, {})
            ind_features["one"] = 1
            ind_features["-one"] = -1
            ind_features["num_ints"] = ind_features.get("num_ints", 0) + len(cur_ind_ints)
            ind_features["num_uniq_ints"] = len(self.interactions.get(ind, {}))
            ind_features["kn"] = -ind_features["num_ints"]
            ind_ints = self.interactions[ind]
            common_interactions = self.build_common_interactions(ind, ind_ints, other_inds)            
            total_group = [(ind1, ints) for ind1, ints in common_interactions.items() if len(ints) > 1]
            cur_total_group_ids = set([ind1 for ind1, _ in total_group])
            total_group_ids = ind_features.setdefault("total_group_ids", set())
            total_group_ids.update(cur_total_group_ids)
            non_dom_set = ind_features.setdefault("non_dom_set", set())
            non_dom_set -= other_inds #probably something changed about them            
            non_dominated = [ind1 for ind1, ints in total_group
                                    if any(ind_outcome > ind1_outcome for _, (ind_outcome, ind1_outcome) in ints.items()) and 
                                        any(ind_outcome < ind1_outcome for _, (ind_outcome, ind1_outcome) in ints.items())]
            non_dom_set.update(non_dominated)
            dom_set = ind_features.setdefault("dom_set", set())
            dom_set -= other_inds    
            dominated = [ind1 for ind1, ints in total_group
                            if ind1 not in non_dom_set
                            if all(ind_outcome >= ind1_outcome for _, (ind_outcome, ind1_outcome) in ints.items()) and 
                            any(ind_outcome > ind1_outcome for _, (ind_outcome, ind1_outcome) in ints.items())]        
            dom_set.update(dominated)
            ind_features["nond"] = 0 if len(total_group) == 0 else round(len(non_dom_set) * 100 / len(total_group_ids))
            ind_features["dom"] = 0 if len(total_group) == 0 else round(len(dom_set) * 100 / len(total_group_ids))
            ind_features["nd"] = sqrt(ind_features["nond"] * ind_features["dom"])

            cfs = [s for s, r in ind_ints.items() if r == 1]
            css = [s for s, r in ind_ints.items() if r == 0]
            simplicity_score = round((0 if len(ind_ints) == 0 else (1 - len(cfs) / len(ind_ints))) * 100)
            difficulty_score = round((0 if len(ind_ints) == 0 else (1 - len(css) / len(ind_ints))) * 100)
            ind_features["sd"] = round(sqrt(simplicity_score * difficulty_score))
            ind_features["d"] = difficulty_score
            ind_features["-d"] = -difficulty_score
            ind_features["s"] = simplicity_score
            ind_features["-s"] = -simplicity_score

        self.t += 1  
        self.population = None    
    
    def is_dupl(self, ind: Any, selected_inds):
        ind_ints = self.interactions[ind]
        common_interactions_with_selected = self.build_common_interactions(ind, ind_ints, selected_inds)
        duplicate_of_selected = any(ind1 for ind1, ints in common_interactions_with_selected.items()
                                    for ind_cfs in [[(ind_outcome, ind1_outcome) for _, (ind_outcome, ind1_outcome) in ints.items() if ind_outcome == 1]] 
                                    if len(ind_cfs) >= 1 and all(ind_outcome == ind1_outcome for (ind_outcome, ind1_outcome) in ind_cfs))
        # scores = dict(dup = 0 if duplicate_of_selected else 1) #prefer non-duplicants
        # scores["kn-rel"] = -(sum(len([test for test in self.interactions.get(ind1, {}).keys() if test in ind_ints]) 
        #                                 for ind1 in [selected_ind]))
        return duplicate_of_selected
    
    def build_selector(self, keys):
        ''' function that builds selector of features based on keys '''
        return lambda scores: tuple(scores[k] for k in keys)
        
    def sample_inds(self) -> list[Any]:
        curr_key_spec = next(spec for spec in self.strategy if self.t >= spec["t"])["keys"]
        seens_interaction_percents = self.get_same_interactions_percent()
        excluded = {ind for ind, p in seens_interaction_percents.items() if rnd.rand() < p }
        selected = set()
        dupls = set()
        self.keys = []
        start_explore_i = self.size
        coef = 2 if len(self.for_group) > 0 else 1 #explore more
        start_explore_i = self.size - max(0, (coef * self.size // 3) * (1 - self.t / self.anneal_max_time))
        while len(selected) < self.size:
            i = len(selected)
            i_key_spec = ["-one", "kn"] if i >= start_explore_i else curr_key_spec[i] if i < len(curr_key_spec) else curr_key_spec[-1]
            if (self.epsilon is not None) and (rnd.rand() < self.epsilon): #apply default epsilon strategy - ordered exploration
                i_key_spec = ["kn", "nond", "d"]
            elif self.softmax is not None:
                softmax_idx = [i for i, k in enumerate(i_key_spec) if k == self.softmax][0]
                i_key_spec = i_key_spec[:softmax_idx]
            key_selector = self.build_selector(i_key_spec)
            default_features_for_noninterracted = tuple(0 for _ in i_key_spec)
            cur_ind_scores = [ {"ind": ind, "scores": ind_features, "key": key_selector(ind_features) }
                                    for ind, ind_features in self.features.items() if ind not in selected and ind not in excluded and ind not in dupls]
            if len(self.interactions) < len(self.all_inds):
                cur_ind_scores.append(dict(noninteracted = True, key=default_features_for_noninterracted))
            # cur_ind_scores.append({"key"})
            # (s * seens_interaction_penalties.get(ind, 1) for s in 
            sorted_inds = sorted(cur_ind_scores, key=lambda x: x["key"], reverse=True)
            if len(sorted_inds) == 0:
                if len(excluded) == 0:
                    break
                excluded = set()
                continue
            if sorted_inds[0].get("noninteracted", False):
                #pick one from noninterracted, search space could be big therefore we resort to enumeration with limited number of times                 
                tries = 0
                while tries < 10:
                    i = rnd.choice(len(self.all_inds))
                    selected_ind = self.all_inds[i]
                    if selected_ind not in selected and selected_ind not in self.interactions:
                        self.keys.append(f"{selected_ind} kn")
                        break
                    tries += 1 
            else:
                best_ind_key = sorted_inds[0]["key"]                
                best_inds = [] 
                i = 0 
                while i < len(sorted_inds) and best_ind_key == sorted_inds[i]["key"]:
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
                    selected_one = rnd.choice(best_inds, p = [d['softmax'] for d in best_inds])
                else:
                    selected_one = rnd.choice(best_inds)
                selected_ind = selected_one["ind"]
                if self.dedupl and self.is_dupl(selected_ind, selected):
                    dupls.add(selected_ind)
                    continue
                selected_features = " ".join("{0} {1}".format(k, round(v)) for k, v in zip(i_key_spec, best_ind_key) if v != 0)
                self.keys.append(f"{selected_ind} {selected_features}")
            # print(f"t={self.t} {i}/{self.n} d={selected_did} from alts {len(best_candidates)} {selected_candidate} kspec: {curr_key_spec[i]}")
            # self.block_similar(selected_did, blocked_dids)
            # print(f"\t selected {qid}: {selected_did}. {selected_dids}")
            selected.add(selected_ind)
        return list(selected)

class ParetoGraphSample(Sample):
    ''' Tracks interactions and from sparced interaction metrics builds Pareto graph 
        directed edges of which define the Pareto relations. Nodes contains tests/candidates 
        Sampling of graph structure happens with the observation 
        1. Origin of CDE Space would have no incoming edges (as well as starts of the axes)
           Weight is minimal for origin (start of the axis)
        2. The further the axis point on the axis from the origin the greater is number of edges from the beginning of axes
           Weight is increased with number of hops
        3. First incomming edge means that the point/node is on the axis. Second and more - it is probably spanned point 
           Weight is descreased with additional incomming edges 
        There should be normalization of weights in range of [0, 1] before further use in calc of probs
    '''
    def __init__(self, all_inds: list[Any], size: int,
                    rank_penalty = 2, min_exploitation_chance = 0.5, max_exploitation_chance = 0.8, max_step = param_steps,
                    **kwargs) -> None:
        super().__init__(all_inds, size, **{"rank_penalty":rank_penalty, "min_exploitation_chance": min_exploitation_chance, "max_exploitation_chance": max_exploitation_chance, "max_step":max_step, **kwargs})
        self.rank_penalty = rank_penalty
        self.min_exploitation_chance = min_exploitation_chance
        self.max_exploitation_chance = max_exploitation_chance
        self.max_step = max_step
        self.keys = []   

    def init_inds(self) -> None:
        super().init_inds() 
        self.ranks = {}
        # self.spanned = set() #could potentially become the end of axes

    # class Node:
    #     def __init__(self) -> None:
    #         self.tests = set() 
    #         self.ints = {}
    #         self.dominates = [] 
    #         self.dominated = [] 
    #         self.axes = {} #dict, axes_id: point_id. For spanned - many of them      
    #         self.ax_max_rank = {}     
    #         # self.rank = 0 

    def update(self, interactions: dict[Any, list[tuple[Any, int]]]) -> None:
        ''' Applies DECA to given local ints and extract axes '''
        super().update(interactions)
        self.keys = []
        tests = [[o for _, o in ind_ints] for _, ind_ints in interactions.items() ]
        inds = [ind for ind, _ in interactions.items() ]
        dims, _, spanned, _ = extract_dims(tests)
        # dom_sets = {best_ind: set(other_ind for other_point in other for other_ind in other_point)
        # to_remove = set()
        # to_add = {}
        present_axes = set(self.ranks.keys())
        for dim in dims:
            for point_id, group in enumerate(dim):
                w = (point_id + 1) / len(dim)
                if len(dim) == (point_id + 1):
                    w += 100
                for i in group:
                    ind = inds[i]
                    if ind not in present_axes: #check if we can remove any other axis by checking common interactions
                        self.ranks[ind] = w
                        # for ind2 in self.ranks.keys():
                        #     common_ints = self.get_common_ints(self.interactions[ind], self.interactions[ind2])
                        #     if len(common_ints) > 1:
                        #         if all(o1 >= o2 for o1, o2 in common_ints.values()) and any(o1 > o2 for o1, o2 in common_ints.values()):
                        #             to_remove.add(ind2)
                        #         elif all(o2 >= o1 for o1, o2 in common_ints.values()) and any(o2 > o1 for o1, o2 in common_ints.values()):
                        #             del to_add[ind]
                    else:
                        self.ranks[ind] = sqrt(w * self.ranks[ind])

        for i in spanned:
            ind = inds[i]
            if ind not in self.ranks:
                self.ranks[ind] = 0.01
            else:
                self.ranks[ind] = sqrt(0.01 * self.ranks[ind])
        # for k, v in to_add.items():              
        #     self.ranks[k] = v
        # self.axes_ends.difference_update(to_remove)
        # filtered_spanned = set([ind for i in spanned for ind in [inds[i]] if ind not in self.axes_ends])
        # self.spanned.update(filtered_spanned)
        # # for ind in self.spanned:
        # #     keys = set(self.interactions[ind].keys())
        # #     combined = {}
        # #     for ind2 in self.axes_ends:
        # #         for k, o in self.interactions[ind2].items():
        # #             if k in keys:
        # #                 combined.setdefault(k, []).append(o)
        # #     combined_vals = {k:max(combined.get(k, []), default=None) for k in keys}
        # #     co, so = zip(*[(combined_vals[k], self.interactions[ind][k]) for k in keys if combined_vals[k] is not None])
        # #     if co != so:
        # #         to_remove.add(ind)
        # self.spanned.difference_update(set(inds) - filtered_spanned)
        
        # dropping low score 
        to_preserve = 2 * self.size
        if len(self.ranks) > to_preserve:
            sorted_ranks = sorted(self.ranks.items(), key = lambda x: x[1])
            for k, _ in sorted_ranks[:-to_preserve]:
                del self.ranks[k]
                
        self.t += 1  
        self.population = None    

    def get_common_ints(self, ind1_ints: dict[Any, int], ind2_ints: dict[Any, int]):
        res = {k: (o, ind2_ints[k]) for k, o in ind1_ints.items() if k in ind2_ints}
        return res

    def sample_inds(self) -> list[Any]:
        
        #exploration vs exploitation - some tests are not in the Pareto graph yet. We need also decide how to balance here
        #we use annealing with game step increase the chance for exploitation with time 

        selected = set()
        # interacted_weights = {ind: 1 - p for ind, p in self.get_same_interactions_percent().items()}

        if len(self.for_group) > 0:
            exploitation_slots = self.size
        else:
            exploitation_percent = self.min_exploitation_chance + (self.t / self.max_step) * (self.max_exploitation_chance - self.min_exploitation_chance)
            exploitation_slots = round(self.size * exploitation_percent)

        # sampling order: axes (exploit), unknown (explore), spanned (exploit, 0.1), origin (explore, 0.1)

        test_weights = {**self.ranks}

        while len(test_weights) > 0 and exploitation_slots > 0:
            sum_node_weights = sum(test_weights.values())
            test_list, p = zip(*[ (ind, w / sum_node_weights) for ind, w in test_weights.items()])
            test_id = rnd.choice(len(test_list), p = list(p))
            selected_test = test_list[test_id]
            if selected_test not in selected:
                selected.add(selected_test)
                is_axes = "ax" if test_weights[selected_test] > 0.5 else "sp"
                self.keys.append(f"{selected_test} {is_axes} {round(p[test_id] * 100)}% {len(test_list)}")                
                exploitation_slots -= 1
            del test_weights[selected_test]

        #here we treet origin as usual 
        j = 0
        while len(selected) < self.size and j < 100:
            was_selected = False
            i = 0
            while i < 10:             
                selected_index = rnd.choice(len(self.all_inds))            
                selected_ind = self.all_inds[selected_index]
                i += 1            
                if selected_ind not in selected:
                    selected.add(selected_ind)
                    self.keys.append(f"{selected_ind} new")
                    was_selected = True 
                    break
            if was_selected:
                continue
            j += 1    
            
        return list(selected)

class ACOPopulation(Sample):
    ''' Population that changes by principle of pheromone distribution 
        We can think of ants as their pheromons - the population is the values of pheromones for each of test 
        At same time, we still have desirability criteria given by interaction matrix. 
        Number of ants is given by popsize. Each of them makes decision what test to pick based on pheromone and desirability 
        No two ants can pick same test. 
        The tests are sampled by such ants but pheremones undergo evolution.
        Note, self.population is still used for sampled test 
        self.pheromones is what evolves   
        Pheromone range [1, inf). It magnifies desirability of the test 
        It goes up when the selected test increases the desirability in the interaction step
        And gradually goes down to 1. 
    '''
    def __init__(self, all_inds: list[Any], size: int,
                    pheromone_decay = 0.8, dom_bonus = 1, span_penalty = 0.25, **kwargs) -> None:
        super().__init__(all_inds, size, **{"pheromone_decay":pheromone_decay, "dom_bonus": dom_bonus, "span_penalty": span_penalty, **kwargs})
        self.pheromones = {}
        self.pheromone_decay = pheromone_decay
        self.dom_bonus = dom_bonus
        self.span_penalty = span_penalty

    def init_inds(self) -> None:
        super().init_inds()
        self.pheromones = {ind: 1 for ind in self.all_inds} #each ind gets a bit of pheromone 
        self.desirability = {}
        self.population_desirability = 0

    def calc_desirability(self) -> None:
        ''' Analysis of interaction matrix for combined criteria how ind is good at moment 
            Desired test is one that dominates many of individuals but do not dominate individuals that Pareto-noncomparable
            For each dominated + dom_bonus, for each dominated that pareto-noncomparable - spanned_penalty
        '''

        #step 1 - compute all common interactions between all pair of tests 
        ind_ids = list(self.interactions.keys())
        common_ints = {}
        for i1 in range(len(ind_ids)):
            ind1 = ind_ids[i1]
            ind1_ints = self.interactions[ind1]
            ind1_int_keys = set(ind1_ints.keys())
            for i2 in range(i1 + 1, len(ind_ids)):
                ind2 = ind_ids[i2]
                ind2_ints = self.interactions[ind2]
                ind2_int_keys = set(ind2_ints.keys())
                common_ids = set.intersection(ind1_int_keys, ind2_int_keys)
                if len(common_ints) > 1:
                    common_ints.setdefault(ind1, {})[ind2] = [(ind1_ints[tid], ind2_ints[tid]) for tid in common_ids] 
                    common_ints.setdefault(ind2, {})[ind1] = [(ind2_ints[tid], ind1_ints[tid]) for tid in common_ids]         
        dominated = {ind:[ind1 for ind1, cints in ind_ints.items() 
                        if all(io >= i1o for io, i1o in cints) and any(io > i1o for io, i1o in cints) ] 
                        for ind, ind_ints in common_ints.items()}
        def non_dominated_pairs(inds: list[Any]):
            result = []
            for i1 in range(len(inds)):
                ind1 = inds[i1]
                for i2 in range(i1 + 1, len(inds)):
                    ind2 = inds[i2]
                    ints = common_ints.get(ind1, {}).get(ind2, [])
                    if len(ints) > 1 and any(o1 > o2 for o1, o2 in ints) and any(o2 > o1 for o1, o2 in ints):
                        result.append((ind1, ind2))
            return result 
        dominated_non_dominant = {ind:non_dominated_pairs(ind_dom) for ind, ind_dom in dominated.items()}

        self.desirability = {ind: self.dom_bonus * len(dominated.get(ind, [])) - self.span_penalty * len(dominated_non_dominant.get(ind, []))
                  for ind in ind_ids}                        

    def build_selector(self, keys):
        ''' function that builds selector of features based on keys '''
        return lambda scores: tuple(scores[k] for k in keys)
        
    def sample_inds(self) -> list[Any]:
        ''' Sample with pheromones and desirability rule 
            Ants book works with random proportional transition rule 
            Here we ignore alpha and beta heuristic parameters (= 1)
        '''        
        #normalize scores from [min, max] -> [0, max2] - spanned point is on level of origin
        min_score = min(self.desirability.values(), default=0)
        d_scores = {k: v - min_score for k, v in self.desirability.items()}

        all_scores = {ind: d_scores.get(ind, 0) for ind in self.all_inds}
        all_scores_shifted = {ind: score + 0.1 for ind, score in all_scores.items()}
        interacted_penalty = {ind: 0.1 + 0.9 * (1 - p) for ind, p in self.get_same_interactions_percent().items()}
        scores_pheromoned = {ind:self.pheromones[ind] * desirability * interacted_penalty.get(ind, 1) for ind, desirability in all_scores_shifted.items()}
        inds, weights = zip(*list(scores_pheromoned.items()))
        sum_weight = sum(weights)
        if sum_weight == 0:
            sum_weight = 1 
        probs = [w / sum_weight for w in weights]
        selected_indexes = rnd.choice(len(inds), size = self.size, replace = False, p = probs)
        selected = [inds[i] for i in selected_indexes]
        self.population_desirability = sum(self.desirability.get(ind, 0) for ind in selected)
        return list(selected)
    
    def update(self, interactions: dict[Any, list[tuple[Any, int]]]) -> None:
        prev_pop_desirability = self.population_desirability
        self.merge_interactions(interactions) 
        self.calc_desirability()
        new_pop_desirability = sum(self.desirability.get(ind, 0) for ind in self.population)
        if new_pop_desirability > prev_pop_desirability:
            for ind in self.population:
                self.pheromones[ind] += 1 
        for ind in self.all_inds:
            self.pheromones[ind] = 1 + (self.pheromones[ind] - 1) * self.pheromone_decay
        self.population = None        

class OneTimeSequential(Population):
    ''' Solits all_inds onto self.size and gives each chunk only once  
        Repeats if there are more requests
    '''
    def __init__(self, all_inds: list[Any],  size: int, shuffle = True) -> None:
        super().__init__(all_inds, size)
        self.shuffle = shuffle            
        self.cur_pos = 0

    def get_group(self) -> list[Any]:
        end_pos = (self.cur_pos + self.size - 1) % len(self.all_inds)
        if end_pos < self.cur_pos:
            selected = [*self.all_inds[:end_pos + 1], *self.all_inds[self.cur_pos:]]
        else: 
            selected = self.all_inds[self.cur_pos:end_pos + 1]
        self.cur_pos = (end_pos + 1) % len(self.all_inds)
        return selected
    
    def init_inds(self) -> None:
        super().init_inds()
        self.cur_pos = 0
        if self.shuffle:
            rnd.shuffle(self.all_inds)
        self.population = self.get_group()

    def update(self, interactions: dict[Any, list[tuple[Any, int]]]) -> None:
        self.population = self.get_group()