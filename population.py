''' Contains classes that define different populations and the way they change 
    See base class Population
'''

from abc import abstractmethod
import json
from math import sqrt
from typing import Any, Iterable, Optional
from de import extract_dims, get_batch_pareto_layers2
from params import PARAM_IND_CHANGES_STORY, PARAM_INTS, PARAM_UNIQ_INTS, rnd, PARAM_UNIQ_INDS, PARAM_MAX_INDS, \
    param_steps, param_selection_size, param_batch_size, param_cut_features, rnd
import approx

class Selection:
    ''' The base class for selection algorithms. The selection from the given pool based on interactions '''
    def __init__(self, pool: list[Any], *, size = param_selection_size, **kwargs) -> None:
        self.size = int(size)
        self.selection = [] #first call of get_inds should initialize
        self.interactions = {}
        self.pool = pool #pool of all possible individuals
        self.pool_set = set(pool)
        self.sel_metrics = {PARAM_UNIQ_INDS: 0, PARAM_MAX_INDS: 0}
        self.seen_inds = set()
        self.sel_params = {"size": self.size, "class":self.__class__.__name__}    

    def get_selection(self, **filters) -> list[Any]:
        return self.selection
    
    def get_pool(self) -> list[Any]:
        return self.pool

    def init_selection(self) -> None:
        self.interactions = {}
        self.sel_metrics[PARAM_UNIQ_INDS] = 0
        self.sel_metrics[PARAM_MAX_INDS] = 0
        self.seen_inds = set()

    def merge_interactions(self, interactions: dict[Any, dict[Any, int]]) -> None:
        num_uniq_ints = 0
        for ind, ints in interactions.items():                        
            ind_ints = self.interactions.setdefault(ind, {})
            self.sel_metrics[PARAM_INTS] = self.sel_metrics.get(PARAM_INTS, 0) + len(ints)
            for t, outcome in ints.items():
                if t not in ind_ints:
                    num_uniq_ints += 1
                ind_ints[t] = outcome
        self.sel_metrics[PARAM_UNIQ_INTS] = self.sel_metrics.get(PARAM_UNIQ_INTS, 0) + num_uniq_ints

    ''' int_keys are keys of inner dictionary and define the order '''
    def update(self, interactions: dict[Any, dict[Any, int]], int_keys: set[Any]) -> None:
        self.merge_interactions(interactions)

    def get_for_drawing(self, role = "") -> list[dict]:
        ''' Called to draw population'''
        return [
            {"xy": self.selection, "class": dict(marker='o', s=30, c='#151fd6'), "legend": [s for s in self.selection[:20]]}
        ]

class HillClimbing(Selection):
    ''' Population that evolves with parent-child hill climbing approach '''
    def __init__(self, pool: list[Any], *,
                    mutation_strategy = "plus_minus_one", 
                    selection_strategy = "pareto_select", 
                    init_strategy = "rand_init", init_range = None, **kwargs) -> None:
        super().__init__(pool, **kwargs) 
        
        self.mutation_strategy = getattr(self, mutation_strategy)
        self.selection_strategy = getattr(self, selection_strategy)
        self.init_strategy = getattr(self, init_strategy)
        self.ind_range = (min(pool), max(pool))
        self.init_range = init_range if init_range is not None else self.ind_range
        self.sel_params.update(mutation_strategy = mutation_strategy, selection_strategy = selection_strategy,
                                    init_strategy = init_strategy)
    
    def plus_minus_one(self, ind: tuple[int,int] | int):
        if type(ind) is int:
            mutated = ind + rnd.choice([-1,1])
        else:
            mutated = tuple(c + rnd.choice([-1,1]) for c in ind)
        return mutated

    def resample(self, ind: Any):
        ''' Picks random other ind from pool '''
        for _ in range(10):
            index = rnd.randint(0, len(self.pool))
            mutated = self.pool[index]
            if mutated != ind:
                return mutated
        return ind 
    
    def pareto_select(self, i, all_ints):
        ''' Version of Pareto selection 
            If parent and child are same by performance - pick child for progress 
            Then, consider domination and preserve parent if non-dominant
        '''
        common_ints = [(co, po) for _, (co, po) in all_ints if co is not None and po is not None]
        # if all(co == po for co, po in common_ints):
        #     return self.children[i] #prefer progress on of performance change
        #prefer domination 
        ind = self.children[i] if len(common_ints) > 1 and \
                            all(co >= po for co, po in common_ints) and \
                            any(co > po for co, po in common_ints) else self.selection[i]
        # todo: what to do on non-dominance? Currently parent is picked - maybe count wins? 
        return ind
    
    def informativeness_select(self, i, all_ints):
        ''' selection based on informativeness score aggregation '''
        def informativeness_score(ints: list[int]):
            ''' counts pairs of interactions of the same outcome '''
            # return sum(1 for i in range(len(ints)) for j in range(i+1, len(ints)) if ints[i] == ints[j])
            return sum(1 for s1 in ints for s2 in ints if s1 == s2)
        common_ints = [(co, po) for _, (co, po) in all_ints if co is not None and po is not None]
        parent_informativeness = informativeness_score([o for _, o in common_ints])
        child_informativeness = informativeness_score([o for o, _ in common_ints])
        ind = self.children[i] if child_informativeness > parent_informativeness else self.selection[i] 
        return ind    
    
    def num_wins(self, i, all_ints):
        parent_score = sum([o for _, (_, o) in all_ints if o is not None])
        child_score = sum([o for _, (o, _) in all_ints if o is not None])
        ind = self.children[i] if child_score > parent_score else self.selection[i] 
        return ind
    
    def rand_init(self):
        parent_indexes = rnd.choice(len(self.pool), size = self.size, replace=False)
        self.selection = [self.pool[i] for i in parent_indexes]

    def zero_init(self):
        self.selection = [tuple(0 for _ in self.pool[0]) if type(self.pool[0]) is tuple else 0 for _ in range(self.size)]

    def range_init(self):
        self.selection = [tuple(rnd.choice(self.init_range) for _ in self.pool[0]) if type(self.pool[0]) is tuple else rnd.choice(self.init_range) for _ in range(self.size)]

    def init_selection(self) -> None:
        super().init_selection()
        self.init_strategy()
        self.children = [self.mutation_strategy(parent) for parent in self.selection]

        self.sel_metrics[PARAM_MAX_INDS] += 2 * self.size
        self.seen_inds.update(self.selection, self.children)
        self.sel_metrics[PARAM_UNIQ_INDS] = len(self.seen_inds)          

    def get_selection(self, *, is_final = False) -> list[Any]:        
        return [*self.selection, *([] if is_final else self.children)]

    def update(self, interactions: dict[Any, dict[Any, int]], int_keys: set[Any]) -> None:
        super().update(interactions, int_keys)
        parent_ints = [interactions.get(p, {}) for p in self.selection]
        child_ints = [interactions.get(c, {}) for c in self.children]   
        def get_union_outcomes(parent_dict: dict[Any, int], child_dict: dict[Any, int]):
            ket_set1 = set(parent_dict.keys())
            ket_set2 = set(child_dict.keys())
            key_set = set.union(ket_set1, ket_set2)
            return [ (k, (child_dict.get(k, None), parent_dict.get(k, None))) for k in key_set ]
        all_ints = [get_union_outcomes(p_ints, c_ints) for p_ints, c_ints in zip(parent_ints, child_ints)]
        
        selected = [ self.selection_strategy(i, ints) for i, ints in enumerate(all_ints)]
        num_changes = sum(1 for s, p in zip(selected, self.selection) if s != p)
        self.sel_metrics.setdefault(PARAM_IND_CHANGES_STORY, []).append(num_changes)
        self.selection = selected 
        self.children = [self.mutation_strategy(parent) for parent in self.selection]
                
        self.sel_metrics[PARAM_MAX_INDS] += 2 * self.size
        self.seen_inds.update(self.selection, self.children)
        self.sel_metrics[PARAM_UNIQ_INDS] = len(self.seen_inds) 

    def get_for_drawing(self, role = "") -> list[dict]:
        class_parent, class_child = (dict(marker='o', s=30, c='#151fd6'), dict(marker='o', s=10, c='#28a4c9')) if role == "cand" else (dict(marker='H', s=30, c='#159e1b'), dict(marker='H', s=10, c='#85ba6a'))
        return [
                {"xy": self.seen_inds, "bg": True}, 
                {"xy": self.selection, "class": class_parent, "legend": [f"{p}".ljust(10) + "|" + f"{c}".ljust(10) for p, c in list(zip(self.selection, self.children))[:20]]},
                {"xy": self.children, "class": class_child}
            ]

class ExploitExploreSelection(Selection):
    ''' Performs selection based on exploitation and exploration strategies and discovered facts in the self.interactions.
        Implements annealing between exploitation and exploration. 
    '''
    def __init__(self, pool: list[Any], *, min_exploit_prop = 0.2, max_exploit_prop = 0.8, max_step = param_steps, 
                    directed_explore_prop = 0, cut_features_prop = param_cut_features, **kwargs) -> None:
        super().__init__(pool, **kwargs)        
        self.min_exploit_prop = float(min_exploit_prop)
        self.max_exploit_prop = float(max_exploit_prop)
        self.max_step = int(max_step)
        self.exploit_prop = self.min_exploit_prop
        self.cut_features_prop = int(cut_features_prop)
        self.directed_explore_prop = float(directed_explore_prop)
        self.sel_params.update(min_exploit_prop = self.min_exploit_prop, max_exploit_prop = self.max_exploit_prop, 
                                directed_explore_prop = self.directed_explore_prop, max_step = self.max_step, 
                                cut_features_prop = self.cut_features_prop)
        self.t = 0        
        self.explore_interacted = True   
        self.ind_groups = {}       
        self.exploited_inds = set()      

    @abstractmethod
    def exploit(self, sample_size: int) -> set[Any]:
        pass

    # def cut_features(self, features: dict[Any, Any], size = None):
    #     ''' Preserves only size best features according to sorted order '''
    #     if size is None:
    #         size = self.cut_features_prop
    #     if len(features) > size:
    #         sorted_features = sorted(features.items(), key = lambda x: x[1])
    #         for k, _ in sorted_features[:-size]:
    #             del features[k]        

    @abstractmethod
    def update_features(self, interactions: dict[Any, dict[Any, int]], int_keys: set[Any]) -> None:
        pass 

    def update(self, interactions: dict[Any, dict[Any, int]], int_keys: set[Any]) -> None:
        ''' Applies DECA to given local ints and extract axes '''
        super().update(interactions, int_keys)
        self.update_features(interactions, int_keys)
        self.t += 1  
        self.selection = None

    def init_selection(self) -> None:
        super().init_selection()        
        self.selection = None
        # self.for_group = None
        self.t = 0        
        #for directed exploration
        self.explore_child_parent = {}       
        self.exploited_inds = set() 

    # def get_same_interactions_percent(self):
    #     ''' We would like to avoid same interactions for the following steps
    #         This method computes the percent of given group in self.for_group with already present interactions
    #         It is up to concrete population how to handle same interactions (current simulations assume uselessness of such interactions)
    #     '''
    #     if len(self.for_group) == 0:
    #         return {}
    #     scores = {ind: existing / len(self.for_group)
    #                 for ind, ind_ints in self.interactions.items()
    #                 for existing in [sum(1 for t in self.for_group if t in ind_ints)]
    #                 if existing > 0}
    #     return scores 
    
    def addDiff(self, addToInd, subFromInd, subInd, mul):
        ''' Representation specific: TODO: move to ind representation '''
        if type(addToInd) is int:
            return addToInd + subFromInd - subInd 
        if type(addToInd) is tuple and len(addToInd) > 0:
            return tuple(a + (b - c) * mul for a,b,c in zip(addToInd, subFromInd, subInd))
        assert False, f"Usupported representation of {addToInd} {type(addToInd)}"

    def addSimpleDirection(self, addToInd):
        ''' Representation specific: TODO: move to ind representation '''
        if type(addToInd) is int:
            all_dirs = [-1,1]
            direction = rnd.choice(all_dirs)
            return addToInd + direction
        if type(addToInd) is tuple and len(addToInd) > 0:
            all_dirs = [-1,1]
            num_poss_to_modify = rnd.randint(1, len(addToInd))
            poss_to_modify = set(rnd.choice(len(addToInd), size = num_poss_to_modify, replace=False))          
            return tuple((v + rnd.choice(all_dirs)) if i in poss_to_modify else v for i, v in enumerate(addToInd))
        assert False, f"Usupported representation of {addToInd} {type(addToInd)}"        
    
    def explore(self, selected: set[Any]) -> None:
        ''' Samples individuals based on interactions exploration. Contrast it to exploit selection 
            We assume that the search space is big and it is expensive to enumerate it/assign distribution per element.
            Adds exploration sample to a given selected. 
            Directed exploration, exploration towards most promising directions, between exploitation and random sampling
        '''
        #directed explore 
        directed_explore_slots = round(max(0, self.size - len(selected)) * self.directed_explore_prop)
        misses = 0
        len_with_directed_explore = len(selected) + directed_explore_slots
        if directed_explore_slots > 0 and len(selected) > 0:            
            # cur_explore_child_parent = {k:v for k, v in self.explore_child_parent.items() if k in selected}
            explore_child_parent = {}
            while len(selected) < len_with_directed_explore and misses < 10:
                selected_list = list(selected)
                selected_index = rnd.randint(0, len(selected_list))
                parent = selected_list[selected_index]
                child = None
                if parent in self.explore_child_parent: #parent was produced by directed explore and has its own parent
                    grandparent, numTimes = self.explore_child_parent[parent]
                    #TODO use representation specific diffAdd 
                    child = self.addDiff(parent, grandparent, parent, numTimes)
                    if child not in self.pool_set or child in selected:
                        child = None
                        # child = parent 
                        # del self.explore_child_parent[parent]
                if child is None: #parent does not have direction - sample simple one
                    child = self.addSimpleDirection(parent)
                    numTimes = 1
                if child not in self.pool_set or child in selected or ((not self.explore_interacted) and child in self.interactions):
                    misses += 1
                else:
                    explore_child_parent[child] = (parent, numTimes + 1)
                    selected.add(child)
                    self.ind_groups.setdefault("directed_explore", []).append(child)
                    misses = 0
            self.explore_child_parent = explore_child_parent

        #random sampling
        misses = 0 # number of times we select same
        while len(selected) < self.size and misses < 100:
            selected_index = rnd.randint(0, len(self.pool))
            selected_ind = self.pool[selected_index]
            if selected_ind in selected or ((not self.explore_interacted) and selected_ind in self.interactions):
                misses += 1
            else:
                selected.add(selected_ind)
                self.ind_groups.setdefault("explore", []).append(selected_ind)
                misses = 0

    def get_selection(self, **filters) -> list[Any]:
        if self.selection is not None and len(self.selection) > 0: # and for_group == self.for_group:
            return self.selection
        self.ind_groups = {}
        # self.for_group = for_group #adjusts sampling by avoiding repetitions of interactions - smaller chance to interruct again
        self.exploit_prop = self.min_exploit_prop + (self.t / self.max_step) * (self.max_exploit_prop - self.min_exploit_prop)
        exploit_slots = round(self.size * self.exploit_prop)
        selected_inds = self.exploit(exploit_slots)
        self.exploited_inds.update(selected_inds)
        self.explore(selected_inds)
        self.selection = list(selected_inds)
        self.sel_metrics[PARAM_MAX_INDS] += self.size
        self.seen_inds.update(self.selection)
        self.sel_metrics[PARAM_UNIQ_INDS] = len(self.seen_inds)
        return self.selection
    
    def get_for_drawing(self, role = "") -> list[dict]:
        pop = self.get_selection() #just to make sure that population is inited
        directed_explore = self.ind_groups.get("directed_explore", [])
        explore = self.ind_groups.get("explore", [])
        exploit_groups = []
        for k,v in self.ind_groups.items():
            if k.startswith("exploit_"):
                s = int(k.split("_")[-1])
                exploit_groups.append({"xy": v, "class": dict(marker='o', s=s, c='#d662e3', alpha=0.5), "legend": [f"{t}".ljust(10) for t in v[:20]]})
        exploit = self.ind_groups.get("exploit", [])
        return [
                {"xy": self.exploited_inds, "bg": True}, 
                {"xy": directed_explore, "class": dict(marker='o', s=10, c='#62c3e3', alpha=0.4), "legend": [f"{t}".ljust(10) for t in directed_explore[:20]]},
                {"xy": explore, "class": dict(marker='o', s=10, c='#73e362', alpha=0.4), "legend": [f"{t}".ljust(10) for t in explore[:20]]},
                *exploit_groups,
                {"xy": exploit, "class": dict(marker='o', s=30, c='#d662e3', alpha=0.5), "legend": [f"{t}".ljust(10) for t in exploit[:20]]},
            ]
    
    def get_discriminating_set(self, selected_inds: set[Any]) -> set[Any]:
        ''' Returns candidates that could distinguish selected individuals based on seen interactions '''
        selected_inds_list = list(selected_inds)
        test_pairs = [ (selected_inds_list[i], selected_inds_list[j]) for i in range(len(selected_inds_list)) for j in range(i + 1, len(selected_inds_list)) ]
        discriminating_lists = {}
        for (tid1, tid2) in test_pairs:
            test1 = self.interactions.get(tid1, {})
            test2 = self.interactions.get(tid2, {})
            common_keys = set.intersection(set(test1.keys()), set(test2.keys()))
            for cid in common_keys:
                if test1[cid] == 0 and test2[cid] == 1:
                    discriminating_lists.setdefault(cid, set()).add((tid1, tid2))
                if test1[cid] == 1 and test2[cid] == 0:
                    discriminating_lists.setdefault(cid, set()).add((tid2, tid1))
        discriminating_set = set()
        while len(discriminating_lists) > 0: 
            max_cid = max(discriminating_lists.keys(), key = lambda x: len(discriminating_lists[x]))
            max_cid_set = discriminating_lists[max_cid]
            discriminating_set.add(max_cid)
            del discriminating_lists[max_cid]
            to_delete = set()
            for cid, cid_set in discriminating_lists.items():
                cid_set -= max_cid_set
                if len(cid_set) == 0:
                    to_delete.add(cid)
            for cid in to_delete:
                del discriminating_lists[cid]
        return discriminating_set
        
class InteractionFeatureOrder(ExploitExploreSelection):
    ''' Sampling of search space based on grouping by interactions features and ordering of the groups 
        Also we called this sampling strategies
    '''
    def __init__(self, pool: list[Any], *, strategy = None, **kwargs) -> None:
        super().__init__(pool, **kwargs)
        self.explore_interacted = False
        self.features = {} #contains set of features per interacted individual 
        default_strategy = ["kn-rel", "kn", "nond", "sd", "dom"]
        strategy = default_strategy if strategy is None else json.loads(strategy) if type(strategy) is str else strategy
        self.strategy = strategy
        self.sel_params.update(strategy = self.strategy)

    def init_selection(self) -> None:
        super().init_selection()
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

    def update_features(self, interactions: dict[Any, dict[Any, int]], int_keys: set[Any]) -> None:
        ''' Also updates individual features from interactions '''    
        other_inds = set(interactions.keys())
        for ind in interactions.keys():
            ind_features = self.features.setdefault(ind, {})
            ind_features["one"] = 1
            ind_features["-one"] = -1
            ind_features["num_ints"] = ind_features.get("num_ints", 0) + len(int_keys)
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
    
    def exploit(self, sample_size) -> set[Any]:
        if len(self.features) == 0:
            return set()
        key_selector = lambda scores: tuple(scores[k] for k in self.strategy)
        cur_ind_scores = [ (ind, key_selector(ind_features))
                                for ind, ind_features in self.features.items() ]        
        cur_ind_scores.sort(key=lambda x: x[1], reverse=True)
        selected = set([f for f, _ in cur_ind_scores[:min(sample_size, len(cur_ind_scores))]])
        self.ind_groups.setdefault(f"exploit", []).extend(selected)
        return selected

class DECASelection(ExploitExploreSelection):
    ''' Implementation that is based on DECA idea 
        Param approx_strategy defines how to compute missing values None from global interactions
        approx_strategy: zero_approx_strategy, one_approx_strategy, majority_approx_strategy, candidate_group_approx_strategy
        cand_sel_strategy defines what candidates to select for new dimension extraction: 
            local - only last batch, global - any candidate with 2 or more tests in the last batch 
            discriminative - those candidates that distinguished axes last time (fallback to local)
    '''
    def __init__(self, pool: list[Any], *, cand_sel_strategy = "local_cand_sel_strategy", 
                                           test_sel_strategy = "local_test_sel_strategy", 
                                           approx_strategy = "majority_approx_strategy", **kwargs) -> None:
        super().__init__(pool, **kwargs)
        self.sel_params.update(cand_sel_strategy = cand_sel_strategy, test_sel_strategy = test_sel_strategy, approx_strategy = approx_strategy)
        self.approx_strategy = getattr(approx, approx_strategy)
        self.cand_sel_strategy = getattr(self, cand_sel_strategy)
        self.test_sel_strategy = getattr(self, test_sel_strategy)
        self.origin = []
        # self.discarded_exploited_tests = [] #ordered from least to most recently discarded
        # self.prev_exploited = set()

    def get_approx_counts(self, interactions):
        approx_counts = [0 for _ in range(len(interactions))]
        for test_id, test in enumerate(interactions):
            for outcome in test:
                if outcome is None:
                    approx_counts[test_id] += 1 
        return approx_counts
    
    def local_test_sel_strategy(self, last_btach_tests:Iterable[Any], last_batch_candidates: set[Any]):
        return list(last_btach_tests)
    
    # NOTE: the following sel_strategy does not work!
    #       When we pick previous discarded test in new batch, almost all values will be unknown and deduced in most conservative way
    #       It does not matter if we switch from discarded to ones that were not selected in exploit function but have a potential
    #       Why this attempt exists: the idea that on new exploitation positon, it could make sense to take previously not selected or discarded tests 
    #       This tests could be better for exploitation than one taken frorm exploration subset. 
    #       This is true for classic annealing schema. But to check relativee performance we need to add previously discarded/not selected in new selection
    #            instead of test from exploration subset placed on axis. 
    #       Decision tree:
    #             New exploitation place appears, what test to select?
    #             1. Just take the next on the axis according to DE procedure
    #             2. Last time when we discovered axes we probably did not sample all best tests on ends. Should we try them instead for a new position?
    #             3. (Bad choice) Last time we discarded some tests, can they be better than one introduced explorativelly? 
    #                Bad choice because the discarded individuals were proven to be worse. 
    #       Since we have only one new position, we need to pick at random between 1 and 2.
    #       For choice 2 we need to track all individuals "similar" to selected at current step. - for later try
    #       As alternative to annealing we also look onto schema similar to PPHC - half of gen to explore and half to exploit
    # def history_test_sel_strategy(self, last_btach_tests:Iterable[Any], last_batch_candidates: set[Any]):
    #     ''' helpful when new exploiting position appears '''
    #     if len(self.discarded_exploited_tests) == 0:
    #         return list(last_btach_tests)
    #     to_retry = self.discarded_exploited_tests[-1:] #take last discarded
    #     return list(set([*to_retry, *last_btach_tests])) #remove duplicates
    
    def local_cand_sel_strategy(self, last_btach_tests:Iterable[Any], last_batch_candidates: set[Any]):
        return list(last_batch_candidates)
    
    def global_cand_sel_strategy(self, last_btach_tests:Iterable[Any], last_batch_candidates: set[Any]):
        candidate_id_counts = {}
        for test_id in last_btach_tests:
            for candidate_id in self.interactions[test_id].keys():
                candidate_id_counts[candidate_id] = candidate_id_counts.get(candidate_id, 0) + 1

        candidate_ids = [candidate_id for candidate_id, count in candidate_id_counts.items() if count > 1]
        return candidate_ids

    def discr_cand_sel_strategy(self, last_btach_tests:Iterable[Any], last_batch_candidates: set[Any]):
        discriminating_candidates = self.get_discriminating_set(set(last_btach_tests))
        return list(discriminating_candidates) # list(set.union(discriminating_candidates, last_batch_candidates))
    
    def init_selection(self):
        super().init_selection()
        self.axes = []
        self.origin = []
        # self.discarded_exploited_tests = []
        # self.prev_exploited = set()

    def update_features(self, interactions: dict[Any, dict[Any, int]], int_keys: set[Any]) -> None:
        ''' Applies DE to a given local or global ints and extract axes '''    
        test_ids = self.test_sel_strategy(interactions.keys(), int_keys)
        candidate_ids = self.cand_sel_strategy(test_ids, int_keys)
        tests = [[self.interactions[test_id].get(candidate_id, None) for candidate_id in candidate_ids] for test_id in  test_ids]
        approx_counts = {test_id: c for test_id, c in zip(test_ids, self.get_approx_counts(tests))}
        self.approx_strategy(tests) #fillin Nones
        
        dims, origin, _, _ = extract_dims(tests)  

        self.origin = [test_ids[i] for i in origin]       
        self.origin.sort(key = lambda x: len(self.interactions[x]), reverse=True)

        self.axes = []
        for dim in dims:
            axes_tests = [(test_ids[i], (len(dim) - point_id - 1, -len(self.interactions[test_ids[i]]), approx_counts.get(test_ids[i], 0))) for point_id, group in enumerate(dim) for i in group]
            axes_tests.sort(key = lambda x: x[1], reverse=True)
            self.axes.append(axes_tests)

        # # fail sets of objectives (ends of axes)
        # cf_sets = [{candidate_ids[i] for i, o in enumerate(tests[dim[-1][0]]) if o == 1} for dim in dims]
        # self.discriminating_candidates = set.union(*cf_sets) - set.intersection(*cf_sets)

    def exploit(self, sample_size) -> set[Any]:

        selected = set()

        axe_id = 0 
        while len(selected) < sample_size and len(self.axes) > 0:
            axe = self.axes[axe_id]
            ind, _ = axe.pop()
            selected.add(ind)
            if len(axe) == 0:
                self.axes.pop(axe_id)
                if len(self.axes) == 0:
                    break
            else:
                axe_id += 1
            axe_id %= len(self.axes)

        for ind in self.origin:
            if len(selected) >= sample_size:
                break
            selected.add(ind)
            
        # self.discriminating_candidates = self.get_discriminating_set(selected)
        self.ind_groups.setdefault(f"exploit", []).extend(selected)
        # removed = self.prev_exploited - selected
        # self.discarded_exploited_tests.extend(removed)
        # self.prev_exploited = set(selected)
        return selected

# class DECARanksSample(ExploitExploreSelection):
#     ''' Applies DECA on each game step to form local axes.
#         Position in such local space gives rank for participated tests.
#         The global ranks than updated, shifting conclusion based on one step performance.
#         Update happens with geometric mean between globally recorded rank and concluded locally on one step.
#         Sampling happens according to ranks. Only 2 *size best ranks are preserved.
#     '''
#     def __init__(self, all_inds: list[Any], 
#                 span_pos_penalty = 5, max_inds_per_obj = 10, obj_bonus = 100, span_penalty = 0.01,
#                 **kwargs) -> None:
#         super().__init__(all_inds, **kwargs)
#         self.obj_bonus = float(obj_bonus)
#         self.span_penalty = float(span_penalty)
#         self.span_pos_penalty = float(span_pos_penalty)
#         self.max_inds_per_obj = int(max_inds_per_obj)
#         # self.ints_bonus = float(ints_bonus)
#         # self.pop_params.update(obj_bonus = self.obj_bonus, span_penalty = self.span_penalty, ints_bonus = self.ints_bonus)
#         self.sel_params.update(obj_bonus = self.obj_bonus, 
#                                 span_pos_penalty = self.span_pos_penalty, max_inds_per_obj = self.max_inds_per_obj,
#                                 span_penalty = self.span_penalty)

#         # self.ind_objs = {} # per each ind, stores id of the objective and the weight on the axis (0 - best)
#         # self.obj_inds = {} # per each obj, stores list of lists of inds, starting from end of axis, back to origin 

#     def init_selection(self) -> None:
#         super().init_selection() 
#         self.ranks = {} 
#         # self.spanned = set() #could potentially become the end of axes

#     def update_features(self, interactions: dict[Any, list[tuple[Any, int]]], int_keys: set[Any]) -> None:
#         ''' Applies DECA to a given local ints and extract axes '''        
#         tests = [[o for _, o in ind_ints] for _, ind_ints in interactions.items() ]
#         inds = [ind for ind, _ in interactions.items() ]
#         dims, _, spanned, _ = extract_dims(tests)
#         present_axes = set(self.ranks.keys())
#         for dim in dims:
#             for point_id, group in enumerate(dim):
#                 w = (point_id + 1) / len(dim)
#                 if len(dim) == (point_id + 1):
#                     w += self.obj_bonus
#                 for i in group:
#                     ind = inds[i]
#                     if ind in present_axes:
#                         self.ranks[ind] = sqrt(w * self.ranks[ind])
#                     else:
#                         self.ranks[ind] = w

#         for i, _ in spanned.items():
#             ind = inds[i]
#             rank = self.span_penalty
#             if ind in self.ranks:
#                 self.ranks[ind] = sqrt(self.ranks[ind] * rank)
#             else:
#                 self.ranks[ind] = rank   

#         self.cut_features(self.ranks)              

#     def exploit(self, sample_size) -> set[Any]:
        
#         if len(self.ranks) == 0:
#             return set()
#         # if len(self.for_group) > 0:
#         #     sample_size = self.size

#         sum_weights = sum(self.ranks.values())
#         ind_list, p = zip(*[ (ind, w / sum_weights) for ind, w in self.ranks.items()])
#         ind_ids = rnd.choice(len(ind_list), size = min(sample_size, len(ind_list)), p = list(p), replace=False)
#         selected = set(ind_list[i] for i in ind_ids)
#         self.ind_groups.setdefault(f"exploit", []).extend(selected)
#         return selected

# class UneqGroupsSample(ExploitExploreSelection):
#     ''' Finds unequality groups and samples them supposing that they are objectivees
#     '''
#     def __init__(self, all_inds: list[Any], **kwargs) -> None:
#         super().__init__(all_inds, **kwargs)
#         # self.ints_bonus = float(ints_bonus)
#         # self.pop_params.update(obj_bonus = self.obj_bonus, span_penalty = self.span_penalty, ints_bonus = self.ints_bonus)
#         # self.pop_params.update()

#         # self.ind_objs = {} # per each ind, stores id of the objective and the weight on the axis (0 - best)
#         # self.obj_inds = {} # per each obj, stores list of lists of inds, starting from end of axis, back to origin 

#     def init_selection(self) -> None:
#         super().init_selection() 
#         self.pair_diffs = {} # for a pair of tests stores candidate ids by which outcomes are different
#         self.uneq_sets = {} #for tried ind, defines detected set of uneq inds by objective (with dim_extraction)        
#         self.main_set = set() 
#         self.all_eq_sets = []
#         # self.num_wins = {} #for each ind, number of 1 in interactions
#         # self.last_num_wins = {}
#         self.win_rate = {}
#         self.main_axes = []
#         self.spanned_axes = []

#     def update_features(self, interactions: dict[Any, list[tuple[Any, int]]], int_keys: set[Any]) -> None:
#         ''' Extract new uneq and trivials '''
#         all_local_ints = [ (tid, [o for _, o in ind_ints]) for tid, ind_ints in interactions.items() ]
#         candidate_ids = [cid for cid, _ in next(iter(interactions.values()))]
#         num_wins = {}
#         for i in range(len(all_local_ints)):
#             ind1, ints1 = all_local_ints[i]
#             if ind1 not in self.main_set:
#                 self.main_set.add(ind1)
#             wins, count = self.win_rate.get(ind1, (0, 0))
#             num_wins[ind1] = sum(ints1)
#             self.win_rate[ind1] = (wins + num_wins[ind1], count + len(ints1))
#             # self.last_num_wins[ind1] = self.num_wins[ind1]
#             for j in range(i + 1, len(all_local_ints)):
#                 ind2, ints2 = all_local_ints[j]
#                 if ind1 in self.uneq_sets and ind2 in self.uneq_sets[ind1]:
#                     continue
#                 pair = (ind1, ind2) if ind1 < ind2 else (ind2, ind1)
#                 cids = [candidate_ids[k] for k, (o1, o2) in enumerate(zip(ints1, ints2)) if o1 != o2]
#                 if len(cids) > 0:
#                     self.pair_diffs.setdefault(pair, set()).update(cids)
#                     if any(self.interactions[pair[0]][cid] > self.interactions[pair[1]][cid] for cid in self.pair_diffs[pair]) and \
#                         any(self.interactions[pair[1]][cid] > self.interactions[pair[0]][cid] for cid in self.pair_diffs[pair]):
#                         self.uneq_sets.setdefault(pair[0], set()).add(pair[1])
#                         self.uneq_sets.setdefault(pair[1], set()).add(pair[0])  
#                         del self.pair_diffs[pair]
#         eq_sets = [[]]        
#         def ind_score(ind):
#             wins, count = self.win_rate.get(ind, (0, 0))
#             rate = wins / (1 if count == 0 else count)
#             return num_wins[ind], rate
#         for ind, _ in sorted(all_local_ints, key=lambda x: ind_score(x[0]), reverse=True):
#             uneq_set = self.uneq_sets.get(ind, set())
#             new_eq_sets = []
#             # eq_sets_to_possibly_add = set()

#             compatible_sets = []
#             incompatible_sets = [] 
#             for eq_set in eq_sets:
#                 if any(el in uneq_set for el in eq_set):
#                     incompatible_sets.append(eq_set)
#                 else:
#                     compatible_sets.append(eq_set)

#             other_candidiates_ordered = []
#             if len(compatible_sets) == 0:
#                 # are there other sets with incompatible elements which could be removed and form a compatible set?
#                 # incompatible element could be removed if there is at least one other incompatible set with it. 
#                 other_candidiates = []
#                 uneq_el_present_count = { el: len([eq_set2 for eq_set2 in incompatible_sets if el in eq_set2]) for el in uneq_set }
#                 for eq_set_id, eq_set in enumerate(incompatible_sets):
#                     present_uneq_els = [ el for el in eq_set if el in uneq_set ]
#                     present_counts = [ uneq_el_present_count[el] for el in present_uneq_els ]
#                     present_uneq_els_set = set(present_uneq_els)
#                     other_candidiates_ordered.append((eq_set, present_uneq_els_set))
#                     if all(c > 1 for c in present_counts):
#                         other_candidiates.append((eq_set_id, eq_set, present_uneq_els_set))
#                 other_candidiates.sort(key = lambda x: len(x[1]) - len(x[2]), reverse=True)
#                 if len(other_candidiates) > 0:
#                     eq_set_id, eq_set, present_uneq_els_set = other_candidiates[0]
#                     del incompatible_sets[eq_set_id]
#                     new_eq_set = [ el for el in eq_set if el not in present_uneq_els_set ]
#                     compatible_sets.append(new_eq_set)                    

#             if len(compatible_sets) == 0:
#                 #take only one such set
#                 other_candidiates_ordered.sort(key = lambda x: len(x[0]) - len(x[1]), reverse=True)
#                 eq_set, present_uneq_els_set = other_candidiates_ordered[0]
#                 new_eq_sets.append(eq_set) #   
#                 new_eq_set = [ el for el in eq_set if el not in present_uneq_els_set ]
#                 new_eq_set.append(ind)
#                 new_eq_sets.append(new_eq_set)
#             else:
#                 for eq_set in compatible_sets:
#                     eq_set.append(ind)
#                     new_eq_sets.append(eq_set)
#                 new_eq_sets.extend(incompatible_sets)
#             # for eq_set in eq_sets_to_possibly_add:
#             #     if any(eq_set.issubset(eq_set1) for eq_set1 in new_eq_sets):
#             #         continue
#             #     new_eq_sets.add(eq_set)            
#             eq_sets = new_eq_sets
#         # searching for axes, we remove all eq_sets that are subsets of any other eq_set

#         ind_num_sets = {}
#         for eq_set in eq_sets:
#             for ind in eq_set:
#                 ind_num_sets[ind] = ind_num_sets.get(ind, 0) + 1

#         main_axes = [[ind for ind in eq_set if ind_num_sets[ind] == 1] for eq_set in eq_sets ]
#         main_eq_sets = [ eq_set for ax, eq_set in zip(main_axes, eq_sets) if len(ax) > 0 ]
#         ind_num_sets = {}
#         for eq_set in main_eq_sets:
#             for ind in eq_set:
#                 ind_num_sets[ind] = ind_num_sets.get(ind, 0) + 1        
#         # spanned_axes = [[ind for ind in eq_set if ind_num_sets[ind] > 1] for eq_set in main_eq_sets ]   
#         self.main_axes = main_axes
#         # self.spanned_axes = spanned_axes
#         self.spanned_axes = []


#         # eq_sets_list = list(eq_sets)
#         # eq_sets_list.sort(key = lambda x: len(x), reverse=True)
#         # axes = []
#         # for eq_set in eq_sets_list:
#         #     ax_candidate = eq_set
#         #     for ax in axes:
#         #         ax_candidate -= ax
#         #     if len(ax_candidate) > 0:
#         #         axes.append(eq_set)
#         # self.all_eq_sets = axes

#         # eq_sets = set()
#         # left_inds = set()
#         # for ind in self.main_set:
#         #     ind_eq_set = frozenset(self.main_set - self.uneq_sets.get(ind, set()))
#         #     eq_sets.add(ind_eq_set)    
#         #     left_inds.add(ind)
#         # axes = []
#         # while len(left_inds) > 0:
#         #     ind, ax = max(((ind, frozenset.intersection(*[eq_set for eq_set in eq_sets if ind in eq_set])) for ind in left_inds), key=lambda x: len(x[1]))
#         #     axes.append(ax)
#         #     left_inds -= ax
#         #     eq_sets = set(new_eq_set for eq_set in eq_sets for new_eq_set in [eq_set - ax] if len(new_eq_set) > 0)

#         # self.all_eq_sets = axes 
            
        

#         # new_inds = [ind for ind, _ in all_local_ints if ind not in self.main_set]
#         # old_inds = [ind for ind, _ in all_local_ints if ind in self.main_set]
#         # for ind1 in old_inds:
#         #     uneq_set = uneq_sets.get(ind1, set())
#         #     ind1_eq_sets = [(i, eq_set) for i, eq_set in enumerate(self.all_eq_sets) if ind1 in eq_set]
#         #     for ind2 in uneq_set:
#         #         shared_eq_set = [(i, eq_set) for (i, eq_set) in ind1_eq_sets if ind2 in eq_set]
#         #         if len(shared_eq_set) == 0:
#         #             if ind2 in self.main_set:
#         #                 uneq_sets[ind2].discard(ind1)
#         #             continue 
#         #         elif len(shared_eq_set) == 1:
#         #             i, eq_set = shared_eq_set[0]
#         #             # eq_set1 = set(eq_set)
#         #             eq_set2 = set(eq_set)
#         #             eq_set.discard(ind2)
#         #             eq_set2.discard(ind1)
#         #             # self.all_eq_sets[i] = eq_set1
#         #             self.all_eq_sets.append(eq_set2)
#         #             uneq_sets[ind2].discard(ind1)
#         #         else:
#         #             inds = [ind2, ind1]
#         #             for j, (i, eq_set) in enumerate(shared_eq_set):
#         #                 ind = inds[j % 2]
#         #                 eq_set.discard(ind)           
#         #             uneq_sets[ind2].discard(ind1) 

#         # new_ind_possible_sets = {} 
#         # for ind1 in new_inds:
#         #     uneq_set = uneq_sets.get(ind1, set())
#         #     old_uneq_set = set()
#         #     new_uneq_set = set()
#         #     for ind2 in uneq_set:
#         #         if ind2 in self.main_set:
#         #             old_uneq_set.add(ind2)
#         #         else:
#         #             new_uneq_set.add(ind2)                
#         #     for eq_set in self.all_eq_sets:
#         #         if any(ind2 in eq_set for ind2 in old_uneq_set):
#         #             continue
#         #         new_ind_possible_sets.setdefault(ind1, []).append(eq_set)
#         #     uneq_sets[ind1] = new_uneq_set
#         # for ind1 in new_inds:
#         #     if ind1 not in new_ind_possible_sets:
#         #         self.all_eq_sets.append({ind1})
#         #     for ind2 in uneq_sets[ind1]:
#         #         self.all_eq_sets.append({ind1, ind2})
#         #     self.main_set.add(ind1)
#         pass #for breakpoint
        
#     def exploit(self, sample_size) -> set[Any]:
#         main_axes = self.main_axes
#         spanned_axes = self.spanned_axes
#         sample_size = max(sample_size, len(main_axes))
#         # 0 if not spanned and 1 if spanned
#         # ind_num_axes = {}
#         # for ax in axes:
#         #     for ind in ax:
#         #         ind_num_axes[ind] = ind_num_axes.get(ind, 0) + 1
#         # # we split axes int many by spanned criteria of ind_rank
#         # main_axes = [[ind for ind in ax if ind_num_axes[ind] == 1] for ax in axes]
#         # spanned_axes = [[ind for ind in ax if ind_num_axes[ind] > 1] for ax in axes]
#         selected = set()
#         pos_id = 0
#         has_axes = True
#         while len(selected) < sample_size and has_axes:
#             has_axes = False
#             for ax in main_axes:
#                 if pos_id < len(ax):
#                     ind = ax[pos_id]
#                     selected.add(ind)
#                     has_axes = True
#                     if len(selected) == sample_size:
#                         break
#             if len(selected) == sample_size:
#                 break
#             for ax in spanned_axes:
#                 if pos_id < len(ax):
#                     ind = ax[pos_id]
#                     selected.add(ind)
#                     has_axes = True
#                     if len(selected) == sample_size:
#                         break
#             pos_id += 1
#         self.ind_groups.setdefault(f"exploit", []).extend(selected)
#         return selected
    
    # def get_for_drawing(self, role = "") -> list[dict]:
    #     pop = self.get_inds()
    #     return [
    #             {"xy": pop, "class": dict(marker='o', s=30, c='#151fd6'), "legend": [f"{t}".ljust(10) for t in pop[:20]]},
    #         ]    

class ParetoLayersSelection(ExploitExploreSelection):
    ''' Sampling is based on pareto ranks (see LAPCA) and corresponding archive '''
    def __init__(self, pool: list[Any], *, max_layers = 10, **kwargs) -> None:
        super().__init__(pool, **kwargs)
        self.max_layers = int(max_layers)
        self.sel_params.update(max_layers = self.max_layers)

    def init_selection(self) -> None:
        super().init_selection() 
        self.ind_order = []
        self.discriminating_candidates = set()
        self.candidate_ids = []

    def update_features(self, interactions: dict[Any, dict[Any, int]], int_keys: set[Any]) -> None:
        ''' Split into pareto layers '''        

        candidate_ids = list(set.union(self.discriminating_candidates, int_keys))
        self.candidate_ids = candidate_ids

        test_ids = list(interactions.keys())
        tests = [[self.interactions[test_id].get(candidate_id, None) for candidate_id in candidate_ids] for test_id in test_ids ]

        layers = get_batch_pareto_layers2(tests, max_layers=self.max_layers)
        
        self.ind_order = []
        for layer in layers:
            tests = [(test_ids[i], sum(o for o in tests[i] if o is not None)) for i in layer]
            tests.sort(key = lambda x: x[1], reverse=True)
            self.ind_order.extend([t for t, _ in tests])


    def exploit(self, sample_size) -> set[Any]:    

        selected_tests = self.ind_order[:sample_size]
        selected = set(selected_tests)

        self.discriminating_candidates = self.get_discriminating_set(selected)
        # for i in range(len(selected_tests)):
        #     test_id1 = selected_tests[i]
        #     test1 = [self.interactions[test_id1].get(cid, None) for cid in self.candidate_ids]
        #     for j in range(i + 1, len(selected_tests)):
        #         test_id2 = selected_tests[j]
        #         test2 = [self.interactions[test_id2].get(cid, None) for cid in self.candidate_ids]
        #         self.discriminating_candidates.update(self.candidate_ids[i] for i, (o1, o2) in enumerate(zip(test1, test2)) if o1 is not None and o2 is not None and o1 != o2)
        
        return selected
    
# class CosaBasedSample(ExploitExploreSelection):
#     ''' Sampling based on COSA archive and way to figure out antichain '''
#     def __init__(self, all_inds: list[Any], archive_bonus = 100, **kwargs) -> None:
#         super().__init__(all_inds, **kwargs)
#         self.archive_bonus = float(archive_bonus)
#         self.sel_params.update(archive_bonus = self.archive_bonus)

#     def init_selection(self) -> None:
#         super().init_selection() 
#         self.ranks = {}

#     def update_features(self, interactions: dict[Any, list[tuple[Any, int]]], int_keys: set[Any]) -> None:
#         ''' Applies DECA to given local ints and extract axes '''        
#         tests = [[o for _, o in ind_ints] for _, ind_ints in interactions.items() ]
#         inds = [ind for ind, _ in interactions.items() ]

#         archive = cosa_extract_archive(tests)

#         for ind_id in self.ranks.keys():
#             self.ranks[ind_id] = self.ranks[ind_id] * 0.9

#         for i in archive:
#             ind = inds[i]
#             if ind in self.ranks:
#                 self.ranks[ind] = sqrt(self.archive_bonus * self.ranks[ind])                
#             else:
#                 self.ranks[ind] = self.archive_bonus
                                 
#         self.cut_features(self.ranks) # dropping low score

#     def exploit(self, sample_size) -> set[Any]:
        
#         if len(self.ranks) == 0:
#             return set()
#         # if len(self.for_group) > 0:
#         #     sample_size = self.size

#         sum_weights = sum(self.ranks.values())
#         ind_list, p = zip(*[ (ind, w / sum_weights) for ind, w in self.ranks.items()])
#         ind_ids = rnd.choice(len(ind_list), size = min(sample_size, len(ind_list)), p = list(p), replace=False)
#         selected = set(ind_list[i] for i in ind_ids)
#         return selected    

# class ACOPopulation(ExploitExploreSelection):
#     ''' Population that changes by principle of pheromone distribution 
#         We can think of ants as their pheromons - the population is the values of pheromones for each of test 
#         At same time, we still have desirability criteria given by interaction matrix. 
#         Number of ants is given by popsize. Each of them makes decision what test to pick based on pheromone and desirability 
#         No two ants can pick same test. 
#         The tests are sampled by such ants but pheremones undergo evolution.
#         Note, self.selection is still used for sampled test 
#         self.pheromones is what evolves   
#         Pheromone range [1, inf). It magnifies desirability of the test 
#         It goes up when the selected test increases the desirability in the interaction step
#         And gradually goes down to 1. 
#     '''
#     def __init__(self, all_inds: list[Any], size: int,
#                     pheromone_decay = 0.5, pheromone_inc = 10, dom_bonus = 1, span_penalty = 1, **kwargs) -> None:
#         super().__init__(all_inds, size, **kwargs)
#         self.sel_params.update(pheromone_decay = pheromone_decay, dom_bonus = dom_bonus, span_penalty = span_penalty, pheromone_inc = pheromone_inc)
#         self.pheromone_decay = float(pheromone_decay)
#         self.dom_bonus = float(dom_bonus)
#         self.span_penalty = float(span_penalty)
#         self.pheromone_inc = float(pheromone_inc)

#     def init_selection(self) -> None:
#         super().init_selection()
#         self.pheromones = {} #{ind: 1 for ind in self.all_inds} #each ind gets a bit of pheromone 
#         self.desirability = {}        

#     def update_features(self, interactions: dict[Any, list[tuple[Any, int]]], int_keys: set[Any]) -> None:
#         #step 1 - compute all common interactions between all pair of tests 
#         ind_ids = list(interactions.keys())
#         common_ints = {}
#         for i1 in range(len(ind_ids)):
#             ind1 = ind_ids[i1]
#             ind1_ints = self.interactions[ind1]
#             ind1_int_keys = set(ind1_ints.keys())
#             for i2 in range(i1 + 1, len(ind_ids)):
#                 ind2 = ind_ids[i2]
#                 ind2_ints = self.interactions[ind2]
#                 ind2_int_keys = set(ind2_ints.keys())
#                 common_ids = set.intersection(ind1_int_keys, ind2_int_keys)
#                 if len(common_ids) > 1:
#                     d1 = common_ints.setdefault(ind1, {})
#                     d1[ind2] = [(ind1_ints[tid], ind2_ints[tid]) for tid in common_ids] 
#                     d2 = common_ints.setdefault(ind2, {})
#                     d2[ind1] = [(ind2_ints[tid], ind1_ints[tid]) for tid in common_ids]         
#         dominated = {ind:[ind1 for ind1, cints in ind_ints.items() 
#                         if all(io >= i1o for io, i1o in cints) and any(io > i1o for io, i1o in cints) ] 
#                         for ind, ind_ints in common_ints.items()}
#         def non_dominated_pairs(inds: list[Any]):
#             result = []
#             for i1 in range(len(inds)):
#                 ind1 = inds[i1]
#                 for i2 in range(i1 + 1, len(inds)):
#                     ind2 = inds[i2]
#                     ints = common_ints.get(ind1, {}).get(ind2, [])
#                     if len(ints) > 1 and any(o1 > o2 for o1, o2 in ints) and any(o2 > o1 for o1, o2 in ints):
#                         result.extend([ind1, ind2])
#             return result 
#         dominated_non_dominant = {ind:non_dominated_pairs(ind_dom) for ind, ind_dom in dominated.items()}
#         spanned = set() 
#         for ind, inds in dominated_non_dominant.items():
#             if len(inds) > 1:
#                 combined_ints = {} 
#                 for ind1 in inds:
#                     for ind2, o in self.interactions[ind1].items():
#                         combined_ints[ind2] = max(combined_ints.get(ind2, 0), o)
#                 if all(o == combined_ints[ind2] for ind2, o in self.interactions[ind].items() if ind2 in combined_ints):
#                     spanned.add(ind)        

#         prev_desirability = self.desirability # {**self.desirability}
#         self.desirability = {}
#         for ind in ind_ids:
#             if len(dominated.get(ind, [])) > 0:
#                 w = self.span_penalty if ind in spanned else (self.dom_bonus * len(dominated.get(ind, [])))
#                 self.desirability[ind] = w

#         # self.cut_features(self.desirability)      

#         for ind in self.pheromones.keys():
#             self.pheromones[ind] = 1 + (self.pheromones[ind] - 1) * self.pheromone_decay

#         common_des = {ind: (self.desirability[ind], prev_desirability[ind]) for ind in self.desirability.keys() if ind in prev_desirability}
#         for ind, (d1, d2) in common_des.items():
#             if d1 > d2:
#                 self.pheromones[ind] = self.pheromones.get(ind, 1) + self.pheromone_inc
        
#     def exploit(self, size) -> set[Any]:
#         ''' Sample with pheromones and desirability rule 
#             Ants book works with random proportional transition rule 
#             Here we ignore alpha and beta heuristic parameters (= 1)
#         '''        
#         #normalize scores from [min, max] -> [0, max2] - spanned point is on level of origin
#         if len(self.desirability) == 0:
#             return set()        
        
#         size_best = self.cut_features_prop
#         best_desirability_list = sorted(self.desirability.items(), key = lambda x: x[1], reverse=True)
#         best_desirability = {k: v for k, v in best_desirability_list[:size_best]}

#         weights = {k: v * self.pheromones.get(k, 1) for k, v in best_desirability.items()}
#         sum_weight = sum(weights.values())     

#         kprobs = [(k, w / sum_weight) for k, w in weights.items()]

#         inds, probs = zip(*kprobs)
#         selected_indexes = rnd.choice(len(inds), size = min(len(inds), size), replace = False, p = probs)
#         selected = set(inds[i] for i in selected_indexes)
#         return selected


class RandSelection(Selection):
    ''' Samples the individuals from game as provide it as result '''
    def __init__(self, pool, **kwargs) -> None:
        super().__init__(pool, **kwargs)
        self.selection = None
        self.sel_metrics[PARAM_MAX_INDS] = self.size
        self.sel_metrics[PARAM_UNIQ_INDS] = self.size

    def init_selection(self) -> None:
        super().init_selection()
        ids = rnd.choice(len(self.pool), self.size, replace=False)
        self.selection = [self.pool[i] for i in ids]

class OneTimeSequential(Selection):
    ''' Solits all_inds onto self.size and gives each chunk only once  
        Repeats if there are more requests
    '''
    def __init__(self, pool: list[Any], *, size = param_batch_size, shuffle = 1, **kwargs) -> None:
        super().__init__(pool, size = size, **kwargs)
        self.shuffle = int(shuffle) != 0
        self.cur_pos = 0
        self.sel_params.update(shuffle = self.shuffle)

    def get_group(self) -> list[Any]:
        end_pos = (self.cur_pos + self.size - 1) % len(self.pool)
        if end_pos < self.cur_pos:
            selected = [*self.pool[:end_pos + 1], *self.pool[self.cur_pos:]]
        else: 
            selected = self.pool[self.cur_pos:end_pos + 1]
        self.cur_pos = (end_pos + 1) % len(self.pool)
        return selected
    
    def init_selection(self) -> None:
        super().init_selection()
        self.cur_pos = 0
        if self.shuffle:
            rnd.shuffle(self.pool)
        self.selection = self.get_group()

    def update(self, interactions: dict[Any, list[tuple[Any, int]]], int_keys: set[Any]) -> None:
        self.selection = self.get_group()

    def get_selection(self, only_children = False, **filters) -> list[Any]:
        if only_children:
            return []    
        return self.selection        
    
    def get_for_drawing(self, role = "") -> list[dict]:
        return [
                {"xy": self.selection, "class": dict(marker='x', s=20, c='#bf5a17')}
            ]