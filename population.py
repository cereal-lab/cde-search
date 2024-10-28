''' Contains classes that define different populations and the way they change 
    See base class Population
'''

from abc import abstractmethod
import json
from math import sqrt
from typing import Any, Iterable, Optional
from de import extract_dims, get_batch_pareto_layers2
from params import PARAM_IND_CHANGES_STORY, PARAM_INTS, PARAM_UNIQ_INTS, rnd, PARAM_UNIQ_INDS, PARAM_MAX_INDS, \
    param_steps, param_selection_size, param_batch_size, rnd
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

    def get_selection(self) -> list[Any]:
        return self.selection
    
    def get_best(self) -> list[Any]:
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

    def get_selection(self) -> list[Any]:        
        return [*self.selection, *self.children]
    
    def get_best(self):
        return self.selection

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
    # min_exploit_prop = 0.2, max_exploit_prop = 0.8, max_step = param_steps, 
    def __init__(self, pool: list[Any], **kwargs) -> None:
        super().__init__(pool, **kwargs)        
        # self.min_exploit_prop = float(min_exploit_prop)
        # self.max_exploit_prop = float(max_exploit_prop)
        # self.max_step = int(max_step)
        # self.exploit_prop = self.min_exploit_prop
        # self.sel_params.update(min_exploit_prop = self.min_exploit_prop, max_exploit_prop = self.max_exploit_prop, max_step = self.max_step)
        # self.t = 0        
        # self.explore_interacted = True   
        self.ind_groups = {}
        self.exploited_inds = set()
        self.exploited = []

    @abstractmethod
    def exploit(self, sample_size: int) -> set[Any]:
        pass

    @abstractmethod
    def update_features(self, interactions: dict[Any, dict[Any, int]], int_keys: set[Any]) -> None:
        pass 

    def update(self, interactions: dict[Any, dict[Any, int]], int_keys: set[Any]) -> None:
        ''' Applies DECA to given local ints and extract axes '''
        super().update(interactions, int_keys)
        self.update_features(interactions, int_keys)
        # self.t += 1  
        self.selection = None

    def init_selection(self) -> None:
        super().init_selection()        
        self.selection = None
        # self.for_group = None
        # self.t = 0
        self.exploited_inds = set() 
        self.exploited = []
    
    def explore(self, selected: set[Any]) -> None:
        ''' Samples individuals based on interactions exploration. Contrast it to exploit selection 
            We assume that the search space is big and it is expensive to enumerate it/assign distribution per element.
            Adds exploration sample to a given selected. 
            Directed exploration, exploration towards most promising directions, between exploitation and random sampling
        '''
        #random sampling
        misses = 0 # number of times we select same
        while len(selected) < (2 * self.size) and misses < 100:
            selected_index = rnd.randint(0, len(self.pool))
            selected_ind = self.pool[selected_index]
            if selected_ind in selected:
                misses += 1
            else:
                selected.add(selected_ind)
                self.ind_groups.setdefault("explore", []).append(selected_ind)
                misses = 0

    def get_selection(self) -> list[Any]:
        if self.selection is not None and len(self.selection) > 0: # and for_group == self.for_group:
            return self.selection
        self.ind_groups = {}
        # self.for_group = for_group #adjusts sampling by avoiding repetitions of interactions - smaller chance to interruct again
        # self.exploit_prop = self.min_exploit_prop + (self.t / self.max_step) * (self.max_exploit_prop - self.min_exploit_prop)
        # exploit_slots = round(self.size * self.exploit_prop)
        selected_inds = self.exploit(self.size)
        self.exploited = list(selected_inds)
        self.exploited_inds.update(selected_inds)
        self.explore(selected_inds)
        self.selection = list(selected_inds)
        self.sel_metrics[PARAM_MAX_INDS] += len(selected_inds)
        self.seen_inds.update(self.selection)
        self.sel_metrics[PARAM_UNIQ_INDS] = len(self.seen_inds)
        return self.selection
    
    def get_best(self):
        return self.exploited
    
    def get_for_drawing(self, role = "") -> list[dict]:
        pop = self.get_selection() #just to make sure that population is inited
        explore = self.ind_groups.get("explore", [])
        exploit_groups = []
        for k,v in self.ind_groups.items():
            if k.startswith("exploit_"):
                s = int(k.split("_")[-1])
                exploit_groups.append({"xy": v, "class": dict(marker='o', s=s, c='#d662e3', alpha=0.5), "legend": [f"{t}".ljust(10) for t in v[:20]]})
        exploit = self.ind_groups.get("exploit", [])
        return [
                {"xy": self.exploited_inds, "bg": True}, 
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
    
    def local_cand_sel_strategy(self, last_btach_tests:Iterable[Any], last_batch_candidates: set[Any]):
        return list(last_batch_candidates)

    def discr_cand_sel_strategy(self, last_btach_tests:Iterable[Any], last_batch_candidates: set[Any]):
        discriminating_candidates = self.get_discriminating_set(set(last_btach_tests))
        return list(discriminating_candidates) # list(set.union(discriminating_candidates, last_batch_candidates))    
        
class InteractionFeatureOrder(ExploitExploreSelection):
    ''' Sampling of search space based on grouping by interactions features and ordering of the groups 
        Also we called this sampling strategies
    '''
    def __init__(self, pool: list[Any], *, strategy = None, **kwargs) -> None:
        super().__init__(pool, **kwargs)
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

class ParetoLayersSelection(ExploitExploreSelection):
    ''' Sampling is based on pareto ranks (see LAPCA) and corresponding archive '''
    def __init__(self, pool: list[Any], *, cand_sel_strategy = "local_cand_sel_strategy", max_layers = 10, **kwargs) -> None:
        super().__init__(pool, **kwargs)
        self.max_layers = int(max_layers)
        self.sel_params.update(max_layers = self.max_layers, cand_sel_strategy = cand_sel_strategy)
        self.cand_sel_strategy = getattr(self, cand_sel_strategy)
        self.discarded = []

    def init_selection(self) -> None:
        super().init_selection() 
        self.ind_order = []
        # self.discriminating_candidates = set()
        self.discarded = []

    def update_features(self, interactions: dict[Any, dict[Any, int]], int_keys: set[Any]) -> None:
        ''' Split into pareto layers '''        

        test_ids = list(interactions.keys())
        candidate_ids = self.cand_sel_strategy(test_ids, int_keys)
        tests = [[self.interactions[test_id].get(candidate_id, None) for candidate_id in candidate_ids] for test_id in test_ids ]

        layers, discarded = get_batch_pareto_layers2(tests, max_layers=self.max_layers)

        self.discarded = [test_ids[tid] for tid in discarded]
        self.discarded.sort(key = lambda x: len(self.interactions[x]), reverse=True)
        
        self.ind_order = []
        for layer in layers:
            tests = [test_ids[i] for i in layer]
            tests.sort(key = lambda x: len(self.interactions[x]), reverse=True)
            self.ind_order.extend(tests)


    def exploit(self, sample_size) -> set[Any]:    

        selected_tests = self.ind_order[:sample_size]
        selected = set(selected_tests)

        if len(selected) < sample_size:
            selected.update(self.discarded[:sample_size - len(selected)])

        self.ind_groups.setdefault(f"exploit", []).extend(selected)

        # self.discriminating_candidates = self.get_discriminating_set(selected)
        # for i in range(len(selected_tests)):
        #     test_id1 = selected_tests[i]
        #     test1 = [self.interactions[test_id1].get(cid, None) for cid in self.candidate_ids]
        #     for j in range(i + 1, len(selected_tests)):
        #         test_id2 = selected_tests[j]
        #         test2 = [self.interactions[test_id2].get(cid, None) for cid in self.candidate_ids]
        #         self.discriminating_candidates.update(self.candidate_ids[i] for i, (o1, o2) in enumerate(zip(test1, test2)) if o1 is not None and o2 is not None and o1 != o2)
        
        return selected

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