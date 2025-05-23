''' Contains classes that define different populations and the way they change 
    See base class Population
'''

from abc import abstractmethod
from dataclasses import dataclass, field
import json
from math import sqrt
import math
from typing import Any, Iterable, Optional

import numpy as np
from de import extract_dims_approx, extract_dims_fix, get_batch_pareto_layers2
from metrics import avg_rank_of_repr, dimension_coverage, duplication, redundancy, trivial
from params import PARAM_IND_CHANGES_STORY, PARAM_INTS, PARAM_UNIQ_INTS, rnd, PARAM_UNIQ_INDS, PARAM_MAX_INDS, \
    param_steps, param_selection_size, param_batch_size, rnd, seq_rnd
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
        self.sel_params = {"size": self.size}

    def get_selection(self) -> list[Any]:
        return self.selection
    
    def get_best(self) -> list[Any]:
        return self.selection
    
    def get_pool(self) -> list[Any]:
        return self.pool

    def init_selection(self) -> None:
        self.selection = []
        self.interactions = {}
        self.sel_metrics = {PARAM_UNIQ_INDS: 0, PARAM_MAX_INDS: 0}
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

    def get_for_drawing(self) -> list[dict]:
        ''' Called to draw population'''
        return [
            {"xy": self.selection, "class": dict(marker='o', s=40, c='black', edgecolor='white'), "legend": [f"{s[0]},{s[1]}" for s in self.selection[:20]]}
        ]
    
    def collect_metrics(self, axes, origin, spanned, *, is_final = False):
        sample = self.get_best()
        if len(sample) == 0:
            return
        DC = dimension_coverage(axes, sample)
        ARR, ARRA = avg_rank_of_repr(axes, sample)
        Dup = duplication(axes, origin, spanned, sample)
        R = redundancy(spanned, sample)
        tr = trivial(axes, sample)
        self.sel_metrics.setdefault("DC", []).append(DC)
        self.sel_metrics.setdefault("ARR", []).append(ARR)
        self.sel_metrics.setdefault("ARRA", []).append(ARRA)
        self.sel_metrics.setdefault("Dup", []).append(Dup)
        self.sel_metrics.setdefault("R", []).append(R)
        self.sel_metrics.setdefault("tr", []).append(tr)
        if is_final:
            self.sel_metrics["sample"] = sample


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
        for _ in range(10):
            if type(ind) is tuple:
                mutated = tuple(c + rnd.choice([-1,1]) for c in ind)
            else:
                mutated = int(ind + rnd.choice([-1,1]))
            if mutated in self.pool_set:
                return mutated
        return ind

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
        self.selection = [tuple(rnd.choice(self.init_range) for _ in self.pool[0]) if type(self.pool[0]) is tuple else int(rnd.choice(self.init_range)) for _ in range(self.size)]

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

    def get_for_drawing(self) -> list[dict]:
        class_parent = dict(marker='o', s=40, c='black', edgecolor='white')
        class_child = dict(marker='o', s=10, c='blue', edgecolor='white')
        return [
                {"xy": self.seen_inds, "bg": True}, 
                {"xy": self.selection, "class": class_parent, "legend": [f"{x[0]},{x[1]}" for x in self.selection[:20]]},
                {"xy": self.children, "class": class_child, "legend": [f"{x[0]},{x[1]}" for x in self.children[:20]]}
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
        super().update(interactions, int_keys)
        self.update_features(interactions, int_keys)
        # self.t += 1  
        self.exploited_inds = [] if self.selection is None else self.selection
        self.selection = None

    def init_selection(self) -> None:
        super().init_selection()        
        self.selection = None
        # self.for_group = None
        # self.t = 0
        self.ind_groups = {}
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
        self.explore(selected_inds)
        self.selection = list(selected_inds)
        self.sel_metrics[PARAM_MAX_INDS] += len(selected_inds)
        self.seen_inds.update(self.selection)
        self.sel_metrics[PARAM_UNIQ_INDS] = len(self.seen_inds)
        return self.selection
    
    def get_best(self):
        return self.exploited
    
    def get_for_drawing(self) -> list[dict]:
        pop = self.get_selection() #just to make sure that population is inited
        explore = self.ind_groups.get("explore", [])
        exploit = self.ind_groups.get("exploit", [])
        return [
                {"xy": self.exploited_inds, "bg": True}, 
                {"xy": explore, "class": dict(marker='o', s=10, c='blue', edgecolor='white'), "legend": [f"{x[0]},{x[1]}" for x in explore[:20]]},
                {"xy": exploit, "class": dict(marker='o', s=40, c='black', edgecolor='white'), "legend": [f"{x[0]},{x[1]}" for x in exploit[:20]]},
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
        return list(discriminating_candidates) # 
        #return list(set.union(discriminating_candidates, last_batch_candidates))    

@dataclass 
class MCTSNode: 
    sample: list[int] # we store ind scores in sample separatelly     
    num_visits: int = 0
    score: float = 0 # total score of sample avgd across sample count
    children: list['MCTSNode'] = field(default_factory=list)

class MCTSSelection(ExploitExploreSelection):
    ''' Implements Monte-Carlo Tree Search selection 
        Idea: tree node is a sample of size test from pool
        The expansion from the node estimate possible improvement of the sample.
        We recognize that some tests in the node sample could be of good position in local coordinate system
        We resemple worst tests to do the exploration 

        p - nonleaf proba in MCTS
        alpha - MCTS len penalty
        beta - minimal reward
    '''      

    def __init__(self, pool: list[Any], *, p = 0, c = 0.1, #sqrt(2) / 2,
                                            spanned_penalty = 0.05,
                                            score_strategy = "mean_score_strategy",
                                            child_sel_strategy = 'uct_selection', **kwargs) -> None:
        ''' p - nonleaf proba in MCTS, alpha - len penalty, beta - minimal reward,
            c - UCT parameter of exploratiion-exploitation tradeoff     
        '''
        super().__init__(pool, **kwargs)
        self.sel_params.update(p = p, c = c, 
                               spanned_penalty = spanned_penalty,
                               score_strategy = score_strategy, child_sel_strategy = child_sel_strategy)
        self.p = float(p)
        # self.alpha = float(alpha)
        # self.beta = float(beta)
        self.c = float(c)
        self.spanned_penalty = float(spanned_penalty)
        self.score_strategy = getattr(self, score_strategy)        
        self.child_sel_strategy = getattr(self, child_sel_strategy)

    def init_selection(self):
        super().init_selection()
        self.axis_scores = {}
        # self.root = MCTSNode(sample = [], num_visits=1, score=0, children = [])
        self.roots = []
        self.total_visits = 0

    def mean_score_strategy(self, test_id: Any) -> float:
        axis_score = np.mean(self.axis_scores[test_id])
        return axis_score
    
    def uct_selection(self, children: list[MCTSNode], parent: Optional[MCTSNode]) -> Optional[MCTSNode]:
        ''' Upper Confidence bounds applied to Trees '''
        if len(children) == 0:
            return None
        
        if parent is not None:
            num_parent_visits = parent.num_visits
        else:
            num_parent_visits = self.total_visits
        if num_parent_visits == 0:
            num_parent_visits = 1
        parent_num_visits_log = np.log2(num_parent_visits)
        total_scores = np.array([ch.score for ch in children])
        num_visits = np.array([ch.num_visits for ch in children])
        explore_part = self.c * np.sqrt(parent_num_visits_log / num_visits)
        exploit_part = total_scores / num_visits
        children_scores = exploit_part + explore_part
        best_child_id = np.argmax(children_scores)
        best_child = children[best_child_id]
        return (best_child, children_scores[best_child_id])
    
    def select_mcts_path(self) -> list[tuple[MCTSNode, float]]:
        ''' Going from root, selects the path that is best according to child_sel_strategy 
            Excludes root node as it is just a container
        '''
        path = [] # path has node and its score in the selection - tuples
        cur_node_with_score = self.child_sel_strategy(self.roots, None)
        while cur_node_with_score is not None:
            path.append(cur_node_with_score)
            if self.p != 0 and rnd.random() < self.p:
                break
            cur_node = cur_node_with_score[0]
            cur_node_with_score = self.child_sel_strategy(cur_node.children, cur_node)
        return path        

    def backpropagate_path(self, sample_score: float, sample: list[int]) -> None:
        ''' Extends path and backpropagates scores '''
        new_node = MCTSNode(sample = sample, num_visits = 0, score = 0, children = [])
        if len(self.selected_path) > 0:
            extended_node = self.selected_path[-1][0]
            extended_node.children.append(new_node)
        else:
            self.roots.append(new_node)
        self.selected_path.append((new_node, 0))
        for node, _ in reversed(self.selected_path):
            node.num_visits += 1
            node.score += sample_score
        self.total_visits += 1

    def update_features(self, interactions: dict[Any, dict[Any, int]], int_keys: set[Any]) -> None:
        test_ids = list(interactions.keys()) #ids off local interactions
        candidate_ids = list(int_keys)
        tests = [[self.interactions[test_id][candidate_id] for candidate_id in candidate_ids] for test_id in test_ids]
        dims, origin, spanned = extract_dims_fix(tests)
        sample_score = 0
        count = 0
        for dim in dims:
            for point_id, group in enumerate(dim):
                for i in group:
                    test_id = test_ids[i]
                    score = (point_id + 1) / len(dim)
                    self.axis_scores.setdefault(test_id, []).append(score)
                    sample_score += score
                    count += 1
        for i in origin:
            test_id = test_ids[i]
            self.axis_scores.setdefault(test_id, []).append(0)
            count += 1
        for i, spanned_dims in spanned.items():
            test_id = test_ids[i]
            span_score = np.mean([(point_id + 1) / len(dims[dim_id]) for dim_id, point_id in spanned_dims.items()]) 
            span_score -= self.spanned_penalty
            if span_score < 0:
                span_score = 0                
            self.axis_scores.setdefault(test_id, []).append(span_score)
            sample_score += span_score
            count += 1
        sample_score /= count
        # backpropagate score through MCTS path
        self.backpropagate_path(sample_score, test_ids)

    # def exploit(self, sample_size) -> set[Any]:
    #     ''' MCTS selection
    #         sample_size could dictate how many samples from node to leave 
    #     '''

    # def explore(self, selected):
    #     ''' Regenerate new samples and create new node '''
    #     return super().explore(selected)
    
    def exploit(self, sample_size) -> set[Any]:
        ''' Select inds with MC tree '''

        best_path = self.select_mcts_path()
        self.selected_path = best_path
        if len(best_path) == 0:
            return set() # create full rand sample
        expansion_node_id = max(range(len(best_path)), key = lambda x: best_path[x][1])
        if expansion_node_id != len(best_path) - 1:
            best_path = best_path[:expansion_node_id + 1]
            self.selected_path = best_path
        expansion_node = best_path[-1][0]
        scores = {}
        for test_id in expansion_node.sample:
            score = self.score_strategy(test_id)
            scores[test_id] = score
        sorted_scores = sorted(scores.keys(), key = lambda x: (scores[x], len(self.interactions[x])), reverse=True)
        selected = set(sorted_scores[:sample_size])
        self.ind_groups.setdefault(f"exploit", []).extend(selected)
        return selected

class DESelection(ExploitExploreSelection):
    ''' Implementation that is based on DECA idea (modified DE algo)
        Param approx_strategy defines how to compute missing values None from global interactions
        approx_strategy: zero_approx_strategy, one_approx_strategy, maj_[c|r|cr]_approx_strategy, candidate_group_approx_strategy, deca_approx_strategy
        cand_sel_strategy defines what candidates to select for new dimension extraction: 
            local - only last batch, global - any candidate with 2 or more tests in the last batch 
            discriminative - those candidates that distinguished axes last time (fallback to local)
        spanned_memory - defines how many times we can forgive the point to be spanned before resampling
                         Originally the o bjective on batch_(n-k), the point being spanned for k times will be discarded
                         Same goes with point which is treated as origin (trivial)
    '''
    def __init__(self, pool: list[Any], *, cand_sel_strategy = "local_cand_sel_strategy", 
                                           test_sel_strategy = "local_test_sel_strategy", 
                                           approx_strategy = "maj_c_approx_strategy", 
                                           spanned_memory = 100,
                                           **kwargs) -> None:
        super().__init__(pool, **kwargs)
        self.spanned_memory = float(spanned_memory)
        self.sel_params.update(cand_sel_strategy = cand_sel_strategy, 
                                test_sel_strategy = test_sel_strategy, approx_strategy = approx_strategy,
                                spanned_memory = self.spanned_memory)
        self.approx_strategy_name = approx_strategy
        self.approx_strategy = None if approx_strategy == "deca_approx_strategy" else getattr(approx, approx_strategy)
        self.cand_sel_strategy = getattr(self, cand_sel_strategy)
        self.test_sel_strategy = getattr(self, test_sel_strategy)
        # self.origin = []
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
        self.prev_selection_spanned = [] 
        self.prev_selection_origin = []
        self.spanned_resampled = {}

    def update_features(self, interactions: dict[Any, dict[Any, int]], int_keys: set[Any]) -> None:
        ''' Applies DE to a given local or global ints and extract axes '''    
        test_ids = self.test_sel_strategy(interactions.keys(), int_keys)
        candidate_ids = self.cand_sel_strategy(test_ids, int_keys)
        tests = [[self.interactions[test_id].get(candidate_id, None) for candidate_id in candidate_ids] for test_id in  test_ids]
        approx_counts = {test_id: c for test_id, c in zip(test_ids, self.get_approx_counts(tests))}
        if self.approx_strategy_name == "deca_approx_strategy":
            dims, origin, spanned = extract_dims_approx(tests) 
        else:
            self.approx_strategy(tests) #fillin Nones
            dims, origin, spanned = extract_dims_fix(tests)
        # approx_dims, _ = self.approx_strategy(tests) #fillin Nones
        
        origin_tests = {test_ids[i] for i in origin}
        spanned_tests = {test_ids[i] for i in spanned.keys()}
        self.prev_selection_spanned = [test_id for test_id in self.prev_selection_spanned if test_id in spanned_tests and self.spanned_resampled.get(test_id, 0) < self.spanned_memory] 
        self.prev_selection_origin = [test_id for test_id in self.prev_selection_origin if test_id in origin_tests and self.spanned_resampled.get(test_id, 0) < self.spanned_memory]
        for test_id in self.prev_selection_spanned:
            self.spanned_resampled[test_id] = self.spanned_resampled.get(test_id, 0) + 1
        for test_id in self.prev_selection_origin:
            self.spanned_resampled[test_id] = self.spanned_resampled.get(test_id, 0) + 1

        # self.origin = [test_ids[i] for i in origin]       
        # self.origin.sort(key = lambda x: len(self.interactions[x]), reverse=True)

        self.axes = []
        for dim in dims:
            axes_tests = [(test_ids[i], (len(dim) - point_id - 1, -len(self.interactions[test_ids[i]]), approx_counts.get(test_ids[i], 0))) for point_id, group in enumerate(dim) for i in group]
            axes_tests.sort(key = lambda x: x[1], reverse=True)
            for ind, _ in axes_tests:
                if ind in self.spanned_resampled:
                    del self.spanned_resampled[ind]
            self.axes.append(axes_tests)

        # # fail sets of objectives (ends of axes)
        # cf_sets = [{candidate_ids[i] for i, o in enumerate(tests[dim[-1][0]]) if o == 1} for dim in dims]
        # self.discriminating_candidates = set.union(*cf_sets) - set.intersection(*cf_sets)

    def exploit(self, sample_size) -> set[Any]:
        
        selected = set(self.prev_selection_spanned)
        axe_id = 0 
        axes = [[point for point in dim] for dim in self.axes]
        axes.sort(key = lambda x: x[-1][1])
        while len(selected) < sample_size and len(axes) > 0:
            axis = axes[axe_id]
            ind, _ = axis.pop()
            selected.add(ind)
            if len(axis) == 0:
                axes.pop(axe_id)
                if len(axes) == 0:
                    break
            else:
                axe_id += 1
            if axe_id == len(axes):
                axe_id = 0
                axes.sort(key = lambda x: x[-1][1])

        for ind in self.prev_selection_origin:
            if len(selected) >= sample_size:
                break
            selected.add(ind)
            
        # self.discriminating_candidates = self.get_discriminating_set(selected)
        self.ind_groups.setdefault(f"exploit", []).extend(selected)
        # removed = self.prev_exploited - selected
        # self.discarded_exploited_tests.extend(removed)
        # self.prev_exploited = set(selected)
        self.prev_selection_spanned = list(selected)
        self.prev_selection_origin = list(selected)
        return selected

class DEScores(ExploitExploreSelection):
    ''' Instead of approximating unknown values, we score each axis '''
    def __init__(self, pool: list[Any], *, score_strategy = "mean_score_strategy",
                                            spanned_penalty = 0.05,
                                           **kwargs) -> None:
        super().__init__(pool, **kwargs)
        score_strategy_name = score_strategy
        self.score_strategy = getattr(self, score_strategy)
        self.spanned_penalty = float(spanned_penalty)
        self.sel_params.update(score_strategy = score_strategy_name, spanned_penalty = spanned_penalty)
        
    def init_selection(self):
        super().init_selection()
        self.axis_scores = {}

    def mean_score_strategy(self, test_id: Any) -> float:
        axis_score = np.mean(self.axis_scores[test_id])
        return axis_score
    
    def median_score_strategy(self, test_id: Any) -> float:     
        axis_score = np.median(self.axis_scores[test_id])
        return axis_score
    
    def mode_score_strategy(self, test_id: Any) -> float:     
        ''' most frequent score '''
        scores = {}
        for score in self.axis_scores[test_id]:
            scores.setdefault(np.floor(score * 100), []).append(score)
        _, axis_score = max([(len(score_group), max(score_group)) for score_group in scores.values()])
        return axis_score 

    def update_features(self, interactions: dict[Any, dict[Any, int]], int_keys: set[Any]) -> None:
        test_ids = list(interactions.keys())
        candidate_ids = list(int_keys)
        tests = [[self.interactions[test_id][candidate_id] for candidate_id in candidate_ids] for test_id in test_ids]
        dims, origin, spanned = extract_dims_fix(tests)
        for dim in dims:
            for point_id, group in enumerate(dim):
                for i in group:
                    test_id = test_ids[i]
                    self.axis_scores.setdefault(test_id, []).append((point_id + 1) / len(dim))                
        for i in origin:
            test_id = test_ids[i]
            self.axis_scores.setdefault(test_id, []).append(0)
        for i, spanned_dims in spanned.items():
            test_id = test_ids[i]
            span_score = np.mean([(point_id + 1) / len(dims[dim_id]) for dim_id, point_id in spanned_dims.items()]) 
            span_score -= self.spanned_penalty
            if span_score < 0:
                span_score = 0
            self.axis_scores.setdefault(test_id, []).append(span_score)

    def exploit(self, sample_size) -> set[Any]:
        scores = {}
        for test_id in self.axis_scores.keys():
            score = self.score_strategy(test_id)
            scores[test_id] = score
        sorted_scores = sorted(scores.keys(), key = lambda x: (scores[x], len(self.interactions[x])), reverse=True)
        selected = set(sorted_scores[:sample_size])
        self.ind_groups.setdefault(f"exploit", []).extend(selected)
        return selected

class ParetoLayersSelection(ExploitExploreSelection):
    ''' Sampling is based on pareto ranks (see LAPCA) and corresponding archive '''
    def __init__(self, pool: list[Any], *, cand_sel_strategy = "local_cand_sel_strategy", max_layers = 10, 
                    discard_spanned = 0, **kwargs) -> None:
        super().__init__(pool, **kwargs)
        self.max_layers = int(max_layers)
        self.discard_spanned = int(discard_spanned) > 0
        self.sel_params.update(max_layers = self.max_layers, cand_sel_strategy = cand_sel_strategy, 
                               discard_spanned = self.discard_spanned)
        self.cand_sel_strategy = getattr(self, cand_sel_strategy)

    def init_selection(self) -> None:
        super().init_selection() 
        self.ind_order = []

    def update_features(self, interactions: dict[Any, dict[Any, int]], int_keys: set[Any]) -> None:
        ''' Split into pareto layers '''        

        test_ids = list(interactions.keys())
        candidate_ids = self.cand_sel_strategy(test_ids, int_keys)
        tests = [[self.interactions[test_id].get(candidate_id, None) for candidate_id in candidate_ids] for test_id in test_ids ]

        layers, _ = get_batch_pareto_layers2(tests, max_layers=self.max_layers, discard_spanned = self.discard_spanned)

        self.ind_order = []
        for layer in layers:
            tests = [test_ids[i] for i in layer]
            tests.sort(key = lambda x: len(self.interactions[x]), reverse=True)
            self.ind_order.extend(tests)

    def exploit(self, sample_size) -> set[Any]:    

        selected = set(self.ind_order[:sample_size])

        self.ind_groups.setdefault(f"exploit", []).extend(selected)
        
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
            seq_rnd.shuffle(self.pool)
        self.selection = self.get_group()

    def update(self, interactions: dict[Any, list[tuple[Any, int]]], int_keys: set[Any]) -> None:
        self.selection = self.get_group()

    def get_selection(self, only_children = False, **filters) -> list[Any]:
        if only_children:
            return []    
        return self.selection        
    
    def get_for_drawing(self) -> list[dict]:
        return [
                {"xy": self.selection, "class": dict(marker='x', s=10, c='#666666')}
            ]
    
    def collect_metrics(self, axes, origin, spanned, *, is_final = False):
        pass