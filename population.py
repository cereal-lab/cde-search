''' Contains classes that define different populations and the way they change 
    See base class Population
'''

from abc import abstractmethod
from collections import deque
from math import sqrt
from typing import Any
from params import PARAM_IND_CHANGES_STORY, PARAM_INTS, PARAM_UNIQ_INTS, rnd, PARAM_UNIQ_INDS, PARAM_MAX_INDS, param_steps

class Population:
    ''' Base class for population of candidates/tests which defines different ways of breeding 
        and going to next generation
    '''            
    def __init__(self, all_inds: list[Any], popsize, **kwargs) -> None:
        self.size = popsize 
        self.population = [] #first call of get_inds should initialize
        self.interactions = {}
        self.total_interaction_count = 0
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
        self.total_interaction_count = 0
        self.pop_metrics[PARAM_UNIQ_INDS] = 0
        self.pop_metrics[PARAM_MAX_INDS] = 0
        self.seen_inds = set()

    def merge_interactions(self, interactions: dict[Any, dict[Any, int]]) -> None:
        for ind, ints in interactions.items():                        
            ind_ints = self.interactions.setdefault(ind, {})
            self.pop_metrics[PARAM_INTS] = self.pop_metrics.get(PARAM_INTS, 0) + len(ints)
            for t, outcome in ints.items():
                if t not in ind_ints:
                    self.total_interaction_count += 1
                ind_ints[t] = outcome
        self.pop_metrics[PARAM_UNIQ_INTS] = self.total_interaction_count        

    def update(self, interactions: dict[Any, dict[Any, int]]) -> None:
        self.merge_interactions(interactions)

class HCPopulation(Population):
    ''' Population that evolves with parent-child hill climbing
        self.population is a set of indexes from self.all_inds
    '''
    def __init__(self, all_inds: list[Any], popsize,
                    mutation = "plus_minus_one", 
                    selection = "pareto_select", **kwargs) -> None:
        super().__init__(all_inds, popsize, **{"mutation":mutation, "selection":selection, **kwargs})
        self.mutation_strategy = getattr(self, mutation)
        self.selection_strategy = getattr(self, selection)

    def plus_minus_one(self, parent_index: int):
        # parent_index = next(i for i, ind in enumerate(pool) if ind == parent)
        child_index = parent_index + rnd.choice([-1,1])
        if child_index < 0: 
            child_index = len(self.all_inds) - 1
        elif child_index >= len(self.all_inds):
            child_index = 0
        return child_index
    
    def resample(self, parent_index: int):
        for _ in range(10):
            child_index = rnd.randint(0, len(self.all_inds))
            if child_index != parent_index:
                return child_index 
        return parent_index    
    
    def pareto_select(self, i, all_ints):
        common_ints = [(co, po) for co, po in all_ints if co is not None and po is not None]
        ind = self.children[i] if len(common_ints) > 1 and \
                            all(co >= po for co, po in common_ints) and \
                            any(co > po for co, po in common_ints) else self.population[i]
        return ind
    
    def informativeness_select(self, i, all_ints):
        ''' selection based on ninformativeness score aggregation '''
        def informativeness_score(ints: list[int]):
            ''' counts pairs of interactions of same outcome '''
            return sum(1 for i in range(len(ints)) for j in range(i+1, len(ints)) if ints[i] == ints[j])
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

    def init_inds(self) -> None:
        super().init_inds()
        self.population = rnd.choice(len(self.all_inds), size = self.size, replace=False)
        self.children = [self.mutation_strategy(parent_index) for parent_index in self.population]

        self.pop_metrics[PARAM_MAX_INDS] += 2 * self.size
        self.seen_inds.update(self.population, self.children)
        self.pop_metrics[PARAM_UNIQ_INDS] = len(self.seen_inds)

    def get_inds(self, *, only_parents = False, only_children = False, **filters) -> list[Any]:        
        return [self.all_inds[i] for group in [([] if only_children else self.population), [] if only_parents else self.children] 
                    for i in group]

    def update(self, interactions: dict[Any, dict[Any, int]]) -> None:
        super().update(interactions)
        parents = [self.all_inds[i] for i in self.population]
        children = [self.all_inds[i] for i in self.children]
        parent_ints = [interactions.get(p, {}) for p in parents]
        child_ints = [interactions.get(c, {}) for c in children]
        all_ints = [[(c_ints.get(t, None), p_ints.get(t, None)) for t in all_int_keys] 
                        for p_ints, c_ints in zip(parent_ints, child_ints)
                        for all_int_keys in [ set([*p_ints.keys(), *c_ints.keys()]) ] ]
        
        selected = [ self.selection_strategy(i, ints) for i, ints in enumerate(all_ints)]
        num_changes = sum(1 for s, p in zip(selected, self.population) if s != p)
        self.pop_metrics.setdefault(PARAM_IND_CHANGES_STORY, []).append(num_changes)
        self.population = selected 
        self.children = [self.mutation_strategy(parent_index) for parent_index in self.population]
        
        
        self.pop_metrics[PARAM_MAX_INDS] += 2 * self.size
        self.seen_inds.update(self.population, self.children)
        self.pop_metrics[PARAM_UNIQ_INDS] = len(self.seen_inds)

class Sample(Population):
    ''' Population is a sample from interaction matrix '''
    def __init__(self, all_inds: list[Any], size:int, **kwargs) -> None:
        super().__init__(all_inds, size, **kwargs)        
        self.t = 0

    @abstractmethod
    def sample_inds(self) -> list[Any]:
        pass

    def init_inds(self) -> None:
        super().init_inds()        
        self.population = None #self.sample_inds()
        self.for_group = None
        self.t = 0

    def update(self, interactions: dict[Any, dict[Any, int]]) -> None:
        super().update(interactions)        
        self.t += 1  
        self.population = None

    def get_same_interactions_percent(self):
        ''' We would like to avoid same interactions for the following steps
            This method computes the percent of given group in self.for_group with already present interactions
            It is up to concrete population how to handle same interactions (current simulations assume uselessness of such interactions)
        '''
        scores = {ind: existing / len(self.for_group)
                    for ind in self.all_inds for ind_ints in [self.interactions.get(ind, {})]
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
        
class SamplingStrategySample(Sample):
    ''' Sampling with strategies from matrix directly '''
    def __init__(self, all_inds: list[Any], size: int,
                    strategy = None, epsilon = None, softmax = None, tau = 1,
                    **kwargs) -> None:
        super().__init__(all_inds, size, **{"strategy":strategy, "epsilon": epsilon, "softmax": softmax, "tau": tau, **kwargs})
        self.strategy = strategy
        self.epsilon = epsilon
        self.softmax = softmax
        self.tau = tau        

    def calc_knowledge_score(self, ind, selected_inds):
        ''' Computes features of interactions about how well ind was alreeady explored '''
        ind_interactions = self.interactions.get(ind, {})
        knowledge_force = 1 if self.total_interaction_count == 0 else (1 - len(ind_interactions) / self.total_interaction_count)
        scores = {}
        # metrics of interactions based on knowledge about ind and already selected inds
        scores["kn"] = knowledge_force    
        scores["rel-kn"] = -(sum(len([oid for oid in selected_ind_ints.keys() if oid in ind_interactions]) 
                                        for selected_ind in selected_inds 
                                        for selected_ind_ints in [self.interactions.get(selected_ind, {})]))
        scores["kn-n"] = len(ind_interactions)
        return scores

    def calc_domination_score(self, ind, selected_inds):
        ''' Computes metrics of individual about how well ind dominates others 
            ind - individual (candidate/test) for which we need to compute score 
            selected_inds - already sampled inds 
            interactions - set of interactions for all inds of ind kind
        '''
        ind_interactions = self.interactions.get(ind, {})
        other_ids = set(ind_interactions.keys()) #other ids represent ids of other kind (for candidate it is test and vise versa)
        def build_common_interactions(pool_inds):
            ''' computes common interactions with other inds of the pool '''
            common_interactions = {ind1: interactions
                for ind1 in pool_inds if ind1 != ind
                for ind1_interactions in [self.interactions.get(ind1, {})]
                for interactions in [{common_other_id: (ind_interactions[common_other_id], ind1_interactions[common_other_id]) 
                    for common_other_id in set.intersection(other_ids, set(ind1_interactions.keys()))}]
                if len(interactions) > 0} #non-empty interactions set
            return common_interactions
        common_interactions = build_common_interactions(self.interactions.keys())
        non_dominated = [ind1 for ind1, ints in common_interactions.items()
                                if len(ints) > 1 and
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

        #NOTE: spanned_of_selected vs spanned: spanned is peramnent while spanned_of_selected is only temporary block

        non_domination_score = len(non_dominated) / max(len(self.interactions), 1)
        domination_score = len(dominated) / max(len(self.interactions), 1)
        scores = {}
        scores["nond"] = non_domination_score
        # scores["nondb"] = 1 if non_domination_score > 0 else 0
        scores["nd"] = sqrt(non_domination_score * domination_score)
        scores["dom"] = domination_score
        scores["dup"] = -1 if duplicate_of_selected else 1 #prefer non-duplicants
        # scores["sp"] = -1 if spanned_of_selected else 1 #prefer nonspanned
        return scores
    
    def calc_complexity_score(self, ind, selected_inds):
        ind_interactions = self.interactions.get(ind, {})
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
        
    def sample_inds(self) -> list[Any]:
        default_strategy = [{"t": 0, "keys": [["rel-kn", "kn", "nond", "sd", "dom"]] * self.size }]
        sample_strategy = default_strategy if self.strategy is None or len(self.strategy) == 0 else self.strategy
        sample_strategy = sorted(self.strategy, key = lambda x: x["t"])
        if sample_strategy[0]["t"] != 0:
            sample_strategy = [*default_strategy, *sample_strategy]        
        curr_key_spec = next(spec for spec in reversed(sample_strategy) if self.t >= spec["t"])["keys"]
        seens_interaction_percents = self.get_same_interactions_percent()
        excluded = {ind for ind, p in seens_interaction_percents.items() if rnd.rand() < p }
        selected = set()
        while len(selected) < self.size:
            i = len(selected)
            
            i_key_spec = curr_key_spec[i] if i < len(curr_key_spec) else curr_key_spec[-1]
            if (self.epsilon is not None) and (rnd.rand() < self.epsilon): #apply default epsilon strategy - ordered exploration
                i_key_spec = ["kn", "nond", "d"]
            elif self.softmax is not None:
                softmax_idx = [i for i, k in enumerate(i_key_spec) if k == self.softmax][0]
                i_key_spec = i_key_spec[:softmax_idx]
            key_selector = self.build_selector(i_key_spec)
            ind_scores = [ {"ind": ind, "scores": scores, "key": key_selector(scores) }
                                    for ind in self.all_inds if ind not in selected and ind not in excluded
                                    for scores in [{**self.calc_knowledge_score(ind, selected), 
                                                    **self.calc_domination_score(ind, selected), 
                                                    **self.calc_complexity_score(ind, selected)}]]
            # (s * seens_interaction_penalties.get(ind, 1) for s in 
            sorted_inds = sorted(ind_scores, key=lambda x: x["key"], reverse=True)
            if len(sorted_inds) == 0:
                if len(excluded) == 0:
                    break
                excluded = set()
                continue
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
            # print(f"t={self.t} {i}/{self.n} d={selected_did} from alts {len(best_candidates)} {selected_candidate} kspec: {curr_key_spec[i]}")
            # self.block_similar(selected_did, blocked_dids)
            # print(f"\t selected {qid}: {selected_did}. {selected_dids}")
            selected.add(selected_ind)
        return list(selected)

class ParetoGraphSample(Sample):
    ''' Tracks interactions and from scarced interaction metricis builds Pareto graph 
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

    class Node:
        def __init__(self) -> None:
            self.tests = set() 
            self.ints = {}
            self.dominates = [] 
            self.dominated = [] 
            self.axes = {} #dict, axes_id: point_id. For spanned - many of them           
            # self.rank = 0 

    def sample_inds(self) -> list[Any]:
        ''' Builds Pareto graph from interactions and sample it '''        
        #first iteration is to build all nodes:
        nodes = []
        sorted_tests = sorted(self.interactions.items(), key=lambda x: (len(x[1]), len([v for v in x[1].values() if v == 1])))
        for test, ints in sorted_tests:
            #first we try to check other nodes
            test_node_common_ints = [(node, [(o, node.ints[c]) for c, o in ints.items() if c in node.ints])
                                        for node in nodes]
            test_common_ints = sorted([(node, common_ints) for node, common_ints in test_node_common_ints
                                        if len(common_ints) > 1 and all(a == b for a,b in common_ints)], 
                                key=lambda x: len(x[1]), reverse=True)
            best_common_ints = len(test_common_ints[0][1]) if len(test_common_ints) > 0 else 0
            selected_nodes = []
            i = 0
            while i < len(test_common_ints) and len(test_common_ints[i][1]) == best_common_ints:
                selected_nodes.append(test_common_ints[i][0])
                i += 1 
            for node in selected_nodes:
                node.tests.add(test)
                node.ints = {**node.ints, **ints} #merge
            if len(selected_nodes) == 0: #create new node
                node = ParetoGraphSample.Node()
                node.tests.add(test)
                node.ints = ints
                nodes.append(node)
        # now we have all list of nodes. Weestablish Pareto dominance between them
        for i in range(len(nodes)):
            node_a = nodes[i]
            for j in range(i + 1, len(nodes)):
                node_b = nodes[j]
                node_node_common_ints = [(o, node_b.ints[c]) for c, o in node_a.ints.items() if c in node_b.ints]
                if len(node_node_common_ints) > 1:
                    if all(a >= b for a, b in node_node_common_ints) and any(a > b for a, b in node_node_common_ints):
                        node_a.dominates.append(node_b)
                        node_b.dominated.append(node_a)
                    elif all(b >= a for a, b in node_node_common_ints) and any(b > a for a, b in node_node_common_ints):
                        node_a.dominated.append(node_b)
                        node_b.dominates.append(node_a)
                    #otherwise noncomparable. or same?? probably cannot be same here 
        # dominates-dominated edges are built now. We can analyse the graph assigning axes 
        # for this we build topological orders of DAGs for each of axis (nodes without incoming edges)                        
        def visit(node: ParetoGraphSample.Node, grayNodes: set[ParetoGraphSample.Node], blackNodes: set[ParetoGraphSample.Node],
                    order: deque[ParetoGraphSample.Node]):
            ''' Walks through dominated edges from node deep down.
                Cuts nodes that have already been processed 
                GrayNodes - nodes we are entered 
                BlackNodes - nodes we existed
                implicit WhiteNodes - nodes that are not yet considered 
                order contains built topological order
            '''
            if node in blackNodes:
                return 
            assert node not in grayNodes, f'There is a cycle (Pareto not DAG). Node: {node.tests}, {node.dominated}'
            grayNodes.add(node)
            for dominator in node.dominated:
                visit(dominator, grayNodes, blackNodes, order)
            grayNodes.remove(node)
            blackNodes.add(node)
            order.appendleft(node)


        axes_starts = [node for node in nodes if len(node.dominates) == 0]
        axes = []
        for axis_id, node in enumerate(axes_starts):
            topological_order = deque([])
            visit(node, set(), set(), topological_order)
            prev_rank = 0
            sum_ranks = 0
            axes_nodes = []
            while len(topological_order) > 0:
                node = topological_order.popleft()
                node.axes[axis_id] = max(0, prev_rank + 1 - self.rank_penalty * max(0, len(node.dominates) - 1))
                prev_rank = node.axes[axis_id] 
                sum_ranks += prev_rank
                axes_nodes.append(node)
            #normalize ranks to probabilities
            for node in axes_nodes:
                node.axes[axis_id] = node.axes[axis_id] / sum_ranks
            axes.append(axes_nodes)
        
        #exploration vs exploitation - some tests are not in the Pareto graph yet. We need also decide how to balance here
        #we use annealing with game step increase the chance for exploitation with time 

        interacted_weights = {ind: 0.1 + 0.9 * (1 - p) for ind, p in self.get_same_interactions_percent().items()}

        exploitation_weight = self.min_exploitation_chance + (self.t / self.max_step) * (self.max_exploitation_chance - self.min_exploitation_chance)
        exploration_weight = 1 - exploitation_weight
        # strategies = rnd.choice([0, 1], size = self.size, p = [1 - exploitation_chance, exploitation_chance])
        # num_exploits = sum(strategies)
        # num_explores = sample_size - num_exploits    
        weights = {}    

        for axis_id, axis in enumerate(axes):
            for node in axis:
                for test in node.tests:
                    weights[test] = exploitation_weight * (node.axes[axis_id] + 1) * interacted_weights.get(test, 1)        

        for ind in self.all_inds:
            if ind not in self.interactions:
                weights[ind] = exploration_weight * interacted_weights.get(ind, 1)

        min_weight = min(weights.values())        
        weights_shifted = {ind: (w - min_weight) + 0.1 for ind, w in weights.items()}
        sum_weights = sum(weights_shifted.values())
        probs = {ind: w / sum_weights for ind, w in weights_shifted.items()}

        inds, p = zip(*list(probs.items()))
        selected_indexes = rnd.choice(len(inds), size = self.size, replace=False, p = list(p))
        selected = [inds[i] for i in selected_indexes]
        return selected    

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
    
    def update(self, interactions: dict[Any, dict[Any, int]]) -> None:
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

    def update(self, interactions: dict[Any, dict[Any, int]]) -> None:
        self.population = self.get_group()