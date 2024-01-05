''' Module that gennerates CDE space and corresponding interaction matrix
    Idea of CDE is taken from: https://link.springer.com/chapter/10.1007/978-3-540-24854-5_53

    Ideally we would like the template to approximate main metrics:
    D - dimension discovery 
    ARR - average rank of representatives 
    R - redundancy 
    Dup - duplication 
    nonInfo - non-informativeness
    Independance  - metric for axes independance in terms of candidates on them
    ---
    D, ARR are defined by axes; R - by spanned points; Dup by tests on same point in space; nonInfo - by tests at origin
    Assumption that number of selected tests |T| >= num_axes for axes to be discovered. 
    More axes in relation to whole number of points, better chance to discover them.

    Default: We work with uniformly distributed points around axes.

    Distributions
    Axes --> points (skew of points on axes - should affect D) --> tests (skew of tests on points - should affect ARR)

    0. RQ0: No skew of points on axes, no spanned, no non-informative, no duplicates, strong independance
    1. RQ1: Under skew of points on axes, how does D change for different algos? 
            Unbalanced representation of objectives/misconceptions by tests
    2. RQ2: Under skew of tests on points, how does ARR change for different algos (extreme to nonInfo)? Noise of test triviality and similarity
    3. RQ3: Under presense of spanned points how R, D, ARR (ARR*) changes? Noise of test complexity
    4. RQ4: Under presense of duplicated tests how Dup, D, ARR (ARR*) changes? Noise of test similarity
    5. RQ5: Under increase number of objectives per candidate, how algo behavior change? Noise of weak independance of axes
    6. RQ6*: consistency of algos around different spaces? Probably under RQ1
    7. --RQ7**: experiments with strategies ==> different setups of algos at first place - all RQs above should consider them

    Algos:
    1. One time random sampling - baseline 
    2. P-PHC - different versions, w archive
    3. Sampling strategies 
    4. ACO    
'''

from itertools import product
import json
from math import prod
import numpy as np

class JSONSerializable:
    def to_json(self):
        return json.dumps(self, default=lambda o: o.to_dict(), sort_keys = True)
    def to_dict(self):
        return self.__dict__

class CDEPoint(JSONSerializable):
    def __init__(self, tests = [], candidates = [], unique_candidates = [], **kwargs) -> None:
        self.tests = tests
        self.candidates = candidates
        self.unique_candidates = unique_candidates
    def __str__(self) -> str:
        return f"Point({self.tests}, {self.candidates})"
    def __repr__(self) -> str:
        return str(self)    

class CDESpace(JSONSerializable):
    ''' Represents the CDE space 
        Constructor builds smaller possible space without duplicates 
        Use with_ methods to add duplication of information, skew of tests, dependency, nonInformativeness and redundancy 
    
        RQ0 space is simplest - does not have duplicates and spanned points
        It can optionally have origin point. Each point has 1 associated test. 
        Objectives are strongly independent, meaning that one candidate is present only on one axis. 
        :param dims - objectives and number of points per axis (with origin)
        :param candidate_distr - for each axis defines distribution of candidates
        :param test_distr - for each axis defines distribution of tests
        :returns - created CDESpace 
    '''    
    def __init__(self, dims: list[int] = [], candidate_remap = [], spanned = [], origin = {}, 
                        axes = [], next_test_id = 0, next_candidate_id = 0, **kwargs) -> None:
        ''' Creates space with len(dims) axes and number of points per axis specified in dims
            Origin is created anyway 
            :param dims - points per axis             
        '''
        self.dims = dims
        self.candidate_remap = {m["b"]: m["a"] for m in candidate_remap}
        self.spanned = {tuple(tuple(x) for x in m["coord"]):CDEPoint(**m["point"]) for m in spanned}
        self.origin = CDEPoint(**origin)
        self.axes = [[CDEPoint(**point) for point in axis] for axis in axes]
        if len(axes) == 0 and len(dims) > 0:
            self.axes = [[CDEPoint() for _ in range(dim)] for dim in dims]
        self.next_test_id = next_test_id
        self.next_candidate_id = next_candidate_id
    def to_dict(self):
        res = dict(**super().to_dict())
        res["spanned"] = [{"coord": coord, "point": point} for coord, point in self.spanned.items()]
        res["candidate_remap"] = [{"a": a, "b": b} for b, a in self.candidate_remap.items()]
        res["DC_prob"] = self.space_dimension_coverage_probability()
        res["ARR_prob"] = self.space_best_ranks_probability()
        res["dependency"] = self.space_dependency()
        res["Dup"] = self.space_duplication()
        res["R"] = self.space_redundancy()
        res["nonI"] = self.space_noninformativeness()
        return res 
    def with_test_distribution(self, origin: int = 0, test_distr: int | list[int] | list[list[int]] = 1):
        ''' Sepcifies counts of tests per each point in the space 
            :param origin - number of tests at origin (non-informative)
            :param test_distr - number of tests on each point OR per point on each axis OR for all axes
            NOTE: the last count will be repeated for each point not covered by test_distr
        '''
        if type(test_distr) == int:
            test_distr = [[test_distr]] #all axes with same number of points 
        elif type(test_distr) == list and type(test_distr[0]) == int:
            test_distr = [test_distr]
        self.origin.tests = list(range(origin))
        self.next_test_id = len(self.origin.tests)
        for axis_id in range(len(self.axes)):
            axis = self.axes[axis_id]
            axis_distr = test_distr[axis_id if axis_id < len(test_distr) else -1]
            for point_id in range(len(axis)):
                point = axis[point_id]
                test_count = axis_distr[point_id if point_id < len(axis_distr) else -1]        
                point.tests = list(range(self.next_test_id, self.next_test_id + test_count))
                self.next_test_id += test_count
        return self 
    def with_candidate_distribution(self, origin: int = 0, cand_distr: int | list[int] | list[list[int]] = 1):
        ''' Specifies counts of candidates per each point in the space 
            Candidates are unique - string independence of axis 
            :param origin - number of candidates at origin - does not fail on any test 
            :param cand_distr - num candidates for each point OR for each axis OR for all space 
        '''
        if type(cand_distr) == int:
            cand_distr = [[cand_distr]] #all axes with same number of points 
        elif type(cand_distr) == list and type(cand_distr[0]) == int:
            cand_distr = [cand_distr]
        self.origin.unique_candidates = list(range(origin))
        self.origin.candidates = list(self.origin.unique_candidates)
        self.next_candidate_id = len(self.origin.unique_candidates)
        for axis_id in range(len(self.axes)):
            axis = self.axes[axis_id]
            axis_distr = cand_distr[axis_id if axis_id < len(cand_distr) else -1]
            prev_candidate = []
            for point_id in range(len(axis)):
                point = axis[point_id]
                candidate_count = axis_distr[point_id if point_id < len(axis_distr) else -1]        
                point.unique_candidates = list(range(self.next_candidate_id, self.next_candidate_id + candidate_count))
                point.candidates = [*prev_candidate, *point.unique_candidates]
                prev_candidate = point.candidates 
                self.next_candidate_id += candidate_count
        return self 
    def with_axes_dependency(self, axis_a: int, point_a: int, axis_b: int, point_b: int):
        ''' Specifies same candidates for two different axes
            This will reassign candidate ids to match on different axes. 
            Note however, that 2 fully dependent axes should be merged into one - should be avoided 
            :param axis_a and point_a - specify the point from which the candidate will be taken
            :param axis_b and point_b - second point 
            NOTE that only nonassigned candidates are taken 
        '''
        assert axis_a != axis_b
        pairs = product([self.candidate_remap.get(c, c) for c in self.axes[axis_a][point_a].candidates], 
                        [self.candidate_remap.get(c, c) for c in self.axes[axis_b][point_b].candidates])

        candidate_a, candidate_b = next((sorted([a, b]) for a, b in pairs if a != b), [None, None])
        if candidate_a is not None and candidate_b is not None:
            self.candidate_remap[candidate_b] = candidate_a
        return self 
    
    def with_spanned_point(self, axes_ids: list[tuple[int, int]], num_tests: int):
        ''' Adds spanned point on intersection of axes and with given number of tests.
            Candidates are union of candidates of axes points 
            :param axes_ids - ids of axes points to combine 
            :param num_tests - number of tests at point
        '''
        axes_ids = tuple(axes_ids)
        assert axes_ids not in self.spanned
        point = CDEPoint()
        self.spanned[axes_ids] = point
        point.tests = list(range(self.next_test_id, self.next_test_id + num_tests))
        self.next_test_id += num_tests
        point.unique_candidates = [] 
        point.candidates = [cid for axis_id, point_id in axes_ids for cid in self.axes[axis_id][point_id].candidates]
        return self

    def get_candidate_fails(self):
        ''' Return fails based on state of the space
            :returns - map, fail set of tests per each candidate
        '''
        interaction_list = [*(el for axis in self.axes for point in axis for el in product(point.candidates, point.tests)),
                             *(el for point in self.spanned.values() for el in product(point.candidates, point.tests))]
        interactions = {}
        for cid, tid in interaction_list:
            interactions.setdefault(self.candidate_remap.get(cid, cid), set()).add(tid)
        return interactions
    
    def get_candidates(self):
        return {*(self.candidate_remap.get(c, c) for axis in self.axes for point in axis for c in point.unique_candidates), 
                *self.origin.unique_candidates}
    
    def get_tests(self):
        return { *self.origin.tests,
                 *(t for spanned in self.spanned.values() for t in spanned.tests),
                 *(t for axis in self.axes for point in axis for t in point.tests)}

    def __str__(self) -> str:
        return f"Space(axes={self.axes}, origin={self.origin}, spanned={self.spanned}, remap={self.candidate_remap})"
    
    def __repr__(self) -> str:
        return str(self)
    
    #NOTE: Monte Carlo method of estimating the expected metric values is implemented in baseline rand selection and Tests sample metricis
    #Space metrics 
    def space_dimension_coverage_probability(self):
        ''' Probablility that uniform sampling without replacement of num=len(axes) tests could cover all dimensions
            This metric is an estimation of how hard to cover the space due to its skews to objectives 
        '''
        num = len(self.axes)
        origin_tests = len(self.origin.tests)
        spanned_tests = sum(len(spanned.tests) for spanned in self.spanned.values())
        axes_tests = [sum(len(point.tests) for point in axis) for axis in self.axes]
        total_tests = origin_tests + spanned_tests + sum(axes_tests)
        denom = prod(total_tests - i for i in range(num))
        nom = prod(axes_tests)
        num_perm = prod(i + 1 for i in range(num))
        res = num_perm * nom / denom 
        return res 
    
    def space_best_ranks_probability(self):
        ''' Probability that uniform selection of num = len(axes) could pick ends of all axes '''
        num = len(self.axes)
        origin_tests = len(self.origin.tests)
        spanned_tests = sum(len(spanned.tests) for spanned in self.spanned.values())
        axes_tests = [sum(len(point.tests) for point in axis) for axis in self.axes]
        total_tests = origin_tests + spanned_tests + sum(axes_tests)
        denom = prod(total_tests - i for i in range(num))
        best_tests = [len(axis[-1].tests) for axis in self.axes]
        nom = prod(best_tests)
        num_perm = prod(i + 1 for i in range(num))
        res = num_perm * nom / denom 
        return res      

    def space_redundancy(self):
        ''' Ratio of spanned tests to all number of tests '''
        origin_tests = len(self.origin.tests)
        spanned_tests = sum(len(spanned.tests) for spanned in self.spanned.values())
        axes_tests = [sum(len(point.tests) for point in axis) for axis in self.axes]
        total_tests = origin_tests + spanned_tests + sum(axes_tests)
        return spanned_tests / total_tests
    
    def space_noninformativeness(self):
        ''' Ratio of origin tests to all number of tests '''
        origin_tests = len(self.origin.tests)
        spanned_tests = sum(len(spanned.tests) for spanned in self.spanned.values())
        axes_tests = [sum(len(point.tests) for point in axis) for axis in self.axes]
        total_tests = origin_tests + spanned_tests + sum(axes_tests)
        return origin_tests / total_tests
    
    def space_duplication(self):
        ''' Percent of duplicated information in the space '''
        origin_tests = len(self.origin.tests)
        spanned_tests = [len(spanned.tests) for spanned in self.spanned.values()]
        axes_tests = [[len(point.tests) for point in axis] for axis in self.axes]
        total_tests = origin_tests + sum(spanned_tests) + sum(sum(axis) for axis in axes_tests)
        duplicates = max(0, origin_tests - 1) + sum(max(0, c - 1) for c in spanned_tests) + sum(sum(max(0, c - 1) for c in axis) for axis in axes_tests)
        return duplicates / total_tests

    def space_dependency(self):
        ''' Metrics that defines how much the axes of the space are independent 
            Expressed through percent of related candidates to total number of candidates
        '''
        return len(self.candidate_remap) / self.next_candidate_id

    # Test sample metrics 
    def dimension_coverage(self, tests_sample: set[int]):
        ''' Computes DC metric based on DECA axes and given tests sample (ex., last population of P-PHC)
            :param tests_sample - ids of tests in the space 
            :returns percent of axes covered by sample
        '''
        sample_axes = [axis for axis in self.axes if any(t in tests_sample for point in axis for t in point.tests)]
        dim_coverage = len(sample_axes) / len(self.axes)
        return dim_coverage

    def avg_rank_of_repr(self, tests_sample:set[int]):
        ''' Computes ARR for a given sample 
            :param tests_sample - set of tests from the space
            :returns - tuple of ARR and ARRA
        '''
        axes_ranks = [next((point_id + 1 for point_id, point in reversed(list(enumerate(axis))) 
                            if any(t in tests_sample for t in point.tests)), 0) / len(axis)
                        for axis in self.axes]
        axes_ranks_no_zero = [r for r in axes_ranks if r > 0]
        arr = 0 if len(axes_ranks_no_zero) == 0 else (sum(axes_ranks_no_zero) / len(axes_ranks_no_zero))
        arra = sum(axes_ranks) / len(axes_ranks)
        return (arr, arra)

    def redundancy(self, tests_sample:set[int]):
        ''' Computes redundancy - percent of spanned points in the selection 
            :param tests_sample - set of tests from the space
            :returns redundancy of the sample
        '''
        spanned_set = {t for spanned in self.spanned.values() for t in spanned.tests}
        spanned_sample = spanned_set & tests_sample
        R = len(spanned_sample) / len(tests_sample) 
        return R

    def duplication(self, tests_sample:set[int]):
        ''' Computes percent of dumplications in tests sample according to CDE space
            :param tests_sample - set of tests from the space
        '''
        space_tests = { **{t:((axis_id, point_id),)
                            for axis_id, axis in enumerate(self.axes) for point_id, point in enumerate(axis)
                            for t in point.tests},
                        **{t:coord_id for coord_id, spanned in self.spanned.items() for t in spanned.tests},
                        **{t:() for t in self.origin.tests}}
        Dup = 0 
        present = set()
        for t in tests_sample:
            pos = space_tests[t]
            if pos in present:
                Dup += 1 
            else:
                present.add(pos)
        Dup = Dup / len(tests_sample)
        return Dup

    def noninformative(self, tests_sample:set[int]):
        ''' percent of population that is not represented by axes or spanned (in space['zero']) '''
        zero_set = set(self.origin.tests)
        noninfo = [t for t in tests_sample if t in zero_set]
        nonI = len(noninfo) / len(tests_sample)
        return nonI    

    @staticmethod
    def from_json(json_str: str):
        data = json.loads(json_str)
        return CDESpace(**data)
    

if __name__ == '__main__':
    #RQ0
    space = CDESpace([5] * 10).with_test_distribution(0, 1).with_candidate_distribution(0, 1).with_spanned_point([(i, -1) for i in range(10)], 1)
    # dc_prob = space.space_dimension_coverage_probability()
    # r_prob = space.space_best_ranks_probability()
    # dup = space.space_duplication()
    # dep = space.space_dependency()
    # noni = space.space_noninformativeness()
    # r = space.space_redundancy()
    print(f"{space}")
    js = space.to_json()
    space2 = CDESpace.from_json(js) 
    print(f"{space2}")

    #RQ1
    # space = CDESpace([6, 6, 6, 6, 6, 4, 4, 4, 4, 4])
    # space = CDESpace([7, 7, 7, 7, 7, 3, 3, 3, 3, 3])
    # space = CDESpace([8, 8, 8, 8, 8, 2, 2, 2, 2, 2])
    # space = CDESpace([9, 9, 9, 9, 9, 1, 1, 1, 1, 1])

    # #RQ2
    # space = CDESpace([5] * 10).with_test_distribution(1) #with 1 non-info 
    # space = CDESpace([5] * 10).with_test_distribution(2) #with 2 non-info 
    # space = CDESpace([5] * 10).with_test_distribution(3) #with 3 non-info 
    # space = CDESpace([5] * 10).with_test_distribution(4) #with 3 non-info 
    # space = CDESpace([5] * 10).with_test_distribution(5) #with 3 non-info 
    # space = CDESpace([5] * 10).with_test_distribution(10) #with 3 non-info 

    # space = CDESpace([5] * 10).with_test_distribution(0, [6,1,1,1,1]) #duplicates
    # space = CDESpace([5] * 10).with_test_distribution(0, [1,6,1,1,1])
    # space = CDESpace([5] * 10).with_test_distribution(0, [1,1,6,1,1])
    # space = CDESpace([5] * 10).with_test_distribution(0, [1,1,1,6,1])
    # space = CDESpace([5] * 10).with_test_distribution(0, [1,1,1,1,6])

    # #RQ3
    # space = CDESpace([5] * 10).with_spanned_point([(i, -1) for i in range(10)], 1)
    # space = CDESpace([5] * 10).with_spanned_point([(i, -1) for i in range(10)], 5)
    # space = CDESpace([5] * 10).with_spanned_point([(0, -1), (1, -1)], 1) #TODO
    # space = CDESpace([5] * 10)
    # for i in range(10):
    #     space = space.with_spanned_point([(i - 1, -1), (i, -1)], 1)

    # #RQ4
    # space = CDESpace([5] * 10).with_test_distribution(0, 1)
    # space = CDESpace([5] * 10).with_test_distribution(0, 2)
    # space = CDESpace([5] * 10).with_test_distribution(0, 3)
    # space = CDESpace([5] * 10).with_test_distribution(0, 4)
    # space = CDESpace([5] * 10).with_test_distribution(0, 5)

    # space = CDESpace([5] * 10).with_candidate_distribution(0, 2)
    # space = CDESpace([5] * 10).with_candidate_distribution(0, 3)
    # space = CDESpace([5] * 10).with_candidate_distribution(0, 4)
    # space = CDESpace([5] * 10).with_candidate_distribution(0, 5)

    # #RQ5
    # # space = CDESpace([5] * 10).with_axes_dependency(0, -1, 1, -1)
    # # space = CDESpace([5] * 10).with_axes_dependency(0, -1, 1, -1).with_axes_dependency(2, -1, 3, -1)
    # space = CDESpace([5] * 10).with_axes_dependency(0, -1, 1, -1).with_axes_dependency(2, -1, 3, -1).with_axes_dependency(4, -1, 5, -1) \
    #                           .with_axes_dependency(6, -1, 7, -1).with_axes_dependency(8, -1, 9, -1)
    # space = CDESpace([5] * 10).with_axes_dependency(0, -1, 1, -1).with_axes_dependency(2, -1, 3, -1).with_axes_dependency(4, -1, 5, -1) \
    #                           .with_axes_dependency(6, -1, 7, -1).with_axes_dependency(8, -1, 9, -1) \
    #                           .with_axes_dependency(0, -2, 1, -2).with_axes_dependency(2, -2, 3, -2).with_axes_dependency(4, -2, 5, -2) \
    #                           .with_axes_dependency(6, -2, 7, -2).with_axes_dependency(8, -2, 9, -2)
        
    # space = CDESpace([5] * 10).with_test_distribution(0, [1,6,1,1,1])


    # space = CDESpace([2,2]).with_test_distribution(0,1).with_candidate_distribution(0, 2).with_axes_dependency(2, 6).with_axes_dependency(3, 7)
    # .with_spanned_point([(0,1),(1,1)], 1)
    # space = CDESpace(4, 5, CDEPoint([0], [0], [0]), [[CDEPoint([1], [1], [1]), CDEPoint([2], [1,2], [2])], [CDEPoint([3], [3], [3])]])
    # print(space.get_candidate_fails(), space.get_num_tests(), space.get_num_candidates())