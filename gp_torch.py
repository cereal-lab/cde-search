'''  
    Reimplelments GP module with torch support, some refactoring of gp.py
'''
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
import inspect
from functools import partial
from itertools import chain, product
from typing import Any, Callable, Optional

import numpy as np
import torch
import utils

from rnd import default_rnd
import utils

# THOUGHTS: we do not need linear representation! ?! 
#           we use our semantic geometric crossover that representation agnostic
#           We need trees to compute paths between nodes and torch expressions to optimize

@dataclass
class SimpleNode():
    ''' We still use trees but simplify them as much as possible 
        All additional attributes should be stored in tensors off RuntimeContext
        Nodes are indexed by int in the system
    '''
    op: Callable
    ''' Operation '''

    args: list[int] = field(default_factory=list)
    ''' ids of arguments in RuntimeContext '''

    # def __init__(self, op: Callable, *args: 'SimpleNode'):
    #     self.op = op
    #     self.args = args # List of Nodes 
    #     self.str = None
    #     self.depth = None
    #     self.nodes = None # list of all nodes for direct access
    #     self.signature = inspect.signature(self.func)
    #     self.return_type = self.signature.return_annotation

    # def __str__(self):
    #     if self.str is None:
    #         if len(self.args) == 0:
    #             self.str = self.func_meta.name()
    #         else:
    #             node_args = ", ".join(arg.__str__() for arg in self.args)
    #             self.str = self.func_meta.name() + "(" + node_args + ")"
    #     return self.str
    # def __repr__(self):
    #     return self.__str__()

    # def get_depth(self):
    #     if self.depth is None:
    #         if len(self.args) == 0:
    #             self.depth = 0
    #         else:
    #             self.depth = max(n.get_depth() for n in self.args) + 1
    #     return self.depth
    # def is_leaf(self):
    #     return len(self.args) == 0
    # def is_of_type(self, node):
    #     return self.return_type == node.return_type
    # def get_nodes(self):
    #     if self.nodes is None:
    #         self.nodes = [(0, self), *((arg_depth + 1, n) for arg in self.args for arg_depth, n in arg.get_nodes())]
    #     return self.nodes
    # def get_node(self, i):
    #     nodes = self.get_nodes()
    #     return nodes[i]        

# Outputs = list[Any] | np.ndarray | torch.Tensor
# Vectors = np.ndarray | torch.Tensor

@dataclass 
class SearchPoint:
    prog_id: int 
    parent_i: int 
    ''' Parent index on its depth level '''
    parent_arg_i: int 
    ''' Argument index in parents arg list '''
    select_koef: float
    ''' Attempted number of optimizations at this point'''

OP_CONST = 0
# OP_VAR = -2

@dataclass 
class ProgramConst:
    value: torch.Tensor
    path: list[int]

@dataclass
class Program:
    prog_op: int 

    args: list[int]


    size: int 
    depth: int 
    # point_selection_koef: dict[int, float] #point id to count
    # points: list[list[SearchPoint]]
    # ''' Points by depth level 
    #     Always has points[0], empty if no args
    # '''

    semantics: torch.Tensor

    consts: dict[int, ProgramConst]
    ''' point id to const tensor '''

    op_counts: torch.Tensor # 1-dim - ops 

    linear_preorder: list[SearchPoint] = field(default_factory=list)
    ''' At start it is [] and expanded on first search of selection point '''
    # expanded: bool = False

@dataclass
class IndexedProgram:
    prog_id: int 
    semantics: torch.Tensor

class SemanticIndex(ABC):
    ''' Base class for spatial indexes in semantic space '''
    
    @abstractmethod
    def rebuild(self, **kwargs):
        ''' Recreates index inner strurctures for new set of arguments '''
        pass

    # @abstractmethod
    # def get_partition(self, semantics: torch.Tensor) -> list[int]:
    #     ''' For a given semantics, fetches all programs in the corresponding partition '''
    #     pass 

    @abstractmethod
    def insert(self, prog_ids: list[int], semantics: list[torch.Tensor]) -> None:
        ''' Adds a program to a partition indexed by semantics '''
        pass

    @abstractmethod
    def query(self, semantics: torch.Tensor, distance: Optional[Any] = None) -> list[IndexedProgram]:
        ''' Finds closest prog_ids (in closest partitions), distance is specific to Index, could be L2 radius,
            When distance is none, retursn programs of current partition
        '''
        pass 

    # TODO: method to get statistics of partitions

# Possible implementations include KDTrees, BallTrees, RTrees etc
# Hierarchical spatial indexes could be used to check semantic equivalence of programs
# We, hovewer, consider partition as bin, which could contain potentially semantically different programs
 
class RCosIndex(SemanticIndex):
    ''' Bins semantics by sphere R + dR and cos distance.
        Such bins could contain many semantically different programs - index cannot be used for seamntic equality check. '''

    def __init__(self, *, center: torch.Tensor, R_epsilon: float = 1e-2, cos_epsilon: float = 1e-2):
        self.zero_vector = torch.zeros_like(center)
        self.ref_point = torch.zeros_like(center) #for cos distance
        if torch.allclose(center, self.zero_vector):
            self.ref_point[0] = 1.0 # change ref point further from center
        self.center = center
        self.ref_vector = self.ref_point - center
        zero_norm = torch.norm(self.ref_vector)
        self.ref_vector /= zero_norm
        self.partitions = None #partition contains prog_ids and semantics 
        self.rebuild(R_epsilon = R_epsilon, cos_epsilon = cos_epsilon)

    def rebuild(self, *, R_epsilon: float, cos_epsilon: float):
        ''' resets partition '''
        self.R_epsilon = R_epsilon
        self.cos_epsilon = cos_epsilon
        self.min_cos_id = 0
        self.max_cos_id = torch.floor(2.0 / self.cos_epsilon).int().item()
        
        old_partitions = self.partitions
        self.partitions = defaultdict(lambda : defaultdict(list))
        if old_partitions is None:
            return
        
        for r_p in old_partitions.values():
            for c_p in r_p.values():
                for (prog_id, prog_semantics) in c_p:
                    self.insert(prog_id, prog_semantics)

    def get_partition(self, semantics: torch.Tensor) -> list[int]:
        ''' Returns partition id '''
        semantic_diff = semantics - self.center
        semantic_norm = torch.norm(semantic_diff)
        R = torch.floor(semantic_norm / self.R_epsilon).int().item()
        if torch.allclose(semantic_diff, self.zero_vector):
            cos_p = 0
        else:
            cos_dist = 1 + (torch.dot(semantic_diff, self.ref_vector) / semantic_norm)
            cos_p = torch.floor(cos_dist / self.cos_epsilon).int().item()
        return [R, cos_p]
    
    def insert(self, prog_ids: list[int], semantics: list[torch.Tensor]) -> None:
        for prog_id, prog_semantics in zip(prog_ids, semantics):
            R_id, cos_id = self.get_partition(prog_semantics)
            self.partitions[R_id][cos_id].append(IndexedProgram(prog_id, prog_semantics))
    
    def query(self, semantics: torch.Tensor, distance: Optional[Any] = None) -> list[IndexedProgram]:
        ''' distance - none - returns records in semantics partition, otherwise computes near partitions '''
        partitions_to_return = set()
        R_id, cos_id = self.get_partition(semantics)
        partitions_to_return.add((R_id, cos_id))
        if distance is not None:
            delta_R, delta_cos = distance
            min_R_id = max(0.0, R_id * self.R_epsilon - delta_R) / self.R_epsilon
            max_R_id = (R_id + delta_R) / self.R_epsilon
            cos_p = cos_id * self.cos_epsilon - 1
            min_cos_id = torch.floor((cos_p - delta_cos) / self.cos_epsilon).int().item()
            max_cos_id = torch.floor((cos_p + delta_cos) / self.cos_epsilon).int().item()
            if max_cos_id > self.max_cos_id:
                max_cos_id = max_cos_id - self.max_cos_id - 1
            if min_cos_id < self.min_cos_id:
                min_cos_id = self.max_cos_id + 1 - min_cos_id
            for r in range(min_R_id, max_R_id + 1):
                for c in range(min_cos_id, max_cos_id + 1):
                    partitions_to_return.add((r, c))
        selected_programs = []
        for (r, c) in partitions_to_return:
            if r in self.partitions and c in self.partitions[r]:
                for prog in self.partitions[r][c]:
                    selected_programs.append(prog)
        return selected_programs

@dataclass(eq=False, unsafe_hash=True)
class CoordinateSystemPoint: 
    interactions: torch.Tensor
    programs: list[IndexedProgram] = field(default_factory=list)
    next_point: Optional['CoordinateSystemPoint'] = None # forward dim chain, uo has no next point
    prev_point: Optional['CoordinateSystemPoint'] = None # backward dim chain, point close to origin has not prev

    def forward_chain(self):
        ''' Iterates over forward chain '''
        cur_point = self
        while cur_point is not None:
            yield cur_point
            cur_point = cur_point.next_point

    def forward_skip(self, n):
        skipped = []
        cur_start = self
        while n > 0: 
            skipped.append(cur_start)
            cur_start = cur_start.next_point
            n -= 1
        return skipped, cur_start
    
    def break_chain(self):
        prev_point = self.prev_point
        self.prev_point = None
        if prev_point is not None:
            prev_point.next_point = None
        return self

    def backward_chain(self):
        ''' Iterates over backward chain '''        
        cur_point = self
        while cur_point is not None:
            yield cur_point
            cur_point = cur_point.prev_point

    def backward_skip(self, n):
        skipped = []
        cur_start = self
        while n > 0:
            skipped.append(cur_start) 
            cur_start = cur_start.prev_point
            n -= 1
        return skipped, cur_start   

# @dataclass 
# class SelectedCoordinateSystemPoint:
#     point: CoordinateSystemPoint
#     pos: list[tuple[int, int]] | frozenset[tuple[int, int]]
#     is_new: bool

@dataclass 
class CoordinateSystem:
    dim_sizes: list[int] = field(default_factory=list)
    dim_starts: list[CoordinateSystemPoint] = field(default_factory=list)
    dim_ends: list[CoordinateSystemPoint] = field(default_factory=list)
    span: dict[frozenset[tuple[int, CoordinateSystemPoint]], CoordinateSystemPoint] = field(default_factory=dict)
    span_lists: dict[CoordinateSystemPoint, list[frozenset[tuple[int, CoordinateSystemPoint]]]] = field(default_factory=dict)
    ''' Relation between coordinate system point and dependent spanned points '''

@dataclass 
class CoordinateSystemDistance: 
    delta_from_max: Optional[int] = None
    with_span_points: bool = False 
    only_span_points: bool = False

class CoordinateSystemIndex(SemanticIndex):
    ''' Organizes all semantics into CoordinateSystem and gradually adds new seamantics '''

    # search - based on underlying objectives and spanned points, by computing Pos(x) and taking neighbors
    # insert - with CSE algo iteration (again, with extension Pos)

    # reindexing at moments or no reindexing???

    def __init__(self, *, center: torch.Tensor, max_dim_size: int = 5, rebuild_size: Optional[int] = None):
        self.epsilons = None # computed automatically
        self.center = center
        self.rebuild_size = rebuild_size
        # self.origin = torch.zeros(self.center, dtype=torch.uint8)
        self.max_dim_size = max_dim_size
        self.cs = CoordinateSystem()
    
    def trim_cs(self) -> None:
        ''' Limits dims to max_dim_size '''
        skipped_points = []
        for dim_id, (dim_start, dim_size) in enumerate(zip(self.cs.dim_starts, self.cs.dim_sizes)):
            if dim_size > self.max_dim_size:
                num_to_skip = dim_size - self.max_dim_size
                skipped, cur_point = dim_start.forward_skip(num_to_skip)
                cur_point = cur_point.break_chain()
                self.cs.dim_sizes[dim_id] = self.max_dim_size
                self.cs.dim_starts[dim_id] = cur_point
                for skip_point in skipped:
                    skipped_points.append(skip_point)
        for point in skipped_points:
            for span_id in self.cs.span_lists[point]:
                del self.cs.span[span_id]
            del self.cs.span_lists[point]
        
    def rebuild(self):
        ''' Index rebuild schedule assumes reduction of epsilons with time '''

        old_cs = self.cs
        self.cs = CoordinateSystem()

        prog_ids = []
        semantics = []
        for dim_start in old_cs.dim_starts:
            for point in dim_start.forward_chain():
                prog_ids.extend([prog.prog_id for prog in point.programs])
                semantics.extend([prog.semantics for prog in point.programs])

        for span_point in old_cs.span.values():
            prog_ids.extend([prog.prog_id for prog in span_point.programs])
            semantics.extend([prog.semantics for prog in span_point.programs])

        self.insert(prog_ids, semantics)

        del old_cs

    def get_interactions(self, semantics: torch.Tensor) -> torch.Tensor:
        ''' This impl returns compressed interactions where outcome is encoded in bit value '''
        if self.epsilons is None:
            min_test_vs = torch.min(semantics, dim=0).values
            max_test_vs = torch.max(semantics, dim=0).values
            self.epsilons = min_test_vs + (max_test_vs - min_test_vs) / 2.0

        plain_interactions = (semantics - self.center) < self.epsilons
        np_interactions = plain_interactions.cpu().numpy()
        compressed_interactions = np.packbits(np_interactions, axis = 1)
        interactions = torch.from_numpy(compressed_interactions)
        return interactions 

    def get_unique_interactions(self, interactions: torch.Tensor) -> tuple[list[int], torch.Tensor, list[list[int]]]:
        unique_interactions, unique_indices = torch.unique(interactions, sorted=True, dim=0, return_inverse = True)
        prog_ids_split = [[] for _ in range(unique_interactions.size(0))]
        for prog_id, unique_id in enumerate(unique_indices):
            prog_ids_split[unique_id].append(prog_id)
        if torch.all(unique_interactions[0] == 0): #ignore oriigin
            unique_interactions = unique_interactions[1:]
            non_informative_ids = prog_ids_split.pop(0)
        else:
            non_informative_ids = []
        return non_informative_ids, unique_interactions, prog_ids_split

    # def has_origin_poss(self, unique_sorted_interactions: torch.Tensor) -> bool:
    #     ''' Check if interactions has CS origin.
    #         Param interactions: 2d tensor, 1d - prog_id, 2d - test_id, cell - int outcome 
    #         Returns: bool, True if interactions has origin
    #     '''
    #     is_origin = unique_sorted_interactions[0] == self.cs.origin.interactions
    #     # NOTE: interactions are sorted, smallest is compared
    #     return is_origin
    
    def get_dupl_poss(self, interactions: torch.Tensor) -> dict[int, CoordinateSystemPoint]:
        '''  Search for a point at dim and spanned points, present in CS 
            Param interactions: 2d tensor, 1d - prog_id, 2d - test_id, cell - int outcome
            Returns found point for each prog_id

            TODO: optimize search - current impl assumes that interactions are not sorted 
        '''
        # is_dupl = torch.zeros(interactions.size(0), dtype = torch.bool)
        poss = {}
        # dims = [list(dim.forward_chain()) for dim in self.cs.dim_starts]
        # poss = torch.full((interactions.size(0), len(dims)), -1, dtype = torch.int)
        if interactions.size(0) == 0:
            return poss
        cur_mask = torch.ones(interactions.size(0), dtype = torch.bool) # active prog_ids 
        cur_ids = torch.where(cur_mask)[0]
        for dim in self.cs.dim_starts:
            if cur_ids.size(0) == 0:
                break
            for point in dim.forward_chain():
                if cur_ids.size(0) == 0:
                    break
                cur_interactions = interactions[cur_ids]
                dupls = torch.all(cur_interactions == point.interactions, dim = 1)
                dupl_ids = cur_ids[dupls]
                if dupl_ids.size(0) > 0:
                    for prog_id in dupl_ids:
                        poss[prog_id] = point
                    cur_mask[dupl_ids] = False
                    cur_ids = torch.where(cur_mask)[0]

        for span_point in self.cs.span.values():
            if cur_ids.size(0) == 0:
                break
            cur_interactions = interactions[cur_ids]
            dupls = torch.all(cur_interactions == span_point.interactions, dim = 1)
            dupl_ids = cur_ids[dupls]
            if dupl_ids.size(0) > 0:
                for prog_id in dupl_ids:
                    poss[prog_id] = span_point
                cur_mask[dupl_ids] = False
                cur_ids = torch.where(cur_mask)[0]
        return poss
    
    def get_poss(self, interactions: torch.Tensor) -> list[list[tuple[int, int, CoordinateSystemPoint]]]:
        ''' Search for positions in CS, calls of get_origin_poss and get_dupl_poss should go before this 
            Here we know that interactions do not match any point in CS 
            Param interactions: 2d tensor, 1d - prog_id, 2d - test_id, cell - int outcome
            Returns for each prog_id, list of dim_id, point_id (from UO) and point on that dim that is dominated by interactions[prog_id]

            NOTE: additional checks on spanned points should be done outside
        '''

        poss = [[] for _ in range(interactions.size(0))]
        if interactions.size(0) == 0:
            return poss
        for dim_id, dim_end in enumerate(self.cs.dim_ends):
            cur_mask = torch.ones(interactions.size(0), dtype = torch.bool) # active prog_ids 
            cur_ids = torch.where(cur_mask)[0]
            for point_id, point in enumerate(dim_end.backward_chain()):
                if cur_ids.size(0) == 0:
                    break
                cur_interactions = interactions[cur_ids]
                dom_mask = torch.all(cur_interactions >= point.interactions, dim = 1)
                dom_ids = cur_ids[dom_mask]
                if dom_ids.size(0) > 0:
                    for prog_id in dom_ids:
                        poss[prog_id].append((dim_id, point_id, point))
                    cur_mask[dom_ids] = False
                    cur_ids = torch.where(cur_mask)[0]

        return poss
    
    def get_span_poss(self, interactions: torch.Tensor, poss: list[list[tuple[int, int, CoordinateSystemPoint]]]) -> list[int]:
        ''' Computes if the prog_id is spanned or not (the point is not necessary in CS)
            Param interactions: prog interactions 2d tensor 
            Param poss - computed previously positions
            Returns list of ids of new spanned points
        '''
        spanned = []
        if len(poss) == 0:
            return spanned 
        for span_id, span_pos in enumerate(poss):
            if len(span_pos) <= 1:
                continue
            span_ints = interactions[span_id]
            match_mask = torch.zeros_like(span_ints, dtype = torch.uint8)
            for _, _, point in span_pos:
                match_mask.bitwise_or_(point.interactions)
            if torch.all(match_mask == span_ints):
                spanned.append(span_id)
        return spanned

    def add_dupl_poss(self, poss: dict[int, CoordinateSystemPoint], prog_ids: list[list[int]], semantics: list[list[torch.Tensor]]) -> None:    
        for dupl_id, (dupl_prog_ids, dupl_semantics) in enumerate(zip(prog_ids, semantics)):
            indexed_programs = [IndexedProgram(p, s) for p, s in zip(dupl_prog_ids, dupl_semantics)]
            dupl_point = poss[dupl_id]
            dupl_point.programs.extend(indexed_programs)
    
    def add_new_span_poss(self, poss: list[list[tuple[int, int, CoordinateSystemPoint]]], 
                          interactions: torch.Tensor, 
                          prog_ids: list[list[int]], semantics: list[list[torch.Tensor]]) -> None:
        for span_pos, span_interactions, span_prog_ids, span_semantics in zip(poss, interactions, prog_ids, semantics):
            indexed_programs = [IndexedProgram(p, s) for p, s in zip(span_prog_ids, span_semantics)]
            pos_id = frozenset((a, c) for a, _, c in span_pos)
            self.cs.span[pos_id] = CoordinateSystemPoint(span_interactions, indexed_programs)
            for _, dim_point in pos_id:
                self.cs.span_lists.setdefault(dim_point, []).append(pos_id)

    def get_ext_poss(self, interactions: torch.Tensor, poss: list[list[tuple[int, int, CoordinateSystemPoint]]]) -> list[list[tuple[int, int, CoordinateSystemPoint]]]:
        ''' Check dimension starts and filters out incompatible points for extension 
            Returns next point before which prog_id can be inserted, or None if new underlying objective
        '''
        ext_poss = [[] for _ in range(interactions.size(0))]
        for ext_pos, interaction, pos in zip(ext_poss, interactions, poss):
            pos_dict = {a: (b, c) for a, b, c in pos}
            for dim_id, dim_starts in enumerate(self.cs.dim_starts):
                if dim_id in pos_dict:
                    cur_point_id, cur_point = pos_dict[dim_id]
                    next_point = cur_point.next_point
                    next_point_id = cur_point_id - 1
                else:
                    next_point = dim_starts
                    next_point_id = self.cs.dim_sizes[dim_id] - 1
                if (next_point is None) or torch.all(next_point.interactions >= interaction):
                    ext_pos.append((dim_id, next_point_id, next_point))
        return ext_poss

    def get_pareto_noncomparable(self, interactions: torch.Tensor) -> list[int]: 
        noncomp_matrix = torch.any(interactions[:, torch.newaxis] < interactions, dim=-1) & torch.any(interactions[:, torch.newaxis] > interactions, dim=-1)
        noncomp_ids = torch.where(noncomp_matrix)[0]
        return noncomp_ids.cpu().tolist()

    # def query(self, semantics: torch.Tensor, distance: Optional[Any] = None) -> list[IndexedProgram]:
    def insert(self, prog_ids: list[int], semantics: list[torch.Tensor]) -> None:

        stacked_semantics = torch.stack(semantics, dim = 0)
        interactions = self.get_interactions(stacked_semantics)

        _, unique_interactions, prog_ids_split_local = self.get_unique_interactions(interactions)

        prog_ids_split = [[prog_ids[i] for i in pg] for pg in prog_ids_split_local]
        # non_informative_ids = set(non_informative_ids)
        # informative_prog_ids = [prog_id for prog_i, prog_id in enumerate(prog_ids) if prog_i not in non_informative_ids]

        all_dupls = self.get_dupl_poss(unique_interactions)
        dupl_prog_ids = [prog_ids_split[i] for i in all_dupls.keys()]
        dupl_semantics = [semantics[i] for i in all_dupls.keys()]
        self.add_dupl_poss(all_dupls, dupl_prog_ids, dupl_semantics)

        nondupl_ids = torch.tensor([int_i for int_i in range(unique_interactions.size(0)) if int_i not in all_dupls], dtype = torch.int)
        nondupl_ints = unique_interactions[nondupl_ids]
        nondupl_prog_ids = [prog_ids_split[i] for i in nondupl_ids]
        nondupl_semantics = [semantics[i] for i in nondupl_ids]
        
        while nondupl_ints.size(0) > 0:
            nondupl_mask = torch.ones(nondupl_ints.size(0), dtype = torch.bool)
            poss = self.get_poss(nondupl_ints)
            new_span_ids = self.get_span_poss(nondupl_ints, poss)  
            new_span_ids_tensor = torch.tensor(new_span_ids, dtype = torch.int)
            nondupl_mask[new_span_ids_tensor] = False

            span_pos = [poss[i] for i in new_span_ids]
            span_ints = nondupl_ints[new_span_ids_tensor]
            span_prog_ids = [nondupl_prog_ids[i] for i in new_span_ids]
            span_semantics = [nondupl_semantics[i] for i in new_span_ids]
            self.add_new_span_poss(span_pos, span_ints, span_prog_ids, span_semantics)

            left_ids = torch.where(nondupl_mask)[0]
            if left_ids.size(0) == 0:
                break
            left_interactions = nondupl_ints[left_ids]
            left_poss = [poss[i] for i in left_ids]
            left_prog_ids = [nondupl_prog_ids[i] for i in left_ids]
            left_semantics = [nondupl_semantics[i] for i in left_ids]
            ext_poss = self.get_ext_poss(left_interactions, left_poss)

            # now we need to insert as many left points as possible 
            ext_poss_lens = torch.tensor([len(ext_pos) for ext_pos in ext_poss], dtype = torch.int)
            min_len_pos_ids = torch.argmin(ext_poss_lens)
            cur_ext_poss = [ext_poss[i] for i in min_len_pos_ids]
            cur_ext_ints = left_interactions[min_len_pos_ids]
            cur_prog_ids = [left_prog_ids[i] for i in min_len_pos_ids]
            cur_semantics = [left_semantics[i] for i in min_len_pos_ids]
            cur_ids = left_ids[min_len_pos_ids]            
            min_len = len(cur_ext_poss[0])
            if min_len == 0: # creation of new dimensions
                # we can pick subset of ints that are pareto non-comparable and create many dims at once.
                nondom_ids = self.get_pareto_noncomparable(cur_ext_ints)
                cur_sel_ids = torch.tensor(nondom_ids, dtype = torch.int)
                for int_i in nondom_ids:
                    indexed_programs = [IndexedProgram(p, s) for p, s in zip(cur_prog_ids[int_i], cur_semantics[int_i])]
                    new_point = CoordinateSystemPoint(cur_ext_ints[int_i], indexed_programs)
                    self.cs.dim_starts.append(new_point)
                    self.cs.dim_ends.append(new_point)
                    self.cs.dim_sizes.append(1)
            else: 
                
                all_possible_extensions = [(point_weight, int_i, dim_id, point) for int_i, prog_dims in enumerate(cur_ext_poss) for dim_id, point_weight, point in prog_dims]
                all_possible_extensions.sort(key = lambda x: x[0], reverse = True)

                collected_exts = []
                covered_dims = {}

                for point_weight, int_i, dim_id, point in all_possible_extensions:
                    if dim_id in covered_dims and covered_dims[dim_id] <= point_weight:
                        continue
                    collected_exts.append((int_i, dim_id, point))
                    covered_dims[dim_id] = point_weight

                cur_sel_ids_list = []
                for int_i, dim_id, point in collected_exts:
                    cur_sel_ids_list.append(int_i)
                    indexed_programs = [IndexedProgram(p, s) for p, s in zip(cur_prog_ids[int_i], cur_semantics[int_i])]
                    new_point = CoordinateSystemPoint(cur_ext_ints[int_i], indexed_programs)
                    if point is None: # new underlying objectiive 
                        uo = self.cs.dim_ends[dim_id]
                        uo.next_point = new_point
                        new_point.prev_point = uo
                        self.cs.dim_ends[dim_id] = new_point
                        self.cs.dim_sizes[dim_id] += 1
                    else: # should insert before point
                        point_prev = point.prev_point 
                        point_prev.next_point = new_point
                        new_point.prev_point = point_prev
                        new_point.next_point = point
                        point.prev_point = new_point
                        self.cs.dim_sizes[dim_id] += 1
                cur_sel_ids = torch.tensor(cur_sel_ids_list, dtype = torch.int)

            nondupl_mask[cur_ids[cur_sel_ids]] = False
            nondupl_ids = torch.where(nondupl_mask)[0]
            nondupl_ints = nondupl_ints[nondupl_ids]
            nondupl_prog_ids = [prog_ids_split[i] for i in nondupl_ids]
            nondupl_semantics = [semantics[i] for i in nondupl_ids]
            self.trim_cs()

        if self.rebuild_size is not None:
            # check counts on UO and corresponding spanned 
            cur_ends = []
            for dim_end in self.cs.dim_ends:
                cur_ends.append(dim_end)
                for span_id in self.cs.span_lists[dim_end]:
                    cur_ends.append(self.cs.span[span_id])
            cur_progs_count = sum(len(p.programs) for p in cur_ends)
            if cur_progs_count >= self.rebuild_size:
                # rebuild with new epsilons 
                cur_ends_semantics = [prog.semantics for p in cur_ends for prog in p.programs]
                stacked_semantics = torch.stack(cur_ends_semantics, dim = 0)
                min_test_vs = torch.min(stacked_semantics, dim=0).values
                max_test_vs = torch.max(stacked_semantics, dim=0).values
                self.epsilons = min_test_vs + (max_test_vs - min_test_vs) / 2.0
                self.rebuild()

    def query(self, semantics: list[torch.Tensor], distance: Optional[CoordinateSystemDistance] = None) -> list[list[IndexedProgram]]:
        ''' Distance from best pos '''
        if distance is None:
            distance = CoordinateSystemDistance()
        # indexed_progs = [[] for _ in range(len(semantics))]
        selected_points = [[] for _ in range(len(semantics))]
        stacked_semantics = torch.stack(semantics, dim = 0)
        interactions = self.get_interactions(stacked_semantics)
        _, unique_interactions, prog_ids_split_local = self.get_unique_interactions(interactions)
        local_dupls = self.get_dupl_poss(unique_interactions)
        for int_i, point in local_dupls.items():
            for prog_i in prog_ids_split_local[int_i]:
                selected_points[prog_i].append(point)

        # dupl_prog_ids = [prog_ids_split_local[i] for i in all_dupls.keys()]
        # dupl_semantics = [semantics[i] for i in all_dupls.keys()]

        nondupl_ids = torch.tensor([int_i for int_i in range(unique_interactions.size(0)) if int_i not in local_dupls], dtype = torch.int)
        nondupl_ints = unique_interactions[nondupl_ids]
        nondupl_prog_ids = [prog_ids_split_local[i] for i in nondupl_ids]
        
        poss = self.get_poss(nondupl_ints)

        if distance.delta_from_max is None: 
            for int_i, prog_pos in enumerate(poss):
                for _, point_weight, point in prog_pos:
                    for prog_i in nondupl_prog_ids[int_i]:
                        selected_points[prog_i].append(point)
                        # indexed_progs[prog_i].extend(point.programs)
        else:
            for int_i, prog_pos in enumerate(poss):
                int_point_w = min([point_weight for _, point_weight, _ in prog_pos]) + distance.delta_from_max
                for _, point_weight, point in prog_pos:
                    if point_weight <= int_point_w:
                        for prog_i in nondupl_prog_ids[int_i]:
                            selected_points[prog_i].append(point)
                            # indexed_progs[prog_i].extend(point.programs)

        selected_span_points = [[] for _ in range(len(semantics))]
        if distance.with_span_points:
            for prog_id, prog_points in enumerate(selected_points):
                for prog_point in prog_points:
                    for span_id in self.cs.span_lists[prog_point]:
                        selected_span_points[prog_id].append(self.cs.span[span_id])
        final_points = selected_points
        if distance.only_span_points:
            final_points = selected_span_points
        elif distance.with_span_points:
            final_points = [a + b for a, b in zip(selected_span_points, selected_points)]

        indexed_progs = [[prog.prog_id for p in ps for prog in p.programs] for ps in final_points]

        return indexed_progs


# NOTE: in test-based approach we are interested to use the coordinate system as spatial index

# @dataclass 
# class PerTestIndex:
#     ''' Stores programs ordered by each tests semantic value '''
#     order: list[list[int]]
#     ''' first dim is test id, second - program id, ordered by semantics[test id] '''

#     #Search is in K log N time, where K is number of tests and N is number of programs in all archive 
#     #Insert is done with insort - O(N) ??? - we would like to avoid this.



@dataclass 
class RuntimeContext2:
    ''' Contains all data necessary for search of syntactic term '''
    
    free_vars: list[torch.Tensor]
    ''' free vars, 1-dim tensors by test '''
    
    gold_outputs: torch.Tensor
    ''' one dimensional - for each test specifies desired outout '''

    op_funcs: list[Callable] 
    ''' List of terminals and non-terminals '''

    op_arities: list[int]
    ''' Defines number of expected args '''

    programs: list[Program]
    ''' Global cache of present programs '''

    semantic_index: SemanticIndex
    ''' Stores good programs. Queried for breeding: selection of parent and selection of replacement'''
        
    # expanded_programs: list[list[list[int]]]
    # ''' first dim is program_id, second - depth level, third - args at that level - prog_ids 
    #     We expand programs gradually during syntax search, from low depth to high depth - to prefer low term complexity'''
    
    # expanded_programs_parent_refs: list[list[list[tuple[int, int]]]]
    # ''' the structure, similar to expanded_programs (prog_id, depth, arg_ids)
    #     Stores index of parent arg on level above current depth and id of arg inside instant parent tree
    #     for d=0, it is 0 as there is one root
    #     Refs are necessary to find path back from deep node to root
    # '''
    
    # search_points: list[list[list[SearchPoint]]]
    # ''' Index program positions by prog_id, depth, and id on that depth'''

    # search_points: list[list[set[int]]]
    # ''' We index search point for each prog_id by depth_level and then id_in_depth.
    #     After attempted search, the point is removed, on expansion to next depth, we add new points
    # '''

    # unexpandable_args: list[list[set[int]]]
    # ''' If we detect that some args should not be expanded, we add them here.
    #     First index - prog_id, second - depth level, third - arg_pos on that level 
    #     Arg in unexpandable when:
    #         1. Previous its variations show no response at tree root (no dependency)
    #         2. ?? Any attempted variation moves away from current position - further from target ?? - makes it worse ==> GP variation will make it worse.
    #         3. ?? Symmetries in target domain, consider tree f(a, b) and f(a1, b) where a and a1 are vary points???
    #         4. ??? Symmetries according to axioms f(a, b) = f(b, a) - current experiments do not assume anything about symbolic relations 
    #         5. ??? Semantic neutrality, terms f1, f2 are semantically same ==> assume that they same anyway ???
    #     ?? - has conterarguments, subject to experiments
    # '''
    
    # semantics: torch.Tensor 
    # ''' device side, outputs of programs, dim1 - program, dim2 - test dim '''

    # ifif necessary? - necessary for CSE method
    # interactions: torch.Tensor #  = field(default_factory=dict)
    # ''' interaction outcome tensor, dim 1 - allocated program, dim 2 - test dim'''    

    syntaxes: dict[tuple, int] 
    ''' Cache for (op_id, *arg_ids) --> existing prog_id, host side construct '''    

    # consts: list[list[torch.Tensor]]
    # ''' Allocated grammar constant ERC accross all programs 
    #     First index is by program id, second - id of constant in the tree, 
    #     tensors are usually 0-d, 1 el
    # '''

    fitness_fns: list[Callable]
    main_fitness_id: int = 0 
    
    # op_counts: torch.Tensor
    # ''' Contains operation numbers inside each program
    #     dim1 - program, dim2 - operation id, cell - count 
    #     Use for count constraint check 
    # '''

    count_constraints: torch.Tensor
    ''' 1-d tensor, allowed number of operations for programs, global constraints
        0-el tensor if not constraints
    '''

    stats: dict[str, Any] = field(default_factory=dict)    

#NOTE: (old) select programs based on observation: (op_id, args) should not be in the syntax
#   Alg1: pick op with smallest op_attempts  --> bad algo??? 
#         then pick least used programs ?? 
#   Alg2: pick batch at random --> filter on CPU and --> filter new -- dummy 
#   Alg3: find outputs for fn that would bring closer to the target (geometric semantic) - output space, but use interaction outcome space 
#   Alg4: Inverse semantic?? - SBP 
#   Alg5: Search semantics with optimizer and then find closest
#   Alg6: Statistical effect of unknown fn on n inputs? How varying of input affects the output?? Gradients? 

# def no_const_optim(semantics: torch.Tensor, consts: list[torch.Tensor]):
#     return semantics

# def prog_unwrap_once(prog_id: int, *, runtime_context: RuntimeContext):
#     ''' Programs are stored as (op, immediate args) in runtime_context.
#         Unwraps are stored in runtime_context.expanded_programs
#         This function expand the current depth level: d --> d + 1
#         prog_id should exist in runtime_context
#         Example of unwrap chain:
#         d=0 [prog_id] --> 
#         d=1 [arg_prog_id1, arg_prog_id2] --> 
#         d=2 [arg_prog_id3, arg_prog_id4, arg_prog_id5, arg_prog_id6] --> ...
#     '''

#     cur_exp_depth = runtime_context.expanded_programs[prog_id]
#     # NOTE: originally when d=0, cur_exp_depth is == runtime_context.program_args[prog_id] - invariant
#     at_depth_arg_prog_ids = cur_exp_depth[-1]
#     at_depth_unexpandable = runtime_context.unexpandable_args[prog_id][-1]
#     next_level_nodes = \
#         [ ((arg_i, arg_arg_i), arg_arg_id) for arg_i, arg_prog_id in enumerate(at_depth_arg_prog_ids) 
#             if arg_i not in at_depth_unexpandable
#             for arg_arg_i, arg_arg_id in enumerate(runtime_context.program_args[arg_prog_id]) ]
#     if len(next_level_nodes) == 0:
#         return False 
#     parent_refs, new_level_args = zip(*next_level_nodes)
#     runtime_context.expanded_programs[prog_id].append(new_level_args)
#     runtime_context.expanded_programs_parent_refs[prog_id].append(parent_refs)
#     new_search_points = set(range(len(new_level_args)))
#     runtime_context.search_points[prog_id].append(new_search_points)
#     runtime_context.unexpandable_args[prog_id].append(set())
#     return True

# def prog_unwrap_to_depth(prog_id: int, target_depth: int, *, runtime_context: RuntimeContext):
#     ''' Unwraps program to target depth, 
#         prog_id should exist in runtime_context
#     '''
#     while len(runtime_context.expanded_programs[prog_id]) < target_depth:
#         if not prog_unwrap_once(prog_id, runtime_context = runtime_context):
#             break

def rand_point(search_points: list[SearchPoint]) -> int:
    probas = np.array([sp.select_koef for sp in search_points])
    probas /= np.sum(probas)
    search_point_id = default_rnd.choice(len(search_points), p = probas)
    return search_point_id

def build_preorder(prog_id: int, runtime_context: RuntimeContext2):
    prog = runtime_context.programs[prog_id]
    if len(prog.linear_preorder) == 0:
        # expand program to linear preorder 
        i = 0 
        prog_to_add = [(prog_id, 0, 0.0)]
        while i < len(prog_to_add):
            cur_prog_id, arg_i, koef = prog_to_add[i]
            search_point = SearchPoint(cur_prog_id, i, arg_i, koef)
            prog.linear_preorder.append(search_point)
            for arg_i, arg_id in enumerate(runtime_context.programs[cur_prog_id].args):
                prog_to_add.append((arg_id, arg_i, 1.0))
            i += 1    
    return prog.linear_preorder

def find_path_to_root(select_point_id: int, linear_preorder: list[SearchPoint]) -> list[int]:
    path = [select_point_id]
    while path[-1] > 0:
        parent_point_id = linear_preorder[path[-1]].parent_i
        path.append(parent_point_id)
    return path

def select_path(prog_id: int, *, select_fn = rand_point, runtime_context: RuntimeContext2) -> list[int]:
    linear_preorder = build_preorder(prog_id, runtime_context)
    select_point_id = select_fn(linear_preorder)
    # now we need to build path to root 
    path = find_path_to_root(select_point_id, linear_preorder)
    return path

def eval_paths(prog_id: int, new_paths: list[list[int]], new_args: list[torch.Tensor],
                *, runtime_context: RuntimeContext2) -> torch.Tensor:
    ''' Run many paths at once     
        new_args: define the pathes for new tensors in linear_preorder '''

    linear_preorder = build_preorder(prog_id, runtime_context)
    path_sps = [[linear_preorder[path_i] for path_i in path] for path in new_paths]
    while len(path_sps) > 0:
        path_i = max(range(len(path_sps)), key = lambda x: len(x))
        path: list[SearchPoint] = path_sps[path_i]
        arg_sp = path[0]
        parent_program: Program = runtime_context.programs[linear_preorder[arg_sp.parent_i].prog_id]
        fn = runtime_context.op_funcs[parent_program.prog_op]
        fn_args = [] 

        mergin_paths = [p_id for p_id, arg_sp2 in enumerate(path_sps) if arg_sp2[0].parent_i == arg_sp.parent_i]
        parent_arg_is = {path_sps[p_id][0].parent_arg_i: p_id for p_id in mergin_paths}

        for parent_arg_i, parent_arg in enumerate(parent_program.args):
            if parent_arg_i in parent_arg_is:
                fn_args.append(new_args[parent_arg_is[parent_arg_i]])
            else:
                fn_args.append(runtime_context.programs[parent_arg].semantics)

        new_instant_semantics = fn(*fn_args)
        for p_id in mergin_paths:
            path_sps.pop(p_id)
            new_args.pop(p_id)
        path.pop(0)
        if len(path) > 1:
            path_sps.append(path)
            new_args.append(new_instant_semantics)

    return new_instant_semantics # new semantics at root    

def mutate_programs(prog_id: int, new_paths: list[list[int]], new_args: list[int], *, runtime_context: RuntimeContext2) -> torch.Tensor:
    ''' Like eval_path, creates new syntaxes which yet to be optimized by const adjustments '''
    new_programs = []
    linear_preorder = build_preorder(prog_id, runtime_context)
    path_sps = [[linear_preorder[path_i] for path_i in path] for path in new_paths]
    while len(path_sps) > 0:
        path_i = max(range(len(path_sps)), key = lambda x: len(x))
        path: list[SearchPoint] = path_sps[path_i]
        arg_sp = path[0]
        parent_program: Program = runtime_context.programs[linear_preorder[arg_sp.parent_i].prog_id]
        # fn = runtime_context.op_funcs[parent_program.prog_op]
        args = [] 

        mergin_paths = [p_id for p_id, arg_sp2 in enumerate(path_sps) if arg_sp2[0].parent_i == arg_sp.parent_i]
        parent_arg_is = {path_sps[p_id][0].parent_arg_i: p_id for p_id in mergin_paths}

        for parent_arg_i, parent_arg in enumerate(parent_program.args):
            if parent_arg_i in parent_arg_is:
                args.append(new_args[parent_arg_is[parent_arg_i]])
            else:
                args.append(parent_arg)

        syntax_key = (parent_program.prog_op, *args)

        if syntax_key in runtime_context.syntaxes:
            new_prog_id = runtime_context.syntaxes[syntax_key]
        else:
            op_counts = torch.zeros(len(runtime_context.op_funcs), dtype = torch.int32)
            for arg in args:
                op_counts += runtime_context.programs[arg].op_counts
            op_counts[parent_program.prog_op] += 1
            if torch.any(op_counts > runtime_context.count_constraints):
                runtime_context.stats['count_constraints_violated'] = runtime_context.stats.get('count_constraints_violated', 0) + 1
                new_prog_id = None
            else:
                new_prog_id = len(runtime_context.programs)
                new_programs.append(new_prog_id)
                runtime_context.syntaxes[syntax_key] = new_prog_id
                consts = {}
                size = 1
                depths = []
                for arg in args:
                    arg_prog = runtime_context.programs[arg]
                    if arg_prog.prog_op == OP_CONST:
                        consts[size] = ProgramConst(arg_prog.semantics.clone(), [])
                    else:
                        for arg_const_point_id, const in arg_prog.consts.items():
                            new_const_value = const.value.clone()                        
                            consts[arg_const_point_id + size] = ProgramConst(new_const_value, [])
                    size += len(arg_prog.size)
                    depths.append(arg_prog.depth)

                depth = max(depths) + 1
                new_program = Program(parent_program.prog_op, args, [], 
                                      size, depth, torch.tensor(0), consts, op_counts)
                runtime_context.programs.append(new_program)

        for p_id in mergin_paths:
            path_sps.pop(p_id)
            new_args.pop(p_id)
        path.pop(0)
        if len(path) > 1 and new_prog_id is not None:
            path_sps.append(path)
            new_args.append(new_prog_id)

    return new_programs # not yet evaluated, in order from most deep to least deep

def no_optim(semantics: torch.Tensor, vars: list[torch.Tensor], *, runtime_context: RuntimeContext2):
    return semantics

def optim_and_eval_programs(prog_ids: list[int], *, optim_fn = no_optim, runtime_context: RuntimeContext2):
    ''' Programs are in order from deepest to shallowest '''
    for prog_id in prog_ids:
        prog = runtime_context.programs[prog_id]
        linear_preorder = build_preorder(prog_id, runtime_context)
        for const_point_id, const in prog.consts.items():
            const.path = find_path_to_root(const_point_id, linear_preorder)
        const_paths = [const.path for const in prog.consts.values()]
        const_values = [const.value for const in prog.consts.values()]
        if len(const_values) == 0: # no constants - exec root
            fn = runtime_context.op_funcs[prog.prog_op]
            optim_semantics = fn(*(runtime_context.programs[arg].semantics for arg in prog.args))
        else:
            new_semantics = eval_paths(prog_id, const_paths, const_values, runtime_context = runtime_context)
            optim_semantics = optim_fn(new_semantics, const_values, runtime_context = runtime_context) #const values are adjusted at place
        prog.semantics = optim_semantics

def find_closest(semantics: torch.Tensor, target: torch.Tensor, *, runtime_context: RuntimeContext2):
    # NOTE: we need some indexes in runtime_context
    runtime_context.semantic_index.query()
    # TODO
   
def init_semantics(*, runtime_context: RuntimeContext2):
    ''' Build all depth 0 and depth 1 trees as starting point and optimizes them 
        Adds the result to initial index and context
    '''    

    start_programs = []
    for var_semantics in runtime_context.free_vars:
        program = Program(OP_VAR, [], [])


    runtime_context.programs.app

    runtime_context.programs
    start_idx = runtime_context.num_programs
    end_idx = start_idx + runtime_context.free_vars.shape[0]
    runtime_context.num_programs = end_idx
    runtime_context.outputs[start_idx:end_idx] = runtime_context.free_vars
    runtime_context.output_to_program[start_idx:end_idx, 0] = 100 + torch.arange(0, runtime_context.free_vars.shape[0])

    if runtime_context.consts_optim is not None:
        last_const = runtime_context.consts[-1]
        new_const = runtime_context.consts_optim(last_const)
        runtime_context.consts.append(new_const)

    start_idx = runtime_context.num_programs
    end_idx = start_idx + len(runtime_context.consts)
    runtime_context.num_programs = end_idx
    for const_id, const_tensor in enumerate(runtime_context.consts):
        runtime_context.outputs[start_idx + const_id] = const_tensor
    runtime_context.output_to_program[start_idx:end_idx, 0] = 10000000 + torch.arange(0, len(runtime_context.consts))

    return
    
def _select_node_id(in_node: Node, filter, select_node_leaf_prob = None) -> Optional[Node]:
    if select_node_leaf_prob is None: 
        places = [(n, i, at_d) for i, (at_d, n) in enumerate(in_node.get_nodes()) if filter(at_d, n) ]
        if len(places) == 0:
            return None, None, None
        selected_idx = default_rnd.choice(len(places))
        selected = places[selected_idx]
    else:
        nonleaves = []
        leaves = []
        for i, (at_d, n) in enumerate(in_node.get_nodes()):
            if filter(at_d, n):
                if n.is_leaf():
                    leaves.append((n, i, at_d))
                else:
                    nonleaves.append((n, i, at_d))
        if len(nonleaves) == 0 and len(leaves) == 0:
            return None, None, None
        if (default_rnd.rand() < select_node_leaf_prob and len(leaves) > 0) or len(nonleaves) == 0:
            selected_idx = default_rnd.choice(len(leaves))
            selected = leaves[selected_idx]
        else:
            selected_idx = default_rnd.choice(len(nonleaves))
            selected = nonleaves[selected_idx]
    return selected

# NOTE: first we do select and then gen muation tree
# TODO: later add to grow and full type constraints on return type
# IDEA: dropout in GP, frozen tree positions which cannot be mutated or crossovered - for later
def subtree_mutation(node, select_node_leaf_prob = 0.1, tree_max_depth = 17, repl_fn = replace_positions, *, runtime_context: RuntimeContext2):
    position, position_id, position_depth = _select_node_id(node, lambda d, n: True, select_node_leaf_prob = select_node_leaf_prob)
    if position is None:
        return node    
    position_func_counts = get_func_counts(position, runtime_context.counts_constraints, runtime_context.counts_cache)
    grow_depth = min(5, tree_max_depth - position_depth)
    if runtime_context.counts_constraints is None:
        grow_counts_constraints = None
    else:
        grow_counts_constraints = {}
        for k, v in runtime_context.counts_constraints.items():
            grow_counts_constraints[k] = v - position_func_counts.get(k, 0)
    new_node = grow(grow_depth = grow_depth, func_list = runtime_context.func_list, terminal_list = runtime_context.terminal_list, 
                    counts_constraints = grow_counts_constraints, grow_leaf_prob = None, node_builder = runtime_context.node_builder)
    # new_node_depth = new_node.get_depth()
    # at_depth, at_node = select_node(leaf_prob, node, lambda d, n: (d > 0) and n.is_of_type(new_node), 
    #                                     lambda d, n: (d + new_node_depth) <= max_depth)
    if new_node is None:
        return node
    res = repl_fn(node, {position_id: new_node}, node_builder = runtime_context.node_builder)
    return res

def no_mutation(node):
    return node
        
def subtree_crossover(parent1: Node, parent2: Node, select_node_leaf_prob = 0.1, tree_max_depth = 17, 
                      repl_fn = replace_positions, *, runtime_context: RuntimeContext2):
    ''' Crossover two trees '''
    # NOTE: we can crossover root nodes
    # if parent1.get_depth() == 0 or parent2.get_depth() == 0:
    #     return parent1, parent2
    parent1, parent2 = sorted([parent1, parent2], key = lambda x: x.get_depth())
    # for _ in range(3):
    # at1_at_depth, at1 = select_node(leaf_prob, parent1, lambda d, n: (d > 0), lambda d, n: True)
    at1, at1_id, at1_at_depth = _select_node_id(parent1, lambda d, n: True, select_node_leaf_prob=select_node_leaf_prob)
    if at1_id is None:
        return parent1, parent2
    # at1_at_depth, at1 = parent1.get_node(at1_id)
    at1_depth = at1.get_depth()
    at2, at2_id, at2_at_depth = _select_node_id(parent2, 
                        lambda d, n: n.is_of_type(at1) and at1.is_of_type(n) and ((n.get_depth() + at1_at_depth) <= tree_max_depth) and (at1_at_depth > 0 or d > 0) and ((d + at1_depth) <= tree_max_depth) \
                                            and are_counts_constraints_satisfied_together(n, at1, runtime_context.counts_constraints, runtime_context.counts_cache), 
                        select_node_leaf_prob=select_node_leaf_prob)
    # at2_depth, at2
    # at2_depth, at2 = select_node(leaf_prob, parent2, 
    #                     lambda d, n: (d > 0) and n.is_of_type(at1) and at1.is_of_type(n), 
    #                     lambda d, n: ((d + at1_depth) <= max_depth) and ((n.get_depth() + at1_at_depth) <= max_depth))
    if at2_id is None:
        # NOTE: should not be here
        # continue # try another pos
        return parent1, parent2 
        # return parent1, parent2
    child1 = repl_fn(parent1, {at1_id: at2}, node_builder = runtime_context.node_builder)
    child2 = repl_fn(parent2, {at2_id: at1}, node_builder = runtime_context.node_builder)
    return child1, child2       

def subtree_breed(size, population, fitnesses,
                    breed_select_fn = tournament_selection, mutation_fn = subtree_mutation, crossover_fn = subtree_crossover,
                    mutation_rate = 0.1, crossover_rate = 0.9, *, runtime_context: RuntimeContext2):
    new_population = []
    runtime_context.parent_child_relations = []
    if runtime_context.select_fitness_ids is not None and fitnesses is not None:
        fitnesses = fitnesses[:, runtime_context.select_fitness_ids]
    collect_parents = ("syntax" in runtime_context.breeding_stats) or ("semantics" in runtime_context.breeding_stats)
    all_parents = []
    while len(new_population) < size:
        # Select parents for the next generation
        parent1_id = breed_select_fn(population, fitnesses, runtime_context = runtime_context)
        parent2_id = breed_select_fn(population, fitnesses, runtime_context = runtime_context)
        parent1 = population[parent1_id]
        parent2 = population[parent2_id]
        if default_rnd.rand() < mutation_rate:
            child1 = mutation_fn(parent1, runtime_context = runtime_context)
        else:
            child1 = parent1
        if default_rnd.rand() < mutation_rate:
            child2 = mutation_fn(parent2, runtime_context = runtime_context)
        else:
            child2 = parent2
        if default_rnd.rand() < crossover_rate:
            child1, child2 = crossover_fn(child1, child2, runtime_context = runtime_context)   
        runtime_context.parent_child_relations.append(([parent1, parent2], [child1, child2]))
        if collect_parents:
            all_parents.extend((parent1, parent2))
        new_population.extend([child1, child2])
    for parent in all_parents:
        if 'syntax' in runtime_context.breeding_stats:
            runtime_context.breeding_stats['syntax'][parent] = runtime_context.breeding_stats['syntax'].get(parent, 0) + 1 
        if 'semantics' in runtime_context.breeding_stats:
            parent_ints = tuple(runtime_context.int_cache[parent])
            runtime_context.breeding_stats['semantics'][parent_ints] = runtime_context.breeding_stats['semantics'].get(parent_ints, 0) + 1 
    return new_population
    
def depth_fitness(interactions, outputs, population = [], **_):
    return [p.get_depth() for p in population]

def hamming_distance_fitness(interactions, outputs, **_):
    return np.sum(1 - interactions, axis = 1)

from torch.nn.functional import mse_loss
def mse_fitness(interaction, outputs, *, gold_outputs, **_):
    losses = []
    for output in outputs:
        loss = mse_loss(torch.tensor(output), gold_outputs)
        losses.append(loss.item())
    res =  np.array(losses)
    res[np.isnan(res)] = np.inf
    return res

def ifs_fitness(interactions, outputs, **_):
    counts = (np.sum(interactions, axis = 0) * interactions).astype(float)
    counts[counts > 0] = 1.0 / counts[counts > 0]
    ifs = np.sum(counts, axis=1)
    return -ifs

# TODO: age fitness: group of fitness functions 
# aging could be simulated differently 
# most interesting case for us is aging of semantics with number of attempted breedings in coevol
# default aging could consider only syntactic tree and number of  generations it exists in lexicographic tournament selection

# def ifs_fitness_fn(interactions, **kwargs):
#     counts = np.sum((interactions[:, None] == interactions) & (interactions == 1), axis = 0).astype(float)
#     counts[counts > 0] = 1 / counts[counts > 0]
#     ifs = np.sum(counts, axis=1)
#     return ifs


# interactions = np.array([[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]])
# ifss = []
# for i in range(interactions.shape[0]):
#     r = []
#     for t in range(interactions.shape[1]):
#         if interactions[i][t] == 1:
#             r.append(np.sum(interactions[:, t] == 1))
#         else:
#             r.append(np.inf)
#     ifss.append(r)
# ifss2 = np.array(ifss)
# np.sum(1 / ifss2, axis=1)

# ifs_fitness_fn(interactions)

# ifss.append(ifs_fitness(0, interactions, eval_idxs = [i]))

# ifs_fitness(0, np.array([[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 1]]))

def collect_additional_stats(stats: dict[str, Any], nodes: list[Node], outputs):
    syntax_counts = {}
    sem_counts = {}
    sem_repr_counts = {}
    for node_id, node in enumerate(nodes):
        # first, syntax stats - number of times each syntax was evaluated
        node_str = str(node)
        syntax_counts[node_str] = syntax_counts.get(node_str, 0) + 1
        # second, number of times semantics appears in evaluation
        node_sem = tuple(outputs[node_id])
        sem_counts[node_sem] = sem_counts.get(node_sem, 0) + 1
        # third, how many representatives this semantics has
        sem_repr_counts.setdefault(node_sem, set()).add(node_str)
    stats.setdefault('stats_nodes', []).append(len(nodes))

    syntax_dupl_rate = sum(c - 1 for c in syntax_counts.values())
    stats.setdefault('syntax_dupl', []).append(syntax_dupl_rate)

    sem_dupl_rate = sum(c - 1 for c in sem_counts.values())
    stats.setdefault('sem_dupl', []).append(sem_dupl_rate)

    sem_repr_rate = np.mean([len(c) for c in sem_repr_counts.values()])
    stats.setdefault('sem_repr_rate', []).append(round(sem_repr_rate, 2))

    num_uniq_syntaxes = len(syntax_counts)
    stats.setdefault('num_uniq_syntaxes', []).append(num_uniq_syntaxes)

    num_uniq_sems = len(sem_counts)
    stats.setdefault('num_uniq_sems', []).append(num_uniq_sems)

def exact_best(fitness_value):
    return fitness_value == 0

def approx_best(fitness_value, epsilon = 1e-6):
    return fitness_value < epsilon

def analyze_population(population, outputs, fitnesses, save_stats = True, best_cond = exact_best, *, runtime_context: RuntimeContext2, **_):
    ''' Get the best program in the population '''
    stats = runtime_context.stats
    fitness_order = np.lexsort(fitnesses.T[::-1])
    best_index = fitness_order[0]
    best_fitness = fitnesses[best_index]
    best = population[best_index]
    stats['best'] = str(best)
    is_best = False 
    if (runtime_context.main_fitness_fn is None) and (len(runtime_context.fitness_fns) > 0):
        main_fitness_fn = runtime_context.fitness_fns[0]
    else:
        main_fitness_fn = runtime_context.main_fitness_fn
    for fitness_idx, fitness_fn in enumerate(runtime_context.fitness_fns):
        if fitness_fn == main_fitness_fn:
            is_best = best_cond(best_fitness[fitness_idx])
        stats.setdefault(fitness_fn.__name__, []).append(best_fitness[fitness_idx])
    if save_stats:
        collect_additional_stats(stats, population, outputs)
        total_best_ch = 0
        total_good_ch = 0 
        total_best_dom_ch = 0
        total_good_dom_ch = 0
        total_bad_ch = 0
        for parents, children in runtime_context.parent_child_relations:
            parent_ints = np.array([ runtime_context.int_cache[n] for n in parents ])
            child_ints = np.array([ runtime_context.int_cache[n] for n in children ])
            best_ch, good_ch, best_dom_ch, good_dom_ch, bad_ch = count_good_bad_children(parent_ints, child_ints)
            total_best_ch += best_ch
            total_good_ch += good_ch
            total_best_dom_ch += best_dom_ch
            total_good_dom_ch += good_dom_ch
            total_bad_ch += bad_ch
        # if total_good_ch > 0 or total_bad_ch > 0:
        stats.setdefault('best_children', []).append(total_best_ch)
        stats.setdefault('good_children', []).append(total_good_ch)
        stats.setdefault('best_dom_children', []).append(total_best_dom_ch)
        stats.setdefault('good_dom_children', []).append(total_good_dom_ch)
        stats.setdefault('bad_children', []).append(total_bad_ch)
    if is_best:
        return population[best_index]
    return None

def evol_loop(max_evals, runtime_context: RuntimeContext2):
    ''' Classic evolution loop '''
    init_semantics(runtime_context = runtime_context)
    gen = 0
    best_ind = None
    while runtime_context.num_programs < max_evals:
        best_ind = analyze_pop_fn(population, outputs, fitnesses)
        if best_ind is not None:
            break        
        population = breed_fn(population_size, population, fitnesses)  
        outputs, fitnesses, *_ = eval_fn(population)
        gen += 1
    
    return best_ind, gen

def evolve(problem_init, *,
                population_size = 1000, max_gens = 100,
                fitness_fns = [hamming_distance_fitness, depth_fitness], main_fitness_fn = hamming_distance_fitness,
                select_fitness_ids = None,
                init_fn = init_each, map_fn = identity_map, breed_fn = subtree_breed, 
                eval_fn = gp_eval, analyze_pop_fn = analyze_population,
                create_runtime_context = RuntimeContext2): 
    runtime_context = create_runtime_context()
    problem_init(runtime_context = runtime_context)
    evo_funcs = [init_fn, map_fn, breed_fn, eval_fn, analyze_pop_fn]
    evo_funcs_bound = [partial(fn, runtime_context = runtime_context) for fn in evo_funcs]
    best_ind, gen = evol_loop(population_size, max_gens, *evo_funcs_bound)
    runtime_context.stats["gen"] = gen 
    runtime_context.stats["best_found"] = best_ind is not None
    return best_ind, runtime_context.stats

gp = koza_evolve
gp_0 = partial(koza_evolve, select_fitness_ids = [0])

ifs = partial(koza_evolve, fitness_fns = [ifs_fitness, hamming_distance_fitness, depth_fitness])
ifs_0 = partial(ifs, select_fitness_ids = [0, 1])

gp_a = partial(koza_evolve, fitness_fns = [mse_fitness, depth_fitness], main_fitness_fn = mse_fitness,
                            eval_fn = partial(gp_eval, int_fn = dist_test_based_interactions, 
                                              output_prep = torch_output_prep),
                            analyze_pop_fn = partial(analyze_population, best_cond = approx_best))

gp_sim_names = [ 'gp', 'ifs', 'gp_0', 'ifs_0' ]

if __name__ == '__main__':
    import gp_benchmarks
    problem_builder = gp_benchmarks.get_benchmark('cmp8')
    best_prog, stats = gp_0(problem_builder)
    print(best_prog)
    print(stats)
    pass    
