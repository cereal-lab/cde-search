''' Defines how good a sample is relative to exising dimension space '''

from typing import Any, Iterable


def dimension_coverage(axes: list[list[set[Any]]], sample: list[Any]):
    ''' Computes DC metric based on DECA axes and given tests sample (ex., last population of P-PHC)
        :param tests_sample - ids of tests in the space 
        :returns percent of axes covered by sample
    '''
    if len(axes) == 0:
        return 0
    sample_axes = [axis for axis in axes if any(t in point for point in axis for t in sample)]
    dim_coverage = len(sample_axes) / len(axes)
    return dim_coverage

def avg_rank_of_repr(axes: list[list[set[Any]]], sample:list[Any]):
    ''' Computes ARR for a given sample 
        :param tests_sample - set of tests from the space
        :returns - tuple of ARR and ARRA
    '''
    if len(axes) == 0:
        return (0, 0)
    axes_ranks = [next((point_id + 1 for point_id, point in reversed(list(enumerate(axis))) 
                        if any(t in point for t in sample)), 0) / len(axis)
                    for axis in axes]
    axes_ranks_no_zero = [r for r in axes_ranks if r > 0]
    arr = 0 if len(axes_ranks_no_zero) == 0 else (sum(axes_ranks_no_zero) / len(axes_ranks_no_zero))
    arra = sum(axes_ranks) / len(axes_ranks)
    return (arr, arra)

def redundancy(spanned: dict[Any, dict[int, int]], sample:list[Any]):
    ''' Computes redundancy - percent of spanned points in the selection 
        :param tests_sample - set of tests from the space
        :returns redundancy of the sample
    '''
    if len(sample) == 0:
        return 0
    spanned_sample = [t for t in sample if t in spanned]
    R = len(spanned_sample) / len(sample) 
    return R

def duplication(axes: list[list[set[Any]]], spanned: dict[Any, dict[int, int]], origin: set[Any], sample:list[Any]):
    ''' Computes percent of dumplications in tests sample according to CDE space
        :param tests_sample - set of tests from the space
    '''
    if len(axes) == 0 or len(sample) == 0:
        return 0
    space_tests = { **{t:((axis_id, point_id),)
                        for axis_id, axis in enumerate(axes) for point_id, point in enumerate(axis)
                        for t in point},
                    **{t:tuple(sorted(spanned.items())) for t, spanned in spanned.items()},
                    **{t:() for t in origin}}
    Dup = 0 
    present = set()
    for t in sample:
        pos = space_tests[t]
        if pos in present:
            Dup += 1 
        else:
            present.add(pos)
    Dup = Dup / len(sample)
    return Dup

def trivial(origin: set[Any], sample:list[Any]):
    ''' percent of population that is not represented by axes or spanned (in space['zero']) '''
    if len(sample) == 0:
        return 0
    noninfo = [t for t in origin if t in sample]
    nonI = len(noninfo) / len(sample)
    return nonI  