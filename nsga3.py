''' based on pymoo - NSGA3 extends NSGA2 with reference directions - we do not implement them manually '''

from functools import partial
import numpy as np
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival #NSGA3
from pymoo.util.ref_dirs import get_reference_directions

from gp import analyze_population, create_runtime_context, depth_fitness, gp_eval, hamming_distance_fitness, \
    identity_map, init_each, subtree_breed
from nsga2 import doc_objectives, dof_w_objectives, dof_wh_objectives, full_objectives, get_pareto_front_indexes, hypervolume_fitness, weighted_hypervolume_fitness
from pymoo.core.problem import Problem
from pymoo.core.population import Population, interleaving_args
from pymoo.core.individual import Individual

# NOTE: The following code is taken from pymoo source: https://github.com/anyoptimization/pymoo/blob/458e29b713c7676ab1c970c1f07f4f3f41045e0f/pymoo/algorithms/moo/nsga3.py#L124
# NOTE: NSGA-3 = NSGA-2 + RefDirections prceossing to do niching 

# population_size = 1000
# nsga3 = NSGA3(pop_size=population_size, ref_dirs=ref_dirs)

# class CustomPopulation(Population):
#     ''' Customize the behavior of get and set - cached values of F '''
#     def __new__(cls, *args, **kwargs):
#         instance = super().__new__(cls, *args)
#         instance.attrs = kwargs
#         return instance

#     def has(self, key):
#         return key in self.attrs
    
#     def set(self, *args, **kwargs):
#         kwargs = interleaving_args(*args, kwargs=kwargs)
#         for k, v in kwargs.items():
#             self.attrs[k] = v 

#     def get(self, *args, to_numpy=True, **kwargs):
#         to_return = []
#         for a in args:
#             to_return.append(self.attrs[a])
#         if len(to_return) == 1:
#             return to_return[0]
#         else:
#             return tuple(to_return)

def nsga3_loop(archive_size, population_size, max_gens, 
              init_fn, map_fn, breed_fn, eval_fn, analyze_pop_fn):
    """ Run NSGA-III algorithm. """
    # Create initial population
    population = init_fn(population_size)
    archive = []
    gen = 0
    best_ind = None
    survive_strategy = None
    fake_problem = Problem()
    prev_n_objs = None
    while gen < max_gens:
        all_inds = population + archive
        all_inds = map_fn(all_inds)
        outputs, fitnesses, interactions, derived_objectives = eval_fn(all_inds) 
        if len(all_inds) > archive_size:
            # new_archive = []
            # all_fronts_indicies = np.array([], dtype=int)
            # fronts = []
            # fronts_size = 0
            # while fronts_size < archive_size:
            #     new_front_indices = get_pareto_front_indexes(derived_objectives, exclude_indexes = all_fronts_indicies)
            #     if len(new_front_indices) == 0:
            #         break            
            #     fronts.append(new_front_indices)
            #     fronts_size += len(new_front_indices)
            #     all_fronts_indicies = np.concatenate([all_fronts_indicies, new_front_indices])
            #     # new_front = [all_inds[i] for i in new_front_indices]
            #     # # if best_front_indexes is None:
            #     # #     best_front_indexes = new_front_indices
            #     # if len(new_archive) + len(new_front_indices) <= archive_size:
            #     #     all_fronts_indicies = np.concatenate([all_fronts_indicies, new_front_indices])
            #     #     new_archive += new_front
            #     # else:
            #     #     left_to_take = archive_size - len(new_archive)
            #     #     front_sparsity = get_sparsity(derived_objectives[new_front_indices])
            #     #     front_id_idx = np.argsort(front_sparsity)[-left_to_take:]
            #     #     front_indexes = new_front_indices[front_id_idx]
            #     #     all_fronts_indicies = np.concatenate([all_fronts_indicies, front_indexes])
            #     #     new_archive += [all_inds[i] for i in front_indexes]
            #     #     break  


            # survive_strategy =  ReferenceDirectionSurvival(ref_dirs)
            moo_F = 1. - derived_objectives # NOTE: here derived objectives cannot be higher than 1
            pop = Population([Individual(ind = ind) for ind in all_inds])
            pop.set("F", moo_F) # because pymoo minimizes F
            n_objs = moo_F.shape[1]
            if survive_strategy is None or prev_n_objs != n_objs: 
                # for many objectives, computing alot of directions are expensive
                prev_n_objs = n_objs
                n_partitions = 2
                if n_objs < 100:
                    n_partitions = 3
                if n_objs < 10:
                    n_partitions = 5
                if n_objs < 5:
                    n_partitions = 10
                ref_dirs = get_reference_directions("das-dennis", n_objs, n_partitions=n_partitions)
                survive_strategy = ReferenceDirectionSurvival(ref_dirs)
            nsga3_survived_ids = survive_strategy.do(fake_problem, pop, n_survive = archive_size, return_indices=True)
            pass
            # non_dominated, last_front = fronts[0], fronts[-1]

            # ideal, _, nadir = hyperplane_normalize(derived_objectives, non_dominated)
            
            archive = [all_inds[i] for i in nsga3_survived_ids]
            archive_fitnesses = fitnesses[nsga3_survived_ids]
            archive_interactions = interactions[nsga3_survived_ids]
            archive_outputs = outputs[nsga3_survived_ids]
        else:
            archive = all_inds
            archive_fitnesses = fitnesses
            archive_interactions = interactions
            archive_outputs = outputs
        # best_front = [all_inds[i] for i in best_front_indexes]
        # best_front_fitnesses = fitnesses[best_front_indexes]
        # best_front_outcomes = outputs[best_front_indexes]
        best_ind = analyze_pop_fn(archive, archive_outputs, archive_fitnesses)
        if best_ind is not None:
            break
        population = breed_fn(population_size, archive, archive_fitnesses, archive_interactions)
        gen += 1
    return best_ind, gen


def gp_nsga3(problem_init, *,
                        population_size = 1000, max_gens = 100, archive_size = 1000,
                        fitness_fns = [hamming_distance_fitness, depth_fitness], main_fitness_fn = hamming_distance_fitness,
                        select_fitness_ids = None, init_fn = init_each, map_fn = identity_map, breed_fn = subtree_breed, 
                        eval_fn = gp_eval, analyze_pop_fn = analyze_population, derive_objs_fn = full_objectives):
    eval_fn = partial(eval_fn, derive_objs_fn = derive_objs_fn)
    runtime_context = create_runtime_context(fitness_fns, main_fitness_fn, select_fitness_ids)
    problem_init(runtime_context = runtime_context)
    evo_funcs = [init_fn, map_fn, breed_fn, eval_fn, analyze_pop_fn]
    evo_funcs_bound = [partial(fn, runtime_context = runtime_context) for fn in evo_funcs]    

    best_ind, gen = nsga3_loop(archive_size, population_size, max_gens, *evo_funcs_bound)
    runtime_context.stats["gen"] = gen
    runtime_context.stats["best_found"] = best_ind is not None
    return best_ind, runtime_context.stats

nsga3_all = partial(gp_nsga3, select_fitness_ids = [0])
nsga3_doc_p = partial(gp_nsga3, derive_objs_fn = doc_objectives, select_fitness_ids = [0, 1],
                fitness_fns = [hypervolume_fitness, hamming_distance_fitness, depth_fitness])

nsga3_doc_d = partial(gp_nsga3, derive_objs_fn = doc_objectives, select_fitness_ids = [0, 1],
                fitness_fns = [weighted_hypervolume_fitness, hamming_distance_fitness, depth_fitness])

nsga3_dof_w_3 = partial(gp_nsga3, derive_objs_fn = partial(dof_w_objectives, k = 3, alpha = 1), 
                        select_fitness_ids = [0])

nsga3_dof_wh_3 = partial(gp_nsga3, derive_objs_fn = partial(dof_wh_objectives, k = 3, alpha = 1), 
                         select_fitness_ids = [0])

nsga3_sim_names = [ 'nsga3_all', 'nsga3_doc_p', 'nsga3_doc_d', 'nsga3_dof_w_3', 'nsga3_dof_wh_3' ]

if __name__ == '__main__':
    import gp_benchmarks
    problem_builder = gp_benchmarks.get_benchmark('cmp8')
    best_prog, stats = nsga3_doc_p(problem_builder)
    print(best_prog)
    print(stats)
    pass    
