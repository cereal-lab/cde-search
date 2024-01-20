''' Implementation of entry points 
    CLI for generating CDE space and running games (Number, CDESpace based)
'''

#TODO: specification for simulation: candidate interracts with set of tests prepared by algo, algo picks set of tests for candidate
#      note that candidate interracts with all tests in selection from algo but only with one batch 
#      should batch be formed at start or with analysis of user interraction? 2 different modes of interaction or mixed mode - multi-batched
# mode - n times gets 1 test for candidate 
# mode - 1 time gets n tests for candidate 
# mode - i times gets j tests (i*j = n) for candidate
#      note that n by itself is varying

#https://click.palletsprojects.com/en/8.1.x/
#conda add click + how to run deps on cluster + bash scripts

import json
import click
import numpy as np
from cde import CDESpace 

@click.group()
def cli():
    pass

@cli.command("space")
@click.option("-o", "--output", type = str)
@click.option("-a", "--axis", type = int, multiple=True, default=[1,1])
@click.option("-t", "--tests", type = int, multiple=True)
@click.option("-c", "--cand", type = int, multiple=True)
@click.option("--spanned", type = str, multiple=True)
@click.option("--spanned-strategy", type = click.Choice(['hardest', '2axes']))
@click.option("--spanned-strategy-count", type = int, default = -1)
@click.option("--spanned-strategy-pos", type = int, default = -1)
@click.option("--spanned-strategy-tests", type = int, default = 1)
@click.option("--dependency", type = str, multiple=True)
@click.option("--dependency-strategy", type = click.Choice(['allaxes', '2axes']))
@click.option("--dependency-strategy-count", type = int, default = -1)
@click.option("--dependency-strategy-pos", type = int, default = -1)
@click.option("--dependency-strategy-pos", type = int, default = -1)
def gen_space(axis:list[int] = [1,1], tests: list[int] = [], cand: list[int] = [], spanned: list[str] = [], 
                    spanned_strategy = None, spanned_strategy_count = -1, spanned_strategy_pos = -1, spanned_strategy_tests = 1,
                    dependency = [], dependency_strategy = None, dependency_strategy_count = -1, dependency_strategy_pos = -1,
                    output = None):    
    click.echo("Creating CDE space...")
    space = CDESpace(axis)
    if len(tests) > 0:
        origin, *axis = tests
        axis = [origin] if len(axis) == 0 else axis
        space = space.with_test_distribution(origin, axis)
    if len(cand) > 0: 
        origin, *axis = cand
        axis = [origin] if len(axis) == 0 else axis 
        space = space.with_candidate_distribution(origin, axis)
    if len(spanned) > 0: 
        for s in spanned:
            *coords, num_tests = s.split(";")
            coords = [tuple(int(el) for el in c.split(",")) for c in coords]
            space = space.with_spanned_point(coords, int(num_tests))
    elif spanned_strategy == 'hardest':
        sign = abs(spanned_strategy_pos) / spanned_strategy_pos
        spanned_coord = [(axis_id, spanned_strategy_pos % (sign * sz)) for axis_id, sz in enumerate(space.dims)]
        if spanned_strategy_count > 0:
            spanned_coord = spanned_coord[:spanned_strategy_count]
        space = space.with_spanned_point(spanned_coord, spanned_strategy_tests)
    elif spanned_strategy == '2axes':
        count = 0 
        sign = abs(spanned_strategy_pos) / spanned_strategy_pos
        for axis_id, sz in enumerate(space.dims):
            prev_sz = space.dims[axis_id - 1]
            space = space.with_spanned_point([(axis_id - 1, spanned_strategy_pos % (sign * prev_sz)), (axis_id, spanned_strategy_pos % (sign * sz))], 
                                                spanned_strategy_tests)
            count += 1 
            if spanned_strategy_count > 0 and count >= spanned_strategy_count:
                break 
    if len(dependency) > 0: 
        for d in dependency:
            axis_point_a, axis_point_b = d.split(";")
            axis_a, point_a = [int(el) for el in axis_point_a.split(",")]
            axis_b, point_b = [int(el) for el in axis_point_b.split(",")]
            space = space.with_axes_dependency(axis_a, point_a, axis_b, point_b)
    elif dependency_strategy == 'allaxes':
        count = 0
        sign = abs(dependency_strategy_pos) / dependency_strategy_pos
        for axis_id, sz in enumerate(space.dims):
            prev_sz = space.dims[axis_id - 1]
            space.with_axes_dependency(axis_id - 1, dependency_strategy_pos % (sign * prev_sz), axis_id, dependency_strategy_pos % (sign * sz))
            count += 1 
            if dependency_strategy_count > 0 and count >= dependency_strategy_count:
                break 
    elif dependency_strategy == '2axes':
        count = 0
        sign = abs(dependency_strategy_pos) / dependency_strategy_pos
        for i in range(len(space.dims) // 2):
            axis_a = 2 * i 
            axis_b = 2 * i + 1
            sz_a = space.dims[axis_a]
            sz_b = space.dims[axis_b]
            space.with_axes_dependency(axis_a, dependency_strategy_pos % (sign * sz_a), axis_b, dependency_strategy_pos % (sign * sz_b))
            count += 1 
            if dependency_strategy_count > 0 and count >= dependency_strategy_count:
                break
    click.echo(f"{space}")
    if output is not None:
        with open(output, "a") as f:
            f.writelines([space.to_json() + "\n"])
    return space

@cli.command("simulate")
@click.option("-i", "--input", type = str, required = True)
@click.option("-n", "--num-interactions", type = int, default = 1)
@click.option("--algo", type = str, default = "algo.RandTestSelector")
@click.option('--algo-params', type = str, default = "{}")
@click.option('--random-seed', type=int, default = 17)
@click.option('--times', type=int, default = 1)
@click.option("-m", "--metrics", type = str, required = True)
def run_simulation(input, metrics, num_interactions = 1, algo = "algo.RandTestSelector", algo_params = "{}", random_seed = 17, times = 1):
    click.echo("Reading space...")
    with open(input, "r") as f:
        spaces = [CDESpace.from_json(line) for line in f.readlines()]
    rnd_state = np.random.RandomState(random_seed)
    click.echo("Initializing algorithm...")
    algo_settings = json.loads(algo_params)
    test_selector = build_test_selector(algo, algo_settings, rnd_state)
    click.echo(f"{test_selector}")
    for space_id, space in enumerate(spaces):
        click.echo(f"\n-- Space:\n {space}")
        all_fails = space.get_candidate_fails()
        test_selector.init_space(space)
        candidates = list(space.get_candidates())
        #simulation loop 
        for i in range(times):         
            rnd_state.shuffle(candidates)   
            for candidate_id in candidates:
                candidate_fails = all_fails.get(candidate_id, set())
                # done_interactions = 0
                # while done_interactions < num_interactions:
                selected_tests = test_selector.get_tests(candidate_id, num_interactions)
                interactions = {t:t in candidate_fails for t in selected_tests}
                test_selector.provide_interactions(candidate_id, interactions)
                # done_interactions += len(selected_tests)        
            #metrics of algo 
            # fetch all tests for next candidate without providing interactions 
            tests_sample = test_selector.get_tests(-1, num_interactions)
            DC = space.dimension_coverage(tests_sample)
            ARR, ARRA = space.avg_rank_of_repr(tests_sample)
            Dup = space.duplication(tests_sample)
            R = space.redundancy(tests_sample)
            nonI = space.noninformative(tests_sample)
            metric_data = json.dumps({"i": i, "input": input, "space_id":space_id, 
                                "DC": DC, "ARR": ARR, "ARRA": ARRA, "Dup": Dup, "R": R, 
                                "nonI": nonI, "algo": algo, "settings": algo_settings, "seed": random_seed,
                                "num_interactions": num_interactions, "eval_sample": [ int(t) for t in tests_sample ]})
            click.echo(f"{metric_data}")
            with open(metrics, "a") as f:
                f.writelines([metric_data + "\n"])

if __name__ == '__main__':
    cli()