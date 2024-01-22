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
from cde import CDESpace
from config import CONFIG, GAME_GROUPS, SIM_GROUPS
from games import GameSimulation, InteractionGame 
from params import param_seed
import fcntl
import time

@click.group()
def cli():
    pass

@cli.command("space")
@click.option("-o", "--output", type = str, required=True)
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

@cli.command("game")
@click.option("-id", type = int, required = True)
@click.option('--times', type=int, default = 1)
@click.option("-m", "--metrics", type = str, default = "metrics.jsonlist")
def run_game(id, times = 1, metrics = ""):
    group, game_name, sim_name = CONFIG[id]    
    game_builder = GAME_GROUPS[group][game_name]
    sim_builder = SIM_GROUPS[group][sim_name]
    click.echo(f"Running simulation {sim_name} on game {game_name} from group '{group}'")
    game : InteractionGame = game_builder() #no kwargs? - see params.py for defaults
    sim : GameSimulation = sim_builder(game)
    for i in range(times): 
        sim.play()
        timestamp = int(time.time())
        metric_data = {"config_id": id, "sim_name": sim_name, "game_name":game_name, "i": i, "timestamp": timestamp,
                            "params":{"seed": param_seed, **game.game_params, **sim.sim_params}, **sim.game_metrics}
        click.echo(f"{metrics}")
        with open(metrics, "a") as f:                
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(metric_data) + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)
            
if __name__ == '__main__':
    ''' Entry for running games and collecting data '''
    cli()