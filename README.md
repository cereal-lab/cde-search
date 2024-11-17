# Experimentation framework for underlying objective search in black-box games

Games are presented as binary interaction matrices between candidates and tests. 
The simulation schema is defined in simulation.py file (method run_game). 
This file also contains benchmark games in GAMES and search algorithms in SIM. 
Game definitions are provided in games.py and search algorithms are in population.py 
To run simulation, consider cli.py.

Coordinate System Extraction CSE algorithm is implemented in de.py, extract_dims_approx function. Also, extract_dims_fix presents CSE on fully defined matrices as fix for originally proposed dimension extraction DE, extract_dims implementation.

## Data folder 

The folder contains observations from the experimentation. 

1. Dynamics: contains animations and snapshots of Number game simulation.
2. Interactions: extracted  dimension spaces from 3x3 number games with CSE algorithm, extract_dims_approx.
3. Matrix-cases: contains interesting interactions, currently one, a cunterexample for the original DE algorithm, extract_dims with 3 axes while minimal space has 2 axes.
4. Metrics: all measured values on runs GAMESxSIM - raw experimentation data. 
5. Num-game-spaces: cache of coordinate systems for 100x100 Number games for efficient metric calculations during runtime. 
6. Plots - raw data postprocessing:
 * hc - hill climbing performance trends;
 * pl - pareto layering performance trends;
 * des - dimension extraction with scoring using CSE;
 * de - dimension extraction with approximation using CSE;
 * de-d-d - veriation of spanned point expiration times;
 * selection - best selected algorithms metric trends;
 * tables in .tex files - analysis of convergence rate %, time to reach threshold and monotonicity.


