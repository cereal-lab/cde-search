#!/bin/bash 
#SBATCH --job-name=sim-space
#SBATCH --time=72:00:00
#SBATCH --output=out/sim-space.out
#SBATCH --mem=8G
#SBATCH --array=0-75

games=('mcts-00:GreaterThanGame'
        'mcts-00:FocusingGame'
        'mcts-00:IntransitiveRegionGame'
        'mcts-00:CompareOnOneGame'
        'mcts-00:ideal'
        'mcts-00:skew-p-1'
        'mcts-00:skew-p-2'
        'mcts-00:skew-p-3'
        'mcts-00:skew-p-4'
        'mcts-00:trivial-5'
        'mcts-00:trivial-25'
        'mcts-00:trivial-50'
        'mcts-00:skew-t-1'
        'mcts-00:skew-t-2'
        'mcts-00:skew-t-3'
        'mcts-00:skew-t-4'
        'mcts-00:skew-t-5'
        'mcts-00:skew-c-1'
        'mcts-00:skew-c-2'
        'mcts-00:skew-c-3'
        'mcts-00:skew-c-4'
        'mcts-00:skew-c-5'
        'mcts-00:span-all-ends-1'
        'mcts-00:span-all-ends-5'
        'mcts-00:span-all-ends-10'
        'mcts-00:span-all-ends-20'
        'mcts-00:span-pairs-1'
        'mcts-00:span-pairs-5'
        'mcts-00:span-pairs-10'
        'mcts-00:span-pairs-20'
        'mcts-00:dupl-t-5'
        'mcts-00:dupl-t-10'
        'mcts-00:dupl-t-50'
        'mcts-00:dupl-c-5'
        'mcts-00:dupl-c-10'
        'mcts-00:dupl-c-50'
        'mcts-00:dependant-all-1'
        'mcts-00:dependant-all-2'
        'mcts-10:GreaterThanGame'
        'mcts-10:FocusingGame'
        'mcts-10:IntransitiveRegionGame'
        'mcts-10:CompareOnOneGame'
        'mcts-10:ideal'
        'mcts-10:skew-p-1'
        'mcts-10:skew-p-2'
        'mcts-10:skew-p-3'
        'mcts-10:skew-p-4'
        'mcts-10:trivial-5'
        'mcts-10:trivial-25'
        'mcts-10:trivial-50'
        'mcts-10:skew-t-1'
        'mcts-10:skew-t-2'
        'mcts-10:skew-t-3'
        'mcts-10:skew-t-4'
        'mcts-10:skew-t-5'
        'mcts-10:skew-c-1'
        'mcts-10:skew-c-2'
        'mcts-10:skew-c-3'
        'mcts-10:skew-c-4'
        'mcts-10:skew-c-5'
        'mcts-10:span-all-ends-1'
        'mcts-10:span-all-ends-5'
        'mcts-10:span-all-ends-10'
        'mcts-10:span-all-ends-20'
        'mcts-10:span-pairs-1'
        'mcts-10:span-pairs-5'
        'mcts-10:span-pairs-10'
        'mcts-10:span-pairs-20'
        'mcts-10:dupl-t-5'
        'mcts-10:dupl-t-10'
        'mcts-10:dupl-t-50'
        'mcts-10:dupl-c-5'
        'mcts-10:dupl-c-10'
        'mcts-10:dupl-c-50'
        'mcts-10:dependant-all-1'
        'mcts-10:dependant-all-2')

game=${games[$SLURM_ARRAY_TASK_ID]}

echo "Starting job $SLURM_ARRAY_TASK_ID with game $game"

module rm apps/python/3.8.5
module load apps/anaconda/5.3.1

source activate cde-search-env

python ~/cde-search/cli.py game -sim "$game" --times 30 --metrics "$WORK/cde-search/mcts.jsonlist"

echo "Done job $SLURM_ARRAY_TASK_ID with game $game"