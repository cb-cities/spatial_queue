#!/bin/bash
#SBATCH -J berkeley_evacuation
#SBATCH -o r0.out
#SBATCH -e r0.err
#SBATCH -t 05:30:00
#SBATCH -p skx-normal
#SBATCH -N 1 -n 33

module purge
module load TACC
module load launcher
module load python3
source $HOME/virtualenvs/cities/bin/activate
# export OMP_NUM_THREADS=35

### folder in $SCRATCH to hold output
scratch_folder=$SCRATCH/spatial_queue/projects/berkeley/simulation_outputs
mkdir -p $scratch_folder
mkdir -p $scratch_folder/log
mkdir -p $scratch_folder/t_stats
mkdir -p $scratch_folder/link_stats

# Launcher to use multicores on node
export LAUNCHER_WORKDIR=$PWD
export LAUNCHER_PPN=33
export LAUNCHER_JOB_FILE=joblist_r0
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

cd $scratch_folder
tar czf ../t_stats_$(date +"%FT%H%M%S").tar.gz t_stats