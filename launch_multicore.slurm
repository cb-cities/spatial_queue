#!/bin/bash
#SBATCH -J berkeley_evacuation
#SBATCH -o r1_50_100.out
#SBATCH -e r1_50_100.err
#SBATCH -t 05:30:00
#SBATCH -p skx-normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=7

module purge
module load TACC
module load launcher
module load python3
source $HOME/virtualenvs/cities/bin/activate
export OMP_NUM_THREADS=7

### folder in $SCRATCH to hold output
scratch_folder=$SCRATCH/spatial_queue/projects/berkeley/simulation_outputs
mkdir -p $scratch_folder
mkdir -p $scratch_folder/log
mkdir -p $scratch_folder/t_stats
mkdir -p $scratch_folder/link_stats

# Launcher to use multicores on node
export LAUNCHER_WORKDIR=$PWD
export LAUNCHER_PPN=6
export LAUNCHER_JOB_FILE=joblist_r1
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

cd $scratch_folder
tar czf ../t_stats_$(date +"%FT%H%M%S").tar.gz t_stats