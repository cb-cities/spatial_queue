#!/bin/bash
#SBATCH -J berkeley_evacuation
#SBATCH -o out_dt.out
#SBATCH -e out_dt.err
#SBATCH -t 4:30:00
#SBATCH -p skx-normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=6

module purge
module load TACC
module load launcher
module load python3
source $HOME/virtualenvs/cities/bin/activate
export OMP_NUM_THREADS=6

### folder in $SCRATCH to hold output
scratch_folder=$SCRATCH/spatial_queue/projects/berkeley/simulation_outputs_surveybaseline_test
mkdir -p $scratch_folder
mkdir -p $scratch_folder/log
mkdir -p $scratch_folder/t_stats
mkdir -p $scratch_folder/link_stats
mkdir -p $scratch_folder/node_stats
export OUTPUT_FOLDER=$scratch_folder

# Launcher to use multicores on node
export LAUNCHER_WORKDIR=$PWD
export LAUNCHER_PPN=15
export LAUNCHER_JOB_FILE=joblist_hhv_r
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

cd $scratch_folder
# tar czf ../t_stats_$(date +"%FT%H%M%S").tar.gz t_stats