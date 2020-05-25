#!/bin/bash
#SBATCH -J test_evacuation
#SBATCH -o job.out
#SBATCH -e job.err
#SBATCH -t 01:30:00
#SBATCH -p skx-normal
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=35

module purge
module load TACC
module load launcher
module load python3
source $HOME/virtualenvs/cities/bin/activate
export OMP_NUM_THREADS=35

### folder in $SCRATCH to hold output
scratch_folder=$SCRATCH/spatial_queue/projects/berkeley/simulation_outputs
mkdir -p $scratch_folder
mkdir -p $scratch_folder/log
mkdir -p $scratch_folder/t_stats
mkdir -p $scratch_folder/link_stats

# rm -f joblist
# touch joblist
# echo "python3 -c 'import dta_meso_berkeley; dta_meso_berkeley.main(random_seed=0, reroute_flag=0, fire_speed=1, dept_time_id=\"imm\", tow_pct=0)'" >> joblist
# echo "python3 -c 'import dta_meso_berkeley; dta_meso_berkeley.main(random_seed=1, reroute_flag=0, fire_speed=1, dept_time_id=\"imm\", tow_pct=0)'" >> joblist
# echo "python3 -c 'import dta_meso_berkeley; dta_meso_berkeley.main(random_seed=3, reroute_flag=0, fire_speed=1, dept_time_id=\"imm\", tow_pct=0)'" >> joblist
# echo "python3 -c 'import dta_meso_berkeley; dta_meso_berkeley.main(random_seed=4, reroute_flag=0, fire_speed=1, dept_time_id=\"imm\", tow_pct=0)'" >> joblist

# Launcher to use multicores on node
export LAUNCHER_WORKDIR=$PWD
export LAUNCHER_PPN=1
export LAUNCHER_JOB_FILE=joblist
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

cd $scratch_folder
tar czf ../simulation_outputs_$(date +"%FT%H%M%S").tar.gz $scratch_folder