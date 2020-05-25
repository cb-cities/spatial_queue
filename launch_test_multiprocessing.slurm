#!/bin/bash
#SBATCH -J test_multiprocessing
#SBATCH -o job.out
#SBATCH -e job.err
#SBATCH -t 00:05:00
#SBATCH -p skx-normal
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3

module purge
module load TACC
module load launcher
module load python3
export OMP_NUM_THREADS=3

rm -f joblist
touch joblist
echo "python3 -c 'import test_multiprocessing; test_multiprocessing.main(1)'" >> joblist
echo "python3 -c 'import test_multiprocessing; test_multiprocessing.main(5)'" >> joblist

# Launcher to use multicores on node
export LAUNCHER_WORKDIR=$PWD
export LAUNCHER_PPN=1
export LAUNCHER_JOB_FILE=joblist
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun