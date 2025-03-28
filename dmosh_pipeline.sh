#!/bin/bash
#SBATCH --ntasks=1   # number of tasks (i.e. number of Galacticus.exe that will run)
#SBATCH --cpus-per-task=24 # number of CPUs to assign to each task
#SBATCH --mem=125G
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "dmosh"   # job name
#SBATCH --mail-user=jlonergan@carnegiescience.edu   # email address
#SBATCH --error=dmosh.log # Send output to a log file
#SBATCH --output=dmosh.log

# Change directory to the location from which this job was submitted
cd $SLURM_SUBMIT_DIR
# Disable core-dumps (not useful unless you know what you're doing with them)
ulimit -c 0
export GFORTRAN_ERROR_DUMPCORE=NO
# Ensure there are no CPU time limits imposed.
ulimit -t unlimited
# Run Galacticus.
/usr/bin/time -v ./Galacticus.exe parameters/tutorials/darkMatterOnlySubHalosPipeline.xml
