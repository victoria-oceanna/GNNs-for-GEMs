#!/bin/bash -l        
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1900m
#SBATCH -p amdsmall
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=jone4254@umn.edu 

module load python3
source activate gt
python -u centrality_batch_job.py

echo $CONDA_DEFAULT_ENV
echo "after calling source: $PATH"
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"