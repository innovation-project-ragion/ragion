#!/bin/bash
#SBATCH --job-name=rag_query
#SBATCH --account=project_2011638
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1

# Load required modules
module load pytorch

# Set working directory
cd /scratch/project_2011638/rag_queries

# Process the query
python process_query.py --input ${SLURM_JOB_ID}_input.json --output ${SLURM_JOB_ID}_output.json

# Cleanup input file
rm ${SLURM_JOB_ID}_input.json