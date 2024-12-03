#!/bin/bash
#SBATCH --job-name=rag_test
#SBATCH --account=project_2011638
#SBATCH --partition=test
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:v100:1

# Print some basic information
echo "Running test job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Time: $(date)"

# Process input file
INPUT_FILE=$1
OUTPUT_FILE=$2

# Read input and write to output
echo "Processing input from: $INPUT_FILE"
echo "Input contents:"
cat $INPUT_FILE

# Create simple output
echo "{\"status\": \"success\", \"job_id\": \"$SLURM_JOB_ID\", \"processed_at\": \"$(date)\"}" > $OUTPUT_FILE

echo "Job completed successfully"