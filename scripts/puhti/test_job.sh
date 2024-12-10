#!/bin/bash
#SBATCH --job-name=rag_test
#SBATCH --account=project_2011638
#SBATCH --partition=gputest      # Use gputest partition for testing
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:v100:1        # Request 1 V100 GPU

# Print some basic information
echo "Running test job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Time: $(date)"

# Process input file
INPUT_FILE=$1
OUTPUT_FILE=$2

# Read input and write to output
if [[ -f "$INPUT_FILE" ]]; then
    echo "Processing input from: $INPUT_FILE"
    cat "$INPUT_FILE"
else
    echo "Input file not found: $INPUT_FILE"
fi

# Create simple output
echo "{\"status\": \"success\", \"job_id\": \"$SLURM_JOB_ID\", \"processed_at\": \"$(date)\"}" > "$OUTPUT_FILE"

echo "Job completed successfully"
