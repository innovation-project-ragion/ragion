from pathlib import Path

def generate_batch_script(job_name: str, input_path: str) -> Path:
    """
    Generate a batch script for a Puhti job.
    """
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=project_2011638
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:v100:1

# Activate the custom environment
export PATH="/scratch/project_2011638/ml_env/bin:$PATH"

# Set cache directory for HuggingFace
export HF_HOME=/scratch/project_2011638/safdarih/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME

# Run embedding generation script with single quotes around the filename to handle spaces and special characters , e.g.: 
python /scratch/project_2011638/embedding_script.py --input '/scratch/project_2011638/Aarne 83v M3-2_.docx'

"""
    # Write script to a temporary file
    script_path = Path(f"/tmp/{job_name}.sh")
    script_path.write_text(script_content)
    return script_path



def generate_llm_script(job_name: str, input_path: str) -> Path:
    """Generate a batch script for LLM generation on Puhti."""
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=project_2011638
#SBATCH --partition=gpu
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:v100:1

# Activate environment
export PATH="/scratch/project_2011638/ml_env/bin:$PATH"

# Set cache directory for HuggingFace
export HF_HOME=/scratch/project_2011638/safdarih/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME

# Run LLM script
python /scratch/project_2011638/llm_script.py --input '{input_path}'
"""
    script_path = Path(f"/tmp/{job_name}.sh")
    script_path.write_text(script_content)
    return script_path