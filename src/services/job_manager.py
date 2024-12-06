## src/services/job_manager.py
import asyncio
import json
import logging
import re
from pathlib import Path
import uuid
from datetime import datetime
from typing import Dict, Optional, Tuple
import paramiko
import torch
from src.core.config import settings

logger = logging.getLogger(__name__)

class DocumentMetadataExtractor:
    @staticmethod
    def extract_from_filename(filename: str) -> Dict:
        """Extract person name, age and document ID from filename."""
        try:
            # Expected format: "Name Age Id.docx", e.g., "Matti 75v M7-54.docx"
            pattern = r"([A-ZÄÖÅa-zäöå]+)\s+(\d+)v\s+([A-Za-z0-9\-]+)"
            match = re.match(pattern, filename)
            
            if match:
                return {
                    "person_name": match.group(1),
                    "person_age": int(match.group(2)),
                    "document_id": match.group(3)
                }
            else:
                logger.warning(f"Could not extract metadata from filename: {filename}")
                return {
                    "person_name": "Unknown",
                    "person_age": 0,
                    "document_id": str(uuid.uuid4())
                }
        except Exception as e:
            logger.error(f"Error extracting metadata from filename: {str(e)}")
            return {
                "person_name": "Unknown",
                "person_age": 0,
                "document_id": str(uuid.uuid4())
            }

class PuhtiJobManager:
    def __init__(self):
        self.ssh_client = None
        self.jobs: Dict[str, Dict] = {}
        self.work_dir = Path("/scratch/project_2011638/input_documents")
        self.script_path = Path("/scratch/project_2011638/embedding_script.py")
        self.metadata_extractor = DocumentMetadataExtractor()

    async def connect(self):
        """Establish SSH connection to Puhti."""
        if not self.ssh_client:
            try:
                self.ssh_client = paramiko.SSHClient()
                self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                self.ssh_client.connect(
                    settings.PUHTI_HOST,
                    username=settings.PUHTI_USERNAME,
                    key_filename=settings.SSH_KEY_PATH,
                )
                logger.info("Connected to Puhti successfully")
            except Exception as e:
                logger.error(f"Failed to connect to Puhti: {str(e)}")
                raise
    async def _cleanup_stalled_jobs(self):
        """Clean up stalled jobs before submitting new ones."""
        try:
            await self.connect()
            # Cancel all jobs for current user
            stdin, stdout, stderr = self.ssh_client.exec_command("scancel -u $USER")
            await asyncio.sleep(2)  # Give time for jobs to be cancelled
            logger.info("Cleaned up stalled jobs")
        except Exception as e:
            logger.error(f"Error cleaning up stalled jobs: {str(e)}")

    async def submit_embedding_job(self, file_path: Path) -> Tuple[str, Dict]:
        """Submit document embedding job and return job ID and metadata."""
        try:
            await self.connect()
            
            # Extract metadata from filename
            metadata = self.metadata_extractor.extract_from_filename(file_path.name)
            job_id = str(uuid.uuid4())
            
            # Prepare remote paths
            remote_filename = f"{metadata['document_id']}_{file_path.name}"
            remote_file_path = self.input_path / remote_filename
            
            # Transfer document file
            sftp = self.ssh_client.open_sftp()
            await self._ensure_remote_dirs(sftp)
            sftp.put(str(file_path), str(remote_file_path))
            
            # Generate and transfer batch script
            batch_script = await self._generate_batch_script(
                job_name=f"embed_{metadata['document_id']}", 
                input_path=str(remote_file_path)
            )
            script_remote_path = self.input_path / f"job_{job_id}.sh"
            sftp.put(str(batch_script), str(script_remote_path))
            
            # Submit batch job
            stdin, stdout, stderr = self.ssh_client.exec_command(
                f"cd {self.input_path} && sbatch {script_remote_path}"
            )
            slurm_job_id = stdout.read().decode().strip().split()[-1]
            
            # Store job information
            self.jobs[job_id] = {
                "slurm_job_id": slurm_job_id,
                "status": "PENDING",
                "input_file": str(remote_file_path),
                "metadata": metadata,
                "submitted_at": datetime.utcnow().isoformat(),
            }
            
            return job_id, metadata
            
        except Exception as e:
            logger.error(f"Failed to submit embedding job: {str(e)}")
            raise

    async def submit_embedding_job(self, file_path: Path) -> Tuple[str, Dict]:
        """Submit document embedding job and return job ID and metadata."""
        try:
            await self.connect()
            
            # Extract metadata from filename
            metadata = self.metadata_extractor.extract_from_filename(file_path.name)
            job_id = str(uuid.uuid4())
            
            # Prepare remote paths
            remote_filename = f"{metadata['document_id']}_{file_path.name}"
            remote_file_path = self.work_dir / remote_filename
            
            # Transfer document file
            sftp = self.ssh_client.open_sftp()
            try:
                # Ensure directory exists
                sftp.stat(str(self.work_dir))
            except FileNotFoundError:
                sftp.mkdir(str(self.work_dir))
                
            # Transfer files
            sftp.put(str(file_path), str(remote_file_path))
            
            # Generate and transfer batch script
            batch_script = await self._generate_batch_script(
                job_name=f"embed_{metadata['document_id']}", 
                input_path=str(remote_file_path)
            )
            script_remote_path = self.work_dir / f"job_{job_id}.sh"
            sftp.put(str(batch_script), str(script_remote_path))
            
            # Submit batch job
            stdin, stdout, stderr = self.ssh_client.exec_command(
                f"cd {self.work_dir} && sbatch {script_remote_path}"
            )
            slurm_job_id = stdout.read().decode().strip().split()[-1]
            
            # Store job information
            self.jobs[job_id] = {
                "slurm_job_id": slurm_job_id,
                "status": "PENDING",
                "input_file": str(remote_file_path),
                "metadata": metadata,
                "submitted_at": datetime.utcnow().isoformat(),
            }
            
            return job_id, metadata
            
        except Exception as e:
            logger.error(f"Failed to submit embedding job: {str(e)}")
            raise
    
    async def check_embedding_job(self, job_id: str) -> Dict:
        """Check job status and retrieve results if complete."""
        try:
            job_info = self.jobs.get(job_id)
            if not job_info:
                raise ValueError(f"No job found for ID {job_id}")
                
            await self.connect()
            
            # Check job status
            stdin, stdout, stderr = self.ssh_client.exec_command(
                f'squeue -j {job_info["slurm_job_id"]} -h'
            )
            job_status = stdout.read().decode().strip()
            
            if not job_status:  # Job completed
                sftp = self.ssh_client.open_sftp()
                document_id = job_info["metadata"]["document_id"]
                
                # Check for output files in work directory
                embedding_path = self.work_dir / f"embeddings_{document_id}_{job_info['metadata']['person_name']} {job_info['metadata']['person_age']}v {document_id}.pt"
                texts_path = self.work_dir / f"texts_{document_id}_{job_info['metadata']['person_name']} {job_info['metadata']['person_age']}v {document_id}.json"
                
                try:
                    # Download results
                    local_embedding_path = Path(f"/tmp/embeddings_{job_id}.pt")
                    local_texts_path = Path(f"/tmp/texts_{job_id}.json")
                    
                    sftp.get(str(embedding_path), str(local_embedding_path))
                    sftp.get(str(texts_path), str(local_texts_path))
                    
                    # Load results
                    embeddings = torch.load(str(local_embedding_path))
                    with open(local_texts_path) as f:
                        texts = json.load(f)["texts"]
                    
                    # Clean up local files
                    local_embedding_path.unlink()
                    local_texts_path.unlink()
                    
                    self.jobs[job_id].update({
                        "status": "COMPLETED",
                        "embeddings": embeddings.numpy(),
                        "texts": texts
                    })
                    
                except FileNotFoundError as e:
                    logger.error(f"Missing output files: {str(e)}")
                    self.jobs[job_id]["status"] = "FAILED"
                except Exception as e:
                    logger.error(f"Error processing results: {str(e)}")
                    self.jobs[job_id]["status"] = "FAILED"
            else:
                self.jobs[job_id]["status"] = "RUNNING"
            
            return self.jobs[job_id]
            
        except Exception as e:
            logger.error(f"Failed to check embedding job: {str(e)}")
            raise


    async def _generate_batch_script(self, job_name: str, input_path: str) -> Path:
        """Generate a batch script for embedding generation."""
        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=project_2011638
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:v100:1

# Activate environment
export PATH="/scratch/project_2011638/ml_env/bin:$PATH"

# Set HuggingFace cache
export HF_HOME=/scratch/project_2011638/safdarih/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME

# Run embedding script
python {self.script_path} --input '{input_path}'
"""
        script_path = Path(f"/tmp/{job_name}.sh")
        script_path.write_text(script_content)
        return script_path

    def cleanup(self):
        """Clean up SSH connection."""
        if self.ssh_client:
            try:
                self.ssh_client.close()
                logger.info("SSH connection closed")
            except Exception as e:
                logger.error(f"Error closing SSH connection: {str(e)}")

    async def check_job_slots(self) -> bool:
        """Check if we have available job slots for GPU partition."""
        try:
            await self.connect()
            
            if not self.ssh_client:
                logger.error("Failed to establish SSH connection")
                return False

            # First get current GPU jobs
            stdin, stdout, stderr = self.ssh_client.exec_command(
                "squeue -u $USER -p gpu -h | wc -l"
            )
            current_jobs = int(stdout.read().decode().strip() or "0")
            logger.info(f"Current GPU jobs: {current_jobs}")

            # Get GPU partition limits
            stdin, stdout, stderr = self.ssh_client.exec_command(
                "sacctmgr show assoc where user=$USER partition=gpu format=account,partition,maxjobs,maxsubmit -n -P"
            )
            output = stdout.read().decode().strip()
            
            # Look for lowest limit as that's what's enforced
            lowest_limit = None
            for line in output.split('\n'):
                if not line.strip():
                    continue
                    
                parts = line.strip().split('|')
                if len(parts) >= 4 and 'project_2011638' in parts[0]:
                    try:
                        max_jobs = int(parts[2]) if parts[2].strip() != '' else float('inf')
                        max_submit = int(parts[3]) if parts[3].strip() != '' else float('inf')
                        limit = min(max_jobs, max_submit)
                        if lowest_limit is None or limit < lowest_limit:
                            lowest_limit = limit
                        logger.info(f"Found GPU limit for {parts[0]}: max_jobs={max_jobs}, max_submit={max_submit}")
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse line {line}: {str(e)}")
                        continue

            if lowest_limit is not None:
                has_slots = current_jobs < lowest_limit
                logger.info(f"Lowest GPU limit: {lowest_limit}, Current GPU jobs: {current_jobs}, Has slots: {has_slots}")
                return has_slots
            else:
                logger.warning("No valid GPU job limits found")
                return False

        except Exception as e:
            logger.error(f"Error checking job slots: {str(e)}")
            return False
    
    async def submit_llm_job(self, input_data: Dict, max_retries: int = 3) -> str:
        """Submit LLM generation job to Puhti with retries."""
        try:
            # First ensure we have a connection
            await self.connect()
            
            # Check for available slots
            if not await self.check_job_slots():
                raise Exception("No available job slots")

            last_error = None
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        logger.info(f"Retry attempt {attempt + 1}/{max_retries}")
                        await asyncio.sleep(5 * attempt)
                    
                    logger.info(f"Attempting to submit LLM job (attempt {attempt + 1}/{max_retries})")
                    
                    await self._cleanup_stalled_jobs()
                    await asyncio.sleep(2)

                    job_id = str(uuid.uuid4())
                    
                    # Prepare input files
                    input_filename = f"llm_input_{job_id}.json"
                    input_path = self.work_dir / input_filename
                    logger.info(f"Input file path: {input_path}")
                    
                    # Transfer input data
                    sftp = self.ssh_client.open_sftp()
                    try:
                        sftp.stat(str(self.work_dir))
                    except FileNotFoundError:
                        logger.info(f"Creating work directory: {self.work_dir}")
                        sftp.mkdir(str(self.work_dir))
                    
                    # Write input data
                    logger.info(f"Writing input data to {input_path}")
                    with sftp.open(str(input_path), 'w') as f:
                        json.dump(input_data, f, ensure_ascii=False, indent=2)
                    
                    # Generate batch script
                    batch_script = await self._generate_llm_batch_script(
                        job_name=f"llm_{job_id}",
                        input_path=str(input_path)
                    )
                    script_path = self.work_dir / f"job_llm_{job_id}.sh"
                    logger.info(f"Generated batch script at: {script_path}")
                    
                    # Log script content for debugging
                    logger.info("Batch script content:")
                    logger.info("-" * 80)
                    logger.info(batch_script.read_text())
                    logger.info("-" * 80)
                    
                    # Transfer batch script
                    sftp.put(str(batch_script), str(script_path))
                    
                    # Check current jobs before submission
                    stdin, stdout, stderr = self.ssh_client.exec_command("squeue -u $USER -h | wc -l")
                    job_count = int(stdout.read().decode().strip() or "0")
                    logger.info(f"Current job count: {job_count}")
                    
                    # Submit job
                    submit_command = f"cd {self.work_dir} && sbatch {script_path}"
                    logger.info(f"Submitting job with command: {submit_command}")
                    
                    stdin, stdout, stderr = self.ssh_client.exec_command(submit_command)
                    stdout_content = stdout.read().decode().strip()
                    stderr_content = stderr.read().decode().strip()
                    
                    logger.info(f"sbatch stdout: {stdout_content}")
                    
                    if stderr_content:
                        logger.warning(f"sbatch stderr: {stderr_content}")
                        if "AssocMaxSubmitJobLimit" in stderr_content:
                            logger.warning(f"Hit job limit, retrying... ({attempt + 1}/{max_retries})")
                            await asyncio.sleep(30 * (attempt + 1))  # Progressive backoff
                            continue
                        else:
                            raise Exception(f"sbatch error: {stderr_content}")

                    if not stdout_content:
                        raise Exception("No output from sbatch command")

                    # Parse job ID
                    try:
                        slurm_job_id = stdout_content.split()[-1]
                        if not slurm_job_id.isdigit():
                            raise ValueError(f"Invalid job ID format: {stdout_content}")
                    except (IndexError, ValueError) as e:
                        raise Exception(f"Failed to parse job ID from output: {stdout_content}")

                    # Store job info
                    self.jobs[job_id] = {
                        "slurm_job_id": slurm_job_id,
                        "status": "PENDING",
                        "input_file": str(input_path),
                        "submitted_at": datetime.utcnow().isoformat(),
                    }

                    logger.info(f"Successfully submitted LLM job {job_id} (Slurm ID: {slurm_job_id})")
                    return job_id

                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Attempt {attempt + 1} failed: {last_error}")
                    await asyncio.sleep(5)
                    
                    if "AssocMaxSubmitJobLimit" not in str(e):
                        # If it's not a job limit error, break the retry loop
                        break
                    
                    continue

            error_msg = f"Failed to submit LLM job after {max_retries} attempts: {last_error}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        except Exception as e:
            logger.error(f"Error in submit_llm_job: {str(e)}")
            raise

    async def check_llm_job(self, job_id: str) -> Dict:
        """Check status of LLM job."""
        try:
            job_info = self.jobs.get(job_id)
            if not job_info:
                raise ValueError(f"No job found for ID {job_id}")
                
            await self.connect()
            
            # Check job status
            stdin, stdout, stderr = self.ssh_client.exec_command(
                f'squeue -j {job_info["slurm_job_id"]} -h'
            )
            job_status = stdout.read().decode().strip()
            
            if not job_status:  # Job completed
                sftp = self.ssh_client.open_sftp()
                response_path = self.work_dir / f"response_{Path(job_info['input_file']).stem}.json"
                
                try:
                    # Download results
                    local_response_path = Path(f"/tmp/response_{job_id}.json")
                    sftp.get(str(response_path), str(local_response_path))
                    
                    # Load results
                    with open(local_response_path) as f:
                        response_data = json.load(f)
                    
                    # Clean up
                    local_response_path.unlink()
                    sftp.remove(str(response_path))
                    sftp.remove(job_info["input_file"])
                    
                    self.jobs[job_id].update({
                        "status": "COMPLETED",
                        "response": response_data["response"]
                    })
                    
                except FileNotFoundError:
                    self.jobs[job_id]["status"] = "FAILED"
                    logger.error(f"LLM job output not found for {job_id}")
                except Exception as e:
                    self.jobs[job_id]["status"] = "FAILED"
                    logger.error(f"Error processing LLM results: {str(e)}")
            else:
                self.jobs[job_id]["status"] = "RUNNING"
                
            return self.jobs[job_id]
            
        except Exception as e:
            logger.error(f"Failed to check LLM job: {str(e)}")
            raise

    async def _generate_llm_batch_script(self, job_name: str, input_path: str) -> Path:
        """Generate batch script for LLM job."""
        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=project_2011638
#SBATCH --partition=gpu
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:v100:1

    # Activate environment
    export PATH="/scratch/project_2011638/ml_env/bin:$PATH"

    # Set cache directory for HuggingFace
    export HF_HOME=/scratch/project_2011638/safdarih/huggingface_cache
    export TRANSFORMERS_CACHE=$HF_HOME

    # Run LLM script with error output
    python /scratch/project_2011638/llm_script.py --input '{input_path}' 2>&1

    # Check exit status
    if [ $? -ne 0 ]; then
        echo "Error: LLM script failed"
        exit 1
    fi
    """
        script_path = Path(f"/tmp/{job_name}.sh")
        script_path.write_text(script_content)
        return script_path
    
    async def test_llm_job_with_context(self):
        """Test LLM job submission with hardcoded Annikki context."""
        try:
            await self.connect()
            
            # Create test input data
            test_input = {
                "query": "Kuinka vanha on Annikki?",
                "context": """# Relevantti konteksti:
                
    Dokumentti 1 (M2042):
    Edelleen välillä kun olen asioilla, minulle tulee kiire kotiin, koska mieheni ei voi olla yksin pitkään, mutta sitten muistan, että nyt minulla ei ole kiire enää. Samoin on vaikeaa laittaa ruokaa vain itselleen niin, ettei sitä tulisi aivan liikaa.

    # Henkilötiedot:

    Annikki:
    - Ikä: 77
    - Suhteet: APPEARS_IN""",
                "params": {
                    "max_tokens": 100,
                    "temperature": 0.1,
                    "top_p": 0.95
                }
            }
            
            # Generate unique ID for this test
            test_id = str(uuid.uuid4())
            
            # Create input file
            sftp = self.ssh_client.open_sftp()
            input_path = self.work_dir / f"test_input_{test_id}.json"
            with sftp.open(str(input_path), 'w') as f:
                json.dump(test_input, f, ensure_ascii=False, indent=2)
            
            # Create batch script
            batch_script = f"""#!/bin/bash
#SBATCH --job-name=test_{test_id}
#SBATCH --account=project_2011638
#SBATCH --partition=gpu
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:v100:1

    # Load required modules
    module purge
    module load cuda

    # Activate environment
    export PATH="/scratch/project_2011638/ml_env/bin:$PATH"

    # Set cache directory for HuggingFace
    export HF_HOME=/scratch/project_2011638/safdarih/huggingface_cache
    export TRANSFORMERS_CACHE=$HF_HOME

    # Run LLM script
    python /scratch/project_2011638/llm_script.py --input '{input_path}' 2>&1

    # Check exit status
    if [ $? -ne 0 ]; then
        echo "Error: Test script failed"
        exit 1
    fi
    """
            # Write batch script
            batch_script_path = self.work_dir / f"test_job_{test_id}.sh"
            with sftp.open(str(batch_script_path), 'w') as f:
                f.write(batch_script)
                
            # Submit job
            submit_command = f"cd {self.work_dir} && sbatch {batch_script_path}"
            stdin, stdout, stderr = self.ssh_client.exec_command(submit_command)
            
            stdout_content = stdout.read().decode().strip()
            stderr_content = stderr.read().decode().strip()
            
            logger.info(f"Test job submission stdout: {stdout_content}")
            if stderr_content:
                logger.warning(f"Test job submission stderr: {stderr_content}")
                
            if stdout_content and not stderr_content:
                try:
                    slurm_job_id = stdout_content.split()[-1]
                    logger.info(f"Test job submitted successfully with ID: {slurm_job_id}")
                    
                    # Wait for job completion
                    while True:
                        await asyncio.sleep(5)
                        stdin, stdout, stderr = self.ssh_client.exec_command(
                            f'squeue -j {slurm_job_id} -h'
                        )
                        if not stdout.read().decode().strip():
                            break
                    
                    # Check output
                    response_path = self.work_dir / f"response_test_input_{test_id}.json"
                    try:
                        with sftp.open(str(response_path), 'r') as f:
                            response = json.load(f)
                        logger.info(f"Test job response: {response}")
                        return True
                    except Exception as e:
                        logger.error(f"Failed to read test job response: {str(e)}")
                        return False
                except Exception as e:
                    logger.error(f"Error monitoring test job: {str(e)}")
                    return False
            return False
            
        except Exception as e:
            logger.error(f"Test job submission failed: {str(e)}")
            return False