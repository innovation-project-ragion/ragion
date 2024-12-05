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

    # async def check_embedding_job(self, job_id: str) -> Dict:
    #     try:
    #         job_info = self.jobs.get(job_id)
    #         if not job_info:
    #             raise ValueError(f"No job found for ID {job_id}")
                
    #         await self.connect()
            
    #         # Check job status
    #         stdin, stdout, stderr = self.ssh_client.exec_command(
    #             f'squeue -j {job_info["slurm_job_id"]} -h'
    #         )
    #         job_status = stdout.read().decode().strip()
            
    #         if not job_status:  # Job completed
    #             sftp = self.ssh_client.open_sftp()
    #             document_id = job_info["metadata"]["document_id"]
                
    #             # Check for output files
    #             embedding_path = self.work_dir / f"embeddings_{document_id}_{job_info['metadata']['person_name']} {job_info['metadata']['person_age']}v {document_id}.pt"
    #             texts_path = self.work_dir / f"texts_{document_id}_{job_info['metadata']['person_name']} {job_info['metadata']['person_age']}v {document_id}.json"
                
    #             try:
    #                 # Download results
    #                 local_embedding_path = Path(f"/tmp/embeddings_{job_id}.pt")
    #                 local_texts_path = Path(f"/tmp/texts_{job_id}.json")
                    
    #                 sftp.get(str(embedding_path), str(local_embedding_path))
    #                 sftp.get(str(texts_path), str(local_texts_path))
                    
    #                 # Load results
    #                 embeddings = torch.load(str(local_embedding_path))
    #                 with open(local_texts_path) as f:
    #                     texts = json.load(f)["texts"]
                    
    #                 # Clean up local files
    #                 local_embedding_path.unlink()
    #                 local_texts_path.unlink()
                    
    #                 self.jobs[job_id].update({
    #                     "status": "COMPLETED",
    #                     "embeddings": embeddings.numpy(),
    #                     "texts": texts
    #                 })
                    
    #             except FileNotFoundError as e:
    #                 logger.error(f"Missing output files: {str(e)}")
    #                 self.jobs[job_id]["status"] = "FAILED"
    #             except Exception as e:
    #                 logger.error(f"Error processing results: {str(e)}")
    #                 self.jobs[job_id]["status"] = "FAILED"
    #         else:
    #             self.jobs[job_id]["status"] = "RUNNING"
            
    #         return self.jobs[job_id]
            
    #     except Exception as e:
    #         logger.error(f"Failed to check embedding job: {str(e)}")
    #         raise
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