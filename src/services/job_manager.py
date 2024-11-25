import asyncio
import json
import logging
import subprocess
from pathlib import Path
import uuid
from datetime import datetime
from typing import Dict, Optional
import paramiko
from src.core.config import settings

logger = logging.getLogger(__name__)

class PuhtiJobManager:
    def __init__(self):
        self.ssh_client = None
        self.jobs: Dict[str, Dict] = {}
        self.query_path = Path("/scratch/project_2011638/rag_queries")
        
    async def connect(self):
        """Establish SSH connection to Puhti for job management."""
        if not self.ssh_client:
            try:
                self.ssh_client = paramiko.SSHClient()
                self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                self.ssh_client.connect(
                    settings.PUHTI_HOST,
                    username=settings.PUHTI_USERNAME,
                    key_filename=settings.SSH_KEY_PATH
                )
            except Exception as e:
                logger.error(f"Failed to connect to Puhti: {str(e)}")
                raise
    
    async def submit_job(self, query: str, params: Dict) -> str:
        """Submit a query processing job to Puhti."""
        try:
            await self.connect()
            
            # Generate unique query ID
            query_id = str(uuid.uuid4())
            
            # Prepare input data
            input_data = {
                "query_id": query_id,
                "prompt": query,
                "max_tokens": params.get("max_tokens", 300),
                "temperature": params.get("temperature", 0.1),
                "top_p": params.get("top_p", 0.95),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Transfer input file
            sftp = self.ssh_client.open_sftp()
            input_path = self.query_path / f"input_{query_id}.json"
            with sftp.file(str(input_path), 'w') as f:
                json.dump(input_data, f)
            
            # Submit batch job
            stdin, stdout, stderr = self.ssh_client.exec_command(
                f'cd {self.query_path} && sbatch submit_query.sh'
            )
            
            # Get job ID
            job_id = stdout.read().decode().strip().split()[-1]
            
            # Store job information
            self.jobs[query_id] = {
                "job_id": job_id,
                "status": "PENDING",
                "submitted_at": datetime.utcnow().isoformat()
            }
            
            return query_id
            
        except Exception as e:
            logger.error(f"Failed to submit job: {str(e)}")
            raise
    
    async def check_job_status(self, query_id: str) -> Dict:
        """Check the status of a submitted job."""
        try:
            job_info = self.jobs.get(query_id)
            if not job_info:
                raise ValueError(f"No job found for query ID {query_id}")
            
            await self.connect()
            
            # Check job status
            stdin, stdout, stderr = self.ssh_client.exec_command(
                f'squeue -j {job_info["job_id"]} -h'
            )
            
            job_status = stdout.read().decode().strip()
            
            if not job_status:
                # Job completed, check for output file
                output_path = self.query_path / f"output_{query_id}.json"
                try:
                    sftp = self.ssh_client.open_sftp()
                    with sftp.file(str(output_path), 'r') as f:
                        results = json.load(f)
                    
                    # Cleanup output file
                    sftp.remove(str(output_path))
                    
                    self.jobs[query_id]["status"] = "COMPLETED"
                    self.jobs[query_id]["results"] = results
                    
                except FileNotFoundError:
                    self.jobs[query_id]["status"] = "FAILED"
            else:
                self.jobs[query_id]["status"] = "RUNNING"
            
            return self.jobs[query_id]
            
        except Exception as e:
            logger.error(f"Failed to check job status: {str(e)}")
            raise
    
    def cleanup(self):
        """Clean up SSH connection."""
        if self.ssh_client:
            self.ssh_client.close()

# Global job manager instance
job_manager = PuhtiJobManager()