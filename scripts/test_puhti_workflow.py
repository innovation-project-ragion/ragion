import os
import json
import asyncio
import argparse
import logging
from pathlib import Path
import paramiko
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PuhtiTestWorkflow:
    def __init__(self, username: str):
        self.username = username
        self.host = "puhti.csc.fi"
        self.project_dir = "/scratch/project_2011638/rag_queries"
        self.ssh_client = None
        
    async def setup_connection(self):
        """Setup SSH connection to Puhti."""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Use default SSH key (~/.ssh/id_rsa)
            self.ssh_client.connect(
                self.host,
                username=self.username,
                key_filename=str(Path.home() / ".ssh" / "id_rsa")
            )
            logger.info("Successfully connected to Puhti")
            
        except Exception as e:
            logger.error(f"Failed to connect to Puhti: {str(e)}")
            raise
            
    async def submit_test_job(self) -> str:
        """Submit a test batch job to Puhti."""
        try:
            # Create unique job identifier
            job_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_file = f"test_input_{job_timestamp}.json"
            output_file = f"test_output_{job_timestamp}.json"
            
            # Create test input data
            test_data = {
                "timestamp": job_timestamp,
                "test_message": "Hello from RAG backend!"
            }
            
            # Upload input file
            sftp = self.ssh_client.open_sftp()
            
            # Ensure directory exists
            stdin, stdout, stderr = self.ssh_client.exec_command(
                f"mkdir -p {self.project_dir}"
            )
            stderr_output = stderr.read().decode()
            if stderr_output:
                logger.warning(f"mkdir warning: {stderr_output}")
            
            # Write input file
            with sftp.file(f"{self.project_dir}/{input_file}", 'w') as f:
                json.dump(test_data, f)
            
            logger.info(f"Uploaded test input file: {input_file}")
            
            # Submit batch job
            submit_cmd = (
                f"cd {self.project_dir} && "
                f"sbatch test_job.sh {input_file} {output_file}"
            )
            
            stdin, stdout, stderr = self.ssh_client.exec_command(submit_cmd)
            
            stderr_output = stderr.read().decode()
            if stderr_output:
                raise Exception(f"Job submission error: {stderr_output}")
            
            # Get job ID
            job_id = stdout.read().decode().strip().split()[-1]
            logger.info(f"Submitted test job with ID: {job_id}")
            
            return job_id, output_file
            
        except Exception as e:
            logger.error(f"Failed to submit test job: {str(e)}")
            raise
            
    async def monitor_job(self, job_id: str, output_file: str):
        """Monitor job status and retrieve results."""
        try:
            while True:
                # Check job status
                stdin, stdout, stderr = self.ssh_client.exec_command(
                    f"squeue -j {job_id} -h"
                )
                
                if not stdout.read().decode().strip():
                    # Job completed, check for output
                    logger.info("Job completed, checking for output file")
                    break
                    
                logger.info(f"Job {job_id} still running...")
                await asyncio.sleep(5)
            
            # Get results
            sftp = self.ssh_client.open_sftp()
            try:
                with sftp.file(f"{self.project_dir}/{output_file}", 'r') as f:
                    results = json.load(f)
                logger.info("Job results:", results)
                
                # Cleanup files
                sftp.remove(f"{self.project_dir}/{output_file}")
                logger.info("Cleaned up output file")
                
                return results
                
            except FileNotFoundError:
                logger.error(f"Output file not found: {output_file}")
                return {"status": "failed", "error": "Output file not found"}
                
        except Exception as e:
            logger.error(f"Error monitoring job: {str(e)}")
            raise
            
    def cleanup(self):
        """Clean up SSH connection."""
        if self.ssh_client:
            self.ssh_client.close()
            logger.info("Closed SSH connection")

async def main(username: str):
    workflow = PuhtiTestWorkflow(username)
    try:
        # Setup connection
        await workflow.setup_connection()
        
        # Submit test job
        job_id, output_file = await workflow.submit_test_job()
        
        # Monitor job and get results
        results = await workflow.monitor_job(job_id, output_file)
        
        logger.info("Test workflow completed successfully!")
        logger.info("Results:", results)
        
    except Exception as e:
        logger.error(f"Test workflow failed: {str(e)}")
    finally:
        workflow.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Puhti batch workflow")
    parser.add_argument("username", help="Your Puhti username")
    args = parser.parse_args()
    
    asyncio.run(main(args.username))