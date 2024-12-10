import argparse
import subprocess
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sync_scripts(username: str, dry_run: bool = False):
    """Sync local scripts to Puhti."""
    
    # Define paths
    local_script_dir = Path(__file__).parent / "puhti"
    puhti_dir = "/scratch/project_2011638/rag_queries"
    
    # Ensure local directory exists
    local_script_dir.mkdir(exist_ok=True)
    
    # Files to sync
    files = {
        "process_query.py": "0644",  # User: read/write, Group/Others: read
        "submit_query.sh": "0755",   # User: all, Group/Others: read/execute
    }
    
    try:
        # Create remote directory if it doesn't exist
        ssh_cmd = f"ssh {username}@puhti.csc.fi 'mkdir -p {puhti_dir}'"
        if not dry_run:
            subprocess.run(ssh_cmd, shell=True, check=True)
        logger.info(f"Created directory: {puhti_dir}")
        
        # Sync each file
        for file, permissions in files.items():
            local_file = local_script_dir / file
            remote_path = f"{username}@puhti.csc.fi:{puhti_dir}/{file}"
            
            # Copy file to Puhti
            scp_cmd = f"scp {local_file} {remote_path}"
            if not dry_run:
                subprocess.run(scp_cmd, shell=True, check=True)
            logger.info(f"Copied {file} to Puhti")
            
            # Set permissions
            chmod_cmd = f"ssh {username}@puhti.csc.fi 'chmod {permissions} {puhti_dir}/{file}'"
            if not dry_run:
                subprocess.run(chmod_cmd, shell=True, check=True)
            logger.info(f"Set permissions {permissions} for {file}")
            
        logger.info("Successfully synced scripts to Puhti")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error syncing scripts: {str(e)}")
        raise
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync scripts to Puhti")
    parser.add_argument("username", help="Your Puhti username")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be done without making changes")
    args = parser.parse_args()
    
    sync_scripts(args.username, args.dry_run)