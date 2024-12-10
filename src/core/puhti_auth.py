import os
from pathlib import Path
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)

class PuhtiAuth:
    def __init__(self):
        self.username = os.getenv("PUHTI_USERNAME")
        self.project = os.getenv("PUHTI_PROJECT", "project_2011638")
        self.ssh_key_path = os.getenv("SSH_KEY_PATH", str(Path.home() / ".ssh" / "id_rsa"))
        
    def validate_credentials(self) -> bool:
        """Validate that all required credentials are present."""
        if not self.username:
            logger.error("PUHTI_USERNAME environment variable not set")
            return False
            
        if not os.path.exists(self.ssh_key_path):
            logger.error(f"SSH key not found at {self.ssh_key_path}")
            return False
            
        return True

    def get_credentials(self) -> dict:
        """Get credentials for Puhti access."""
        return {
            "username": self.username,
            "project": self.project,
            "ssh_key_path": self.ssh_key_path
        }