# src/backend/auth/auth_handler.py
import bcrypt
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict


class AuthHandler:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        # test users data this is just for testing purposes and should not be used in production
        self.users: Dict[str, Dict] = {}

    def register_user(self, name: str, email: str, password: str) -> bool:
        if email in self.users:
            return False

        # Hash the password
        salt = bcrypt.gensalt()
        hashed_password = bcrypt.hashpw(password.encode("utf-8"), salt)

        # Store user data
        self.users[email] = {
            "name": name,
            "email": email,
            "password": hashed_password,
            "created_at": datetime.now(tz=timezone.utc),
        }
        return True

    def authenticate_user(self, email: str, password: str) -> bool:
        user = self.users.get(email)
        if not user:
            return False

        return bcrypt.checkpw(password.encode("utf-8"), user["password"])

    def create_token(self, email: str) -> str:
        payload = {
            "email": email,
            "exp": datetime.now(tz=timezone.utc) + timedelta(days=1),
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def verify_token(self, token: str) -> Optional[str]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload["email"]
        except:
            return None
