import os
from typing import Optional
from dataclasses import dataclass
import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

load_dotenv()

CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")

optional_security = HTTPBearer(auto_error=False)


@dataclass
class AuthUser:
    user_id: str
    auth_type: str  # "auth_user" or "anon_user"


async def get_auth_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_security),
) -> AuthUser:
    """
    ?anon=<user_id> in query param → anon user
    Bearer token in header        → auth user, extract user_id from JWT
    """
    anon_user_id = request.query_params.get("anon_id")
    if anon_user_id:
        return AuthUser(user_id=anon_user_id, auth_type="anon_user")

    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization token required")
    try:
        payload = jwt.decode(
            credentials.credentials,
            options={"verify_signature": False},
            algorithms=["RS256"],
        )
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: no user_id found")
        return AuthUser(user_id=user_id, auth_type="auth_user")
    except jwt.exceptions.DecodeError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token format: {str(e)}")