import json
import hashlib
import time
from pathlib import Path
from itsdangerous import TimestampSigner, BadSignature, SignatureExpired
from fastapi import Request, HTTPException, Response
from backend.config import (
    SECRET_KEY, SESSION_MAX_AGE, USERS_FILE,
    ADMIN_USERNAME, ADMIN_PIN
)

signer = TimestampSigner(SECRET_KEY)


def _hash_pin(pin: str) -> str:
    return hashlib.sha256(pin.encode()).hexdigest()


def _load_users() -> dict:
    if not USERS_FILE.exists():
        # Create default users file with admin
        users = {
            ADMIN_USERNAME: {
                "pin_hash": _hash_pin(ADMIN_PIN),
                "display_name": "Admin",
                "role": "admin"
            }
        }
        _save_users(users)
        return users
    with open(USERS_FILE) as f:
        return json.load(f)


def _save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def verify_login(username: str, pin: str) -> dict | None:
    """Returns user dict if valid, None if invalid."""
    users = _load_users()
    user = users.get(username.lower())
    if not user:
        return None
    if user["pin_hash"] != _hash_pin(pin):
        return None
    return {
        "username": username.lower(),
        "display_name": user["display_name"],
        "role": user["role"]
    }


def create_session_cookie(response: Response, username: str):
    """Signs username into a session cookie."""
    token = signer.sign(username).decode()
    response.set_cookie(
        key="session",
        value=token,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax"
    )


def get_current_user(request: Request) -> dict:
    """Reads and validates session cookie. Raises 401 if invalid."""
    token = request.cookies.get("session")
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in")
    try:
        username = signer.unsign(
            token, max_age=SESSION_MAX_AGE
        ).decode()
    except (BadSignature, SignatureExpired):
        raise HTTPException(status_code=401, detail="Session expired")
    users = _load_users()
    user = users.get(username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return {
        "username": username,
        "display_name": user["display_name"],
        "role": user["role"]
    }


def require_admin(request: Request) -> dict:
    """Like get_current_user but also checks admin role."""
    user = get_current_user(request)
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return user


def add_user(username: str, pin: str, display_name: str, role: str = "user"):
    """Admin utility to add a new user."""
    users = _load_users()
    users[username.lower()] = {
        "pin_hash": _hash_pin(pin),
        "display_name": display_name,
        "role": role
    }
    _save_users(users)


def list_users() -> list:
    users = _load_users()
    return [
        {"username": k, "display_name": v["display_name"], "role": v["role"]}
        for k, v in users.items()
    ]
