import os
import requests

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
ADMIN_CREDS = {
    "username": os.getenv("ADMIN_USERNAME", "admin"),
    "pin": os.getenv("ADMIN_PIN", "1234")
}

TEST_USERS = [
    {"username": "testuser1", "pin": "1111", "display_name": "Test User 1"},
    {"username": "testuser2", "pin": "2222", "display_name": "Test User 2"},
    {"username": "testuser3", "pin": "3333", "display_name": "Test User 3"},
    {"username": "testuser4", "pin": "4444", "display_name": "Test User 4"},
    {"username": "testuser5", "pin": "5555", "display_name": "Test User 5"},
    {"username": "testuser6", "pin": "6666", "display_name": "Test User 6"},
]

session = requests.Session()
login_res = session.post(f"{BASE_URL}/auth/login", json=ADMIN_CREDS)
if login_res.status_code != 200:
    print(f"❌ Admin login failed: {login_res.status_code} - {login_res.text}")
    exit(1)

print("✅ Admin logged in successfully")
for user in TEST_USERS:
    res = session.post(f"{BASE_URL}/admin/users", json={**user, "role": "user"})
    print(f"{'✅' if res.status_code == 200 else '❌'} {user['username']}: {res.status_code}")

