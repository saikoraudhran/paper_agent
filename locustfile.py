import json
import random
import os
from locust import HttpUser, task, between, events

# ── Test users — created in the app first ──
TEST_USERS = [
    {"username": "testuser1", "pin": "1111"},
    {"username": "testuser2", "pin": "2222"},
    {"username": "testuser3", "pin": "3333"},
    {"username": "testuser4", "pin": "4444"},
    {"username": "testuser5", "pin": "5555"},
    {"username": "testuser6", "pin": "6666"},
]

# ── Sample questions to simulate real usage ──
SAMPLE_QUESTIONS = [
    "What is the main methodology used in the papers?",
    "Summarize the key findings",
    "What are the research gaps?",
    "What do all papers agree on?",
    "Explain the introduction section",
    "What statistical methods were used?",
    "Compare the results across papers",
    "What future work is suggested?",
]

SAMPLE_MODES = ["qa", "patterns", "common", "gaps"]


class ResearchUser(HttpUser):
    """
    Simulates a real researcher using the app.
    Each user logs in, sends chat messages, 
    and reads paper list.
    """
    wait_time = between(5, 15)  # realistic think time between actions

    def on_start(self):
        """Called once when a simulated user starts."""
        # Pick a random test user
        self.user_creds = random.choice(TEST_USERS)
        self.logged_in = False
        self.active_paper_ids = []
        self.login()

    def login(self):
        with self.client.post(
            "/auth/login",
            json=self.user_creds,
            catch_response=True
        ) as res:
            if res.status_code == 200:
                self.logged_in = True
                # Get active paper IDs for chat
                papers_res = self.client.get("/papers")
                if papers_res.status_code == 200:
                    papers = papers_res.json()
                    self.active_paper_ids = [
                        p["paper_id"] for p in papers if p.get("active")
                    ]
            else:
                res.failure(f"Login failed: {res.status_code}")

    @task(5)  # weight 5 — most common action
    def send_chat_message(self):
        """Send a chat message and consume the SSE stream."""
        if not self.logged_in or not self.active_paper_ids:
            return

        question = random.choice(SAMPLE_QUESTIONS)
        mode = random.choice(SAMPLE_MODES)

        # SSE streaming — we consume the full stream
        with self.client.post(
            "/chat",
            json={
                "message": question,
                "mode": mode,
                "active_paper_ids": self.active_paper_ids
            },
            stream=True,
            catch_response=True,
            timeout=120  # LLM can take up to 2 min
        ) as res:
            if res.status_code == 200:
                # Consume full stream
                full_content = ""
                for line in res.iter_lines():
                    if line and b"content" in line:
                        try:
                            data = json.loads(line.decode(errors="ignore").replace("data: ", ""))
                            full_content += data.get("text", "")
                        except Exception:
                            pass
                
                if full_content:
                    res.success()
                else:
                    res.failure("Empty response from LLM")
            else:
                res.failure(f"Chat failed: {res.status_code}")

    @task(2)  # weight 2
    def view_papers(self):
        """List all papers."""
        with self.client.get("/papers", catch_response=True) as res:
            if res.status_code == 200:
                res.success()
            else:
                res.failure(f"Papers failed: {res.status_code}")

    @task(1)  # weight 1 — least common
    def check_session(self):
        """Verify session is still valid."""
        with self.client.get("/auth/me", catch_response=True) as res:
            if res.status_code == 200:
                res.success()
            elif res.status_code == 401:
                self.login()  # re-login if session expired
            else:
                res.failure(f"Session check failed: {res.status_code}")

    def on_stop(self):
        """Logout when test ends."""
        self.client.post("/auth/logout")

