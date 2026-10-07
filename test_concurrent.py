import asyncio
import aiohttp
import time
import os

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

USERS = [
    {"username": "testuser1", "pin": "1111"},
    {"username": "testuser2", "pin": "2222"},
    {"username": "testuser3", "pin": "3333"},
    {"username": "testuser4", "pin": "4444"},
    {"username": "testuser5", "pin": "5555"},
    {"username": "testuser6", "pin": "6666"},
]

QUESTIONS = [
    "What is the main methodology?",
    "What are the key findings?",
    "What research gaps exist?",
    "Summarize the introduction",
    "What do all papers agree on?",
    "What future work is suggested?",
]


async def simulate_user(session, user_creds, question, user_num):
    """Simulate one user: login → get papers → chat."""
    start = time.time()
    results = {"user": user_num, "username": user_creds["username"], "errors": []}

    # 1. Login
    async with session.post(f"{BASE_URL}/auth/login", json=user_creds) as res:
        if res.status != 200:
            results["errors"].append(f"Login failed: {res.status}")
            return results

    # 2. Get papers
    async with session.get(f"{BASE_URL}/papers") as res:
        if res.status != 200:
            results["errors"].append(f"Get papers failed: {res.status}")
            return results
        papers = await res.json()
        active_ids = [p["paper_id"] for p in papers if p.get("active")]

    if not active_ids:
        results["errors"].append("No active papers found")
        return results

    # 3. Send chat message — consume full SSE stream
    chat_start = time.time()
    content_received = False

    async with session.post(
        f"{BASE_URL}/chat",
        json={
            "message": question,
            "mode": "qa",
            "active_paper_ids": active_ids
        },
        timeout=aiohttp.ClientTimeout(total=180)
    ) as res:
        if res.status != 200:
            results["errors"].append(f"Chat failed: {res.status}")
            return results

        async for line in res.content:
            decoded = line.decode(errors="ignore").strip()
            if "content" in decoded and "data:" in decoded:
                content_received = True

    results["chat_time"] = round(time.time() - chat_start, 2)
    results["total_time"] = round(time.time() - start, 2)
    results["got_response"] = content_received

    return results


async def run_concurrent_test():
    """Run all 6 users simultaneously."""
    print(f"\n{'='*55}")
    print(f"Running concurrent test with {len(USERS)} users against {BASE_URL}")
    print(f"{'='*55}\n")

    connector = aiohttp.TCPConnector(limit=20)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            simulate_user(session, USERS[i], QUESTIONS[i], i + 1)
            for i in range(len(USERS))
        ]

        start = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total = round(time.time() - start, 2)

    # Report
    print(f"All {len(USERS)} users completed in {total}s\n")
    all_passed = True
    for r in results:
        if isinstance(r, Exception):
            print(f"❌ Exception: {r}")
            all_passed = False
        elif r.get("errors"):
            print(f"❌ User {r['user']} ({r['username']}): FAILED — {r['errors']}")
            all_passed = False
        else:
            print(
                f"✅ User {r['user']} ({r['username']}): "
                f"chat={r.get('chat_time')}s  "
                f"total={r.get('total_time')}s  "
                f"response={'yes' if r.get('got_response') else 'NO'}"
            )
            if not r.get("got_response"):
                all_passed = False

    print("\n" + ("=" * 55))
    if all_passed:
        print("🎉 SUCCESS: All 6 concurrent users handled flawlessly!")
    else:
        print("⚠️ Some concurrent requests encountered issues. Check errors above.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    asyncio.run(run_concurrent_test())

