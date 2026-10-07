# ResearchAI

RAG-powered academic research assistant for paper writing teams.

## Setup

### 1. Clone and install
```bash
git clone <your-repo>
cd research-rag
pip install -r requirements.txt
```

### 2. Create .env
```bash
cp .env.example .env
# Fill in NVIDIA_API_KEY, ADMIN_PIN, SECRET_KEY
```

### 3. Run locally
```bash
uvicorn backend.main:app --reload --port 8000
```

### 4. Add users (run once after first start)
```
POST /admin/users with admin session cookie
or edit data/users.json directly for initial setup
```

## Deploy to Render
1. Push to GitHub
2. Connect repo on render.com → New Web Service
3. Render auto-detects render.yaml
4. Set secret env vars in Render dashboard:
   - NVIDIA_API_KEY
   - ADMIN_PIN
   - SECRET_KEY
5. Deploy

## First Login
- Username: `admin`
- PIN: (whatever you set in ADMIN_PIN env var)

## Adding Users
```
POST /admin/users
Body: { "username": "alice", "pin": "1234",
        "display_name": "Alice", "role": "user" }
```

