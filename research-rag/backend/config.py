import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Tiktoken cache directory for offline tokenizer loading
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(__file__).resolve().parent.parent / "data" / "tiktoken_cache"))

# NVIDIA NIM
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
LLM_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"
EMBEDDING_DIMENSION = 2048

# Auth
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PIN = os.getenv("ADMIN_PIN", "1234")
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
SESSION_MAX_AGE = 8 * 60 * 60  # 8 hours in seconds

# Paths
DATA_PATH = Path(os.getenv("DATA_PATH", "./data"))
UPLOADS_PATH = DATA_PATH / "uploads"
CHROMA_PATH = DATA_PATH / "chroma"
HISTORIES_PATH = DATA_PATH / "histories"
USERS_FILE = DATA_PATH / "users.json"

# Ensure all directories exist
for path in [UPLOADS_PATH, CHROMA_PATH, HISTORIES_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Chunking
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
TOP_K_RETRIEVAL = 5

# Chat
HISTORY_DEPTH = 6  # last 6 messages (3 user + 3 AI)

# Section headings to detect in PDFs
SECTION_HEADINGS = [
    "abstract", "introduction", "background",
    "literature review", "related work",
    "methodology", "methods", "approach",
    "results", "findings", "evaluation",
    "discussion", "analysis",
    "conclusion", "conclusions", "future work",
    "references", "bibliography", "appendix"
]

# Humanize
HUMANIZE_MAX_WORDS = 3000
HUMANIZE_STRENGTHS = ["light", "medium", "strong"]
HUMANIZE_TONES = ["academic", "neutral", "conversational"]
