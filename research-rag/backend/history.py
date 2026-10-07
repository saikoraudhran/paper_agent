import json
from pathlib import Path
from backend.config import HISTORIES_PATH, HISTORY_DEPTH


def _history_file(username: str) -> Path:
    return HISTORIES_PATH / f"{username}.json"


def load_history(username: str) -> list[dict]:
    """Load full chat history for a user."""
    f = _history_file(username)
    if not f.exists():
        return []
    with open(f) as file:
        return json.load(file)


def save_history(username: str, history: list[dict]):
    """Save full chat history for a user."""
    with open(_history_file(username), "w") as f:
        json.dump(history, f, indent=2)


def append_message(username: str, role: str, content: str,
                   reasoning: str = None, mode: str = None,
                   citations: list = None):
    """Append a single message to user history."""
    history = load_history(username)
    entry = {
        "role": role,
        "content": content,
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S")
    }
    if reasoning:
        entry["reasoning"] = reasoning
    if mode:
        entry["mode"] = mode
    if citations:
        entry["citations"] = citations
    history.append(entry)
    save_history(username, history)


def get_recent_for_llm(username: str) -> list[dict]:
    """
    Get last N messages formatted for LLM context.
    Only includes role + content (no reasoning/metadata).
    """
    history = load_history(username)
    recent = history[-HISTORY_DEPTH:] if len(history) > HISTORY_DEPTH else history
    return [{"role": h["role"], "content": h["content"]} for h in recent]


def clear_history(username: str):
    """Clear all chat history for a user."""
    save_history(username, [])


def export_history_markdown(username: str) -> str:
    """Export full history as markdown string."""
    history = load_history(username)
    lines = [f"# Chat Export — {username}\n"]

    for msg in history:
        role_label = "**You**" if msg["role"] == "user" else "**ResearchAI**"
        timestamp = msg.get("timestamp", "")
        mode = msg.get("mode", "")

        lines.append(f"### {role_label} _{timestamp}_ {f'[{mode}]' if mode else ''}")

        if msg.get("reasoning"):
            lines.append(f"\n> \U0001f9e0 **Reasoning:**\n> {msg['reasoning'][:500]}...\n")

        lines.append(f"\n{msg['content']}\n")

        if msg.get("citations"):
            lines.append("\n**Sources:**")
            for c in msg["citations"]:
                lines.append(
                    f"- {c['filename']} — p.{c['page_num']} — §{c['section_name']}"
                )

        lines.append("\n---\n")

    return "\n".join(lines)
