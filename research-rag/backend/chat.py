from openai import OpenAI
from backend.config import (
    NVIDIA_API_KEY, NVIDIA_BASE_URL, LLM_MODEL
)

nvidia_client = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY
)
HUMANIZE_SYSTEM_PROMPT = (
    "You are a meticulous academic copy editor. You rewrite text so it reads "
    "like it was written by a careful human researcher, without changing its meaning."
)
# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert academic research assistant with deep \
knowledge in analyzing, synthesizing, and writing scientific papers. \
You are helping a research team working on their paper.

CORE RULES:
1. Answer ONLY from the provided context chunks. Never hallucinate or use \
outside knowledge unless explicitly asked.
2. ALWAYS cite every claim: [Source: filename.pdf, Page X, Section: Y]
3. Never truncate, summarize, or shorten your answer. Give the COMPLETE, \
DETAILED response every time.
4. If your answer is getting long, finish the current logical section \
completely, then end with:
   ⏭️ CONTINUE PROMPT: "continue [topic]" to get the next part.
5. If the answer cannot be found in the provided context, say exactly:
   "This information is not present in the uploaded papers."
6. Write in formal academic English unless told otherwise.
7. Never say "based on the context" — just state the facts with citations.
8. Format all mathematical formulas, symbols, variables, and equations using standard clean LaTeX ($...$ for inline math, $$...$$ for display blocks) to ensure crisp, elegant mathematical rendering.
9. Organize answers with clear Markdown structure (bold headings, bullet points, and neat paragraphs). Avoid repeating text or phrases.

CITATION FORMAT:
- Inline: [paper2.pdf, p.12, §Methods]
- End of answer:
  ── References Used ──
  [1] filename.pdf — Page X — Section Y — "exact chunk excerpt"
"""

# ─────────────────────────────────────────────
# MODE PROMPTS
# ─────────────────────────────────────────────

MODE_PROMPTS = {

"qa": """MODE: RESEARCH Q&A

Structure your answer as:

DIRECT ANSWER:
[2-3 sentences giving the direct answer immediately]

DETAILED EXPLANATION:
[Full explanation with all relevant details, sub-points, context,
and nuance — as long as needed, never cut short]

SUPPORTING EVIDENCE:
[Every claim backed by citation: paper, page, section, quote]

IMPLICATIONS FOR YOUR RESEARCH:
[How this answer is useful for writing the research paper — 2-3 sentences]

── References Used ──
[Full reference block]

If the answer is long, complete the current section fully then add:
⏭️ CONTINUE PROMPT: "continue answer [your question]"

Context Chunks:
{chunks}

User Request: {user_message}""",

"retrieval": """MODE: DIRECT RETRIEVAL

User wants to retrieve specific text from the papers.

INSTRUCTIONS:
- Return the FULL content of the requested section from the context chunks.
- Do NOT paraphrase. Return content as close to the original as possible,
  preserving structure, numbering, equations, and terminology.
- If the section spans multiple chunks, reconstruct it completely in order.
- Preserve all subsections, bullet points, numbered lists, and paragraphs.
- If exact section not found: "Exact section not found. Closest match: \
[section name, page X]"
- End with full reference block.

If content is long, finish current subsection fully then add:
⏭️ CONTINUE PROMPT: "continue retrieval [section name]"

Context Chunks:
{chunks}

User Request: {user_message}""",

"patterns": """MODE: PATTERN ANALYSIS

INSTRUCTIONS:
- Analyze ALL provided context chunks carefully.
- Identify and group recurring patterns under clear named categories.
- For each pattern:

    PATTERN NAME: [descriptive name]
    DESCRIPTION: [detailed explanation — minimum 3-4 sentences]
    APPEARS IN:
      → [paper1.pdf, p.X, §Section] — "[quote or close paraphrase]"
      → [paper2.pdf, p.X, §Section] — "[quote or close paraphrase]"
    SIGNIFICANCE: [why this pattern matters — 2-3 sentences]
    ──────────────────────────────────────────────

- Cover ALL patterns found. Never stop early.
- If a pattern appears in only 2 papers, include it and note it is partial.
- After all patterns add:

    SYNTHESIS:
    [Detailed paragraph connecting all patterns and what they collectively
     suggest about the field]

If patterns are many, complete each block fully then add:
⏭️ CONTINUE PROMPT: "continue patterns [focus area]"

Context Chunks:
{chunks}

User Request: {user_message}""",

"common": """MODE: COMMON POINTS SYNTHESIS

INSTRUCTIONS:
- A "common point" qualifies if it appears in 3+ papers, OR in 2 papers
  with strong conceptual alignment.
- For each common point:

    ── COMMON POINT #[N] ──────────────────────────
    STATEMENT:
    [Precise, detailed statement — 2-3 sentences]

    EVIDENCE FROM PAPERS:
    → [paper1.pdf, p.X, §Section]:
      "[Direct quote or very close paraphrase]"
      Explanation: [How this paper expresses this point — 2 sentences]

    → [paper2.pdf, p.X, §Section]:
      "[Direct quote or very close paraphrase]"
      Explanation: [How this paper expresses this point — 2 sentences]

    CONSENSUS STRENGTH: [Strong / Moderate / Emerging]
    WHY IT MATTERS:
    [2-3 sentences on significance for your research]
    ────────────────────────────────────────────────

- After all common points add:

    ── DIVERGENCE NOTE ─────────────────────────────
    [Points where papers DISAGREE or contradict each other, with citations]

If list is long, complete each block fully then add:
⏭️ CONTINUE PROMPT: "continue common points"

Context Chunks:
{chunks}

User Request: {user_message}""",

"draft": """MODE: SECTION DRAFTING

Target Section: extracted from user message

INSTRUCTIONS:
- Write a complete, publication-ready draft of the requested section.
- Use ONLY information from provided context chunks as source material.
- Follow standard academic structure for the section type:
    Introduction → Background, Gap, Objective, Contribution
    Literature Review → Thematic grouping, chronological flow, gap analysis
    Methodology → Design, data, procedure, justification
    Results → Findings presented objectively
    Discussion → Interpretation, comparison to literature, implications
    Conclusion → Summary, contributions, future work
- Write in formal academic English, third person.
- Every factual claim MUST have an inline citation: [paper.pdf, p.X]
- Never pad, never cut. Length = what the section genuinely requires.
- Include smooth transitions between paragraphs.
- End with:

    DRAFT NOTES:
    - Sections needing more evidence: [list]
    - Suggested additions from outside the uploaded papers: [list]
    - Potential reviewer concerns: [list]

If draft is long, complete each subsection fully then add:
⏭️ CONTINUE PROMPT: "continue draft [section name]"

Context Chunks:
{chunks}

User Request: {user_message}""",

"gaps": """MODE: RESEARCH GAP ANALYSIS

INSTRUCTIONS:
- Analyze ALL chunks for: limitations stated by authors, future work
  suggestions, contradictions between papers, missing variables,
  understudied areas, unexplored methods, unanswered questions.
- For each gap:

    ── GAP #[N] ────────────────────────────────────
    GAP TITLE: [Short descriptive name]
    TYPE: [Methodological / Theoretical / Empirical / Contextual / Replication]

    DESCRIPTION:
    [What is missing, unexplored, or contradicted — minimum 4-5 sentences]

    EVIDENCE THIS GAP EXISTS:
    → [paper1.pdf, p.X, §Section]: "[quote showing gap or limitation]"
    → [paper2.pdf, p.X, §Section]: "[quote showing gap or limitation]"

    HOW YOUR RESEARCH CAN ADDRESS IT:
    [2-3 sentences on how this gap is a contribution opportunity]

    RESEARCH QUESTION THIS OPENS:
    "[A specific, well-formed research question this gap generates]"
    ────────────────────────────────────────────────

- After all gaps:

    ── PRIORITY GAPS ───────────────────────────────
    [Rank top 3 gaps by research impact with explanation]

If gaps are many, complete each block fully then add:
⏭️ CONTINUE PROMPT: "continue gaps"

Context Chunks:
{chunks}

User Request: {user_message}""",

"continue": """MODE: CONTINUATION

The previous response was cut at a logical break point.
Previous topic: {topic}
Last completed section: {last_section}

INSTRUCTIONS:
- Pick up EXACTLY where the previous answer ended.
- Do NOT repeat anything already said.
- Do NOT add a new introduction or preamble — jump straight into the content.
- Continue with the same format, depth, and citation style as before.
- Apply the same mode rules as the original request: {original_mode}
- If this is the final part, end with:
  ✅ COMPLETE — Full answer delivered across {part_num} messages.

Context Chunks:
{chunks}

User Request: {user_message}""",

"humanize": """MODE: HUMANIZE TEXT
Settings: strength={strength}; tone={tone}

RULES:
- Preserve every fact, number, statistic, equation, abbreviation, technical term.
- Preserve every citation EXACTLY as written e.g. [paper.pdf, p.12, §Methods].
- Do NOT add claims, sources, examples, or opinions.
- Vary sentence length and structure. Mix short and long sentences.
- Avoid: Moreover, Furthermore, In conclusion, It is important to note, filler phrases, stacked hedging.
- Prefer concrete verbs over abstract noun phrases.
- Keep paragraph breaks, headings, and lists intact.
- Strength guide:
    light  = fix stiffness and repetition, keep most sentences
    medium = restructure many sentences, change rhythm
    strong = rewrite substantially while keeping exact meaning
- Output ONLY the rewritten text. No preface, no notes, no explanation.

TEXT TO REWRITE:
{user_message}""",
}


def build_messages(
    mode: str,
    user_message: str,
    chunks_text: str,
    history: list[dict],
    continuation_meta: dict = None,
    humanize_options: dict = None,
) -> list[dict]:
    """
    Build the full messages array for the LLM call.
    Includes system prompt, chat history, and current turn.
    """
    if mode == "humanize":
        # Humanize mode does not use context chunks or history
        opts = humanize_options or {}
        strength = opts.get("strength", "light")
        tone = opts.get("tone", "academic")
        template = MODE_PROMPTS["humanize"]
        user_prompt = template.format(
            strength=strength,
            tone=tone,
            user_message=user_message,
        )
        messages = [{"role": "system", "content": HUMANIZE_SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": user_prompt})
        return messages
    if mode == "continue" and continuation_meta:
        user_prompt = MODE_PROMPTS["continue"].format(
            topic=continuation_meta.get("topic", ""),
            last_section=continuation_meta.get("last_section", ""),
            original_mode=continuation_meta.get("original_mode", "qa"),
            part_num=continuation_meta.get("part_num", 2),
            chunks=chunks_text,
            user_message=user_message,
        )
    else:
        template = MODE_PROMPTS.get(mode, MODE_PROMPTS["qa"])
        user_prompt = template.format(
            chunks=chunks_text,
            user_message=user_message,
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    return messages


def stream_response(messages: list[dict], thinking: bool = True, temperature: float = 0.6):
    """
    Generator that streams reasoning + content tokens
    from NVIDIA Nemotron.
    Yields: {"type": "reasoning"|"content"|"done", "text": str}
    """
    completion = nvidia_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        top_p=0.95,
        max_tokens=4096,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        stream=True
    )

    for chunk in completion:
        if not chunk.choices:
            continue

        reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
        content = chunk.choices[0].delta.content

        if reasoning:
            yield {"type": "reasoning", "text": reasoning}
        if content:
            yield {"type": "content", "text": content}

    yield {"type": "done", "text": ""}
