"""
AI Co-Author - collaborative storytelling with an LLM co-writer.

Uses OpenRouter (Qwen 3 235B) for story generation. Built with Streamlit.

"""

import os
import re
import json
import html
import time
import uuid
import random
import logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager  # for the processing_state lock

import streamlit as st
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# log format is minimal on purpose - streamlit's own logging is noisy enough
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("story_weaver")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = "qwen/qwen3-235b-a22b"  # switched from smaller Qwen models after quality issues

# kept to 6 genres - tried adding Thriller and Western but the LLM couldn't
# reliably distinguish them from Mystery and Fantasy respectively
GENRES = ["Fantasy", "Sci-Fi", "Mystery", "Romance", "Horror", "Comedy"]
HISTORY_FILE = Path(__file__).parent / "story_history.json"
MAX_UNDO_LEVELS = 10  # anything more than this and the user probably needs to start over

VOCAB_LABELS = {1: "Simple", 2: "Moderate", 3: "Rich"}

# short blurbs shown in the sidebar - just so the user knows what they're getting
GENRE_RULES = {
    "Fantasy": "Magic, quests, and imagined worlds.",
    "Sci-Fi": "Future tech, space, and science.",
    "Mystery": "Clues, secrets, and whodunits.",
    "Romance": "Love, chemistry, and connection.",
    "Horror": "Dread, darkness, and survival.",
    "Comedy": "Laughs, chaos, and bad luck.",
}


# Custom exceptions for LLM errors (so the UI layer can catch specific cases)
class LLMRateLimitExceeded(Exception): pass
class LLMRequestError(Exception): pass



# Theme CSS - Warm Minimalism
# Most of these selectors were found by inspecting Streamlit's DOM.
# They use baseweb components internally which is why the selectors look weird.
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* tested on streamlit 1.45 - some of these selectors break across versions
   because streamlit doesn't expose official styling hooks for everything */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=Lora:ital,wght@0,400;0,500;0,600;1,400&display=swap');

:root {
    --alabaster: #F8F5F2;
    --taupe: #D5CABD;
    --taupe-light: #EDE8E3;
    --terracotta: #C87961;
    --terracotta-hover: #B56A53;
    --charcoal: #332E2C;
    --charcoal-light: #5A5350;
    --cream: #FAF8F6;
    --radius: 6px;  /* used everywhere, easier to change in one place */
}

/* global styles - Lora for body text, Cormorant Garamond for headings.
   Picked these because they feel literary without being stuffy. */
.stApp {
    font-family: 'Lora', Georgia, serif !important;
    color: var(--charcoal) !important;
    background-color: var(--alabaster) !important;
}

/* keep deploy button visible, hide everything else in the header */
header[data-testid="stHeader"] {
    background: var(--alabaster) !important;
}
.stDeployButton { visibility: visible !important; }

/* Streamlit uses baseweb modals under the hood - these selectors
   are fragile and will probably break on a major streamlit update */
div[data-baseweb="modal"] div[data-baseweb="modal-body"],
div[data-baseweb="modal"] div[data-baseweb="modal-header"],
div[data-baseweb="modal"] div[data-baseweb="modal-footer"],
div[data-baseweb="modal"] > div > div {
    background: var(--cream) !important;
    background-color: var(--cream) !important;
    color: var(--charcoal) !important;
}
div[data-baseweb="modal"] h1,
div[data-baseweb="modal"] h2,
div[data-baseweb="modal"] h3,
div[data-baseweb="modal"] p,
div[data-baseweb="modal"] span,
div[data-baseweb="modal"] label,
div[data-baseweb="modal"] div {
    color: var(--charcoal) !important;
}
div[data-baseweb="modal"] code {
    background: var(--taupe-light) !important;
    color: var(--charcoal) !important;
    border: none !important;
}
/* modal buttons */
div[data-baseweb="modal"] button {
    font-family: 'Lora', Georgia, serif !important;
    border-radius: 6px !important;
}
div[data-baseweb="modal"] button[kind="primary"],
div[data-baseweb="modal"] button[data-baseweb="button"][kind="primary"] {
    background: var(--terracotta) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--terracotta) !important;
}
div[data-baseweb="modal"] button[kind="primary"]:hover {
    background: var(--terracotta-hover) !important;
}
div[data-baseweb="modal"] button[kind="secondary"],
div[data-baseweb="modal"] button[kind="tertiary"],
div[data-baseweb="modal"] button:not([kind="primary"]) {
    background: var(--cream) !important;
    color: var(--charcoal) !important;
    border: 1px solid var(--taupe) !important;
}
div[data-baseweb="modal"] button:not([kind="primary"]):hover {
    background: var(--taupe-light) !important;
}
/* modal inputs */
div[data-baseweb="modal"] input,
div[data-baseweb="modal"] select,
div[data-baseweb="modal"] div[data-baseweb="select"] > div {
    background: var(--cream) !important;
    color: var(--charcoal) !important;
    border: 1px solid var(--taupe) !important;
    border-radius: 6px !important;
}
/* overlay backdrop */
div[data-baseweb="modal-backdrop"] {
    background: rgba(51, 46, 44, 0.4) !important;
}
/* hide the theme picker in settings */
div[data-baseweb="modal"] div[data-testid="stThemeSettings"],
div[data-baseweb="modal"] .stThemeSettings {
    display: none !important;
}

/* top-right settings menu */
div[data-testid="stMainMenu"] ul,
div[data-testid="stMainMenu"] li,
div[data-testid="stMainMenu"] div {
    background: var(--cream) !important;
    color: var(--charcoal) !important;
}
div[data-testid="stMainMenu"] li:hover {
    background: var(--taupe-light) !important;
}

/* Dropdown - found these selectors through Chrome DevTools.
   Streamlit uses baseweb's Select component internally */
div[data-baseweb="popover"] {
    background: var(--cream) !important;
    border: 1px solid var(--taupe) !important;
    border-radius: 6px !important;
}
div[data-baseweb="popover"] ul {
    background: var(--cream) !important;
}
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] li span,
ul[role="listbox"] li,
ul[role="listbox"] li span {
    background: var(--cream) !important;
    color: var(--charcoal) !important;
    font-family: 'Lora', Georgia, serif !important;
}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] li:focus,
div[data-baseweb="popover"] li[aria-selected="true"],
ul[role="listbox"] li:hover,
ul[role="listbox"] li:focus,
ul[role="listbox"] li[aria-selected="true"] {
    background: var(--taupe-light) !important;
    color: var(--charcoal) !important;
}
/* selected/hovered state */
div[data-baseweb="menu"] {
    background: var(--cream) !important;
}
div[data-baseweb="menu"] li[data-highlighted="true"],
div[data-baseweb="menu"] li[data-highlighted="true"] span {
    background: var(--taupe-light) !important;
    color: var(--charcoal) !important;
}

/* Tooltips - these were annoying to style, streamlit renders them
   outside the normal DOM hierarchy */
/* tried div[data-baseweb="tooltip"] alone first but it missed the
   nested content divs, had to target both levels */
div[data-baseweb="tooltip"],
div[data-baseweb="tooltip"] div,
div[role="tooltip"],
div[role="tooltip"] div,
div[data-testid="stTooltipContent"] {
    background: var(--charcoal) !important;
    background-color: var(--charcoal) !important;
    color: var(--alabaster) !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Lora', Georgia, serif !important;
    font-size: 0.82rem !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
}
div[data-baseweb="tooltip"] p,
div[data-baseweb="tooltip"] span {
    color: var(--alabaster) !important;
}

/* code blocks */
pre,
code,
.stCode,
div[data-testid="stCode"],
div[data-testid="stCode"] pre,
div[data-testid="stCode"] code {
    background: var(--taupe-light) !important;
    background-color: var(--taupe-light) !important;
    color: var(--charcoal) !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
}
/* copy button inside code blocks */
div[data-testid="stCode"] button {
    color: var(--charcoal-light) !important;
    background: transparent !important;
    border: none !important;
}

/* hide the spinner gif but keep the status text */
.stStatusWidget svg {
    display: none !important;
}
.stStatusWidget img {
    display: none !important;
}

/* sidebar */
section[data-testid="stSidebar"] {
    left: 0 !important;
    background-color: var(--taupe-light) !important;
    border-right: 1px solid var(--taupe) !important;
}
section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1.2rem !important;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.03em !important;
    color: var(--charcoal) !important;
    text-transform: uppercase !important;
    margin-bottom: 0.6rem !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span {
    font-size: 0.88rem !important;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--taupe) !important;
    opacity: 0.5 !important;
}

/* center the main content area */
.stMainBlockContainer {
    max-width: 920px !important;
    margin: 0 auto !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
.block-container {
    padding-top: 2rem !important;
}

/* typography */
h1, h2, h3 {
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    color: var(--charcoal) !important;
}
h2 {
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
}
p, li, span, div {
    color: var(--charcoal) !important;
}

/* buttons */
div[data-testid="stButton"] > button {
    font-family: 'Lora', Georgia, serif !important;
    font-size: 0.85rem !important;
    border-radius: 6px !important;
    border: 1px solid var(--taupe) !important;
    background: var(--cream) !important;
    color: var(--charcoal) !important;
    transition: all 0.2s ease !important;
    padding: 0.5rem 1rem !important;
}
div[data-testid="stButton"] > button:hover {
    background: var(--taupe-light) !important;
    border-color: var(--charcoal-light) !important;
}

/* primary action button */
div[data-testid="stButton"] > button[kind="primary"] {
    background: var(--terracotta) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--terracotta) !important;
    font-weight: 600 !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: var(--terracotta-hover) !important;
    border-color: var(--terracotta-hover) !important;
}

/* disabled state */
div[data-testid="stButton"] > button:disabled {
    opacity: 0.45 !important;
    cursor: not-allowed !important;
}

/* text inputs and textareas */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    font-family: 'Lora', Georgia, serif !important;
    background: var(--cream) !important;
    border: 1px solid var(--taupe) !important;
    border-radius: 6px !important;
    color: var(--charcoal) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--terracotta) !important;
    box-shadow: 0 0 0 1px var(--terracotta) !important;
}


.stSlider > div > div > div > div {
    background: var(--terracotta) !important;
}

/* alerts */
div[data-testid="stAlert"] {
    font-family: 'Lora', Georgia, serif !important;
    border-radius: 6px !important;
    font-size: 0.88rem !important;
}


hr {
    border-color: var(--taupe) !important;
    opacity: 0.4 !important;
}

/* download btn */
.stDownloadButton > button {
    font-family: 'Lora', Georgia, serif !important;
    background: var(--cream) !important;
    border: 1px solid var(--taupe) !important;
    color: var(--charcoal) !important;
    border-radius: 6px !important;
}
.stDownloadButton > button:hover {
    background: var(--taupe-light) !important;
}

/* story text blocks - user segments get a left border to visually
   separate "your words" from "AI words". Tried background color
   difference first but it was too distracting while reading. */
.story-segment-ai {
    padding: 0.6rem 0;  /* 1rem felt too loose, 0.4 too tight */
    margin-bottom: 1rem;
    line-height: 1.85;  /* generous for readability - standard 1.6 felt cramped for fiction */
    text-align: justify;
    font-size: 1.02rem;
    color: var(--charcoal);
}
.story-segment-user {
    background: rgba(213, 202, 189, 0.2);
    border-left: 3px solid var(--terracotta);
    padding: 0.8rem 1.2rem;
    border-radius: 4px;
    margin-bottom: 1rem;
    text-align: justify;
    line-height: 1.85;
    font-size: 1.02rem;
    color: var(--charcoal);
}

/* branching choice cards */
.choice-btn {
    background: var(--cream);
    border: 1px solid var(--taupe);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: left;
    font-family: 'Lora', Georgia, serif;
    color: var(--charcoal);
}
.choice-btn:hover {
    border-color: var(--terracotta);
    background: rgba(200, 121, 97, 0.06);
}

/* history cards in sidebar */
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div:first-child button {
    text-align: center !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 0.75rem !important;
    line-height: 1.35 !important;
    white-space: normal !important;
    border: 1px solid var(--taupe) !important;
    background: var(--cream) !important;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div:first-child button:hover {
    background: var(--alabaster) !important;
    border-color: var(--terracotta) !important;
}
/* Delete button on history cards: hidden until hover.
   Spent forever figuring out that streamlit wraps each column
   in stHorizontalBlock > div, not a direct child. */
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div:last-child {
    opacity: 0;
    transition: opacity 0.25s ease;
    display: flex;
    align-items: center;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:hover > div:last-child {
    opacity: 1;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div:last-child button {
    border: none !important;
    background: transparent !important;
    color: var(--charcoal-light) !important;
    font-size: 0.7rem !important;
    padding: 0.15rem 0.35rem !important;
    min-height: 0 !important;
    opacity: 0.5;
    transition: opacity 0.2s ease;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div:last-child button:hover {
    opacity: 1 !important;
    background: transparent !important;
    color: var(--terracotta) !important;
}

/* story page header */
.story-header {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--charcoal);
    margin: 0 0 0.15rem 0;
    letter-spacing: -0.01em;
}
.story-meta {
    font-family: 'Lora', Georgia, serif;
    color: var(--charcoal-light);
    font-size: 0.85rem;
    margin: 0 0 0.5rem 0;
    letter-spacing: 0.02em;
}

/* setup/landing page */
.setup-title {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--charcoal);
    margin-bottom: 0.1rem;
}
.setup-subtitle {
    font-family: 'Lora', Georgia, serif;
    color: var(--charcoal-light);
    font-size: 0.95rem;
    font-style: italic;
    margin-bottom: 1.5rem;
}

/* processing status text */
.processing-msg {
    font-family: 'Lora', Georgia, serif;
    color: var(--charcoal-light);
    font-style: italic;
    font-size: 0.9rem;
    padding: 0.5rem 0;
}
</style>
"""


# -- json / validation helpers --

def extract_json(raw_text):
    """Tries to pull JSON out of whatever the LLM spits back."""
    s = raw_text.strip()
    # peel off markdown fences - Qwen wraps JSON in these more often than not
    if s.startswith("```"):
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline + 1:]
        else:
            s = s[3:]  # edge case: everything on one line like ```[1,2,3]```
    if s.rstrip().endswith("```"):
        s = s.rstrip()[:-3]
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # plan B: the model sometimes puts "Here is the JSON:" or similar
    # before the actual data - find the earliest [ or { and try from there
    first_bracket = s.find("[")
    first_brace = s.find("{")
    valid_starts = [p for p in (first_bracket, first_brace) if p > 0]
    if valid_starts:
        start_at = min(valid_starts)
        try:
            return json.loads(s[start_at:])
        except json.JSONDecodeError as e:
            log.debug(f"fallback parse failed at position {start_at}: {e}")
    log.warning("gave up trying to parse JSON from LLM output")
    return None


def validate_choices(choices_data):
    """Make sure we got 3 usable choices. The model sometimes returns
    4 or 5 so we only look at the first 3, and sometimes it gives
    empty labels which look broken in the UI."""
    if not isinstance(choices_data, list) or len(choices_data) < 3:
        return False
    for choice in choices_data[:3]:
        if not isinstance(choice, dict):
            return False
        label = choice.get("label", "")
        desc = choice.get("description", "")
        if not label or not desc:
            return False
    return True


def validate_characters(char_data):
    """Keep only well-formed character entries. The model occasionally
    returns entries with just a name and no description, or vice versa."""
    clean = []
    for c in char_data:
        if not isinstance(c, dict):
            continue
        name = c.get("name", "").strip()
        desc = c.get("description", "").strip()
        if name and desc:
            clean.append({"name": name, "description": desc})
    return clean



# History - saves stories to a local JSON file so users can pick up where
# they left off. Not the most scalable solution, but works great for a
# single-user prototype.
# ---------------------------------------------------------------------------

def load_history():
    """Pull saved stories from disk. Returns empty list if anything goes wrong -
    we've seen the file get corrupted when streamlit crashes mid-write."""
    if not HISTORY_FILE.exists():
        return []
    try:
        raw = HISTORY_FILE.read_text(encoding="utf-8")
        if not raw.strip():
            return []  # empty file, don't bother parsing
        stories = json.loads(raw)
        return stories if isinstance(stories, list) else []
    except (json.JSONDecodeError, IOError, OSError):
        return []


def save_history(stories):
    """Write stories to disk. indent=2 so I can actually read the file
    when debugging. Non-critical so we just log if it fails."""
    try:
        HISTORY_FILE.write_text(
            json.dumps(stories, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except (IOError, OSError) as e:
        log.warning("History file write failed: %s", e)


def save_current_story():
    """Snapshot the current story to the history file.
    If the story already exists (by id), we update it in place."""
    if not st.session_state.story_segments:
        return

    snapshot = {
        "id": st.session_state.story_id,
        "title": st.session_state.story_title,
        "genre": st.session_state.genre,
        "initial_hook": st.session_state.initial_hook,
        "temperature": st.session_state.get("temperature", 0.8),
        "vocab_level": st.session_state.get("vocab_level", 1),
        "segments": st.session_state.story_segments,
        "characters": st.session_state.characters,
        "updated_at": datetime.now().isoformat(),
    }

    stories = load_history()
    # update in place if this story already exists
    for i, existing in enumerate(stories):
        if existing.get("id") == snapshot["id"]:
            stories[i] = snapshot
            save_history(stories)
            return
    stories.insert(0, snapshot)
    # don't let the history file grow forever
    if len(stories) > 50:
        stories = stories[:50]
    save_history(stories)


def delete_story(story_id):
    stories = [s for s in load_history() if s.get("id") != story_id]
    save_history(stories)


def load_story_into_session(story):
    """Restore a saved story. Uses .get() with defaults because older saves
    might be missing fields we added later (like vocab_level)."""
    # pull story fields into session state
    # note: JSON keys don't always match session_state keys (e.g. "id" vs "story_id",
    # "segments" vs "story_segments") - this was an early naming mistake I never fixed
    fields = {
        "story_id": story.get("id", str(uuid.uuid4())),
        "story_title": story.get("title", "Untitled"),
        "genre": story.get("genre", "Fantasy"),
        "initial_hook": story.get("initial_hook", ""),
        "story_segments": story.get("segments", []),
        "characters": story.get("characters", []),
        "temperature": story.get("temperature", 0.8),
        "vocab_level": story.get("vocab_level", 1),
    }
    st.session_state.update(fields)
    # wipe transient state from whatever story was loaded before
    st.session_state.undo_stack = []
    st.session_state.pending_choices = None
    st.session_state.error_msg = ""
    st.session_state.last_viz_prompt = ""
    st.session_state.processing = False
    st.session_state.page = "story"


# ---------------------------------------------------------------------------
# LLM Client - OpenRouter wrapper. The visible countdown was a deliberate
# choice over silent retries because users kept thinking the app had frozen.
# ---------------------------------------------------------------------------

@st.cache_resource
def get_client():
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)


def call_llm(system_prompt, user_prompt, temperature=0.8, max_retries=3, base_wait=15):
    """Hit the LLM and return the text. Retries with a visible countdown
    if we get rate-limited or the connection drops."""
    client = get_client()
    last_error = None

    attempt = 0
    while attempt < max_retries:
        try:
            # not using streaming because it complicates error handling
            # and the responses are short enough that it doesn't matter
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=1024,  # higher values made the model ramble
            )
            output = resp.choices[0].message.content or ""
            # Qwen3 has a "thinking" mode that wraps reasoning in <think> tags.
            # Users don't need to see the model arguing with itself
            if "<think>" in output:
                output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
            return output

        except RateLimitError as e:
            last_error = e
            err_type = "rate_limit"
        except (APIConnectionError, APITimeoutError) as e:
            last_error = e
            err_type = "connection"
        except Exception as e:
            raise LLMRequestError(f"Unexpected: {e}")

        # if we get here, something went wrong but it might be transient
        attempt += 1
        if attempt >= max_retries:
            break

        wait_for = base_wait * attempt
        msg = "Rate limit hit" if err_type == "rate_limit" else "Connection issue"
        ticker = st.empty()
        for sec_left in range(wait_for, 0, -1):
            ticker.warning(f"{msg} - retrying in **{sec_left}s** (attempt {attempt + 1}/{max_retries})")
            time.sleep(1)
        ticker.empty()

    # exhausted all retries
    if isinstance(last_error, RateLimitError):
        raise LLMRateLimitExceeded("Still rate-limited after all retries")
    raise LLMRequestError(f"Gave up after {max_retries} attempts: {last_error}")



# Prompts - each vocab level gets its own prompt because a
# single prompt with a "be simple" footnote doesn't actually work. Learned
# that the hard way after multiple iterations.
#
# The caps and headers in the prompts aren't for aesthetics - Qwen pays
# more attention to stuff at the top and stuff that's visually distinct.
# ---------------------------------------------------------------------------

def _build_simple_prompt(genre: str, title: str) -> str:
    """Prompt for Simple mode. The bad/good examples below are actual
    outputs from earlier versions that kept ignoring the 'be simple' instruction.
    Adding concrete examples was the only thing that reliably worked."""
    return f"""IMPORTANT - this story must be written at a simple reading level.

You are a children's book author writing a {genre} story titled "{title}".
Write exactly like Percy Jackson, Goosebumps, or Magic Tree House.
Use ONLY simple, everyday words that a 9-year-old would use in daily life.
Every sentence must be short - 12 words max.
No similes (no "like" or "as" comparisons). No metaphors. Just say what happens.

Here's what I mean. The model kept producing stuff like this, and I need you to avoid it:

  Don't write: "The amber glow of dusk seeped through gauzy curtains, casting long shadows"
  Write this:  "The evening sun came through the thin curtains. It made long shadows on the floor."

  Don't write: "dust motes danced like spectral confetti in the fading light"
  Write this:  "Tiny bits of dust floated in the air."

  Don't write: "his grief a second skin, unrelenting and visceral"
  Write this:  "He felt sad all the time. It never went away."

  Don't write: "the car shot forward, swallowing the horizon whole"
  Write this:  "The car went really fast. Everything behind them got smaller."

  Don't write: "its hull pulsing like live wires"
  Write this:  "Its body glowed and buzzed."

  Don't write: "a wolf with fur like liquid starlight and eyes that burned with ancient hunger"
  Write this:  "A wolf stood there. Its fur was bright silver. Its eyes glowed."

Match the "Write this" examples above. Every sentence.
If you catch yourself writing "like" or "as" to compare two things, rewrite it.

The story is {genre}, but stick to everyday basic words - no complex {genre.lower()} jargon, ancient-sounding language, or heavy atmosphere. Write in third person, 1-2 paragraphs (100-200 words). Keep the plot moving, don't repeat what already happened, and weave in any text the reader contributed. Output only story text, nothing else.

Before you respond, check: would a 9-year-old understand every word? Is any sentence over 12 words? Did you use "like" or "as" to compare things? Fix any that fail."""


def _build_moderate_prompt(genre: str, title: str) -> str:
    """Middle ground - Stephen King level. Clear and engaging, no purple prose."""
    return f"""You are a collaborative storyteller co-writing a mainstream {genre} story titled "{title}".
Write like a fast-paced modern paperback - think Stephen King or Brandon Sanderson. Clear, engaging prose for a general adult audience. The vocabulary can be varied and adult, but never overly poetic or purple. Focus on action and concrete descriptions.

Stay consistent with everything established so far - review the story before writing. Write in third person, 1-2 paragraphs (~100-200 words). Keep things in the {genre} genre with standard tropes, but keep the language grounded. No meta-commentary, no "Chapter X" headers unless asked. Advance the plot, don't rehash. If the reader adds text, treat it as part of the story.

Output only the story continuation. Keep every word accessible - if a simpler word works just as well, use it."""


def _build_rich_prompt(genre: str, title: str) -> str:
    """Full literary mode - Tolkien, McCarthy, Rothfuss. Go wild."""
    return f"""You are a masterful literary author co-writing an epic {genre} story titled "{title}".

Consistency comes first - never contradict anything established in the story so far. Review every prior paragraph before writing. Write in vivid, poetic third-person prose with rich descriptions. Stay firmly in the {genre} genre and lean into deep atmosphere and imagery. Keep continuations to 1-2 focused paragraphs (~100-200 words).

Advance the plot meaningfully - no filler. Maintain the tone throughout. If the reader contributes text, work it naturally into the narrative. Output only the story continuation, no commentary or labels.

Write with the prose quality of Cormac McCarthy, Tolkien, or Patrick Rothfuss - expansive vocabulary, vivid metaphors, sensory details, complex rhythmic sentences."""


def build_system_prompt(genre: str, title: str) -> str:
    """Pick the right prompt based on vocab level. Each level is a totally
    different prompt structure - not just a different footnote."""
    vocab = st.session_state.get("vocab_level", 1)
    if vocab == 1:
        return _build_simple_prompt(genre, title)
    elif vocab == 2:
        return _build_moderate_prompt(genre, title)
    return _build_rich_prompt(genre, title)


def build_choices_system_prompt(genre: str, title: str) -> str:
    """Asks the LLM for 3 branching story options in JSON format."""
    return f"""You are co-writing a {genre} story titled "{title}".

Based on the story so far, come up with 3 different directions the plot could go next. Make them meaningfully different from each other, consistent with established facts, and fitting for {genre}. Each one should be exciting and move things forward.

Return them as JSON, exactly like this (nothing else):
[
  {{"label": "Short title (3-6 words)", "description": "One sentence describing what happens"}},
  {{"label": "Short title (3-6 words)", "description": "One sentence describing what happens"}},
  {{"label": "Short title (3-6 words)", "description": "One sentence describing what happens"}}
]"""


def build_character_extraction_prompt() -> str:
    """Gets the LLM to pull out character names + descriptions as JSON."""
    return """Look through the story below and find every named character.
Give me a one-sentence description of each, based only on what's actually in the story so far.

Return them as JSON like this:
[
  {"name": "Character Name", "description": "Brief description based on story events"}
]

If there aren't any named characters yet, just return []"""


def build_genre_remix_prompt(target_genre):
    """Rewrites the latest paragraph in a different genre - keeps plot, changes vibe."""
    return (
        f"Rewrite just the last paragraph of the story in the {target_genre} genre. "
        f"Keep all the same plot events, characters, and outcomes - only change "
        f"the tone and atmosphere to fit {target_genre}. Give me just the "
        f"rewritten paragraph, nothing else."
    )


def build_viz_prompt():
    """Tells the LLM to write an image-gen prompt for the latest scene."""
    return (
        "Look at the latest paragraph of the story and write an image generation "
        "prompt for it (the kind you'd use with DALL-E or Midjourney). Pick the "
        "most visually interesting moment. Include style, mood, lighting, "
        "composition. Keep it under 100 words. Just the prompt, nothing else."
    )


# -- story helpers --

def effective_temperature() -> float:
    """Cap temperature at 0.5 when Simple vocab is active. High temp makes the
    model reach for unusual words, which defeats the whole point of Simple."""
    temp = st.session_state.get("temperature", 0.8)
    if st.session_state.get("vocab_level", 1) == 1:
        return min(temp, 0.5)
    return temp


@contextmanager
def processing_state(msg):
    """Lock the UI, show a status message, clean up when done.
    Got tired of repeating the same try/finally in every button handler."""
    status = st.empty()
    status.markdown(f'<div class="processing-msg">{msg}</div>', unsafe_allow_html=True)
    st.session_state.processing = True
    try:
        yield
    finally:
        st.session_state.processing = False
        status.empty()


def safe_render_text(raw):
    """Escape HTML then convert *markdown italics* to <em> tags.
    Order matters - escape first, THEN add our own tags."""
    safe = html.escape(raw)
    return re.sub(r"\*([^*]+)\*", r"<em>\1</em>", safe)


def get_full_story_text():
    """Join all segments with blank lines - this is what gets sent to the LLM
    as the 'story so far' context."""
    return "\n\n".join(seg["text"] for seg in st.session_state.story_segments)


def push_undo(prev):
    """Stash a copy of segments so the user can undo. Capped at
    MAX_UNDO_LEVELS so long sessions don't eat memory."""
    st.session_state.undo_stack.append(prev)
    if len(st.session_state.undo_stack) > MAX_UNDO_LEVELS:
        st.session_state.undo_stack.pop(0)


def add_segment(text, author="ai"):
    """Append a segment and immediately persist to disk.
    This way if streamlit crashes, the user doesn't lose their last turn."""
    st.session_state.story_segments.append({
        "text": text,
        "author": author,
        "timestamp": datetime.now().isoformat(),
    })
    save_current_story()


def export_story_markdown():
    """Build a markdown export. User contributions get italicized
    so you can tell them apart from AI-generated text."""
    # TODO: might want to add character list to the export at some point
    title = st.session_state.story_title
    genre = st.session_state.genre
    hook = st.session_state.initial_hook

    header = f"# {title}\n*Genre: {genre}*\n\n> {hook}\n\n---\n"
    body_parts = []
    for seg in st.session_state.story_segments:
        t = seg["text"]
        body_parts.append(f"*{t}*" if seg["author"] == "user" else t)

    total_words = sum(len(seg["text"].split()) for seg in st.session_state.story_segments)
    footer = (
        f"---\n*{total_words} words - exported from AI Co-Author on "
        f"{datetime.now().strftime('%b %d, %Y at %I:%M %p')}*"
    )
    return header + "\n\n".join(body_parts) + "\n\n" + footer


# Streamlit reruns the script on every click, so we need session_state
# to keep track of everything between reruns.

def init_state():
    s = st.session_state

    # navigation + story identity
    s.setdefault("page", "setup")
    s.setdefault("story_id", "")
    s.setdefault("story_title", "")
    s.setdefault("genre", "Fantasy")
    s.setdefault("initial_hook", "")

    # story content
    s.setdefault("story_segments", [])
    s.setdefault("characters", [])
    s.setdefault("undo_stack", [])
    s.setdefault("pending_choices", None)

    # controls - vocab_level was originally 3 (Rich) but testers found
    # Simple more intuitive as a starting point
    s.setdefault("temperature", 0.8)
    s.setdefault("vocab_level", 1)

    # transient UI state - these reset on every fresh session anyway
    s.setdefault("error_msg", "")
    s.setdefault("last_viz_prompt", "")
    s.setdefault("processing", False)



# Setup page - where users create a new story or load an existing one
# ---------------------------------------------------------------------------

def render_setup() -> None:
    # sidebar shows saved stories - added this late in development
    # because testers kept losing their work
    with st.sidebar:
        st.markdown("### History")
        history = load_history()
        if history:
            for story in history:
                updated = story.get("updated_at", "")
                date_str = ""
                if updated:
                    try:
                        dt = datetime.fromisoformat(updated)
                        date_str = dt.strftime("%b %d")
                    except ValueError:
                        pass
                seg_count = len(story.get("segments", []))
                title = story.get("title", "Untitled")
                meta = story.get("genre", "")
                if date_str:
                    meta += f" · {date_str}"
                meta += f" · {seg_count} parts"

                card = st.container()
                with card:
                    col_main, col_x = st.columns([6, 1])
                    with col_main:
                        if st.button(
                            f"{title}\n\n{meta}",
                            key=f"load_{story.get('id', '')}",
                            use_container_width=True,
                        ):
                            load_story_into_session(story)
                            st.rerun()
                    with col_x:
                        if st.button(
                            "x",
                            key=f"del_{story.get('id', '')}",
                        ):
                            delete_story(story.get("id", ""))
                            st.rerun()
        else:
            st.caption("No stories yet.")

    # Main
    st.markdown(
        '<div class="setup-title">AI Co-Author</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="setup-subtitle">'
        "A collaborative storytelling experience</div>",
        unsafe_allow_html=True,
    )

    st.session_state.story_title = st.text_input(
        "Story Title",
        value=st.session_state.story_title,
        placeholder="The Last Lighthouse",
    )
    st.session_state.genre = st.selectbox(
        "Genre",
        GENRES,
        index=GENRES.index(st.session_state.genre),
    )
    # text_area instead of text_input so people can write multi-sentence hooks
    st.session_state.initial_hook = st.text_area(
        "Initial Hook / Setting",
        value=st.session_state.initial_hook,
        placeholder=(
            # this placeholder doubles as an example of what a good hook looks like
            "A lonely keeper tends a lighthouse on a cliff above a sea "
            "that has been dry for a hundred years. One night, the water "
            "returns - but it glows."
        ),
        height=120,
    )

    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your-openrouter-api-key-here":
        st.warning("Set your OPENROUTER_API_KEY in the .env file to get started.")

    if st.button("Start the Story", type="primary", use_container_width=True):
        if not st.session_state.story_title.strip():
            st.error("Please enter a title.")
            return
        if not st.session_state.initial_hook.strip():
            st.error("Please provide an initial hook or setting.")
            return
        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your-openrouter-api-key-here":
            st.error("Please set your OpenRouter API key in .env.")
            return

        with processing_state("Crafting the opening..."):
            try:
                system = build_system_prompt(
                    st.session_state.genre, st.session_state.story_title
                )
                # adapt user message to vocab level - "mood" primes literary language
                vocab_level = st.session_state.get("vocab_level", 3)
                if vocab_level == 1:
                    tone_instruction = (
                        "Focus on simple actions and basic descriptions. "
                        "Introduce at least one character."
                    )
                else:
                    tone_instruction = (
                        "Establish the setting, mood, and at least one character."
                    )
                user_msg = (
                    f'Here is the premise for our story:\n\n'
                    f'"{st.session_state.initial_hook}"\n\n'
                    f"Write a strong opening paragraph (150-250 words). "
                    f"{tone_instruction} "
                    f"End on a note that invites continuation."
                )
                result = call_llm(system, user_msg, effective_temperature())
            except LLMRateLimitExceeded:
                log.warning("LLM rate limit exceeded during story start")
                st.error("Rate limit reached - please wait a minute and try again.")
                return
            except LLMRequestError as e:
                log.warning("LLM request error during story start: %s", e)
                st.error(f"Something went wrong: {e}")
                return

        st.session_state.story_id = str(uuid.uuid4())
        st.session_state.story_segments = []
        st.session_state.undo_stack = []
        st.session_state.characters = []
        st.session_state.pending_choices = None
        st.session_state.last_viz_prompt = ""
        st.session_state.processing = False
        add_segment(result, author="ai")
        st.session_state.page = "story"
        st.rerun()



# Story page - the main writing interface where the collaboration happens
# ---------------------------------------------------------------------------

def render_story() -> None:
    is_busy = st.session_state.processing

    # title + genre display
    st.markdown(
        f'<div class="story-header">{st.session_state.story_title}</div>'
        f'<div class="story-meta">'
        f"{st.session_state.genre} &middot; "
        f"{len(st.session_state.story_segments)} segments</div>",
        unsafe_allow_html=True,
    )

    # intentionally not type="primary" - don't want people accidentally nuking their story
    if st.button("New Story", disabled=is_busy):
        save_current_story()
        # wipe story state but keep settings (temperature, vocab_level)
        st.session_state.page = "setup"
        st.session_state.story_title = ""
        st.session_state.initial_hook = ""
        st.session_state.story_segments = []
        st.session_state.story_id = ""
        st.session_state.processing = False
        st.rerun()

    st.markdown("---")

    # sidebar controls
    with st.sidebar:
        st.markdown("### Controls")

        # called it "Creativity" instead of "Temperature" because
        # non-technical users had no idea what temperature meant
        st.slider(
            "Creativity",
            min_value=0.1,
            max_value=1.5,
            step=0.1,
            key="temperature",
            help="Lower = more predictable, Higher = more creative",
        )

        # select_slider instead of radio buttons - takes up less vertical space
        st.select_slider(
            "Vocabulary",
            options=[1, 2, 3],
            format_func=lambda x: VOCAB_LABELS[x],
            key="vocab_level",
            help="Simple = everyday words, Rich = literary language",
        )

        st.markdown("---")

        st.markdown(f"**Genre:** {st.session_state.genre}")
        st.info(GENRE_RULES.get(st.session_state.genre, ""))

        st.markdown("---")

        st.markdown("### Characters")
        if st.session_state.characters:
            for char in st.session_state.characters:
                st.markdown(f"**{char['name']}** - {char['description']}")
        else:
            st.caption("Characters appear here as the story develops.")

        if st.button("Update Characters", use_container_width=True, disabled=is_busy):
            if not st.session_state.story_segments:
                st.warning("No story content yet.")
            else:
                with processing_state("Scanning for characters..."):
                    try:
                        story_text = get_full_story_text()
                        result = call_llm(
                            build_character_extraction_prompt(),
                            story_text,
                            temperature=0.2,
                        )
                        parsed = extract_json(result)
                        if isinstance(parsed, list):
                            st.session_state.characters = validate_characters(parsed)
                            save_current_story()
                        else:
                            st.warning("Could not parse character data.")
                    except LLMRateLimitExceeded:
                        log.warning("rate limited during character scan")
                        st.warning("Rate limit - try again shortly.")
                    except LLMRequestError as e:
                        log.warning("character extraction failed: %s", e)
                        st.warning("Couldn't extract characters right now.")
                st.rerun()

        st.markdown("---")

        st.markdown("### Visualize")
        if st.button("Generate Image Prompt", use_container_width=True, disabled=is_busy):
            if not st.session_state.story_segments:
                st.warning("No story content yet.")
            else:
                with processing_state("Generating image prompt..."):
                    try:
                        story_text = get_full_story_text()
                        result = call_llm(
                            build_viz_prompt(),
                            story_text,
                            temperature=0.7,
                        )
                        # strip markdown artifacts the model sometimes adds
                        cleaned = result.replace("**", "").strip()
                        for prefix in ["Image Prompt:", "Prompt:", "Image:"]:
                            if cleaned.startswith(prefix):
                                cleaned = cleaned[len(prefix):].strip()
                        st.session_state.last_viz_prompt = cleaned
                    except LLMRateLimitExceeded:
                        log.warning("rate limited on image prompt gen")
                        st.warning("Rate limited - try the image prompt again in a sec.")
                    except LLMRequestError as e:
                        log.warning("viz prompt failed: %s", e)
                        st.warning("Image prompt generation failed.")
                st.rerun()

        if st.session_state.last_viz_prompt:
            st.code(st.session_state.last_viz_prompt, language=None)

        st.markdown("---")

        st.markdown("### Export")
        md = export_story_markdown()
        st.download_button(
            "Download as Markdown",
            data=md,
            file_name=f"{st.session_state.story_title.replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # render the story segments
    # (originally used st.write() but it doesn't support custom CSS classes)
    for seg in st.session_state.story_segments:
        css_class = "story-segment-user" if seg["author"] == "user" else "story-segment-ai"
        safe_text = safe_render_text(seg["text"])
        st.markdown(f'<div class="{css_class}">{safe_text}</div>', unsafe_allow_html=True)

    st.markdown("---")


    if st.session_state.error_msg:
        st.error(st.session_state.error_msg)
        st.session_state.error_msg = ""

    # show branching choices if the user asked for them
    if st.session_state.pending_choices:
        st.markdown("**Choose a path:**")
        choices = st.session_state.pending_choices
        for idx, choice in enumerate(choices):
            if st.button(
                f"{choice['label']} - {choice['description']}",
                key=f"choice_{idx}",
                use_container_width=True,
                disabled=is_busy,
            ):
                st.session_state.pending_choices = None
                with processing_state("Continuing the story..."):
                    try:
                        system = build_system_prompt(
                            st.session_state.genre, st.session_state.story_title
                        )
                        story_text = get_full_story_text()
                        user_msg = (
                            f"STORY SO FAR:\n\n{story_text}\n\n---\n\n"
                            f"The reader chose this path: "
                            f"\"{choice['label']}: {choice['description']}\"\n\n"
                            f"Write 1-2 paragraphs continuing the story along "
                            f"this chosen path."
                        )
                        result = call_llm(system, user_msg, effective_temperature())
                        push_undo(list(st.session_state.story_segments))
                        add_segment(result, author="ai")
                    except LLMRateLimitExceeded:
                        log.warning("rate limited after choosing path")
                        st.session_state.error_msg = "Too many requests - wait a bit and pick again."
                    except LLMRequestError as e:
                        log.warning("path continuation failed: %s", e)
                        st.session_state.error_msg = f"Failed to continue on that path: {e}"
                st.rerun()
        return


    # label is hidden because it's obvious from context what this is for
    user_text = st.text_area(
        "Your contribution",
        placeholder="Add to the story in your own words, or leave blank and let the AI continue...",
        height=100,
        key="user_input",
        label_visibility="collapsed",
    )

    # main action gets more space since it's the primary workflow
    c_continue, c_choices, c_remix = st.columns([3, 2, 2])

    with c_continue:
        if st.button(
            "Continue with AI",
            type="primary",
            use_container_width=True,
            disabled=is_busy,
        ):
            snapshot = list(st.session_state.story_segments)
            with processing_state("Writing..."):
                try:
                    if user_text.strip():
                        add_segment(user_text.strip(), author="user")

                    system = build_system_prompt(
                        st.session_state.genre, st.session_state.story_title
                    )
                    story_text = get_full_story_text()
                    user_msg = (
                        f"STORY SO FAR:\n\n{story_text}\n\n---\n\n"
                        f"Continue the story with 1-2 paragraphs. "
                        f"Advance the plot meaningfully."
                    )
                    result = call_llm(system, user_msg, effective_temperature())
                    add_segment(result, author="ai")
                    push_undo(snapshot)

                except LLMRateLimitExceeded:
                    # rollback - user text was added but AI failed
                    log.warning("LLM rate limit exceeded during continue")
                    st.session_state.story_segments = snapshot
                    save_current_story()
                    st.session_state.error_msg = "Rate limit reached - please wait and try again."
                except LLMRequestError as e:
                    log.warning("LLM request error during continue: %s", e)
                    st.session_state.story_segments = snapshot
                    save_current_story()
                    st.session_state.error_msg = f"Something went wrong: {e}"
            st.rerun()

    with c_choices:
        if st.button("Give Me Choices", use_container_width=True, disabled=is_busy):
            with processing_state("Generating choices..."):
                try:
                    if user_text.strip():
                        push_undo(list(st.session_state.story_segments))
                        add_segment(user_text.strip(), author="user")

                    system = build_choices_system_prompt(
                        st.session_state.genre, st.session_state.story_title
                    )
                    story_text = get_full_story_text()
                    user_msg = (
                        f"STORY SO FAR:\n\n{story_text}\n\n---\n\n"
                        f"Propose 3 branching options."
                    )
                    result = call_llm(system, user_msg, temperature=0.9)

                    parsed = extract_json(result)
                    if isinstance(parsed, list) and validate_choices(parsed):
                        st.session_state.pending_choices = parsed[:3]
                    else:
                        st.session_state.error_msg = "Invalid response format. Try again."

                except LLMRateLimitExceeded:
                    log.warning("rate limited while generating choices")
                    st.session_state.error_msg = "Hit the rate limit - give it a moment and try again."
                except LLMRequestError as e:
                    log.warning("choices request failed: %s", e)
                    st.session_state.error_msg = f"Couldn't generate choices: {e}"
            st.rerun()

    with c_remix:
        if st.button("Genre Remix", use_container_width=True, disabled=is_busy):
            other_genres = [g for g in GENRES if g != st.session_state.genre]
            remix_genre = random.choice(other_genres)

            with processing_state(f"Remixing as {remix_genre}..."):
                try:
                    system = build_genre_remix_prompt(remix_genre)
                    story_text = get_full_story_text()
                    result = call_llm(system, story_text, temperature=1.0)

                    push_undo(list(st.session_state.story_segments))
                    if st.session_state.story_segments:
                        st.session_state.story_segments[-1] = {
                            "text": f"[Remixed as {remix_genre}] {result}",
                            "author": "ai",
                            "timestamp": datetime.now().isoformat(),
                        }
                        save_current_story()

                except LLMRateLimitExceeded:
                    log.warning("rate limit hit during remix")
                    st.session_state.error_msg = "Rate limited - the remix will have to wait."
                except LLMRequestError as e:
                    log.warning("remix failed: %s", e)
                    st.session_state.error_msg = f"Remix didn't work: {e}"
            st.rerun()

    # undo is separate from the creative actions - it's a utility, not a workflow step
    if st.session_state.undo_stack and not is_busy:
        if st.button("Undo Last", use_container_width=False):
            st.session_state.story_segments = st.session_state.undo_stack.pop()
            st.session_state.pending_choices = None  # stale choices would reference old story state
            save_current_story()
            st.rerun()


# ---------------------------------------------------------------------------
# streamlit wants set_page_config to be the very first st.* call,
# so it goes here at module level rather than inside main()
st.set_page_config(page_title="AI Co-Author", page_icon=None, layout="wide")


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_state()
    # only two pages - setup and story. Didn't bother with streamlit's
    # multipage app system since it's overkill for just two screens
    if st.session_state.page == "setup":
        render_setup()
    else:
        render_story()


main()
