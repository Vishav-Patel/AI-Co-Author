# AI Co-Author

Collaborative storytelling app. You and an AI take turns writing fiction together.

Built with **Streamlit** and **OpenRouter** (Qwen 3 235B).

---

## Setup

```bash
git clone <your-repo-url>
cd story-weaver
pip install -r requirements.txt
streamlit run app.py
```

Put your OpenRouter API key in `.env` before running. You can grab one at https://openrouter.ai/keys.

Opens at `http://localhost:8501`.

## Model

I'm using **Qwen 3 235B** through OpenRouter, but you can swap in any model you want. The app uses the OpenAI-compatible SDK, so any provider that speaks that protocol works. Just change the `MODEL` string and `base_url` in `app.py`. If you wanted to use Claude, GPT-4, or a local model through Ollama, you'd just update those two lines. The prompts don't rely on anything Qwen-specific.

I landed on Qwen 3 235B after trying a few smaller models that couldn't follow the vocabulary constraints reliably. OpenRouter made it painless to switch between models during testing.

## System Prompt

There are three separate prompts, one per vocabulary level. Not the same prompt with a "be simple" footnote - each one is built from scratch.

I learned this the hard way. A single prompt with vocabulary instructions at the bottom just gets ignored. The model writes literary prose regardless. Took about 8 rounds of iteration to get the Simple prompt working.

Here's the **Rich** version (the shortest of the three):

```
You are a masterful literary author co-writing an epic {genre} story titled "{title}".

Consistency comes first - never contradict anything established in the story so far.
Review every prior paragraph before writing. Write in vivid, poetic third-person prose
with rich descriptions. Stay firmly in the {genre} genre and lean into deep atmosphere
and imagery. Keep continuations to 1-2 focused paragraphs (~100-200 words).

Advance the plot meaningfully - no filler. Maintain the tone throughout. If the reader
contributes text, work it naturally into the narrative. Output only the story
continuation, no commentary or labels.

Write with the prose quality of Cormac McCarthy, Tolkien, or Patrick Rothfuss -
expansive vocabulary, vivid metaphors, sensory details, complex rhythmic sentences.
```

The **Simple** version is twice as long. It opens with constraints, includes 6 "Don't write / Write this" example pairs from actual bad outputs, gives the model a children's book author persona (Percy Jackson, Goosebumps), bans similes and metaphors, and ends with a self-check: "would a 9-year-old understand every word?" The few-shot examples were the breakthrough - nothing else worked.

The **Moderate** version sits in between. Uses Stephen King and Brandon Sanderson as style anchors.

Consistency is rule #1 in every prompt. "Review every prior paragraph" nudges the model toward checking itself before writing. Every prompt ends with "output only the story continuation" because otherwise meta-text leaks into the display.

## How Memory Works

The full story gets sent with every API call:

```
STORY SO FAR:

[all paragraphs joined together]

---

Continue the story with 1-2 paragraphs. Advance the plot meaningfully.
```

Token usage grows with every turn, but Qwen 3 has a big context window so it's fine for a prototype. For 50+ turn stories you'd want a sliding window with a compressed summary, but that's not implemented here.

Other things that help with consistency: there's a character tracker in the sidebar that extracts names and descriptions from the story, genre blurbs remind the user what kind of story they're writing, and the undo button lets you roll back if the AI goes off track.

## Features

| What | How it works |
|------|-------------|
| Vocabulary slider | Simple / Moderate / Rich - completely different prompts |
| Creativity slider | LLM temperature (0.1-1.5), capped at 0.5 for Simple |
| Branching choices | 3 plot directions as JSON, pick one |
| Genre Remix | Rewrites the last paragraph in a random genre |
| Character Tracker | Pulls named characters into the sidebar |
| Image Prompt | Generates a DALL-E/Midjourney prompt for the latest scene |
| Markdown Export | Download as .md with word count |
| Undo | Rolls back the last turn, including on failure |
| Story History | Local JSON file, click to resume, hover to delete |
| Rate Limit Retry | Up to 3x with a visible countdown |

## What Went Wrong

The vocab slider was the biggest headache. The model ignored "use simple words" completely when the rest of the prompt talked about "atmosphere and pacing." Rebuilding the Simple prompt from scratch took most of a day - constraints first, few-shot examples from actual bad outputs, dynamic user messages ("focus on simple actions" instead of "establish the mood"), and a temperature cap. All of that together finally got it working.

Other stuff that broke:

Qwen wraps its internal reasoning in `<think>` tags. Those showed up in the story until I added a regex to strip them.

The Streamlit slider needed to be moved twice before the value changed. Turns out you're supposed to use the `key` parameter instead of assigning to session_state manually.

Rate limits were caught by checking `if "rate" in error_string` which was fragile. Switched to the SDK's typed exceptions.

The `processing` flag got stuck after errors, disabling all buttons permanently. Wrote a context manager to guarantee cleanup in `finally`.

## Given More Time

Streaming responses first. Users stare at nothing for 5-10 seconds while the model thinks. With `stream=True` the text would appear word by word. That alone would make the app feel completely different.

I'd also add writer personality modes - an Introvert/Extrovert toggle. Right now genre controls what happens and vocabulary controls the language, but nothing controls how the narrator actually thinks. An introverted writer goes deep: inner monologue, emotional weight, slow atmospheric pacing. An extroverted writer stays on the surface: punchy dialogue, fast action, humor. The interesting part is these are independent from genre. An introverted comedy is dry wit and character study. An extroverted horror is jump scares and relentless pace. That would give users way more control over how their story feels.

The other big thing is a backup LLM with live summary sync. Rate limits are real on free tiers. A secondary model (something small and cheap) would continuously maintain a compressed summary of the story in the background. If the primary hits a rate limit or goes down, the secondary picks up instantly using its own summary. The user never notices the switch. More of a production concern than a prototype one, but it's what separates a demo from something people can rely on.

Beyond that: sliding-window memory so stories can run indefinitely (keep last 5-6 turns verbatim, compress everything older), and a story branching tree where users can go back to any choice point and explore alternate paths.

## Code Notes

Single file, on purpose. Streamlit apps don't benefit much from splitting into modules at this scale.

The `processing_state` context manager exists because I got tired of the same try/finally in 7 different button handlers. `extract_json` has a fallback that scans for `[` or `{` because the model sometimes writes "Here is the JSON:" before the actual data. `effective_temperature()` caps temp at 0.5 for Simple vocab so the model doesn't reach for unusual words.

History is a local JSON file. Not scalable, but it works.

## Project Structure

```
story-weaver/
├── .streamlit/
│   └── config.toml       # theme (warm minimalism)
├── app.py                 # everything
├── requirements.txt
├── .env                   # your API key (not committed)
├── .gitignore
├── story_history.json     # starts empty, auto-populated
└── README.md
```
