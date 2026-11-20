# Legal Sentence Game

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/ky2p6n8p2m/badge.svg)](https://ask.thili.ai/b/ky2p6n8p2m)

This project creates a command-line game designed for law students to hone their legal drafting skills. Players complete partial legal sentences using n-gram predictions, then receive immediate feedback on the clause type from a fine-tuned GPT-2 model. It's an interactive way to practice legal language and understand common sentence structures.

## What you are building

**Goal.** Develop a command-line interface (CLI) game where law students practice drafting legal sentences by completing partial sentences using n-gram predictions, receiving instant feedback on their generated clause types from a fine-tuned GPT-2 model.

**Done when.** The CLI game allows players to complete legal sentences using n-gram predictions. After completion, the fine-tuned GPT-2 model successfully classifies the full sentence and provides feedback on the clause type.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a legal text corpus for n-gram model training.**  
      Hint: Apply text preprocessing and tokenization techniques from the 'Build a Next-Word Predictor from Scratch' drill.
- [ ] **2. Implement the n-gram next-word predictor.**  
      Hint: Build the n-gram model and implement next-word prediction logic as shown in the 'Build a Next-Word Predictor from Scratch' drill.
- [ ] **3. Load the fine-tuned GPT-2 model for legal clause classification.**  
      Hint: Load the pre-trained and fine-tuned GPT-2 model for text classification, similar to the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.
- [ ] **4. Create a CLI game interface presenting partial legal sentences for completion.**  
      Hint: Design an interactive command-line interface for user input and game flow.
- [ ] **5. After user completion, classify the full sentence and provide feedback.**  
      Hint: Use the loaded GPT-2 model to classify the completed sentence and display feedback, referencing the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Implement a scoring system based on the GPT-2 feedback to gamify the experience further.
- Allow users to choose different legal sub-domains for sentence generation, if the corpus supports it.
- Add a feature to display the most common predicted next words as hints for the player.
