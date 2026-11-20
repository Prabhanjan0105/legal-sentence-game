
import os
import re
from pathlib import Path
from collections import defaultdict, Counter


# ==========================================
# MILESTONE 1: LEGAL CORPUS PREPROCESSING
# ==========================================

def preprocess_text(text):
    """Clean and tokenize legal text."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def load_legal_corpus(file_path):
    """Load sentences from the legal corpus."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Legal corpus not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        sentences = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return sentences


def milestone_1():
    """Return tokenized legal sentences."""
    project_root = Path(__file__).resolve().parent.parent

    corpus_path = (
        project_root / "data" / "legal_corpus.txt"
    )

    sentences = load_legal_corpus(corpus_path)

    return [
        preprocess_text(sentence)
        for sentence in sentences
    ]


# ==========================================
# MILESTONE 2: N-GRAM NEXT-WORD PREDICTOR
# ==========================================

def build_ngram_model(tokenized_sentences, n=3):
    """Build a frequency-based n-gram model."""
    if n < 2:
        raise ValueError("n must be at least 2")

    model = defaultdict(Counter)

    for tokens in tokenized_sentences:
        for i in range(len(tokens) - n + 1):
            context = tuple(tokens[i:i + n - 1])
            next_word = tokens[i + n - 1]

            model[context][next_word] += 1

    return dict(model)


def predict_next_word(model, text, n=3):
    """Predict the most common next word."""
    tokens = preprocess_text(text)

    if len(tokens) < n - 1:
        return None

    context = tuple(tokens[-(n - 1):])

    if context not in model:
        return None

    return model[context].most_common(1)[0][0]


def milestone_2():
    """Train a trigram model."""
    corpus = milestone_1()
    return build_ngram_model(corpus, n=3)


# ==========================================
# MILESTONE 3: GPT-2 CLASSIFIER LOADING
# ==========================================

def milestone_3(model_path=None):
    """Load the fine-tuned GPT-2 classifier."""
    from app.classifier import LegalClauseClassifier

    if model_path is None:
        model_path = os.environ.get(
            "LEGAL_GPT2_MODEL",
            "models/legal-gpt2"
        )

    return LegalClauseClassifier(model_path)


# ==========================================
# MILESTONE 4: INTERACTIVE GAME LOGIC
# ==========================================

LEGAL_PROMPTS = [
    "The parties agree to",
    "The employee shall",
    "The contractor shall",
    "The tenant shall",
    "The agreement shall",
]


def complete_sentence(prompt, answer):
    """Combine the prompt and player's answer."""
    prompt = prompt.strip()
    answer = answer.strip()

    if not answer:
        raise ValueError("Answer cannot be empty.")

    sentence = f"{prompt} {answer}"

    if sentence[-1] not in ".!?":
        sentence += "."

    return sentence


def get_next_word_hint(prompt, model):
    """Generate an n-gram hint."""
    return predict_next_word(model, prompt)


def milestone_4():
    """Prepare game prompts and prediction model."""
    model = milestone_2()

    return {
        "prompts": LEGAL_PROMPTS.copy(),
        "ngram_model": model,
    }


# ==========================================
# MILESTONE 5: CLASSIFICATION AND FEEDBACK
# ==========================================

def generate_feedback(result):
    """Generate feedback for a classified clause."""
    clause_type = result["clause_type"]
    confidence = result["confidence"]

    feedback = (
        f"This sentence was classified as a "
        f"{clause_type} clause."
    )

    if confidence >= 0.80:
        feedback += (
            "\nThe model reports high confidence "
            "in this prediction."
        )

    elif confidence >= 0.50:
        feedback += (
            "\nThe model reports moderate confidence. "
            "Consider reviewing the clause."
        )

    else:
        feedback += (
            "\nThe model reports low confidence. "
            "The clause may require further review."
        )

    return feedback


def milestone_5(sentence, classifier=None):
    """Classify a completed sentence and give feedback."""
    if not sentence or not sentence.strip():
        raise ValueError("Sentence cannot be empty.")

    if classifier is None:
        classifier = milestone_3()

    result = classifier.predict(sentence)

    return {
        "sentence": sentence,
        "clause_type": result["clause_type"],
        "confidence": result["confidence"],
        "feedback": generate_feedback(result),
    }
