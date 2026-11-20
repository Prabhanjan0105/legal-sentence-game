
import re
from pathlib import Path
from collections import defaultdict, Counter

def preprocess_text(text):
    """Clean and tokenize legal text."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


def load_legal_corpus(file_path):
    """Read legal sentences from a text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        sentences = [
            line.strip()
            for line in file
            if line.strip()
        ]
    return sentences


def milestone_1():
    """Prepare tokenized legal sentences."""
    project_root = Path(__file__).resolve().parent.parent
    corpus_path = project_root / "data" / "legal_corpus.txt"

    sentences = load_legal_corpus(corpus_path)

    tokenized_sentences = [
        preprocess_text(sentence)
        for sentence in sentences
    ]

    return tokenized_sentences





# --- Milestone 2: N-gram Next-Word Predictor ---

def build_ngram_model(tokenized_sentences, n=3):
    """Build an n-gram frequency model."""
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
    """Predict the most frequent next word."""
    tokens = preprocess_text(text)

    if len(tokens) < n - 1:
        return None

    context = tuple(tokens[-(n - 1):])

    if context not in model:
        return None

    predictions = model[context]
    return predictions.most_common(1)[0][0]


def milestone_2():
    """Train a trigram model using the legal corpus."""
    tokenized_sentences = milestone_1()

    model = build_ngram_model(
        tokenized_sentences,
        n=3
    )

    return model



def milestone_3(model_path=None):
    """Load the fine-tuned legal clause classifier."""
    import os
    from app.classifier import LegalClauseClassifier

    if model_path is None:
        model_path = os.environ.get(
            "LEGAL_GPT2_MODEL",
            "models/legal-gpt2"
        )

    return LegalClauseClassifier(model_path)



# --- Milestone 4: Create a CLI game interface presenting partial legal sentences for completion.
# Hint: Design an interactive command-line interface for user input and game flow.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: After user completion, classify the full sentence and provide feedback.
# Hint: Use the loaded GPT-2 model to classify the completed sentence and display feedback, referencing the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
