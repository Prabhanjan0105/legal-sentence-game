
import re
from pathlib import Path


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




# --- Milestone 2: Implement the n-gram next-word predictor.
# Hint: Build the n-gram model and implement next-word prediction logic as shown in the 'Build a Next-Word Predictor from Scratch' drill.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Load the fine-tuned GPT-2 model for legal clause classification.
# Hint: Load the pre-trained and fine-tuned GPT-2 model for text classification, similar to the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Create a CLI game interface presenting partial legal sentences for completion.
# Hint: Design an interactive command-line interface for user input and game flow.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: After user completion, classify the full sentence and provide feedback.
# Hint: Use the loaded GPT-2 model to classify the completed sentence and display feedback, referencing the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
