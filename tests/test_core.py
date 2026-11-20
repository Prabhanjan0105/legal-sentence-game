
import pytest
import torch

from unittest.mock import MagicMock

from app.core import (
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    preprocess_text,
    build_ngram_model,
    predict_next_word,
    complete_sentence,
    get_next_word_hint,
)


# ==========================================
# MILESTONE 1 TEST
# ==========================================

def test_milestone_1():
    tokens = preprocess_text(
        "The Parties AGREE!"
    )

    assert tokens == [
        "the", "parties", "agree"
    ]

    corpus = milestone_1()

    assert len(corpus) == 20
    assert all(
        isinstance(sentence, list)
        for sentence in corpus
    )
    assert all(len(s) > 0 for s in corpus)


# ==========================================
# MILESTONE 2 TEST
# ==========================================

def test_milestone_2():
    model = milestone_2()

    assert len(model) > 0

    assert predict_next_word(
        model, "parties agree"
    ) == "to"

    assert predict_next_word(
        model, "the landlord"
    ) == "shall"

    assert predict_next_word(
        model, "unknown legalphrase"
    ) is None

    sample = [
        ["the", "party", "shall", "pay"],
        ["the", "party", "shall", "comply"],
    ]

    sample_model = build_ngram_model(sample, n=3)

    assert sample_model[
        ("the", "party")
    ]["shall"] == 2


# ==========================================
# MILESTONE 3 TEST
# ==========================================

def test_milestone_3(monkeypatch):
    from app import classifier as classifier_module

    fake_tokenizer = MagicMock()
    fake_tokenizer.pad_token = "<pad>"
    fake_tokenizer.pad_token_id = 0

    fake_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]])
    }

    fake_model = MagicMock()
    fake_model.config.pad_token_id = 0
    fake_model.config.id2label = {
        0: "Confidentiality",
        1: "Termination",
    }

    fake_model.return_value.logits = torch.tensor(
        [[3.0, 1.0]]
    )

    monkeypatch.setattr(
        classifier_module.AutoTokenizer,
        "from_pretrained",
        lambda path: fake_tokenizer
    )

    monkeypatch.setattr(
        classifier_module.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda path: fake_model
    )

    classifier = milestone_3("fake-model")

    result = classifier.predict(
        "The employee shall maintain confidentiality."
    )

    assert result["clause_type"] == "Confidentiality"
    assert 0 <= result["confidence"] <= 1

    fake_model.eval.assert_called_once()


# ==========================================
# SHARED FAKE CLASSIFIER FOR CLI TESTS
# ==========================================

class FakeClassifier:
    def predict(self, sentence):
        return {
            "sentence": sentence,
            "clause_type": "Confidentiality",
            "confidence": 0.91,
        }


# ==========================================
# MILESTONE 4 TEST
# ==========================================

def test_milestone_4(monkeypatch, capsys):
    from app.cli import main

    game = milestone_4()

    assert "prompts" in game
    assert "ngram_model" in game
    assert len(game["prompts"]) == 5

    sentence = complete_sentence(
        "The parties agree to",
        "comply with all laws"
    )

    assert sentence == (
        "The parties agree to comply with all laws."
    )

    hint = get_next_word_hint(
        "The parties agree to",
        game["ngram_model"]
    )

    assert hint == "comply"

    monkeypatch.setattr(
        "app.cli.milestone_3",
        lambda: FakeClassifier()
    )

    answers = iter([
        "comply with all laws", "",
        "maintain confidentiality", "",
        "complete the work", "",
        "pay rent on time", "",
        "remain valid",
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": next(answers)
    )

    main(["play"])

    output = capsys.readouterr().out

    assert "LEGAL SENTENCE GAME" in output
    assert "Completed sentence:" in output
    assert "Rounds completed: 5/5" in output


# ==========================================
# MILESTONE 5 TEST
# ==========================================

def test_milestone_5(monkeypatch, capsys):
    from app.cli import main

    classifier = FakeClassifier()

    result = milestone_5(
        "The employee shall maintain confidentiality.",
        classifier=classifier
    )

    assert result["clause_type"] == "Confidentiality"
    assert result["confidence"] == 0.91
    assert "Confidentiality" in result["feedback"]

    with pytest.raises(ValueError):
        milestone_5("", classifier=classifier)

    monkeypatch.setattr(
        "app.cli.milestone_3",
        lambda: classifier
    )

    answers = iter([
        "maintain confidentiality",
        "",
        "quit",
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": next(answers)
    )

    main(["play"])

    output = capsys.readouterr().out

    assert "GPT-2 CLASSIFICATION" in output
    assert "Confidentiality" in output
    assert "91.00%" in output
    assert "Feedback:" in output
