"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest



def test_milestone_1():
    from app.core import milestone_1, preprocess_text

    tokens = preprocess_text("The Parties AGREE!")
    assert tokens == ["the", "parties", "agree"]

    corpus = milestone_1()

    assert len(corpus) == 20
    assert all(isinstance(sentence, list) for sentence in corpus)
    assert all(len(sentence) > 0 for sentence in corpus)
    assert corpus[0][0] == "the"



def test_milestone_2():
    from app.core import (
        milestone_2,
        build_ngram_model,
        predict_next_word
    )

    model = milestone_2()

    assert len(model) > 0

    # Test the trained legal corpus
    prediction = predict_next_word(
        model, "parties agree"
    )
    assert prediction == "to"

    # Test another legal phrase
    prediction = predict_next_word(
        model, "the landlord"
    )
    assert prediction == "shall"

    # Test an unknown context
    prediction = predict_next_word(
        model, "unknown legalphrase"
    )
    assert prediction is None

    # Test the n-gram builder independently
    sample = [
        ["the", "party", "shall", "pay"],
        ["the", "party", "shall", "comply"]
    ]

    sample_model = build_ngram_model(sample, n=3)

    assert sample_model[("the", "party")]["shall"] == 2



def test_milestone_3(monkeypatch):
    from unittest.mock import MagicMock
    from app import classifier as classifier_module
    from app.core import milestone_3

    fake_tokenizer = MagicMock()
    fake_tokenizer.pad_token = "<pad>"
    fake_tokenizer.pad_token_id = 0
    fake_tokenizer.return_value = {
        "input_ids": __import__("torch").tensor(
            [[1, 2, 3]]
        )
    }

    fake_model = MagicMock()
    fake_model.config.pad_token_id = 0
    fake_model.config.id2label = {
        0: "Confidentiality",
        1: "Termination",
    }

    import torch
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




def test_milestone_4(monkeypatch, capsys):
    from app.core import (
        milestone_4,
        complete_sentence,
        get_next_word_hint,
    )
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

    # Simulate completing all five rounds
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



# Milestone 5: After user completion, classify the full sentence and provide feedback.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

