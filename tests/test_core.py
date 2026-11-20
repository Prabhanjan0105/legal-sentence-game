"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest


def test_milestone_1():
    from app.core import milestone_1, preprocess_text

    tokens = preprocess_text(
        "The Parties AGREE to comply!"
    )

    assert tokens == [
        "the", "parties", "agree", "to", "comply"
    ]

    corpus = milestone_1()

    assert len(corpus) >= 20
    assert all(isinstance(s, list) for s in corpus)
    assert all(len(s) > 0 for s in corpus)
    assert "agreement" in corpus[1]


# Milestone 2: Implement the n-gram next-word predictor.
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Load the fine-tuned GPT-2 model for legal clause classification.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Create a CLI game interface presenting partial legal sentences for completion.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: After user completion, classify the full sentence and provide feedback.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

