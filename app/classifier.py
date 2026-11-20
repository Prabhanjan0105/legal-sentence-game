
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


class LegalClauseClassifier:
    def __init__(self, model_path):
        """
        Load a fine-tuned GPT-2 classification model.
        model_path can be a local directory or
        a Hugging Face model repository ID.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(model_path)
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = (
                self.tokenizer.pad_token_id
            )

        self.model.eval()

    def predict(self, sentence):
        """Classify a legal sentence."""
        if not sentence.strip():
            raise ValueError("Sentence cannot be empty.")

        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits, dim=-1
        )

        predicted_id = int(
            torch.argmax(probabilities, dim=-1).item()
        )

        confidence = float(
            probabilities[0, predicted_id].item()
        )

        label = self.model.config.id2label[
            predicted_id
        ]

        return {
            "sentence": sentence,
            "clause_type": label,
            "confidence": round(confidence, 4),
        }
