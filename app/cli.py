
import argparse

from app.core import (
    milestone_3,
    milestone_4,
    milestone_5,
    complete_sentence,
    get_next_word_hint,
)


def run_game():
    """Run the interactive Legal Sentence Game."""

    game = milestone_4()

    prompts = game["prompts"]
    model = game["ngram_model"]

    print("\n" + "=" * 45)
    print("          LEGAL SENTENCE GAME")
    print("=" * 45)

    print("\nWelcome to the Legal Sentence Game!")
    print("Complete each legal sentence.")
    print("Type 'quit' to exit the game.\n")

    # Load classifier once before the game starts.
    try:
        classifier = milestone_3()

    except Exception as error:
        print("\nUnable to load GPT-2 classifier.")
        print(f"Error: {error}")
        print(
            "\nPlace your fine-tuned GPT-2 model in "
            "'models/legal-gpt2' or configure "
            "LEGAL_GPT2_MODEL."
        )
        return 0

    completed = 0

    for round_number, prompt in enumerate(
        prompts, start=1
    ):
        print("\n" + "-" * 45)
        print(f"Round {round_number}/{len(prompts)}")
        print("-" * 45)

        print("\nComplete the legal sentence:")
        print(f"\n{prompt} ______")

        hint = get_next_word_hint(prompt, model)

        if hint:
            print(f"\nHint (n-gram): {hint}")
        else:
            print("\nHint: No prediction available.")

        while True:
            try:
                answer = input("\nYour answer: ").strip()

            except (EOFError, KeyboardInterrupt):
                print("\nGame exited.")
                return completed

            if answer.lower() == "quit":
                print("\nThanks for playing!")
                return completed

            if answer:
                break

            print("Please enter a valid answer.")

        sentence = complete_sentence(
            prompt,
            answer
        )

        print("\nCompleted sentence:")
        print(sentence)

        print("\nAnalyzing legal clause...")

        try:
            result = milestone_5(
                sentence,
                classifier=classifier
            )

            print("\n" + "=" * 45)
            print("         GPT-2 CLASSIFICATION")
            print("=" * 45)

            print(
                f"\nClause Type: {result['clause_type']}"
            )

            print(
                f"Confidence: "
                f"{result['confidence'] * 100:.2f}%"
            )

            print("\nFeedback:")
            print(result["feedback"])

        except Exception as error:
            print("\nClassification failed.")
            print(f"Error: {error}")

        completed += 1

        print("\nRound completed!")

        if round_number < len(prompts):
            try:
                input(
                    "\nPress Enter for the next round..."
                )

            except (EOFError, KeyboardInterrupt):
                print("\nGame exited.")
                return completed

    print("\n" + "=" * 45)
    print("              GAME OVER")
    print("=" * 45)

    print(
        f"\nRounds completed: "
        f"{completed}/{len(prompts)}"
    )

    print("\nThank you for playing!")

    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="legal-sentence-game",
        description=(
            "Interactive legal sentence completion "
            "game with GPT-2 classification"
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True
    )

    sub.add_parser(
        "hello",
        help="Check that the project is working"
    )

    sub.add_parser(
        "play",
        help="Start the Legal Sentence Game"
    )

    args = parser.parse_args(argv)

    if args.command == "hello":
        print(
            "Legal Sentence Game is ready. "
            "Open README.md for your milestones."
        )

    elif args.command == "play":
        run_game()


if __name__ == "__main__":
    main()
