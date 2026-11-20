
import argparse

from app.core import (
    milestone_4,
    complete_sentence,
    get_next_word_hint,
)


def run_game():
    """Run the interactive legal sentence game."""

    game = milestone_4()

    prompts = game["prompts"]
    model = game["ngram_model"]

    print("\n" + "=" * 45)
    print("          LEGAL SENTENCE GAME")
    print("=" * 45)

    print("\nWelcome to the Legal Sentence Game!")
    print("Complete the legal sentences.")
    print("Type 'quit' to exit the game.\n")

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

        sentence = complete_sentence(prompt, answer)

        print("\nCompleted sentence:")
        print(sentence)

        completed += 1

        print("\nRound completed!")

        if round_number < len(prompts):
            try:
                input("\nPress Enter for the next round...")
            except (EOFError, KeyboardInterrupt):
                print("\nGame exited.")
                return completed

    print("\n" + "=" * 45)
    print("             GAME OVER")
    print("=" * 45)
    print(f"\nRounds completed: {completed}/{len(prompts)}")
    print("\nThank you for playing!")

    return completed


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="legal-sentence-game",
        description="Interactive legal sentence completion game"
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
        help="Start the legal sentence game"
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
