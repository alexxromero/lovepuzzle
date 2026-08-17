"""
Runs the full puzzle pipeline (equation + narration) on N random US phone
numbers with random domains, and saves each generated puzzle to a
human-readable text file. No accuracy grading here — just generation.
"""

import os
import random

from clue_generator import load_model, MODEL_ID
from puzzle import generate_puzzle, validate_phone_number
from verifier import load_verifier, VERIFIER_MODEL_ID

N_PUZZLES = 20
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "test_pipeline_output.txt")

DOMAINS = [
    "sports", "history", "music", "science", "geography", "math",
    "movies", "food", "animals", "technology", "art", "literature",
]


def _random_phone():
    for _ in range(100):
        area = random.randint(200, 999)
        exch = random.randint(200, 999)
        sub  = random.randint(0, 9999)
        raw  = f"({area}) {exch}-{sub:04d}"
        try:
            validate_phone_number(raw)
            return raw
        except ValueError:
            continue
    raise RuntimeError("Could not generate a valid phone number after 100 tries")


def _clue_source_counts(clues_info):
    counts = {"generated": 0, "hardcoded": 0, "alternative": 0, "none": 0}
    for _, _, clue, _, _, source in clues_info:
        if clue is None:
            counts["none"] += 1
        elif source in counts:
            counts[source] += 1
    return counts


def main():
    print(f"Loading generator ({MODEL_ID})...")
    g_model, g_tokenizer = load_model()

    print(f"Loading verifier ({VERIFIER_MODEL_ID})...")
    v_model, v_tokenizer = load_verifier()

    with open(OUTPUT_FILE, "w") as f:
        for i in range(N_PUZZLES):
            phone_str = _random_phone()
            phone_int = validate_phone_number(phone_str)
            domains = random.sample(DOMAINS, 3)

            print(f"[{i + 1}/{N_PUZZLES}] {phone_str} | {domains}")

            header = (
                f"{'=' * 70}\n"
                f"Puzzle {i + 1}/{N_PUZZLES}\n"
                f"{'-' * 70}\n"
                f"Phone number : {phone_str}\n"
                f"Domains      : {', '.join(domains)}\n"
            )

            try:
                _, puzzle, clues_info, _ = generate_puzzle(
                    phone_int, domains, g_model, g_tokenizer, v_model, v_tokenizer
                )
                counts = _clue_source_counts(clues_info)
                body = (
                    f"Clue sources : generated={counts['generated']}  "
                    f"hardcoded={counts['hardcoded']}  "
                    f"alternative={counts['alternative']}  "
                    f"none={counts['none']}\n\n"
                    f"{puzzle}\n\n"
                )
            except ValueError as e:
                body = f"FAILED: {e}\n\n"

            f.write(header + body)
            f.flush()

    print(f"\nSaved {N_PUZZLES} puzzles → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
