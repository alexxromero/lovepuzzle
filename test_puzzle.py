import csv
import random
import traceback
import sys

from clue_generator import load_model, MODEL_ID
from verifier import load_verifier, VERIFIER_MODEL_ID
from puzzle import generate_puzzle

DOMAINS_POOL = [
    "sports", "history", "music", "science", "geography",
    "movies", "literature", "mathematics", "food", "politics",
]

OUTPUT_FILE = "test_results.csv"
N_TESTS = 5
N_DOMAINS = 3


def random_phone():
    """Generate a random valid US phone number as a 10-digit integer."""
    area = random.randint(200, 999)
    exchange = random.randint(200, 999)
    subscriber = random.randint(0, 9999)
    return int(f"{area}{exchange}{subscriber:04d}")


def main():
    rng = random.Random(42)

    print(f"Loading generator ({MODEL_ID})...")
    g_model, g_tokenizer = load_model()

    print(f"Loading verifier ({VERIFIER_MODEL_ID})...")
    v_model, v_tokenizer = load_verifier()

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phone", "domains", "equation", "step",
            "op", "const", "step_domain", "clue_text", "diff", "source",
        ])

        for i in range(N_TESTS):
            phone = random_phone()
            domains = rng.sample(DOMAINS_POOL, N_DOMAINS)
            print(f"\n[{i+1}/{N_TESTS}] phone={phone}  domains={domains}")

            try:
                eq, puzzle, steps_with_clues, step_domains = generate_puzzle(
                    phone, domains, g_model, g_tokenizer, v_model, v_tokenizer
                )
                for step_num, ((op, const, clue_text, diff, source), domain) in enumerate(
                    zip(steps_with_clues, step_domains), start=1
                ):
                    writer.writerow([
                        phone, domains, eq["infix"], step_num,
                        op, const, domain, clue_text, diff, source,
                    ])
                f.flush()
            except Exception as e:
                print(f"ERROR: {e}", file=sys.stderr)
                traceback.print_exc()
                writer.writerow([phone, domains, "", "", "", "", "", "", "", f"ERROR: {e}"])
                f.flush()

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
