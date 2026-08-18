"""
Pulls previously-generated (clue, verifier_guess) pairs out of
test_batch_results.csv (plain narrated puzzle text from an old batch run,
predating the fact-checking prompt fix), samples a subset, and runs each
one through fact_check() to see what the search API says.

Records clue / verifier's guess / search API's response side by side, so
they can be compared directly. Needs SERPER_API_KEY set to get real search
results -- without it, every row will show "disabled".
"""

import ast
import csv
import os
import random
import re

from fact_checker import fact_check, SERPER_API_KEY
from verifier import load_verifier, VERIFIER_MODEL_ID

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "..", "test_batch_results.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "fact_check_results.csv")
SAMPLE_SIZE = 20

# Matches: "{Verb}: {clue text}, plus N." / "{Verb}: {clue text}, minus N."
_CORRECTED_RE = re.compile(r"^\d+\.\s+.+?:\s+(.+?),\s+(plus|minus)\s+(\d+)\.$")
# Matches: "{Verb} {clue text}" (no correction -- diff is 0)
_PLAIN_RE = re.compile(
    r"^\d+\.\s+(?:Add|Subtract|Multiply by|Divide by)\s+(.+)$"
)


def _extract_clue_guess_pairs(path):
    """Parse test_batch_results.csv into a list of (clue, verifier_guess).

    Each puzzle block starts with "Equation : [(...), ...]" -- the (op, val)
    chain -- followed by narrated lines in the same order as equation_chain[1:].
    Lines that don't mention "the number of" are alternative representations
    or exponent steps with no clue, and are skipped.
    """
    pairs = []
    with open(path) as f:
        text = f.read()

    blocks = re.split(r"\n(?=Equation :)", text)
    for block in blocks:
        block = block.strip()
        if not block.startswith("Equation :"):
            continue

        eq_line, _, rest = block.partition("\n")
        chain_str = eq_line.split("Equation :", 1)[1].strip()
        chain = ast.literal_eval(chain_str)
        steps = chain[1:]  # skip ("seed", val)

        # Drop the "1. Start with N." line -- it narrates the seed, which
        # equation_chain[1:] (i.e. `steps`) doesn't include.
        narration_lines = [l for l in rest.splitlines() if re.match(r"^\d+\.\s", l)][1:]
        for (op, val), line in zip(steps, narration_lines):
            if "the number of" not in line.lower():
                continue

            m = _CORRECTED_RE.match(line)
            if m:
                clue_lower, sign, amount = m.group(1), m.group(2), int(m.group(3))
                diff = amount if sign == "plus" else -amount
            else:
                m = _PLAIN_RE.match(line)
                if not m:
                    continue
                clue_lower, diff = m.group(1), 0

            clue = clue_lower[0].upper() + clue_lower[1:]
            if not clue.endswith("."):
                clue += "."
            guess = val - diff
            pairs.append((clue, guess))

    return pairs


def main():
    if not SERPER_API_KEY:
        print(
            "WARNING: SERPER_API_KEY is not set -- fact_check() will return "
            "'disabled' for every clue below. Set the key first if you want "
            "to see real search results.\n"
        )

    pairs = _extract_clue_guess_pairs(RESULTS_CSV)
    print(f"Parsed {len(pairs)} (clue, guess) pairs from {RESULTS_CSV}")

    sample = random.sample(pairs, min(SAMPLE_SIZE, len(pairs)))
    print(f"Sampled {len(sample)} for fact-checking\n")

    print(f"Loading verifier ({VERIFIER_MODEL_ID})...")
    v_model, v_tokenizer = load_verifier()

    rows = []
    for i, (clue, guess) in enumerate(sample):
        print(f"[{i + 1}/{len(sample)}] {clue}", end=" ... ", flush=True)
        verdict, api_number = fact_check(clue, v_model, v_tokenizer)
        print(f"{verdict} ({api_number})")
        rows.append({
            "clue": clue,
            "verifier_guess": guess,
            "search_verdict": verdict,
            "search_number": api_number if api_number is not None else "",
        })

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clue", "verifier_guess", "search_verdict", "search_number"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} results -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
