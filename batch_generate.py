import csv
import sys
import traceback
from clue_generator import load_model, MODEL_ID, generate_clues, find_best
from verifier import load_verifier, VERIFIER_MODEL_ID

INPUTS = [
    # (number, domain)
    (7,    "sports"),
    (42,   "history"),
    (13,   "history"),
    (100,  "science"),
    (7,    "music"),
    (12,   "sports"),
    (1969, "history"),
    (8,    "science"),
    (23,   "sports"),
    (360,  "science"),
    (88,   "music"),
    (3,    "music"),
]

N_CLUES = 5
OUTPUT_FILE = "results.csv"


def main():
    print(f"Loading {MODEL_ID}...")
    g_model, g_tokenizer = load_model()

    print(f"Loading {VERIFIER_MODEL_ID}...")
    v_model, v_tokenizer = load_verifier()

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["number", "domain", 
                         "clues", "best_clue_corrected",
                         "verifier_guess", "verifier_confidence"])

        for i, (number, domain) in enumerate(INPUTS):
            print(f"[{i+1}/{len(INPUTS)}] {number} | {domain}", end=" ... ", flush=True)
            try:
                clues = generate_clues(g_model, g_tokenizer, number, domain, N_CLUES)
                best_clue, diff, v_confidence = find_best(v_model, v_tokenizer, clues, number)

                if best_clue is None:
                    writer.writerow([number, domain, clues, "NO VALID CLUE", "", ""])
                    f.flush()
                    print("no valid clue")
                    continue

                best_clue_corrected = ""
                if diff:
                    hint = f"Plus {diff}" if diff > 0 else f"Minus {abs(diff)}"
                    best_clue_corrected = f"{best_clue} {hint}."

                guessed = number - diff
                writer.writerow([
                    number, domain, clues, best_clue_corrected, guessed, f"{v_confidence:.3f}"
                ])
                f.flush()
                status = "✓" if diff == 0 else f"delta={diff:+d}"
                print(f"{status}  confidence={v_confidence:.3f}")
                
            except Exception as e:
                print(f"ERROR", file=sys.stderr)
                traceback.print_exc()
                writer.writerow([number, domain, "", "", f"ERROR: {e}", ""])

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
