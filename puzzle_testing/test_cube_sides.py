"""Quick test: ask the verifier model several paraphrases of "how many
sides does a cube have" and average the per-number probabilities across them.
Does not modify any existing pipeline code.
"""
import csv
import re
from collections import defaultdict

import torch

from verifier import load_verifier, VERIFIER_SYSTEM_PROMPT

CLUES = [
    "How many sides does a cube have?",
    "What is the number of sides on a cube?",
    "How many sides are there on a cube?",
    "Count the number of sides a cube has.",
    "How many sides exist on a cube?",
]
DETAIL_CSV = "test_cube_sides_results.csv"
SUMMARY_CSV = "test_cube_sides_avg_results.csv"
MAX_NEW_TOKENS = 16
TOP_K = 20  # how many first-token candidates to inspect before filtering to numeric ones


def _parse_number(text):
    match = re.search(r"-?\d+", text)
    return int(match.group()) if match else None


def _continue_greedy(model, tokenizer, prompt_ids, prompt_mask, forced_token_id, max_new_tokens):
    """Force the first generated token, then greedily complete the rest."""
    device = model.device
    forced = torch.tensor([[forced_token_id]], device=device)
    input_ids = torch.cat([prompt_ids, forced], dim=1)
    attention_mask = torch.cat([prompt_mask, torch.ones_like(forced)], dim=1)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens - 1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output[0][prompt_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def analyze_clue(model, tokenizer, clue):
    """Return a list of numeric candidate rows for a single clue."""
    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Clue: {clue}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    prompt_ids = inputs["input_ids"]
    prompt_mask = inputs["attention_mask"]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = output.sequences[0][prompt_ids.shape[1]:]
    rank1_full = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    probs = torch.softmax(output.scores[0][0], dim=-1)
    topk = torch.topk(probs, k=TOP_K)
    token_ids = topk.indices.tolist()
    probabilities = topk.values.tolist()

    rows = []
    for rank, (token_id, probability) in enumerate(zip(token_ids, probabilities), start=1):
        token_text = tokenizer.decode([token_id])
        # only chase completions for tokens that could start a number
        if not re.match(r"^-?\d", token_text.strip()):
            continue

        full_answer = rank1_full if rank == 1 else _continue_greedy(
            model, tokenizer, prompt_ids, prompt_mask, token_id, MAX_NEW_TOKENS
        )
        parsed = _parse_number(full_answer)
        if parsed is None:
            continue
        rows.append({
            "clue": clue,
            "rank": rank,
            "token_text": token_text,
            "probability": probability,
            "full_answer": full_answer,
            "parsed_number": parsed,
        })
    return rows


def main():
    model, tokenizer = load_verifier()

    all_rows = []
    per_number = defaultdict(list)  # parsed_number -> list of probabilities (one per clue, 0 if absent)

    for clue in CLUES:
        rows = analyze_clue(model, tokenizer, clue)
        all_rows.extend(rows)

        print(f"Clue: {clue}")
        for row in rows:
            print(
                f"  #{row['rank']}  token={row['token_text']!r}  prob={row['probability']:.4f}  "
                f"full_answer={row['full_answer']!r}  parsed={row['parsed_number']}"
            )

        seen = defaultdict(float)
        for row in rows:
            seen[row["parsed_number"]] += row["probability"]
        for number in seen:
            per_number.setdefault(number, [])
        for number in per_number:
            per_number[number].append(seen.get(number, 0.0))

    print()
    with open(DETAIL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clue", "rank", "token_text", "probability", "full_answer", "parsed_number"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved per-clue detail to {DETAIL_CSV}")

    summary = [
        {
            "parsed_number": number,
            "avg_probability": sum(probs) / len(CLUES),
            "times_seen": sum(1 for p in probs if p > 0),
            "out_of": len(CLUES),
        }
        for number, probs in per_number.items()
    ]
    summary.sort(key=lambda r: -r["avg_probability"])

    print("\nAveraged across all paraphrases:")
    for row in summary:
        print(
            f"  {row['parsed_number']:>3}  avg_prob={row['avg_probability']:.4f}  "
            f"seen in {row['times_seen']}/{row['out_of']} clues"
        )

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["parsed_number", "avg_probability", "times_seen", "out_of"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSaved averaged summary to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
