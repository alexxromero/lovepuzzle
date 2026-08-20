"""
Interactively look up a clue via the Serper search API and show exactly
where the number came from: Serper's answer box (deterministic, no model
involved) or the LLM extractor reading organic snippets -- plus the raw
search results and, for the snippet path, the actual confidence margin
(even when it falls short of the threshold). Also runs the same clue
through the verifier directly (the same call clue_generator.py's
validate_clues() makes), so you can see side by side whether the verifier's
own guess agrees with what the search API found -- helpful for spotting a
confidently-wrong verifier guess versus a genuinely weird/ambiguous clue.

Needs SERPER_API_KEY set. Run: python -m puzzle_testing.inspect_clue
"""

import re
import sys

import torch

from fact_checker import (
    SERPER_API_KEY,
    EXTRACT_SYSTEM_PROMPT,
    EXTRACT_MARGIN_THRESHOLD,
    _search,
    _extract_number,
)
from verifier import load_verifier, VERIFIER_MODEL_ID, first_token_stats, verify


def _extract_number_from_snippets_verbose(model, tokenizer, clue, snippets):
    """Same as fact_checker._extract_number_from_snippets, but returns the
    raw decoded text and margin too instead of just the final gated number."""
    snippet_text = "\n".join(f"- {s}" for s in snippets)
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Claim: {clue}\nSearch snippets:\n{snippet_text}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = output.sequences[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    _, margin = first_token_stats(output.scores)
    match = re.search(r"-?\d[\d,]*", raw)
    number = int(match.group().replace(",", "")) if match else None
    return raw, margin, number


def inspect_clue(clue, v_model, v_tokenizer):
    guessed, top_prob, margin = verify(v_model, v_tokenizer, clue)
    print("\n--- Verifier's own guess (same call validate_clues() makes) ---")
    print(f"  guess            : {guessed}")
    print(f"  top-token prob   : {top_prob:.4f}")
    print(f"  confidence margin: {margin:.4f}  (validate_clues() threshold: 0.65)")

    query = clue.rstrip(". ")
    print(f'\nSearch query: "{query}"')

    result = _search(query)
    if result is None:
        print(">>> SOURCE: none -- search request failed (network/API error)")
        return

    answer_box = result.get("answerBox")
    if answer_box:
        print("\n--- Answer box (raw) ---")
        print(answer_box)
        number = _extract_number(answer_box)
        if number is not None:
            print(f"\n>>> SOURCE: answer box (deterministic, no model involved)")
            print(f">>> Extracted number: {number}")
            return
        print("(answer box present, but no number could be parsed out of it)")

    organic = result.get("organic") or []
    snippets = [o["snippet"] for o in organic[:5] if o.get("snippet")]
    print("\n--- Top organic snippets ---")
    if not snippets:
        print("  (none)")
        print("\n>>> SOURCE: none -- no answer box and no usable snippets")
        return
    for s in snippets:
        print(f"  - {s}")

    raw, margin, number = _extract_number_from_snippets_verbose(v_model, v_tokenizer, clue, snippets)
    print(f"\n--- LLM extractor ---")
    print(f"  raw model output : {raw!r}")
    print(f"  confidence margin: {margin:.4f}  (threshold: {EXTRACT_MARGIN_THRESHOLD})")

    if margin < EXTRACT_MARGIN_THRESHOLD:
        print(f"\n>>> SOURCE: none -- extractor found {number!r} but margin {margin:.4f} "
              f"is below the {EXTRACT_MARGIN_THRESHOLD} threshold, so it's discarded "
              f"(fact_check() would treat this as inconclusive)")
        return

    if number is None:
        print("\n>>> SOURCE: none -- extractor's output had no parseable number")
        return

    print(f"\n>>> SOURCE: snippet (LLM extractor)")
    print(f">>> Extracted number: {number}")


def main():
    if not SERPER_API_KEY:
        print("SERPER_API_KEY is not set -- export it first, then rerun.")
        sys.exit(1)

    print(f"Loading verifier ({VERIFIER_MODEL_ID})...")
    v_model, v_tokenizer = load_verifier()

    print("\nEnter a clue to inspect (blank line or Ctrl+D to quit).")
    while True:
        try:
            clue = input("\nClue> ").strip()
        except EOFError:
            break
        if not clue:
            break
        try:
            inspect_clue(clue, v_model, v_tokenizer)
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
