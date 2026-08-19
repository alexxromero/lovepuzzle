import os
import re

import requests
import torch

from verifier import first_token_stats

SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
SERPER_URL = "https://google.serper.dev/search"


EXTRACT_MARGIN_THRESHOLD = 0.95

EXTRACT_SYSTEM_PROMPT = (
    "You are a fact-checking assistant. You will be given a factual claim and some "
    "search result snippets. Based only on the snippets, state the number the claim "
    "refers to. Respond with only the number, or UNKNOWN if the snippets don't say."
)


def _search(clue, timeout=8):
    try:
        resp = requests.post(
            SERPER_URL,
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": clue},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def _extract_number(answer_box):
    """Pull the first integer out of a Serper answerBox, checking the fields
    most likely to hold a direct numeric answer first."""
    candidates = []
    for key in ("answer", "snippet", "title"):
        val = answer_box.get(key)
        if val:
            candidates.append(str(val))
    for val in answer_box.get("snippetHighlighted") or []:
        candidates.append(str(val))

    for text in candidates:
        match = re.search(r"-?\d[\d,]*", text)
        if match:
            return int(match.group().replace(",", ""))
    return None


def _extract_number_from_snippets(model, tokenizer, clue, snippets):
    """Ask the verifier model what number the clue's claim refers to, based only
    on search snippets. Only returns a number if the model's first-token margin 
    (confidence) is higher than the EXTRACT_MARGIN_THRESHOLD.
    Else, returns None.
    """
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
    if margin < EXTRACT_MARGIN_THRESHOLD:
        return None

    match = re.search(r"-?\d[\d,]*", raw)
    return int(match.group().replace(",", "")) if match else None


def fact_check(clue, v_model, v_tokenizer):
    """Look up the clue's via online search -- mimics what someone solving the
    puzzle would actually do: google it.

    Returns (verdict, number):
      - ("disabled", None): SERPER_API_KEY isn't set, or the request itself
        failed (network/API error). The fallback is to trust the verifier's guess.
      - ("conclusive", api_number): search was successful
      - ("inconclusive", None): search ran but couldn't resolve to a number.
        The online search might be ambiguous. Better to reject such clues.
    """
    if not SERPER_API_KEY:
        return "disabled", None

    try:
        result = _search(clue.rstrip(". "))
        if result is None:
            return "disabled", None

        api_number = None
        answer_box = result.get("answerBox")
        if answer_box:
            api_number = _extract_number(answer_box)

        if api_number is None:
            organic = result.get("organic") or []
            snippets = [o["snippet"] for o in organic[:5] if o.get("snippet")]
            if snippets:
                api_number = _extract_number_from_snippets(v_model, v_tokenizer, clue, snippets)

        if api_number is None:
            return "inconclusive", None
        return "conclusive", api_number

    except Exception as e:
        print(f"Fact-check error for clue {clue!r}: {e}")
        return "disabled", None
