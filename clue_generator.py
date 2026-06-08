import argparse
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from verifier import load_verifier, verify, VERIFIER_MODEL_ID

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

_SMALL_NUMBER_WORDS = {
    1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
    11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen',
    15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen',
    19: 'nineteen', 20: 'twenty',
}


def _reveals_answer(clue: str, target: int) -> bool:
    """Return True if the clue explicitly contains the target number."""
    lower = clue.lower()
    if re.search(rf'(?<!\d){re.escape(str(target))}(?!\d)', lower):
        return True
    word = _SMALL_NUMBER_WORDS.get(target)
    return bool(word and re.search(rf'\b{word}\b', lower))

def _system_prompt(num_clues: int) -> str:
    return (
        f"You are a trivia expert. When given a number and a domain, you generate "
        f"{num_clues} distinct, short (<20 words), fun, and factual clues that each connect "
        f"that number to the domain. Each clue must begin with 'The number of'. "
        f"The clues must be distinct — do not repeat the same fact. "
        f"Do not include the number itself anywhere in the clue, either as a digit or spelled out. "
        f"Output only a numbered list, one clue per line, with no extra explanation.\n"
        f"Example format:\n1. The number of ...\n2. The number of ..."
    )


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate_clues(model, tokenizer, number, domain, num_clues):
    """Generate num_clues distinct clues in a single inference call."""
    messages = [
        {"role": "system", "content": _system_prompt(num_clues)},
        {"role": "user", "content": f"Number: {number}\nDomain: {domain}"},
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
            max_new_tokens=num_clues * 40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return _parse_clues(raw, num_clues)


def _parse_clues(text: str, num_clues: int) -> list[str]:
    """Extract clues from a numbered list response."""
    clues = []
    for line in text.splitlines():
        line = line.strip()
        # Match lines like "1. The number of ..." or "1) The number of ..."
        match = re.match(r"^\d+[.)]\s*(.+)", line)
        if match:
            clues.append(match.group(1).strip())
    # Fall back: return non-empty lines if numbered parsing found nothing
    if not clues:
        clues = [l.strip() for l in text.splitlines() if l.strip()]
    return clues[:num_clues]

def find_best(model, tokenizer, clues, target_number):
    """Verify clues by feeding them to the verifier model and select the best one.

    Returns (clue, diff, confidence), or (None, None, None) if no valid clue found.
    diff = target_number - guessed_number (0 means perfect match).

    Ranking: smallest |diff| first, then highest confidence as a tiebreaker.
    """
    valid = []
    for clue in clues:
        if _reveals_answer(clue, target_number):
            continue

        guessed, confidence = verify(model, tokenizer, clue)

        if guessed is None:
            continue

        diff = target_number - guessed

        # reject clues where the guess is too far off (>50% of target)
        if abs(diff) / max(abs(target_number), 1) > 0.5:
            continue

        valid.append((clue, diff, confidence))

    if not valid:
        return None, None, None

    valid.sort(key=lambda x: (abs(x[1]), -x[2]))
    return valid[0]



def main():
    """
    Here we generate clues for a number and domain. Basic steps:
    1. Ask a model to generate a total of 'num_clues' clues based on a target number and domain
    2. Run the clues through a verifier: 
       - Ask a small model to guess each clue.
       - Return the verifier's guesses and teir confidence.
    3. Select the best clue:
       - For each clue, we compare the guessed number to the target number.
       - Ideally, the guessed number would be equal to the target number. Their diff=0.
       - We sort first by the diff and then by the verifier's confidence. 
       - Meaning, if multiple clues are a perfect match, keep the one with the highest confidence.
       - Reject any clue where the difference between the guessed and target numbers is more than 50%.
    4. Apply a correction (if needed):
       - If the guessed number matches the target number, no correction needed.
       - If they differ by N, then we add a correction to the clue.
         For example: "Add one" or "Subtract seven"
    If no clues were accepted then it's better to try again with a different number or domain.
    """
    
    parser = argparse.ArgumentParser(description="Generate a fun clue for a number and domain using Llama 3.2 3B.")
    parser.add_argument("--number", type=int, help="The target number.")
    parser.add_argument("--domain", type=str, help="The domain/category (e.g. sports, history, music).")
    parser.add_argument("--num_clues", type=int, help="The number of clues to generate.", default=3)
    parser.add_argument("-vo", "--verbose_output", action="store_true", help="Output everything. For debugging.")
    args = parser.parse_args()

    print(f"Loading the model {MODEL_ID}...")
    g_model, g_tokenizer = load_model()  # generator

    print(f"Loading the verifier {VERIFIER_MODEL_ID}...")
    v_model, v_tokenizer = load_verifier()  # verifier

    # 1. generate clues
    clues = generate_clues(g_model, g_tokenizer, args.number, args.domain, num_clues=args.num_clues)
    # 2. & 3. verify clues and select the best one
    best_clue, diff, v_confidence = find_best(v_model, v_tokenizer, clues)
    # 4. apply a correction
    best_clue_corrected = None
    if best_clue and diff:
        correction = f"Plus {diff}" if diff > 0 else f"Minus {abs(diff)}"
        best_clue_corrected = f"{best_clue} {correction}."

    if args.verbose_output:
        output_dicc = {
            "target_num": args.number,
            "target_domain": args.domain,
            "clues": clues,
            "best_clue": best_clue,
            "best_clue_corrected": best_clue_corrected,
            "best_clue_diff": diff,
            "best_clue_confidence": v_confidence
        }
        return output_dicc
    return best_clue_corrected if best_clue_corrected else best_clue

if __name__ == "__main__":
    main()
