import argparse
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

from verifier import load_verifier, verify, VERIFIER_MODEL_ID
from number_words import reveals_answer

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

def _system_prompt(num_clues):
    return (
        f"You are a trivia expert. When given a number and a domain, you generate "
        f"{num_clues} distinct, short, fun, and factual clues that each connect "
        f"that number to the domain.\n"
        f"Each clue must be 'The number of' followed by only the noun phrase being counted, "
        f"nothing else. Stop right after the noun phrase. Do not add a verb, a clause starting "
        f"with 'is'/'are'/'that'/'which', or any explanation of why the fact is true.\n"
        f"Good: 'The number of points on a Cartesian coordinate system.'\n"
        f"Bad: 'The number of points on a Cartesian coordinate system is determined by its dimensions.'\n"
        f"Keep each clue under 12 words.\n"
        f"The clues must be distinct — do not repeat the same fact.\n"
        f"Do not include the number itself anywhere in the clue, either as a digit or spelled out.\n"
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
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return _parse_clues(raw, num_clues)


def _parse_clues(raw_clues, num_clues):
    """Parse the generated clues."""
    clues = []
    for line in raw_clues.splitlines():
        line = line.strip()
        # look for strings of the form 1. CLUE. or 1) CLUE.
        match = re.match(r"^\d+[.)]\s*(.+)", line)
        if match:
            # save only the clue, not the delimiting numbers
            clues.append(match.group(1).strip())

    # if no clues then split the raw clues, strip, and return them as-is
    if not clues:
        clues = [l.strip() for l in raw_clues.splitlines() if l.strip()]
    return clues[:num_clues]


def validate_clues(model, tokenizer, clues, target_number):
    """Verify clues by feeding them to the verifier model.

    Returns all valid clues, orderd by confidence in desc order.
    """
    valid = []
    for clue in clues:
        if reveals_answer(clue, target_number):
            continue

        guessed, top_prob, margin = verify(model, tokenizer, clue)
        if guessed is None:
            continue

        if margin < 0.65:
            continue

        valid.append((clue, guessed, margin))

    valid.sort(key=lambda x: -x[2])
    return valid



def main():
    """Here we generate clues given a number and domain. Basic steps:
    1. Ask a LLM to generate a total of 'num_clues' clues that relate the number and the domain
    2. Run the clues through a verifier: 
       - Ask a small LLM to guess the number each clue is referring to.
       - Return the verifier's guesses and their confidence.
    3. Select the best clue:
       - For each clue, we compare the guessed number to the target number.
       - We keep clues with a confident guess.
       - Then order clues according to confidence. The best clue is the most confident one.
    4. Apply a correction (if needed):
       - If the guessed number matches the target number, no correction needed.
       - If they differ by N, then we add a correction to the clue.
    5. If for some reason very few clues were generated/accepted, then we force clues by using 
       an alternative representation of the target number, like Roman numerals or Spanish. 
       We can also use Fibonacci and prime ordinal numbers.
    """
    
    parser = argparse.ArgumentParser(description="Generate a fun clue for a number and domain.")
    parser.add_argument("--number", type=int, help="The target number.")
    parser.add_argument("--domain", type=str, help="The domain/category (e.g. sports, history, music).")
    parser.add_argument("--num_clues", type=int, help="The number of clues to generate.", default=5)
    parser.add_argument("--output", type=str, default="clue_output.txt", help="Path to save the result (JSON for verbose, plain text otherwise).")
    parser.add_argument("-vo", "--verbose_output", action="store_true", help="Output everything. For debugging.")
    args = parser.parse_args()

    print(f"Loading the model {MODEL_ID}...")
    g_model, g_tokenizer = load_model()  # generator

    print(f"Loading the verifier {VERIFIER_MODEL_ID}...")
    v_model, v_tokenizer = load_verifier()  # verifier

    # 1. generate clues
    clues = generate_clues(g_model, g_tokenizer, args.number, args.domain, num_clues=args.num_clues)
    # 2. & 3. verify clues and select the best one
    valid_clues = validate_clues(v_model, v_tokenizer, clues, args.number)
    best_clue, best_guess, best_margin = valid_clues[0] if valid_clues else (None, None, None)
    diff = args.number - best_guess if best_guess is not None else None
    # 4. apply a correction
    best_clue_corrected = None
    if best_clue and diff:
        correction = f"Plus {diff}" if diff > 0 else f"Minus {abs(diff)}"
        best_clue_corrected = f"{best_clue} {correction}."

    if args.verbose_output:
        result = {
            "target_num": args.number,
            "target_domain": args.domain,
            "all_clues":  [
                {"clue": c, "guess": g, "diff": args.number - g, "margin": m}
                for c, g, m in valid_clues
            ],
            "best_clue": best_clue,
            "best_clue_corrected": best_clue_corrected,
            "best_clue_diff": diff,
            "best_clue_margin": best_margin,
        }
        output_text = json.dumps(result, indent=2)
    else:
        lines = [
            f"{c} | guess={g} diff={args.number - g:+d} margin={m:.3f}"
            for c, g, m in valid_clues
        ]
        lines.append("")
        lines.append(f"Best: {best_clue_corrected if best_clue_corrected else best_clue or 'No valid clue found.'}")
        output_text = "\n".join(lines)

    print(output_text)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text + "\n")
        print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
