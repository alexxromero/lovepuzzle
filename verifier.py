import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

VERIFIER_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

VERIFIER_SYSTEM_PROMPT = (
    "You are a number-guessing expert. You will be given a clue that describes a number. "
    "Your task is to guess what number the clue is referring to. "
    "Respond with only the number — no explanation."
)

# A clue is invalid if the verifier's guess deviates by more than this
# fraction of the expected number (e.g. 0.2 = 20% off).
INVALID_THRESHOLD = 0.2


def load_verifier():
    tokenizer = AutoTokenizer.from_pretrained(VERIFIER_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        VERIFIER_MODEL_ID,
        dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    return model, tokenizer

def _parse_number(text):
    """Get the model's answer. It should be one integer"""
    match = re.search(r"-?\d+", text)
    return int(match.group()) if match else None

def _first_token_margin(scores):
    """Get the delta between the probabilities of the first- and second-best tokens"""
    if not scores:
        return 0.0
    probs = torch.softmax(scores[0][0], dim=-1)
    top2 = torch.topk(probs, k=2).values
    delta = top2[0] - top2[1]
    return delta.item()

def verify(model, tokenizer, clue):
    """Feed a generated clue to the verifier model. 
    Return the verifier's guess and its confidence.

    The 'confidence' is the delta between the first- and second-best token's probabilities.
    Token prob ranges from [0-1], with 1 the most certain.
    """
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

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = output.sequences[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    guessed = _parse_number(raw)
    confidence = _first_token_margin(output.scores)
    return guessed, confidence


