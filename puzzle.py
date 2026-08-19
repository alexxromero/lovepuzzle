import re
import random
import json
import base64
import phonenumbers

from equation_maker import EquationGenerator, PREF_INTS
from clue_generator import load_model, generate_clues, validate_clues, MODEL_ID
from verifier import load_verifier, VERIFIER_MODEL_ID
from hardcoded_clues import get_hardcoded_clue
from alternative_representations import (
    roman_clue, binary_clue, prime_ordinal_clue,
    fibonacci_ordinal_clue, spanish_clue,
)
from fact_checker import fact_check

N_CLUES = 5  # number of clues to ask the generator LLM for
MAX_RETRIES = 3  # number of tries for the whole puzzle generation

CLUE_SOURCE_GENERATED   = "generated"
CLUE_SOURCE_HARDCODED   = "hardcoded"
CLUE_SOURCE_ALTERNATIVE = "alternative"

OP_VERB = {
    '+':  'Add',
    '-':  'Subtract',
    '*':  'Multiply by',
    '/':  'Divide by',
    'e2': 'Square it',
    'e3': 'Cube it',
    'e4': 'Raise it to the 4th power',
    'e5': 'Raise it to the 5th power',
}

STOPWORDS = {
    'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with',
    'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'shall', 'should', 'may', 'might', 'can', 'could',
    'number', 'that', 'this', 'which', 'who', 'their', 'they', 'it',
    'its', 'by', 'from', 'as', 'into', 'during', 'before', 'after',
    'each', 'every', 'all', 'some', 'no', 'not', 'only', 'so', 'than',
    'just', 'about', 'any', 'how', 'what', 'when', 'where', 'up',
}


def _content_words(clue):
    tokens = re.sub(r"[^a-z0-9 ]", "", clue.lower()).split()
    return frozenset(t for t in tokens if t not in STOPWORDS and len(t) > 1)

class ClueRegistry:
    """Tracks clues used in a puzzle to avoid duplicates."""

    DUPLICATE_THRESHOLD = 0.4

    def __init__(self):
        # for each string, we save the set of relevant content words
        self._entries: dict[str, frozenset] = {}

    def register(self, clue):
        self._entries[clue] = _content_words(clue)

    def is_duplicate(self, clue):
        # use Jaccard similarity to compare the concepts between clues
        new_concept = _content_words(clue)
        if not new_concept:
            return False
        for seen_concept in self._entries.values():
            union = new_concept | seen_concept
            if not union:
                continue
            if len(new_concept & seen_concept) / len(union) >= self.DUPLICATE_THRESHOLD:
                return True
        return False


def validate_phone_number(raw):
    # we use Google's phone number library for validating the phone numbers
    try:
        parsed = phonenumbers.parse(raw, "US")
    except phonenumbers.NumberParseException as e:
        raise ValueError(f"Could not parse phone number: {e}")

    if not phonenumbers.is_valid_number(parsed):
        raise ValueError("Not a valid phone number.")

    if not phonenumbers.is_possible_number(parsed):
        raise ValueError("Phone number is not possible (wrong length for region).")

    digits = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.NATIONAL)
    digits = digits.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    return int(digits)


def get_inputs():
    raw = input("Enter a phone number: ").strip()
    phone_num = validate_phone_number(raw)

    print("Enter three interests (e.g. sports, history, music):")
    domains = []
    for i in range(1, 4):
        domain = ""
        while not domain:
            domain = input(f"  Interest {i}: ").strip()
        domains.append(domain)

    return phone_num, domains


def narrate_step(op, val, clue, correction):
    """Turn a puzzle step into a sentence."""
    verb = OP_VERB[op]
    if op in ('e2', 'e3', 'e4', 'e5'):
        return f"{verb}."
    if clue:
        inline = clue[0].lower() + clue[1:]
        if correction:
            base = inline.rstrip(". ")
            suffix = f"plus {correction}" if correction > 0 else f"minus {abs(correction)}"
            return f"{verb}: {base}, {suffix}."
        return f"{verb} {inline}"
    return f"{verb} {val}"


def format_equation(equation_chain):
    """Render an equation chain like [("seed", 123), ("+", 7), ...] as an
    infix string, e.g. "((123 + 7) * 12) - 14".
    """
    seed_op, seed_val = equation_chain[0]
    assert seed_op == "seed"
    expr = str(seed_val)
    for op, val in equation_chain[1:]:
        if op == "+":
            expr = f"({expr} + {val})"
        elif op == "-":
            expr = f"({expr} - {val})"
        elif op == "*":
            expr = f"({expr} * {val})"
        elif op == "/":
            expr = f"({expr} / {val})"
        elif op in ("e2", "e3", "e4", "e5"):
            expr = f"{expr}^{op[1:]}"
        else:
            raise ValueError(f"Unknown operator: {op}")
    return expr


def encode_puzzle(puzzle_text, phone, equation_str):
    """Pack a generated puzzle into a URL-safe token so it can be shared as a
    link -- solving/checking a shared puzzle needs none of the generator/
    verifier models, just this data. Not encryption: anyone who decodes the
    token can read the target phone number directly.
    """
    payload = {"p": puzzle_text, "n": phone, "e": equation_str}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_puzzle(token):
    """Inverse of encode_puzzle. Returns (puzzle_text, phone, equation_str).
    Raises ValueError on a malformed/tampered token.
    """
    try:
        padded = token + "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode(padded)
        payload = json.loads(raw.decode("utf-8"))
        return payload["p"], payload["n"], payload["e"]
    except Exception as e:
        raise ValueError(f"Malformed puzzle token: {e}")


def build_puzzle_text(seed, clues_info):
    """Returns the full narrated puzzle as a string."""
    lines = [f"1. Start with {seed}."]
    i = 2
    for (op, val, clue, correction, _, _, _) in clues_info:
        lines.append(f"{i}. {narrate_step(op, val, clue, correction)}")
        i += 1
    return "\n".join(lines)


def _build_puzzle(phone_num, domains, seed, g_model, g_tokenizer, v_model, v_tokenizer):
    """Build a puzzle for the given phone number and seed.
    Returns (eq, puzzle, n_valid_clues).
    """
    equation_chain = EquationGenerator(seed=seed).sample(phone_num)
    seed_op, seed_val = equation_chain[0]
    assert(seed_op == "seed")

    registry = ClueRegistry()  # track clues to avoid duplicates

    clues_info = []  #(operator, value, best clue, correction, domain, clue source, fact_checked)

    # First pass: LMM clue generator
    for op, val in equation_chain[1:]:
        if op in ('e2', 'e3', 'e4', 'e5') or not (val in PREF_INTS or val < 100):
            # no clue generation
            clues_info.append((op, val, None, None, None, None, None))
            continue

        domain = random.choice(domains)
        candidates = generate_clues(g_model, g_tokenizer, val, domain, N_CLUES)
        # validate_clues returns the clues that passed the verifier's
        # confidence check, ordered by highest confidence
        valid_clues = validate_clues(v_model, v_tokenizer, candidates, val)
        best_clue, best_clue_diff, fact_checked = None, None, None
        for clue, guess, _ in valid_clues:
            if registry.is_duplicate(clue):
                continue

            # last check: use a search API to guess the clue. This mimics what
            # a user is likely to do -- search the clues on google. If the search
            # ran but came up empty, a user would likely be stuck too, so we
            # discard the clue. If fact-checking is off (no key configured, or
            # the request itself failed), fall back on the verifier's own guess.
            veredict, api_number = fact_check(clue, v_model, v_tokenizer)
            if veredict == "inconclusive":
                continue
            if veredict == "conclusive":
                guess = api_number

            best_clue = clue
            best_clue_diff = val - guess
            fact_checked = (veredict == "conclusive")
            registry.register(clue)
            break

        source = CLUE_SOURCE_GENERATED if best_clue is not None else None
        clues_info.append((op, val, best_clue, best_clue_diff, domain, source, fact_checked))

    # Second pass: for values without a clue, look up if there are any hard-coded clues
    # that we can use
    for i, (op, val, clue_text, _, domain, _, _) in enumerate(clues_info):
        if clue_text is not None:  # already has a clue
            continue
        if domain is None:  # likely not a fact-rich number
            continue
        hardcoded, diff = get_hardcoded_clue(val, domain)
        if hardcoded and not registry.is_duplicate(hardcoded):
            registry.register(hardcoded)
            clues_info[i] = (op, val, hardcoded, diff, "misc", CLUE_SOURCE_HARDCODED, None)

    # Third pass: try alternative representations.
    # Each representation type may appear at most once per puzzle.
    used_reps = set()
    representations = [
        ("prime_rep", prime_ordinal_clue),
        ("fibonacci_rep", fibonacci_ordinal_clue),
        ("roman_rep", roman_clue),
        ("spanish_rep", spanish_clue),
        ("binary_rep", binary_clue),
    ]
    for i, (op, val, clue_text, _, domain, _, _) in enumerate(clues_info):
        if clue_text is not None or val is None:
            continue

        clue, rep_name = None, None
        for name, rep_fn in representations:
            if name in used_reps:
                continue
            clue = rep_fn(val)
            if clue is not None:
                rep_name = name
                break

        if clue is not None and not registry.is_duplicate(clue):
            registry.register(clue)
            used_reps.add(rep_name)
            clues_info[i] = (op, val, clue, None, rep_name, CLUE_SOURCE_ALTERNATIVE, None)

    n_valid = sum(1 for _, _, clue, _, _, _, _ in clues_info if clue is not None)
    puzzle = build_puzzle_text(seed_val, clues_info)
    return equation_chain, puzzle, n_valid, clues_info


def fact_check_summary(clues_info):
    """How many of this puzzle's LLM-generated clues were confirmed via the
    search API. Returns (n_confirmed, n_generated).
    """
    n_generated = sum(1 for (*_, source, _) in clues_info if source == CLUE_SOURCE_GENERATED)
    n_confirmed = sum(1 for (*_, fact_checked) in clues_info if fact_checked)
    return n_confirmed, n_generated


def generate_puzzle(phone_num, domains, g_model, g_tokenizer, v_model, v_tokenizer):
    # Do many tries in case a puzzle is invalid (has no value clues, n_valid=0)
    for attempt in range(MAX_RETRIES):
        seed = random.randint(0, 10000)
        eq, puzzle, n_valid, clues_info = _build_puzzle(
            phone_num, domains, seed, g_model, g_tokenizer, v_model, v_tokenizer
        )
        if n_valid > 0:
            n_fact_checked, n_generated = fact_check_summary(clues_info)
            print(f"\nEquation : {eq}\n")
            print(puzzle)
            print(f"\nFact-checked: {n_fact_checked}/{n_generated} generated clues confirmed via search API")
            step_domains = [domain for (_, _, _, _, domain, _, _) in clues_info]
            return eq, puzzle, clues_info, step_domains
        print(f"Attempt {attempt + 1}: no valid clues found, retrying...")

    raise ValueError(
        f"Could not generate any valid clues after {MAX_RETRIES} attempts. "
        f"Try different domains."
    )


if __name__ == "__main__":
    phone_num, domains = get_inputs()
    print(f"\nPhone number : {phone_num}")
    print(f"Domains      : {domains}")

    print(f"\nLoading generator ({MODEL_ID})...")
    g_model, g_tokenizer = load_model()

    print(f"Loading verifier ({VERIFIER_MODEL_ID})...")
    v_model, v_tokenizer = load_verifier()

    # returns (equation, puzzle, clues_info)
    generate_puzzle(phone_num, domains, g_model, g_tokenizer, v_model, v_tokenizer)
