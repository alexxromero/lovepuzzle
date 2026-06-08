import re
import random
import phonenumbers

from equation_maker import EquationGenerator, expr_to_chain, PREF_INTS
from clue_generator import load_model, generate_clues, find_best, MODEL_ID
from verifier import load_verifier, VERIFIER_MODEL_ID
from hardcoded_clues import get_hardcoded_clue
from alternative_representations import (
    roman_clue, binary_clue, prime_ordinal_clue,
    fibonacci_ordinal_clue, spanish_clue,
)

N_CLUES = 5

CLUE_SOURCE_GENERATED   = "generated"
CLUE_SOURCE_HARDCODED   = "hardcoded"
CLUE_SOURCE_ALTERNATIVE = "alternative"

_STOPWORDS = {
    'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with',
    'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'shall', 'should', 'may', 'might', 'can', 'could',
    'number', 'that', 'this', 'which', 'who', 'their', 'they', 'it',
    'its', 'by', 'from', 'as', 'into', 'during', 'before', 'after',
    'each', 'every', 'all', 'some', 'no', 'not', 'only', 'so', 'than',
    'just', 'about', 'any', 'how', 'what', 'when', 'where', 'up',
}


def _content_words(clue: str) -> frozenset:
    tokens = re.sub(r"[^a-z0-9 ]", "", clue.lower()).split()
    return frozenset(t for t in tokens if t not in _STOPWORDS and len(t) > 1)


class ClueRegistry:
    """Tracks clues used in a single puzzle and detects conceptual duplicates."""

    _DUPLICATE_THRESHOLD = 0.6

    def __init__(self):
        self._entries: dict[str, frozenset] = {}  # clue_text -> content_words

    def register(self, clue: str) -> None:
        self._entries[clue] = _content_words(clue)

    def is_duplicate(self, clue: str) -> bool:
        words = _content_words(clue)
        if not words:
            return False
        for stored_words in self._entries.values():
            union = words | stored_words
            if not union:
                continue
            if len(words & stored_words) / len(union) >= self._DUPLICATE_THRESHOLD:
                return True
        return False


_OP_VERB = {
    '+':  'Add',
    '-':  'Subtract',
    '*':  'Multiply by',
    '/':  'Divide by',
    'e2': 'Square it',
    'e3': 'Cube it',
    'e4': 'Raise it to the 4th power',
    'e5': 'Raise it to the 5th power',
}

def _validate_phone(raw: str) -> int:
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
    phone = _validate_phone(raw)

    print("Enter three domains (e.g. sports, history, music):")
    domains = []
    for i in range(1, 4):
        domain = input(f"  Domain {i}: ").strip()
        domains.append(domain)

    return phone, domains


def _step_line(op, const, clue_text, diff):
    """Format one puzzle step into a sentence."""
    verb = _OP_VERB[op]
    if op in ('e2', 'e3', 'e4', 'e5'):
        return f"{verb}."
    if clue_text:
        inline = clue_text[0].lower() + clue_text[1:]
        if diff:
            base = inline.rstrip(". ")
            suffix = f"plus {diff}" if diff > 0 else f"minus {abs(diff)}"
            return f"{verb}: {base}, {suffix}."
        return f"{verb} {inline}"
    return f"{verb} {const}"


def build_puzzle_text(seed, steps_with_clues):
    """
    steps_with_clues: list of (op, const, clue_text, diff, source)
    Returns the full puzzle as a string.
    """
    lines = [f"1. Start with {seed}."]
    for i, (op, const, clue_text, diff, _source) in enumerate(steps_with_clues, start=2):
        lines.append(f"{i}. {_step_line(op, const, clue_text, diff)}")
    return "\n".join(lines)


MAX_RETRIES = 3


def _try_build_puzzle(phone, domains, seed, g_model, g_tokenizer, v_model, v_tokenizer):
    """Try to build a puzzle for the given phone number and seed.
    Returns (eq, puzzle, n_valid_clues).
    """
    eq = EquationGenerator(seed=seed).sample(phone)
    chain_seed, steps = expr_to_chain(eq['expr'])

    registry = ClueRegistry()

    # First pass: generate clues via the model for every PREF_INT constant.
    # step_domains tracks which domain was chosen per step (None for skipped steps).
    steps_with_clues = []
    step_domains = []
    for op, const in steps:
        if op in ('e2', 'e3', 'e4', 'e5') or const not in PREF_INTS:
            steps_with_clues.append((op, const, None, None, None))
            step_domains.append(None)
            continue

        domain = random.choice(domains)
        candidates = generate_clues(g_model, g_tokenizer, const, domain, N_CLUES)
        best, diff, _ = find_best(v_model, v_tokenizer, candidates, const)
        if best is not None and registry.is_duplicate(best):
            best = None
        if best is not None:
            registry.register(best)
        source = CLUE_SOURCE_GENERATED if best is not None else None
        steps_with_clues.append((op, const, best, diff, source))
        step_domains.append(domain)

    # Second pass: for PREF_INT steps still without a clue, try the hardcoded lookup.
    for i, ((op, const, clue_text, diff, source), domain) in enumerate(
        zip(steps_with_clues, step_domains)
    ):
        if clue_text is not None or domain is None:
            continue
        hardcoded = get_hardcoded_clue(const, domain)
        if hardcoded and not registry.is_duplicate(hardcoded):
            registry.register(hardcoded)
            steps_with_clues[i] = (op, const, hardcoded, None, CLUE_SOURCE_HARDCODED)

    # Third pass: for PREF_INT steps still without a clue, try alternative representations.
    # Binary is domain-gated (tech/math only) and limited to one use per puzzle.
    _BINARY_DOMAINS = {'math', 'mathematics', 'science', 'computers', 'computing',
                       'technology', 'engineering', 'physics', 'programming'}
    binary_used = False
    for i, (op, const, clue_text, diff, source) in enumerate(steps_with_clues):
        if clue_text is not None or const is None:
            continue
        domain = step_domains[i]

        clue = prime_ordinal_clue(const)
        if clue is None:
            clue = fibonacci_ordinal_clue(const)
        if clue is None:
            clue = roman_clue(const)
        if clue is None:
            clue = spanish_clue(const)
        if clue is None and not binary_used and domain and domain.lower().strip() in _BINARY_DOMAINS:
            clue = binary_clue(const)
            binary_used = True

        if clue is not None and not registry.is_duplicate(clue):
            registry.register(clue)
            steps_with_clues[i] = (op, const, clue, None, CLUE_SOURCE_ALTERNATIVE)

    n_valid = sum(1 for _, _, clue_text, _, _ in steps_with_clues if clue_text is not None)
    puzzle = build_puzzle_text(chain_seed, steps_with_clues)
    return eq, puzzle, n_valid, steps_with_clues, step_domains


def generate_puzzle(phone, domains, g_model, g_tokenizer, v_model, v_tokenizer):
    for attempt in range(MAX_RETRIES):
        seed = random.randint(0, 10_000)
        eq, puzzle, n_valid, steps_with_clues, step_domains = _try_build_puzzle(
            phone, domains, seed, g_model, g_tokenizer, v_model, v_tokenizer
        )
        if n_valid > 0:
            print(f"\nEquation : {eq['infix']}\n")
            print(puzzle)
            return eq, puzzle, steps_with_clues, step_domains
        print(f"Attempt {attempt + 1}: no valid clues found, retrying...")

    raise ValueError(
        f"Could not generate any valid clues after {MAX_RETRIES} attempts. "
        f"Try different domains."
    )


if __name__ == "__main__":
    phone, domains = get_inputs()
    print(f"\nPhone number : {phone}")
    print(f"Domains      : {domains}")

    print(f"\nLoading generator ({MODEL_ID})...")
    g_model, g_tokenizer = load_model()

    print(f"Loading verifier ({VERIFIER_MODEL_ID})...")
    v_model, v_tokenizer = load_verifier()

    generate_puzzle(phone, domains, g_model, g_tokenizer, v_model, v_tokenizer)
