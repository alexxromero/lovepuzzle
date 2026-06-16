import re
import random
import phonenumbers

from equation_maker import EquationGenerator, PREF_INTS
from clue_generator import load_model, generate_clues, find_best, MODEL_ID
from verifier import load_verifier, VERIFIER_MODEL_ID
from hardcoded_clues import get_hardcoded_clue
from alternative_representations import (
    roman_clue, binary_clue, prime_ordinal_clue,
    fibonacci_ordinal_clue, spanish_clue,
)

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

    DUPLICATE_THRESHOLD = 0.6

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
    phone = validate_phone_number(raw)

    print("Enter three interests (e.g. sports, history, music):")
    domains = []
    for i in range(1, 4):
        domain = input(f"  Interest {i}: ").strip()
        domains.append(domain)

    return phone, domains


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


def build_puzzle_text(seed, clues_info):
    """Returns the full narrated puzzle as a string."""
    lines = [f"1. Start with {seed}."]
    i = 2
    for (op, val, clue, correction, _, _) in clues_info:
        lines.append(f"{i}. {narrate_step(op, val, clue, correction)}")
        i += 1
    return "\n".join(lines)


def _try_build_puzzle(phone, domains, seed, g_model, g_tokenizer, v_model, v_tokenizer):
    """Try to build a puzzle for the given phone number and seed.
    Returns (eq, puzzle, n_valid_clues).
    """
    equation_chain = EquationGenerator(seed=seed).sample(phone)
    seed_op, seed_val = equation_chain[0]
    assert(seed_op == "seed")

    registry = ClueRegistry()  # track clues to avoid duplicates

    clues_info = []  #(operator, value, best clue, correction, domain, clue source)

    # First pass: LMM clue generator
    for op, val in equation_chain[1:]:
        if op in ('e2', 'e3', 'e4', 'e5') or val not in PREF_INTS:
            # no clue generation
            clues_info.append((op, val, None, None, None, None))
            continue

        domain = random.choice(domains)
        candidates = generate_clues(g_model, g_tokenizer, val, domain, N_CLUES)
        best, diff, _ = find_best(v_model, v_tokenizer, candidates, val)
        # TODO: don't throw away the rest of the generated clues. If the best clue is duplicate, move on to the next
        if best is not None and registry.is_duplicate(best):
            best = None
        if best is not None:
            registry.register(best)
        source = CLUE_SOURCE_GENERATED if best is not None else None
        clues_info.append((op, val, best, diff, domain, source))
        
    # Second pass: for values without a clue, look up if there are any hard-coded clues
    # that we can use
    for i, (op, val, clue_text, _, domain, _) in enumerate(clues_info):
        if clue_text is not None:  # already has a clue
            continue
        if domain is None:  # likely not a fact-rich number
            continue
        hardcoded = get_hardcoded_clue(val, domain)
        if hardcoded and not registry.is_duplicate(hardcoded):
            registry.register(hardcoded)
            clues_info[i] = (op, val, hardcoded, None, "misc", CLUE_SOURCE_HARDCODED)

    # Third pass: try alternative representations.
    # Binary is domain-gated (tech/math only) and limited to one use per puzzle.
    _BINARY_DOMAINS = {'math', 'mathematics', 'science', 'computers', 'computing',
                       'technology', 'engineering', 'physics', 'programming'}
    binary_used = False
    for i, (op, val, clue_text, _, domain, _) in enumerate(clues_info):
        if clue_text is not None or val is None:
            continue

        clue = prime_ordinal_clue(val)
        if clue is not None:
            domain = "prime_rep"
        if clue is None:
            clue = fibonacci_ordinal_clue(val)
            if clue is not None:
                domain = "fibonacci_rep"
        if clue is None:
            clue = roman_clue(val)
            if clue is not None:
                domain = "roman_rep"
        if clue is None:
            clue = spanish_clue(val)
            if clue is not None:
                domain = "spanish_rep"
        if clue is None and not binary_used and domain and domain.lower().strip() in _BINARY_DOMAINS:
            clue = binary_clue(val)
            domain = "binary_rep"
            binary_used = True

        if clue is not None and not registry.is_duplicate(clue):
            registry.register(clue)
            clues_info[i] = (op, val, clue, None, domain, CLUE_SOURCE_ALTERNATIVE)

    n_valid = sum(1 for _, _, clue, _, _, _ in clues_info if clue is not None)
    puzzle = build_puzzle_text(seed_val, clues_info)
    return equation_chain, puzzle, n_valid, clues_info


def generate_puzzle(phone, domains, g_model, g_tokenizer, v_model, v_tokenizer):
    for attempt in range(MAX_RETRIES):
        seed = random.randint(0, 10000)
        eq, puzzle, n_valid, clues_info = _try_build_puzzle(
            phone, domains, seed, g_model, g_tokenizer, v_model, v_tokenizer
        )
        if n_valid > 0:
            print(f"\nEquation : {eq}\n")
            print(puzzle)
            return eq, puzzle, clues_info
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
