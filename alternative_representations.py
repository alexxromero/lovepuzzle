_ROMAN_VALUES = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100,  "C"), (90,  "XC"), (50,  "L"), (40,  "XL"),
    (10,   "X"), (9,   "IX"), (5,   "V"), (4,   "IV"),
    (1,    "I"),
]


def to_roman(n: int) -> str | None:
    """Convert a positive integer to a Roman numeral string.
    Returns None for numbers outside the range 1–3999.
    """
    if not 1 <= n <= 3999:
        return None
    result = []
    for value, numeral in _ROMAN_VALUES:
        while n >= value:
            result.append(numeral)
            n -= value
    return "".join(result)


def to_binary(n: int) -> str:
    """Convert a non-negative integer to its binary string (e.g. 42 → '101010')."""
    if n < 0:
        raise ValueError("Binary representation is only defined for non-negative integers.")
    return bin(n)[2:]


# ---------------------------------------------------------------------------
# Prime ordinals (primes up to 100, mapped to their 1-based position)
# ---------------------------------------------------------------------------
PRIME_ORDINALS = {
     2: "1st",   3: "2nd",   5: "3rd",   7: "4th",  11: "5th",
    13: "6th",  17: "7th",  19: "8th",  23: "9th",  29: "10th",
    31: "11th", 37: "12th", 41: "13th", 43: "14th", 47: "15th",
    53: "16th", 59: "17th", 61: "18th", 67: "19th", 71: "20th",
    73: "21st", 79: "22nd", 83: "23rd", 89: "24th", 97: "25th",
}

# ---------------------------------------------------------------------------
# Fibonacci ordinals (Fibonacci numbers up to 100, mapped to their position)
# Sequence: F(1)=1, F(2)=1, F(3)=2, F(4)=3, F(5)=5, ...
# Duplicate value 1 maps to its first occurrence (1st).
# ---------------------------------------------------------------------------
FIBONACCI_ORDINALS = {
     0: 0,
     1: 1,  # and 2
     2: 3,
     3: 4,
     5: 5,
     8: 6,
    13: 7,
    21: 8,
}

# ---------------------------------------------------------------------------
# Spanish number words 1–100
# ---------------------------------------------------------------------------
SPANISH_NUMBERS = {
     1: "uno",          2: "dos",          3: "tres",         4: "cuatro",
     5: "cinco",        6: "seis",         7: "siete",        8: "ocho",
     9: "nueve",       10: "diez",        11: "once",        12: "doce",
    13: "trece",       14: "catorce",     15: "quince",      16: "dieciséis",
    17: "diecisiete",  18: "dieciocho",   19: "diecinueve",  20: "veinte",
    21: "veintiuno",   22: "veintidós",   23: "veintitrés",  24: "veinticuatro",
    25: "veinticinco", 26: "veintiséis",  27: "veintisiete", 28: "veintiocho",
    29: "veintinueve", 30: "treinta",
    31: "treinta y uno",    32: "treinta y dos",    33: "treinta y tres",
    34: "treinta y cuatro", 35: "treinta y cinco",  36: "treinta y seis",
    37: "treinta y siete",  38: "treinta y ocho",   39: "treinta y nueve",
    40: "cuarenta",
    41: "cuarenta y uno",   42: "cuarenta y dos",   43: "cuarenta y tres",
    44: "cuarenta y cuatro",45: "cuarenta y cinco", 46: "cuarenta y seis",
    47: "cuarenta y siete", 48: "cuarenta y ocho",  49: "cuarenta y nueve",
    50: "cincuenta",
    51: "cincuenta y uno",  52: "cincuenta y dos",  53: "cincuenta y tres",
    54: "cincuenta y cuatro",55:"cincuenta y cinco",56: "cincuenta y seis",
    57: "cincuenta y siete",58: "cincuenta y ocho", 59: "cincuenta y nueve",
    60: "sesenta",
    61: "sesenta y uno",    62: "sesenta y dos",    63: "sesenta y tres",
    64: "sesenta y cuatro", 65: "sesenta y cinco",  66: "sesenta y seis",
    67: "sesenta y siete",  68: "sesenta y ocho",   69: "sesenta y nueve",
    70: "setenta",
    71: "setenta y uno",    72: "setenta y dos",    73: "setenta y tres",
    74: "setenta y cuatro", 75: "setenta y cinco",  76: "setenta y seis",
    77: "setenta y siete",  78: "setenta y ocho",   79: "setenta y nueve",
    80: "ochenta",
    81: "ochenta y uno",    82: "ochenta y dos",    83: "ochenta y tres",
    84: "ochenta y cuatro", 85: "ochenta y cinco",  86: "ochenta y seis",
    87: "ochenta y siete",  88: "ochenta y ocho",   89: "ochenta y nueve",
    90: "noventa",
    91: "noventa y uno",    92: "noventa y dos",    93: "noventa y tres",
    94: "noventa y cuatro", 95: "noventa y cinco",  96: "noventa y seis",
    97: "noventa y siete",  98: "noventa y ocho",   99: "noventa y nueve",
   100: "cien",
}


def prime_ordinal_clue(n: int) -> str | None:
    """Return a clue string if n is a prime ≤ 100, else None.
    Example: 7 → 'the 4th prime number'
    """
    ordinal = PRIME_ORDINALS.get(n)
    if ordinal is None:
        return None
    return f"the {ordinal} prime number"

def sub(n):
    return str(n).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))

def fibonacci_ordinal_clue(n: int) -> str | None:
    """Return a clue string if n is a Fibonacci number ≤ 100, else None.
    """
    ordinal = FIBONACCI_ORDINALS.get(n)
    if ordinal is None:
        return None
    return f"the F{sub(ordinal)} Fibonacci number (F{sub(0)}=0, F{sub(1)}=1, ...)"


def spanish_clue(n: int) -> str | None:
    """Return a clue string using the Spanish word for n (1–100), else None.
    Example: 42 → 'cuarenta y dos in Spanish'
    """
    word = SPANISH_NUMBERS.get(n)
    if word is None:
        return None
    return f"{word} in Spanish"


def roman_clue(n: int) -> str | None:
    """Return a clue string using the Roman numeral representation of n.
    Returns None if n is outside the range 1–3999.
    """
    roman = to_roman(n)
    if roman is None:
        return None
    return f"the Roman numeral {roman}"


def binary_clue(n: int) -> str:
    """Return a clue string using the binary representation of n."""
    return f"the binary number {to_binary(n)}"
