import random 
import math
from dataclasses import dataclass
from typing import Union, List

# ------------------------------------------------------------------------- #
# ----- FACT-RICH INTEGERS ------------------------------------------------ #
# ------------------------------------------------------------------------- #
# To bias the equations towards interesting numbers.
# PREF_INTS -> dicc with key=int number, val=freq
# The frequency is not normalized.
PREF_INTS = {
    1: 10, 2: 10, 3: 10, 4:10, 5:10, 7: 10, 10: 10,
    6: 8, 8: 7, 9: 7, 
    11: 5, 12: 7, 13: 5, 14: 4, 15: 5, 16: 6, 18: 5, 
    20: 5, 21: 4, 24: 5, 25: 5, 26: 4, 27: 4, 28: 4, 
    30: 4, 32: 3, 36: 3, 
    42: 2, 
    50: 4, 52: 3, 
    60: 3, 66: 2, 
    80: 4, 88: 2, 
    100: 4, 101: 2, 
    180: 2, 360: 2, 1000: 2,
    1776: 1, 1969: 1, 2000: 1, 2020: 1, 2024: 1
}

pref_ints = list(PREF_INTS.keys())
# ------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Const:  # Constant
    value: int

@dataclass(frozen=True)
class BinaryOp:  # Binary Operators
    op: str  # {'+', '-', '*', '/'}
    left: "Expr"
    right: "Expr"

@dataclass(frozen=True)
class UnaryOp:  # Unary Operators (e.g., exponentiation)
    op: str  # {'e2', 'e3', 'e4', 'e5'}
    operand: "Expr"

Expr = Union[Const, BinaryOp, UnaryOp]

def _count_digits(y):
    if y == 0:
        return 1
    
    y = -y if y < 0 else y
    count = 0
    while y > 0:
        y //= 10
        count += 1
    return count

def perfect_square_root(n):
    if n < 0:
        return None
    root = math.isqrt(n)
    return root if root * root == n else None

def perfect_cube_root(n):
    if n < 0:
        return False
    root = round(n ** (1/3))
    return root if root * root * root == n else None

def find_closest_root(n):
    exponents = [2, 3, 4, 5]
    min_dist = float('inf')
    for exp in exponents:
        root = round(n ** (1/exp))
        dist = n - root ** exp
        if abs(dist) < abs(min_dist):
            min_dist = dist
            closest_root = root
            closest_exp = exp
    return closest_exp, closest_root, min_dist

def eval_expr(e):
    """
    Evaluate an expression tree and return the integer result.
    """
    # Evaluating constants
    if isinstance(e, Const):
        return e.value
    
    # Evaluating unary operations
    if isinstance(e, UnaryOp):
        val = eval_expr(e.operand)
        if e.op == 'e2':
            return val * val
        elif e.op == 'e3':
            return val * val * val
        elif e.op == 'e4':
            return val ** 4
        elif e.op == 'e5':
            return val ** 5
        else:
            raise ValueError(f"Unknown unary operator: {e.op}")
    
    # Evaluating binary operations
    l = eval_expr(e.left)
    r = eval_expr(e.right)
    if e.op == '+':
        return l + r
    elif e.op == '-':
        return l - r
    elif e.op == '*':
        return l * r
    elif e.op == '/':
        if r == 0:
            raise ZeroDivisionError("Division by zero")
        if l % r != 0:
            raise ValueError("Non-integer division")
        return l // r
    else:
        raise ValueError(f"Unknown operator: {e.op}")


def to_infix(e):
    """
    Convert expression to infix notation string.
    That's the usual mathematical notation: (a + b) * c, etc.
    """
    if isinstance(e, Const):
        return str(e.value)
    
    if isinstance(e, UnaryOp):
        inner = to_infix(e.operand)
        if e.op == 'e2':
            return f"({inner}^{2})"
        if e.op == 'e3':
            return f"({inner}^{3})"
        if e.op == 'e4':
            return f"({inner}^{4})"
        if e.op == 'e5':
            return f"({inner}^{5})"
    return f"({to_infix(e.left)} {e.op} {to_infix(e.right)})"

def to_prefix_tokens(e, const_token="CONST", add_eos=True):
    """
    Convert expression to (tokens, constants) format.
    The tokens are in prefix notation with 'CONST' as placeholders for constants.
    In prefix notation, the operator precedes its operands, e.g. + 3 4 instead of 3 + 4.
    """
    tokens = []
    constants = []

    def rec(e):
        if isinstance(e, Const):
            tokens.append(const_token)
            constants.append(e.value)
            return 
        if isinstance(e, UnaryOp):
            tokens.append(e.op)
            rec(e.operand)
            return
        tokens.append(e.op)
        rec(e.left)
        rec(e.right)

    rec(e)
    if add_eos:
        tokens.append("EOS")
    return tokens, constants

def expr_to_chain(e):
    """Convert an expression tree into a (seed_value, steps) chain.
    Useful when narrating the steps of the equation.
    """
    steps = []
    node = e
    def unwind(x):
        nonlocal steps 
        if isinstance(x, Const):
            return x.value
        if isinstance(x, UnaryOp):
            v = unwind(x.operand)
            steps.append((x.op, None))
            return v
        
        l = unwind(x.left)
        if not isinstance(x.right, Const):
            cval = eval_expr(x.right)  # turn it into a constant
        else:
            cval = x.right.value
        steps.append((x.op, cval))
        return l
    seed = unwind(node)
    return seed, steps

class EquationGenerator:
    """
    This class generates a randomized equation that evaluates to a target value.  
    For simplicity, we limit the operators to {+, -, *, /, ^2, ^3, ^4, ^5}.
    We want to keep the math simple, avoiding negative values. For example, 
    if we want to subtract 5, we pair (-, 5) instead of (+, -5).
    We also avoid equations with multiple subtrees, meaning we avoid equations 
    where the distributive quality applies over multiple parenthesis, like
    ((1 + 2) / 3 + (4 - 5)) / 6. This makes for simple equations that are easy 
    to type on a regular non-scientific calculator.

    We bias the numbers in the equation toward fact-rich numbers. See PREF_INTS.
    But to make things more interesting, ocasionally draw random numbers from 
    [_int_min, _int_max]. And with less frequency, we draw from an extended 
    range [_int_min_ext, _int_max_ext].
    """
    def __init__(
            self, 
            seed=None
    ):
        self.rng = random.Random(seed)
        self._pref_ints = pref_ints
        self._norm = sum(PREF_INTS.values())
        self._pref_ints_freq = [val / self._norm for val in PREF_INTS.values()]
        self._pref_ints_bias = 0.8
        self._int_min = 1
        self._int_max = 100
        self._small_range_bias = 0.8
        self._int_min_ext = self._int_max + 1
        self._int_max_ext = 150
        self.ops = {'+', '-', '*', '/', 'e2', 'e3'}

    def _ran_int_weighted(self):
        if self.rng.random() < self._pref_ints_bias:
            # draw from the preferred integers
            return self.rng.choices(self._pref_ints, weights=self._pref_ints_freq, k=1)[0]
        else:
            # draw from either the small or the extended range
            if self.rng.random() < self._small_range_bias:
                return self.rng.randint(self._int_min, self._int_max)
            else:
                return self.rng.randint(self._int_min_ext, self._int_max_ext)

    def _ran_op_weighted(self):
        weights = {'+': 4.0, '-': 3.0, '*': 5.0, '/': 1.0}
        total_weight = sum(weights.values())
        r = self.rng.uniform(0, total_weight)
        cumulative = 0.0
        for op, weight in weights.items():
            cumulative += weight
            if r < cumulative:
                return op

    def _get_divisors(self, n):
        # we find divisors up until _int_max
        if n == 0:
            return [d for d in range(1, self._int_max + 1)]
        out = set()
        for d in range(1, min(n, self._int_max)):
            if n % d == 0:
                out.add(d)
        return list(out) if out else [1]

    def _build(self, y, max_depth):
        """Returns a TREE representing an expression that evaluates to y.
        """
        candidates = []  # (operator, constant, previous value)

        # build the chain backwards from the target value y
        curr = y
        c = self._ran_int_weighted()  # random seed value
        
        # To prevent very long equations or large integers, 
        # we enforce an exponent early on, in the first 0-3 operations. 
        inject_exp_idx = self.rng.randint(0, 3)
        i = 0
        while i < max_depth:
            if i == inject_exp_idx:
                # 1. To enforce an exponent, we find the closest root (e^2 to e^5)
                # and add/subtract what's necessary to get to it
                exp, root, dist = find_closest_root(curr)
                if dist != 0:
                    if dist > 0: 
                        prev = curr - dist
                        candidates.append(("+", dist, prev))
                    else:
                        prev = curr + abs(dist)
                        candidates.append(("-", abs(dist), prev))
                curr = prev
                i += 1

                # 2. Apply the exponent
                fop = f"e{exp}"
                candidates.append((fop, None, root))
                curr = root
                i += 1
                continue

            # If the current number is a perfect square or root, use that
            if curr > 2:
                r = perfect_square_root(curr)
                if r is not None:
                    candidates.append(('e2', None, r))
                    curr = r
                    i = + 1
                    continue

                r = perfect_cube_root(curr)
                if r is not None:
                    candidates.append(('e3', None, r))
                    curr = r
                    i = + 1
                    continue

            # Else select a random binary operators (+, -, *, /)
            op = self._ran_op_weighted()
            if op == '+':
                c = self._ran_int_weighted()
                prev = curr - c
                candidates.append(('+', c, prev))
                curr = prev
                i += 1
            
            elif op == '-':
                # We want to avoid subtractig large numbers to reach the target faster
                c = self._ran_int_weighted()
                while c > 20:
                    c = self._ran_int_weighted()
                prev = curr + c
                candidates.append(('-', c, prev))
                curr = prev
                i += 1

            elif op == '*':
                divisors = self._get_divisors(curr, self._int_max)
                if divisors == [1]:  # prime number. Skip multiplication and try another operator
                    continue

                pref_divisors = [d for d in divisors if d in PREF_INTS]
                if pref_divisors:
                    c = self.rng.choice(pref_divisors)
                else:
                    c = self.rng.choice(divisors)
                    
                prev = curr // c
                candidates.append(('*', c, prev))
                curr = prev 
                i += 1

            elif op == '/':
                # we avoid dividing by large numbers to reach the target faster
                c = self.rng.randint(2, 10)
                prev = curr * c
                candidates.append(('/', c, prev))
                curr = prev
                i += 1

            else:
                raise ValueError(f"Unknown operator: {op}")
            
        # Now reverse the chain to generate the equation
        seed_val = curr 
        eq_tree = Const(seed_val)

        for op, c, _ in reversed(candidates):
            if op in {'e2', 'e3', 'e4', 'e5'}:
                eq_tree = UnaryOp(op, eq_tree)
            else:
                eq_tree = BinaryOp(op, eq_tree, Const(c))
        
        return eq_tree
    
    def sample(self, y, max_depth=8):
        """Returns a randomized expression that evaluates to y.
        Output:
        **eq**: The generated expression as a tree.
        **infix**: The generated expression in infix notation (regular math format).
        **tokens**: The generated equation as a list of prefix tokens, 
                    with 'CONST' as placeholders for constants.
        **consts**: The list of constant values corresponding to the 'CONST' tokens.
        **value**: The integer value that the generated equation evaluates to (should equal y).
        """
        if not isinstance(y, int) or y < 0:
            raise ValueError("The target value must be a positive integer.")
        
        ndigits = _count_digits(y)
        if ndigits < 10:
            raise ValueError("The target value must have at least 10 digits.")
        if ndigits > 15:
            raise ValueError("The target value must have at most 15 digits.")
        
        if max_depth < 5:
            raise ValueError("max_depth must be at least 5.")

        eq = self._build(y, max_depth)
        eval = eval_expr(eq)
        if eval != y:
            print(to_infix(eq))
            raise ValueError(f"Generated expression does not evaluate to target value: {eval} != {y}")
        
        tokens, consts = to_prefix_tokens(eq, const_token="CONST", add_eos=True)

        return {"expr": eq, 
                "infix": to_infix(eq), 
                "tokens": tokens,
                "consts": consts}