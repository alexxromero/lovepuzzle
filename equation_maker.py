import random 
import math

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
# ------------------------------------------------------------------------- 

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

class EquationGenerator:
    """
    This class generates a randomized equation that evaluates to a target value.  
    For simplicity, we limit the operators to {+, -, *, /, ^2, ^3, ^4, ^5}.
    We want to keep the math simple, without distributive operations acting on 
    multiple parenthesis. The goal is to have a simple equation that can be typed
    on a basic calculator.

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
        """Builds an expression that evaluates to y.
        Example: y = ((123 + 7) * 12) - 14
        
        An expression chained of (op, val) steps is returned.
        Example chain: [("seed", 123), ("+", 7), ("*", 12), ("-", 14)]
        """
        candidates = []  # (operator, value)

        curr = y  # we build the chain backwards from the target value y
        
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
                        comp = curr - dist
                        candidates.append(("+", dist))
                    else:
                        comp = curr + abs(dist)
                        candidates.append(("-", abs(dist)))
                    curr = comp
                    i += 1

                # 2. Apply the exponent
                fop = f"e{exp}"
                candidates.append((fop, root))
                curr = root
                i += 1
                continue

            # If the current number is a perfect square or root, use that
            if curr > 2:
                r = perfect_square_root(curr)
                if r is not None:
                    candidates.append(('e2', r))
                    curr = r
                    i += 1
                    continue

                r = perfect_cube_root(curr)
                if r is not None:
                    candidates.append(('e3', r))
                    curr = r
                    i += 1
                    continue

            # Else select a random binary operators (+, -, *, /)
            op = self._ran_op_weighted()
            if op == '+':
                c = self._ran_int_weighted()
                comp = curr - c
                candidates.append(('+', c))
                curr = comp
                i += 1
            
            elif op == '-':
                # We want to avoid subtractig large numbers to reach the target faster
                c = self._ran_int_weighted()
                while c > 20:
                    c = self._ran_int_weighted()
                comp = curr + c
                candidates.append(('-', c))
                curr = comp
                i += 1

            elif op == '*':
                divisors = self._get_divisors(curr)
                if divisors == [1]:  # prime number. Skip multiplication and try another operator
                    continue

                pref_divisors = [d for d in divisors if d in PREF_INTS]
                if pref_divisors:
                    c = self.rng.choice(pref_divisors)
                else:
                    c = self.rng.choice(divisors)
                    
                comp = curr // c
                candidates.append(('*', c))
                curr = comp 
                i += 1

            elif op == '/':
                # we avoid dividing by large numbers to reach the target faster
                c = self.rng .randint(2, 10)
                comp = curr * c
                candidates.append(('/', c))
                curr = comp
                i += 1

            else:
                raise ValueError(f"Unknown operator: {op}")
        
        steps = [(op, val) for op, val in candidates]

        # Get the leftover value. 
        # Once we reverse the chain, this will be the frist value in the chain
        seed = curr
        steps.append(("seed", seed))
        return steps[::-1]
    
    def sample(self, y, max_depth=8):
        """Returns a randomized expression that evaluates to y. 
        The expression is an ordered chain of (op, val) steps. See _build.
        max_depth determines the maximum number of operations.
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

        chain = self._build(y, max_depth)
        return chain