"""Check the local answer key using exact arithmetic and enumeration."""
import json
import math
from fractions import Fraction
from itertools import product
from pathlib import Path

root = Path(__file__).resolve().parents[1]
answers = {row["id"]: row["answer"] for row in json.loads((root / "data/math_checks.json").read_text())}
recurrence = [2, 5]
for i in range(2, 9):
    recurrence.append(5 * recurrence[-1] - 6 * recurrence[-2])
powers = [2, 3]
for i in range(2, 6):
    powers.append(3 * powers[-1] - powers[-2])
dice = [(a, b) for a in range(1, 7) for b in range(1, 7) if (a+b) % 2 == 0]
paths = 0
for steps in product([-1, 1], repeat=12):
    balance = 0
    for step in steps:
        balance += step
        if balance < 0:
            break
    else:
        paths += balance == 0
actual = {
    "remainder": pow(7, 2026, 13),
    "divisors": sum(360 % d == 0 for d in range(1, 361)),
    "quadratic": sum(x**3 for x in range(8) if x*x-7*x+10 == 0),
    "geometry": math.isqrt(21*8*7*6),
    "combinatorics": sum(sum(s) == 4 and all(not(s[i] and s[i+1]) for i in range(9)) for s in product([0,1], repeat=10)),
    "probability": Fraction(sum(a+b == 8 for a,b in dice), len(dice)),
    "diophantine": sum(Fraction(1,x)+Fraction(1,y) == Fraction(1,12) for x in range(1,157) for y in range(1,157)),
    "recurrence": recurrence[8],
    "factorial": sum(math.factorial(i) for i in range(1,21)) % 1000,
    "lattice": paths,
    "crt": next(n for n in range(1,106) if n%3 == 2 and n%5 == 3 and n%7 == 2),
    "polynomial": powers[5],
}
assert {k: str(v) for k,v in actual.items()} == answers
print("All 12 mathematical answer keys verified.")
