"""
Test-set size: how many cases are needed for a given precision, using the rough
rule margin ≈ 1/sqrt(n), checked with the Wilson interval at a 70% pass rate.
"""
import math
from ai_qa.stats import wilson

for margin in [0.20, 0.10, 0.05, 0.03]:
    n = math.ceil(1 / margin ** 2)
    passes = round(0.7 * n)
    low, high = wilson(passes, n)
    print(f"±{margin:.0%} -> about {n} cases   (check at 70%: {low:.0%} to {high:.0%})")