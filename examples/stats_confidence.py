"""
Confidence interval demo: turns one measured pass rate (passes out of n) into a
95% range using the Wilson score interval, which works well for small test sets.
"""
import math


def wilson(passes, n, z=1.96):
    p = passes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    margin = (z / (1 + z * z / n)) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return centre - margin, centre + margin


for passes, n in [(7, 10), (70, 100), (700, 1000), (9, 10), (10, 10)]:
    low, high = wilson(passes, n)
    print(f"{passes}/{n} = {passes / n:.0%}  ->  95% range {low:.0%} to {high:.0%}")