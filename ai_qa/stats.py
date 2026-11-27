"""
Statistics helpers for evals: the Wilson 95% confidence interval for a pass rate.
"""
import math


def wilson(passes, n, z=1.96):
    p = passes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    margin = (z / (1 + z * z / n)) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return centre - margin, centre + margin

def cohen_kappa(a, b):
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    chance = sum((a.count(label) / n) * (b.count(label) / n) for label in labels)
    return (observed - chance) / (1 - chance)