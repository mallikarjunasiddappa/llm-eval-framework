"""
Judge noise demo: the same correct answer scored on several runs (real scores
from Session 6). Compares single runs, the mean and the median against the threshold.
"""
import statistics

THRESHOLD = 0.7
SCORES = [0.7, 0.9, 0.7, 0.3]

print(f"scores: {SCORES}")
print(f"single runs passing: {sum(1 for s in SCORES if s >= THRESHOLD)}/{len(SCORES)}")
print(f"spread: {min(SCORES)} to {max(SCORES)}")
print(f"mean:   {statistics.mean(SCORES):.2f}")
print(f"median: {statistics.median(SCORES):.2f}")