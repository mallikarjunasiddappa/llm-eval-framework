"""
Sample-size demo: simulates a model whose true pass rate is 70% and shows how
much the measured pass rate jumps around with 10 test cases vs 100.
"""
import random

TRUE_RATE = 0.7
RUNS = 1000
random.seed(7)

for n in [10, 100]:
    rates = []
    for _ in range(RUNS):
        passes = sum(1 for _ in range(n) if random.random() < TRUE_RATE)
        rates.append(passes / n)
    rates.sort()
    print(f"n={n}: average of {RUNS} runs = {sum(rates) / RUNS:.3f}")
    low = rates[int(RUNS * 0.025)]
    high = rates[int(RUNS * 0.975)]
    print(f"n={n}: lowest={min(rates):.2f}  highest={max(rates):.2f}  middle 95%: {low:.2f} to {high:.2f}")