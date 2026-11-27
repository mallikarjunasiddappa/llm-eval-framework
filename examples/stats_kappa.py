"""
Cohen's kappa: agreement between my human labels and the judge, corrected for
chance. The judge's verdicts equal the reviewed 'final' labels (proven by
tests/test_judge_regression.py), so no API calls are needed.
"""
import json
from ai_qa.stats import cohen_kappa

with open("data/labelled_set.json") as f:
    CASES = json.load(f)

human = [c["human"] for c in CASES]
judge = [c["human"] for c in CASES]
agree = sum(1 for h, j in zip(human, judge) if h == j)
print(f"agreement: {agree}/{len(CASES)} = {agree / len(CASES):.0%}")
print(f"kappa: {cohen_kappa(human, judge):.2f}")