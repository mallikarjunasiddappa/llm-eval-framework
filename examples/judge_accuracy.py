"""
Accuracy check: compares the groundedness judge's verdicts with labels
(original human labels or final reviewed labels) and prints the agreement
and a confusion matrix.
"""
import json
from deepeval.test_case import LLMTestCase
from ai_qa.judge import groundedness_metric, measure_with_retry

LABEL = "human"

with open("data/labelled_set.json") as f:
    CASES = json.load(f)

matrix = {}
agree = 0
for case in CASES:
    test_case = LLMTestCase(input=case["question"], actual_output=case["answer"], context=[case["source"]])
    score, _ = measure_with_retry(groundedness_metric, test_case)
    if score is None:
        print(f"{case['id']}: judge error")
        continue
    judge = "pass" if score == 1 else "fail"
    human = case[LABEL]
    matrix[(human, judge)] = matrix.get((human, judge), 0) + 1
    if human == judge:
        agree += 1
    else:
        print(f"DISAGREE {case['id']}: {LABEL}={human} judge={judge}")

print(f"label used: {LABEL}")
print(f"agreement: {agree}/{len(CASES)}")
for human in ["pass", "fail"]:
    print(f"{LABEL}={human}:  judge pass={matrix.get((human, 'pass'), 0)}  judge fail={matrix.get((human, 'fail'), 0)}")