"""
Repeatability check: runs the groundedness judge N times on each fixed case
and counts how often it gives the same verdict.
"""
import json
from deepeval.test_case import LLMTestCase
from ai_qa.judge import groundedness_metric, measure_with_retry

N = 5

with open("data/groundedness_cases.json") as f:
    CASES = json.load(f)

for case in CASES:
    test_case = LLMTestCase(input=case["question"], actual_output=case["answer"], context=[case["source"]])
    verdicts = []
    errors = 0
    retries = 0
    for _ in range(N):
        score, attempts = measure_with_retry(groundedness_metric, test_case)
        if score is None:
            errors += 1
            continue
        retries += attempts - 1
        verdicts.append(int(score))
    passes = sum(verdicts)
    print(f"{case['id']}: {verdicts}  passes={passes}/{len(verdicts)}  errors={errors}  retries={retries}  expected={case['expected']}")