# metrics.py - eval metrics: turn a set of scores into one number for a quality gate.
#   pass_rate(scores, threshold): share of cases with score >= threshold, rounded to 2 decimals
#                                 (2 of 3 -> 0.67). Empty dict -> ValueError (fail loudly).
#   average_score(scores):        mean of all scores, NOT rounded -> compare with pytest.approx.

def pass_rate(scores, threshold):
    if not scores:
        raise ValueError("No test cases to evaluate")
    passed = 0
    for name, score in scores.items():
        if score >= threshold:
            passed += 1
    return round(passed / len(scores), 2)

def average_score(scores):
    if not scores:
        raise ValueError("No test cases to evaluate")
    return sum(scores.values()) / len(scores)
