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

