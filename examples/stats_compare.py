"""
Compare two versions on the same test cases: pass rates with 95% ranges,
and which cases the new version fixed or broke.
"""
from ai_qa.stats import wilson

V1 = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]
V2 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]

for name, results in [("v1", V1), ("v2", V2)]:
    passes = sum(results)
    n = len(results)
    low, high = wilson(passes, n)
    print(f"{name}: {passes}/{n} = {passes / n:.0%}  ->  95% range {low:.0%} to {high:.0%}")

fixed = [i for i in range(len(V1)) if V1[i] == 0 and V2[i] == 1]
broken = [i for i in range(len(V1)) if V1[i] == 1 and V2[i] == 0]
print(f"v2 fixed {len(fixed)} cases: {fixed}")
print(f"v2 broke {len(broken)} cases: {broken}")