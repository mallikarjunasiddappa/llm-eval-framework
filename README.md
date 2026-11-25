# llm-eval-framework

[![tests](https://github.com/mallikarjunasiddappa/llm-eval-framework/actions/workflows/tests.yml/badge.svg)](https://github.com/mallikarjunasiddappa/llm-eval-framework/actions/workflows/tests.yml)

A Python + pytest framework for evaluating AI / LLM output: a golden dataset, an LLM-as-a-judge (DeepEval), quality and safety gates, prompt regression testing and a deterministic PII check, all running in CI.

> **Status:** in progress. This is a self-taught project: I'm applying 12+ years of test automation to AI quality engineering, step by step.

## How it works

```
data/golden_set.json   questions + expert reference answers
data/safety_set.json   provocative prompts (rude customer, stereotypes, personal data)
        │
        ▼
ask()                  gpt-4o-mini answers, with a versioned system prompt   (AI under test)
        │
        ├──► find_pii()            regex: phone numbers / emails in the answer   (deterministic, free)
        │
        ▼
Claude Haiku judge     scores every answer with DeepEval                     (independent judge)
   ├─ Answer Relevancy  does the answer address the question?
   ├─ G-Eval Correctness factually consistent with the reference? (fixed steps + rubric)
   ├─ G-Eval fun_fact_rule product rule: fun fact only when no format is requested (strict yes/no)
   └─ Toxicity · Bias · PII Leakage   is the answer harmful?
        │
        ▼
Quality gates          correctness: zero tolerance (any wrong answer fails, named)
                       relevancy:   pass rate ≥ 0.8
                       safety:      zero tolerance (threshold 1.0) + no contact details
```

## What it does today

**Model under test** (`ai_qa/llm.py`)
- `ask(prompt)` is the only place that talks to the LLM. The model is pinned (`gpt-4o-mini`).
- A **system prompt** (`SYSTEM_PROMPT`, sent as `instructions=`) is part of the product and is regression-tested like code.
- OpenAI errors are re-raised as a clear `RuntimeError` that names the model; clients are created lazily, so tests import without API keys.

**Judge and metrics** (`ai_qa/judge.py`), all thresholds named in one place
- `relevancy_metric()`: built-in Answer Relevancy (reference-free).
- `correctness_metric()`: G-Eval with **fixed evaluation steps** (no judge-generated checklist drift) and a **4-band rubric**: contradiction 0–2, key fact missing 3–6, minor issues 7–9, fully correct 10. Optional `verbose` and `strict` modes.
- `fun_fact_metric()`: a product-rule metric in **strict mode** (yes/no): a fun fact only when the user asks for no format, never beyond a requested format.
- `toxicity_metric()`, `bias_metric()`, `pii_metric()`: safety metrics at a **zero-tolerance threshold of 1.0**.
- The judge (Claude Haiku) comes from a different company than the model under test, to reduce self-preference bias.

**Deterministic PII check** (`ai_qa/pii.py`)
- `find_pii(text)` finds email addresses and Swedish mobile numbers with regex: free, instant, the same result every run. Unit-tested, including a red check.

**Tests**
- `tests/test_deepeval.py`: the original quality gate (zero-tolerance correctness + relevancy pass rate).
- `tests/test_deepeval_native.py`: the same golden set with DeepEval's `assert_test`, parametrized and named per case; run with `deepeval test run` for a results table with scores and judge reasons.
- `tests/test_safety.py`: the safety set against Toxicity, Bias and PII Leakage, plus the regex check before any paid judge call.
- `tests/test_golden.py`: golden cases plus data validation (types, non-empty reference answers) before any paid API call.
- `tests/test_pii.py`, `tests/test_metrics.py`, `tests/test_llm.py`: fast unit and error-handling tests.
- pytest design: an `llm` marker separates paid tests from free ones; `--strict-markers`; readable parametrize ids.

## CI (GitHub Actions)

| Job | When | What |
|---|---|---|
| `fast-tests` | every push and pull request | `pytest -m "not llm"`: free, about 25 s |
| `llm-evals` | manually (**Run workflow**) | the LLM quality gate; API keys come from GitHub Secrets |

## Lessons built into the design

- **Relevant ≠ correct.** A wrong answer ("Oslo" for Sweden's capital) scored 1.0 on relevancy and 0.0 on correctness. That's why correctness is a separate, zero-tolerance gate.
- **Fix the metric, not the threshold.** A correct but longer answer scored 0.7, 0.9, 0.7 and 0.3 across runs: the judge punished true extra details. Refining the criteria, then fixed evaluation steps and a rubric, moved correct answers to 0.9–1.0 while wrong ones stay at 0.0. Every metric change is verified in both directions (correct stays green, injected wrong answer still fails).
- **The judge is part of the test system.** Without fixed steps, G-Eval rewrote its own checklist between runs. Even with fixed steps, the same answer scored 0.4, 0.4 and 0.3, so thresholds need margin.
- **Prompt changes are releases.** A "always add a fun fact" system prompt dropped relevancy from 1.0 to 0.25–0.33 on one-word questions; the first fix still failed 3 of 4 cases; putting the format rule first fixed it. A green regression suite only protects old behaviour, so the new feature got its own case and metric.
- **Verify a metric's direction before trusting it.** DeepEval's safety metrics changed direction between versions (old: share of violations; now: 1 = safe). With the default threshold of 0.5, an answer calling the user an "idiot" passed. Safety now uses zero tolerance, which later caught a nationality stereotype (Bias 0.75) that a 0.7 threshold would have missed.
- **Use code for hard rules, the judge for grey areas.** The LLM PII metric gave reasons that contradicted its scores (reported upstream to DeepEval), so contact details are also checked with a deterministic regex.
- **An LLM judge isn't a fact oracle.** Both the model and the judge accepted an outdated fact (Copenhagen's abandoned 2025 carbon-neutral goal). Fact checks need a source: hallucination / faithfulness metrics are next.
- **Test data needs tests.** A string instead of a list once made a check compare letters, a silent false green.

## Project layout

```
llm-eval-framework/
├── .github/workflows/tests.yml   # CI: fast tests on push, LLM evals on demand
├── ai_qa/
│   ├── llm.py                    # ask() + SYSTEM_PROMPT: the model under test
│   ├── judge.py                  # judge, thresholds and all metric builders
│   ├── metrics.py                # pass_rate, average_score
│   └── pii.py                    # find_pii(): deterministic email/phone check
├── data/
│   ├── golden_set.json           # golden dataset with reference answers
│   └── safety_set.json           # provocative prompts for the safety gate
├── examples/                     # experiments: G-Eval steps, safety metric direction, red checks
├── tests/
│   ├── test_metrics.py           # unit tests for the metrics
│   ├── test_golden.py            # golden cases + data validation
│   ├── test_llm.py               # basic LLM tests + error handling
│   ├── test_deepeval.py          # quality gate (custom gate rules)
│   ├── test_deepeval_native.py   # quality gate with DeepEval assert_test
│   ├── test_safety.py            # safety gate: toxicity, bias, PII + regex
│   └── test_pii.py               # unit tests for find_pii
├── pytest.ini
└── requirements.txt
```

## Run it

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
# .venv\Scripts\Activate.ps1       # Windows PowerShell
# source .venv/bin/activate        # macOS / Linux

python -m pip install -r requirements.txt
```

Create a `.env` file (git-ignored) with your keys:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

```bash
python -m pytest -v -m "not llm"                        # fast tests, no keys needed
deepeval test run tests/test_deepeval_native.py         # quality gate with a results table
deepeval test run tests/test_safety.py                  # safety gate
python -m pytest -s -m llm tests/test_safety.py         # same, printing the AI's answers
```

Run the examples as modules, from the project root: `python -m examples.geval_steps`

## Roadmap

- [x] Metrics with an empty-set guard; pytest setup with strict markers
- [x] Call an LLM from Python, with clear error handling
- [x] Golden dataset with data validation and reference answers
- [x] DeepEval: Answer Relevancy + G-Eval correctness with an independent Claude judge
- [x] Quality gate: zero-tolerance correctness + relevancy pass rate
- [x] GitHub Actions: fast tests on every push, LLM evals on demand with secrets
- [x] DeepEval `assert_test` and `deepeval test run`, parametrized per case
- [x] G-Eval in depth: fixed evaluation steps, rubric, strict and verbose modes
- [x] Prompt regression testing with a versioned system prompt; feature metric in strict mode
- [x] Safety gate: toxicity, bias, PII leakage at zero tolerance + deterministic regex PII check
- [ ] Hallucination: check facts against a given source
- [ ] Judge reliability: pinned judge version, median-of-N gating, scored rubric examples, judge-vs-human agreement
- [ ] Prompt injection and red-teaming (DeepTeam)
- [ ] Small RAG app tested for faithfulness and retrieval quality
- [ ] Agent testing: task success, tool correctness, cost and latency
- [ ] Production monitoring / tracing (Langfuse)
