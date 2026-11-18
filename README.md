# llm-eval-framework

A Python + pytest framework for evaluating the quality of AI / LLM output: eval metrics, quality gates and, later, CI.

> **Status:** early, in progress. This is a self-taught learning project. I'm building it step by step on the way to AI QA engineering.

## What it does today

- **`pass_rate(scores, threshold)`**: given a set of eval scores, returns the share of cases that meet the threshold, rounded to 2 decimals.
  - Raises a clear `ValueError` for an empty set instead of crashing, because an empty eval set usually means something upstream is broken.
- **Tests (pytest)**: normal cases, a data-driven (parametrized) set of cases, and the empty-set error.
- **Strict markers**: tests are tagged (`metrics`), and `--strict-markers` turns any marker typo into an error, so tests can't be skipped silently.

## Project layout

```
llm-eval-framework/
├── ai_qa/
│   ├── __init__.py
│   └── metrics.py        # pass_rate
├── tests/
│   └── test_metrics.py
├── pytest.ini            # import path, test folder, markers
└── requirements.txt
```

## Run it

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
# .venv\Scripts\Activate.ps1       # Windows PowerShell
# source .venv/bin/activate        # macOS / Linux

python -m pip install -r requirements.txt
python -m pytest -v
```

Run only the metric tests:

```bash
python -m pytest -v -m metrics
```

## Roadmap

- [x] Pass-rate metric with an empty-set guard
- [x] pytest setup: parametrized cases, strict markers
- [ ] Call an LLM API from Python and evaluate real responses
- [ ] Golden dataset + pass rate over repeated runs
- [ ] DeepEval metrics (answer relevancy, faithfulness, LLM-as-judge)
- [ ] Small RAG app (LangChain) tested with Ragas / DeepEval
- [ ] Safety tests: prompt injection, jailbreaks, data leakage
- [ ] Eval quality gate in GitHub Actions
