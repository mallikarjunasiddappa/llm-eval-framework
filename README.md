# llm-eval-framework

[![tests](https://github.com/mallikarjunasiddappa/llm-eval-framework/actions/workflows/tests.yml/badge.svg)](https://github.com/mallikarjunasiddappa/llm-eval-framework/actions/workflows/tests.yml)

A Python + pytest framework for evaluating the quality of AI / LLM output, with a golden dataset, an LLM-as-a-judge (DeepEval) and quality gates that run in CI.

> **Status:** in progress. This is a self-taught project: I'm applying 12+ years of test automation to AI quality engineering, step by step.

## How it works

```
data/golden_set.json   questions + expert reference answers
        │
        ▼
ask()                  gpt-4o-mini answers each question   (AI under test)
        │
        ▼
Claude Haiku judge     scores every answer with DeepEval   (independent judge)
   ├─ Answer Relevancy  does the answer address the question?
   └─ G-Eval Correctness is it factually consistent with the reference answer?
        │
        ▼
Quality gate           correctness: zero tolerance (any wrong answer fails, named)
                       relevancy:   pass rate ≥ 0.8
```

## What it does today

- **`ai_qa/llm.py`, `ask(prompt)`**: the only place that talks to the LLM.
  - The model is pinned (`gpt-4o-mini`), so results don't change because "latest" moved.
  - OpenAI errors (400/401/404/429) are re-raised as a clear `RuntimeError` that names the model.
  - The client is created lazily, so tests import without API keys.
- **`ai_qa/judge.py`**: the judge (Claude Haiku via DeepEval) and **named thresholds**, defined in one place.
  - `relevancy_metric()`: is the answer on topic? (no reference needed)
  - `correctness_metric()`: G-Eval, is the answer factually consistent with the expected output?
  - The judge comes from a different company than the model under test, to reduce self-preference bias.
- **`ai_qa/metrics.py`**: `pass_rate()` and `average_score()`, with a clear `ValueError` on an empty eval set.
- **Golden dataset** (`data/golden_set.json`): each case has an id, a prompt, required words and a reference answer. A fast test validates the data itself before any paid API call.
- **Quality gate** (`tests/test_deepeval.py`): relevancy and correctness per case. Correctness is zero-tolerance; relevancy must reach a pass rate of at least 0.8.
- **pytest design**: an `llm` marker separates slow, paid tests from fast ones; `--strict-markers` catches marker typos; parametrized cases with readable ids; `pytest.approx` for scores.

## CI (GitHub Actions)

| Job | When | What |
|---|---|---|
| `fast-tests` | every push and pull request | `pytest -m "not llm"`: free, about 25 s |
| `llm-evals` | manually (**Run workflow**) | the full LLM quality gate; API keys come from GitHub Secrets |

## Lessons built into the design

- **Relevant ≠ correct.** A wrong answer ("Oslo" for Sweden's capital) scored 1.0 on relevancy. That's why correctness is a separate, zero-tolerance gate.
- **Meaning, not keywords.** A keyword check passed "Stockholm is *not* the capital of Sweden"; G-Eval scored it 0.0.
- **Thresholds come from data.** Thresholds sit between the scores of known-bad answers (0.0–0.1) and good ones (0.8–1.0). A threshold of 0.9 made the gate flaky, because the judge itself varies.
- **Test data needs tests.** A string instead of a list once made a check compare letters, a silent false green.

## Project layout

```
llm-eval-framework/
├── .github/workflows/tests.yml   # CI: fast tests on push, LLM evals on demand
├── ai_qa/
│   ├── llm.py                    # ask(): the model under test
│   ├── judge.py                  # judge + thresholds + metric builders
│   └── metrics.py                # pass_rate, average_score
├── data/golden_set.json          # golden dataset with reference answers
├── examples/                     # small experiments (first call, first judgements)
├── tests/
│   ├── test_metrics.py           # unit tests for the metrics
│   ├── test_golden.py            # golden cases + data validation
│   ├── test_llm.py               # basic LLM tests + error handling
│   └── test_deepeval.py          # the DeepEval quality gate
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
python -m pytest -v -m "not llm"                   # fast tests, no keys needed
python -m pytest -v -s -m llm -k relevancy         # the DeepEval quality gate
```

Run the examples as modules, from the project root: `python -m examples.geval_correctness`

## Roadmap

- [x] Metrics with an empty-set guard; pytest setup with strict markers
- [x] Call an LLM from Python, with clear error handling
- [x] Golden dataset with data validation and reference answers
- [x] DeepEval: Answer Relevancy + G-Eval correctness with an independent Claude judge
- [x] Quality gate: zero-tolerance correctness + relevancy pass rate
- [x] GitHub Actions: fast tests on every push, LLM evals on demand with secrets
- [ ] More DeepEval metrics: toxicity, bias, PII leakage, hallucination
- [ ] Judge calibration against human-labelled samples
- [ ] Small RAG app (LangChain) tested for faithfulness and retrieval quality
- [ ] Agent testing (tool calls), safety and red-teaming
