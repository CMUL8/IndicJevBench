# IndicJevBench — How the Benchmark is Built

**What it is:** The first open benchmark for Indic-language structured decision AI. It tests whether a model can return calibrated probabilities for typed questions (intent, urgency, escalation, routing) given a customer message — in Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, and Hinglish.

Released *before* the Nirṇaya model so the community has a leaderboard to run against before we ship.

---

## Why This Benchmark Exists

Most NLP benchmarks test reading comprehension or generation. IndicJevBench tests something different: **decision quality under uncertainty**, specifically for Indian enterprise use cases (payments, e-commerce, customer support, agent routing).

Existing models score poorly on Indic structured decisions because:
- They were never trained to return calibrated probability distributions
- They don't understand Hinglish (code-mixed Roman Hindi)
- Their training data has no UPI/payments or Indian e-commerce context

IndicJevBench makes this gap visible and measurable.

---

## The Three Question Types

Every benchmark item is a `(state, question)` pair. The question is one of three types:

| Type | What it asks | Output |
|------|-------------|--------|
| `choice` | Pick one option from 2–64 | Probability per option |
| `score` | Rate on an ordered scale (e.g. urgency 1–5) | Distribution + expected level |
| `noul` | Yes/no (escalate? in-scope?) | P(true) |

A single customer message can have multiple questions — intent + urgency + escalation together.

---

## Data Sources

### What goes directly into the benchmark (redistributable)

| Source | License | Tasks | Languages |
|--------|---------|-------|-----------|
| MASSIVE test split | CC BY 4.0 | Intent classification (60 intents) | hi-Deva, bn-Beng, ta-Taml, te-Telu, kn-Knda, ml-Mlym |
| Banking77 | CC BY 4.0 | Fintech intent (77 intents) | en-Latn (translated to Indic via NLLB) |
| COMI-LINGUA | CC BY 4.0 | Language ID, code-mix detection | hi-Latn (Hinglish) |
| Synthetic (test_synth) | CC BY 4.0 (ours) | UPI urgency/escalation, e-commerce intent, agent routing | hi-Latn, hi-Deva |

### What ships as a download script only (v1)

| Source | License | Reason | Script |
|--------|---------|--------|--------|
| Bitext customer support | CDLA-Sharing-1.0 | Share-alike clause would infect the entire benchmark under CDLA — can't release | `scripts/fetch_bitext.py` |
| IndicXNLI | CC BY-NC 4.0 | Non-commercial clause — can't include in a CC BY release | `scripts/fetch_indicxnli.py` |

Users run these scripts locally to reconstruct those splits. The harness detects missing splits and prints instructions.

### v2 plan (after initial release)

Replace all Bitext-derived tasks with fully synthetic CC BY 4.0 equivalents. This makes IndicJevBench 100% redistributable with no share-alike or NC strings attached. v2 will be a complete re-release under clean CC BY 4.0.

---

## How the Test Data Was Created

### Step 1 — Original English datasets (Phase 1–2)

Raw datasets were ingested and converted into the unified `Example` schema:
- `state`: the customer message
- `questions`: typed questions with labels

Labels for score/noul that didn't exist in the original data (e.g. bitext only had intent) were derived by rule — urgency from intent type, escalation from whether the issue involves money loss or account access. These rules are documented in the converter code.

### Step 2 — Machine translation to Indic languages (Phase 3A)

English rows from bitext, Banking77, and CLINC150 were translated into **hi, bn, ta, te, kn, ml** using `facebook/nllb-200-distilled-1.3B`.

- Labels are unchanged — translation preserves meaning, not labels
- ChrF quality filter: examples with ChrF < 10 and length < 5 chars are rejected
- Result: 311,324 translated examples, 130 rejected

NLLB was chosen over IndicTrans2 because IndicTrans2 is incompatible with transformers≥4.40 (removed `transformers.onnx` module). NLLB achieves within 2–3 BLEU on our target languages for short enterprise text.

### Step 3 — Synthetic Hinglish and Hindi data (Phase 3B)

Hinglish (`hi-Latn`, Roman code-mixed Hindi) cannot be produced by NLLB. It was generated synthetically using a two-model pipeline:

**Generator** (`qwen/qwen3-235b-a22b`): given a task slot definition and a few seed examples, generates batches of customer messages with labels.

**Checker** (`deepseek/deepseek-chat-v3-0324`): independently verifies that every label in a generated example is correct. Only examples where checker agrees are kept.

Five task slots were generated:

| Slot | Language | Domain | Questions |
|------|----------|--------|-----------|
| `hinglish_upi_score_noul` | hi-Latn | UPI/digital payments | urgency (score), escalate (noul) |
| `hinglish_ecom_intent` | hi-Latn | Indian e-commerce | intent (choice, 9 options), urgency (score) |
| `hinglish_agent_routing` | hi-Latn | Customer support routing | department (choice, 6 options), escalate (noul) |
| `hindi_upi_score_noul` | hi-Deva | UPI/digital payments | urgency (score), escalate (noul) |
| `hindi_ecom_score_noul` | hi-Deva | Indian e-commerce | urgency (score), escalate (noul) |

Target: 500 accepted examples per slot → 2,500 synthetic examples total.

The questions (instructions, options, qids) are **fixed per slot** — the LLM only generates the customer message (`state`) and predicts the label. The checker then validates label correctness.

**100 synthetic test examples are hand-reviewed before being included in the benchmark.**

### Step 4 — Freezing the test split (Phase 2)

Test splits were created once, sha256-hashed, and frozen in `SPLIT_HASHES.json`. They are never retouched. Training, calibration, and evaluation use strictly separate splits.

---

## What the Benchmark Measures

### Metrics per task

| Metric | What it measures |
|--------|-----------------|
| Accuracy | Fraction of correct argmax predictions |
| Macro-F1 | Per-class F1 averaged — handles class imbalance |
| NLL | Negative log-likelihood — penalises confident wrong answers |
| Brier score | Mean squared error of probability distributions |
| ECE (15 bins) | Calibration error — are stated confidences accurate? |
| MAE of expected level | For score questions: how far is the predicted mean from true level |
| Automatable share | Fraction of items auto-decidable at <5% error rate (confidence threshold) |

### Breakdown dimensions

Every metric is reported broken down by:
- **Language** (hi-Deva, hi-Latn, bn-Beng, ta-Taml, te-Telu, kn-Knda, ml-Mlym, en-Latn)
- **Question type** (choice, score, noul)
- **Source** (MASSIVE, Banking77, COMI-LINGUA, synthetic)

---

## Baselines Evaluated on the Benchmark

Before releasing, we run three external baselines:

1. **Zero-shot Qwen3-4B-Instruct option-logprob** — same prompt, score options by log-probability. Shows what fine-tuning adds vs a raw instruct model.

2. **Laya-multilingual** (`convaiinnovations/laya-multilingual`) — the closest prior work on Indic intent classification. Run on our MASSIVE Indic test, or cite published numbers labeled "as published by convaiinnovations".

3. **Generative API LLM** — a frontier model (GPT-4o or equivalent) asked the same questions in JSON output mode. Records accuracy, latency, and $/1k decisions to show the cost of using generation for what should be a one-pass decision.

Competitor numbers are labeled "as published by \<source\>" unless we reproduced them ourselves.

---

## Directory Layout

```
bench/indicjevbench/
  tasks/
    intent_massive/          # MASSIVE Indic intent, CC BY 4.0
    fintech_banking77/       # Banking77 translated, CC BY 4.0
    hinglish_lid/            # COMI-LINGUA, CC BY 4.0
    synthetic_enterprise/    # test_synth, CC BY 4.0
  scripts/
    fetch_bitext.py          # reconstruct Bitext-derived tasks locally
    fetch_indicxnli.py       # reconstruct IndicXNLI tasks locally
  run_bench.py               # harness: --endpoint <url>
  leaderboard.json           # results from all evaluated models
  DATASHEET.md               # sources, licenses, known biases, synthetic share
  BENCHMARK_EXPLAINED.md     # this file
```

---

## Release Timeline

| Event | When |
|-------|------|
| IndicJevBench released on HuggingFace as `cmul8/IndicJevBench` | **Before** Nirṇaya model ships |
| Baseline numbers published in dataset card | Same time |
| Short blog: "first benchmark for Indic structured decision AI" | Same time |
| Nirṇaya model released | 2–4 weeks later |
| v2: Bitext tasks replaced with synthetic CC BY 4.0 equivalents | Post-release |

Releasing the benchmark first establishes cmul8 as the authority on Indic decision AI evaluation before any model competition, and means the model ships into an ecosystem that already has a leaderboard.

---

## What Is Not in the Benchmark

- **Training data** — only frozen test splits; no train/calib rows
- **Model weights** — benchmark is model-agnostic, works against any `/v1/systemone` endpoint
- **Generation tasks** — only structured decision outputs (probabilities), no free-text generation
- **Mr/Gu languages** — Marathi and Gujarati are absent from the MASSIVE mirror we could access; not included in v1
