# IndicJevBench — How the Benchmark is Built

**What it is:** The first open benchmark for Indic-language structured decision AI. It tests whether a model can return calibrated probabilities for typed questions (intent, urgency, escalation, routing) given a customer message — in Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, and Hinglish.

---

## Why This Benchmark Exists

Most NLP benchmarks test reading comprehension or generation. IndicJevBench tests something different: **decision quality under uncertainty**, specifically for Indian enterprise use cases (payments, e-commerce, customer support, agent routing).

Existing models score poorly on Indic structured decisions because:
- They were never trained to return calibrated probability distributions
- They do not understand Hinglish (code-mixed Roman Hindi)
- Their training data has no UPI/payments or Indian e-commerce context

IndicJevBench makes this gap visible and measurable.

---

## The Three Question Types

Every benchmark item is a `(state, question)` pair. The question is one of three types:

| Type | What it asks | Output |
|------|-------------|--------|
| `choice` | Pick one option from a fixed list (2–77 options in v1) | Probability per option |
| `score` | Rate on an ordered scale (e.g. urgency 1–5) | Distribution + expected level |
| `noul` | Yes/no (escalate? in-scope?) | P(true) |

A single customer message can have multiple questions — intent + urgency + escalation together.

---

## Data Sources

| Source | License | Tasks | Languages |
|--------|---------|-------|-----------|
| MASSIVE test split | CC BY 4.0 | Intent classification (60 intents) | hi-Deva, bn-Beng, ta-Taml, te-Telu, kn-Knda, ml-Mlym, en-Latn |
| Banking77 | CC BY 4.0 | Fintech intent (77 intents) | Translated to Indic via NLLB-200 |
| COMI-LINGUA | CC BY 4.0 | Language identification, code-mix detection | hi-Latn (Hinglish) |
| Synthetic (cmul8) | CC BY 4.0 | UPI urgency/escalation, e-commerce intent, agent routing | hi-Latn, hi-Deva |

---

## How the Test Data Was Created

**MASSIVE and Banking77** come from existing publicly licensed datasets. Banking77 was originally English-only — we translated it into six Indic languages using `facebook/nllb-200-distilled-1.3B`. Labels are unchanged by translation. Examples where the translated text was too short or linguistically malformed were filtered out automatically.

**COMI-LINGUA** is a Hinglish language identification dataset from IIT Gandhinagar. We use its test split directly.

**Synthetic enterprise data** covers the domains where no public Indic dataset exists: UPI payments, Indian e-commerce, and customer support agent routing in Hinglish and Hindi. These were generated using a two-model pipeline — a large language model writes the customer message and predicts the label, and a second independent model verifies the label is correct. Only examples where both models agree are kept. A sample of 100 examples was hand-reviewed before inclusion.

The test split was frozen once, checksummed, and never modified. No training or calibration data touches the test split.

---

## What the Benchmark Measures

| Metric | What it measures |
|--------|-----------------|
| Accuracy | Fraction of correct predictions |
| Macro-F1 | Per-class F1 averaged — handles class imbalance |
| NLL | Negative log-likelihood — penalises confident wrong answers |
| Brier score | Mean squared error of probability distributions |
| ECE (15 bins) | Calibration error — are stated confidences accurate? |
| MAE of expected level | For score questions: how far is the predicted mean from true level |
| Automatable share | Fraction of items auto-decidable at less than 5% error rate |

Every metric is reported broken down by language, question type, and source dataset.

---

## What Is Not in the Benchmark

- **Training data** — only frozen test splits
- **Model weights** — benchmark is model-agnostic, works against any `/v1/systemone` endpoint
- **Generation tasks** — only structured decision outputs (probabilities), no free-text generation
- **Marathi and Gujarati** — absent from the MASSIVE mirror used for v1; planned for v2
