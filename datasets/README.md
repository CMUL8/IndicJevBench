---
license: cc-by-4.0
language:
- hi
- bn
- ta
- te
- kn
- ml
tags:
- intent-classification
- structured-prediction
- indic-languages
- hinglish
- decision-ai
pretty_name: IndicJevBench
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: intent_massive
    path: v1/intent_massive.jsonl
  - split: fintech_banking77
    path: v1/fintech_banking77.jsonl
  - split: hinglish_lid
    path: v1/hinglish_lid.jsonl
  - split: synthetic_enterprise
    path: v1/synthetic_enterprise.jsonl
---

# IndicJevBench

The first open benchmark for structured decision AI in Indian languages.

IndicJevBench tests whether a model can answer typed questions about a customer message and return calibrated probabilities. Questions cover intent classification, urgency scoring, escalation detection, and agent routing.

Languages covered: Hindi (Devanagari), Hindi (Romanized / Hinglish), Bengali, Tamil, Telugu, Kannada, Malayalam.

---

## Tasks

| Dataset | Type | Languages | Items | License |
|---------|------|-----------|-------|---------|
| intent_massive | Intent classification | hi, bn, ta, te, kn, ml | 57,922 | CC BY 4.0 |
| fintech_banking77 | Banking intent | hi, bn, ta, te, kn, ml | 7,816 | CC BY 4.0 |
| hinglish_lid | Language identification | hi-Latn (Hinglish) | 3,220 | CC BY 4.0 |
| synthetic_enterprise | Urgency, escalation, routing | hi, hi-Latn | 444 | CC BY 4.0 |

Total: 69,402 items.

---

## Format

Each item looks like this:

```json
{
  "id": "massive-hi-Deva-0001-q0",
  "family": "intent",
  "lang": "hi-Deva",
  "state": "मुझे अपनी फ्लाइट बुक करनी है",
  "question": {
    "type": "choice",
    "instructions": "What is the customer's intent?",
    "options": ["book_flight", "cancel_flight", "check_status", "other"]
  },
  "expected": 0,
  "license": "CC BY 4.0"
}
```

Question types:
- **choice**: pick one option from a list
- **score**: rate on an ordered scale (urgency 1 to 5)
- **noul**: yes or no decision (should this be escalated?)

---

## Scoring

Models are scored on the **IndicJevScore**, a single number from 0 to 100:

| Axis | Weight |
|------|--------|
| Accuracy | 35% |
| Calibration (ECE + Brier) | 25% |
| Speed | 20% |
| Cost per 1k decisions | 20% |

The score is the geometric mean of all four axes.

---

## Sources

- **MASSIVE** (Amazon, CC BY 4.0) - intent classification across Indic languages
- **Banking77** (PolyAI, CC BY 4.0) - banking intent, translated to Indic via NLLB-200
- **COMI-LINGUA** (IIT Gandhinagar, CC BY 4.0) - Hinglish language identification
- **Synthetic** (cmul8, CC BY 4.0) - enterprise support data generated for UPI payments, e-commerce, and agent routing

---

## Leaderboard

Measured by cmul8 on 2026-09-22 using 1000 items per dataset (where available).

| Model | fintech_banking77 acc | fintech_banking77 IJScore | synthetic_enterprise acc | synthetic_enterprise IJScore | intent_massive | hinglish_lid |
|-------|----------------------|--------------------------|-------------------------|------------------------------|----------------|--------------|
| OpenJev (Qwen3.5-4B) | 0.620 | 78.3 | 0.606 | 79.6 | N/A (exceeds 16-option limit) | 0.998 / 87.2 |

Notes:
- OpenJev has a hard limit of 16 options. intent_massive has up to 60 options and cannot be evaluated.
- hinglish_lid accuracy is near-perfect for all language models as language identification is trivial. Not a useful differentiator.
- The meaningful comparison tasks are fintech_banking77 and synthetic_enterprise.

---

## Running the Benchmark

The benchmark harness and full code are coming soon. Watch this space.

---

## License

This dataset is released under CC BY 4.0. Individual task licenses are listed in the table above.

Built by [cmul8](https://cmul8.com). Inspired by [JevBench](https://github.com/fstandhartinger/jevbench) by Florian Standhartinger.
