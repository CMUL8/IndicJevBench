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

| Model | Type | fintech_banking77 | synthetic_enterprise | intent_massive | hinglish_lid |
|-------|------|-------------------|----------------------|----------------|--------------|
| OpenJev (Qwen3.5-4B) | local | acc=0.620 / IJScore=78.3 | acc=0.606 / IJScore=79.6 | N/A (>16-option limit) | acc=0.998 / IJScore=87.2 |
| Laya-multilingual | local | acc=0.455 / IJScore=69.3 | acc=0.351 / IJScore=58.7 | acc=0.350 / IJScore=61.7 | acc=0.493 / IJScore=73.7 |
| GPT-4o-mini (OpenRouter) | API | acc=0.775 / IJScore=56.1 | acc=0.670 / IJScore=46.6 | acc=0.580 / IJScore=51.3 | acc=0.175 / IJScore=21.2‡ |
| Laya (base) | local | acc=0.256 / IJScore=56.2 | acc=0.369 / IJScore=64.5 | acc=0.199 / IJScore=50.2 | acc=0.569 / IJScore=74.0 |
| Qwen3-4B (zero-shot logprob) | local | acc=0.373 / IJScore=55.3 | acc=0.525 / IJScore=68.5 | acc=0.316 / IJScore=49.6 | acc=0.840 / IJScore=2.3† |

† Qwen3-4B logprob scoring runs one forward pass per option. hinglish_lid has many language classes, pushing latency to ~16s/item and collapsing the speed axis.
‡ GPT-4o-mini hinglish_lid had 46/200 JSON parse errors (pre-fix run); accuracy and IJScore are underestimates.

Notes:
- OpenJev has a hard limit of 16 options; intent_massive (up to 60 options) cannot be evaluated with OpenJev.
- The primary comparison tasks are fintech_banking77, intent_massive, and synthetic_enterprise.

---

## Running the Benchmark

The benchmark harness and full code are coming soon.

---

## License

This dataset is released under CC BY 4.0. Individual task licenses are listed in the table above.

Built by [cmul8](https://cmul8.com). Inspired by [JevBench](https://github.com/fstandhartinger/jevbench) by Florian Standhartinger.
