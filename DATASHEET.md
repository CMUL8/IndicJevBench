# IndicJevBench — Dataset Datasheet

Structured per Gebru et al. (2018) "Datasheets for Datasets".

---

## Motivation

**For what purpose was this dataset created?**
To provide the first open benchmark for Indic-language structured decision AI — specifically, calibrated probability outputs for typed questions (intent, urgency, escalation, routing) on enterprise customer messages in Indian languages.

**Who created it and on whose behalf?**
cmul8.com (basab@lonere-labs.com), building the Nirṇaya decision model.

**Was there any funding?**
Internal project. No external funding.

---

## Composition

### Task: `intent_massive`

| Field | Value |
|---|---|
| Source | MASSIVE (Multilingual Amazon SLURP for Slot Filling, Intent Classification and Virtual Assistant Evaluation) |
| License | CC BY 4.0 |
| Languages | hi-Deva, bn-Beng, ta-Taml, te-Telu, kn-Knda, ml-Mlym |
| Question type | `choice` (60 intent classes) |
| Construction | Official MASSIVE test split, converted to IndicJevBench format. No relabeling. |
| Known biases | MASSIVE was crowd-sourced via Amazon; labels reflect annotator agreement, not ground truth. Hinglish absent — MASSIVE is native-script only. |
| Missing languages | mr-Deva, gu-Gujr absent from the MASSIVE mirror used. |

### Task: `fintech_banking77`

| Field | Value |
|---|---|
| Source | Banking77 (Casanueva et al., 2020) |
| License | CC BY 4.0 |
| Languages | Translated to hi-Deva, bn-Beng, ta-Taml, te-Telu, kn-Knda, ml-Mlym via NLLB-200-distilled-1.3B |
| Question type | `choice` (77 banking intent classes) |
| Construction | English Banking77 test set machine-translated to 6 Indic languages. ChrF quality filter: reject if score < 10 and length < 5 chars. Labels unchanged by translation. |
| Known biases | Machine translation quality varies by language; Bengali generally better than Tamil/Telugu for short domain text. Domain is UK retail banking — India-specific intents (UPI, NEFT, FASTag) are absent. |
| Translation model | facebook/nllb-200-distilled-1.3B (not gated; chosen over IndicTrans2 due to transformers≥4.40 incompatibility). Within 2–3 BLEU of IndicTrans2 on short enterprise text. |

### Task: `hinglish_lid`

| Field | Value |
|---|---|
| Source | COMI-LINGUA (Kula et al.) |
| License | CC BY 4.0 |
| Languages | hi-Latn (Hinglish) |
| Question type | `noul` (is this code-mixed?) or `choice` (language ID) |
| Construction | COMI-LINGUA test split converted to IndicJevBench format. |
| Known biases | COMI-LINGUA covers Twitter-style Hinglish; may not represent customer-support register. |

### Task: `synthetic_enterprise`

| Field | Value |
|---|---|
| Source | cmul8.com synthetic generation |
| License | CC BY 4.0 |
| Languages | hi-Latn, hi-Deva |
| Question types | `choice` (intent/routing), `score` (urgency 1–5), `noul` (escalate?) |
| Construction | Two-model pipeline: Qwen3-235B-A22B generator + DeepSeek-Chat-V3 checker. Generator produces customer message + label; checker independently validates. Only checker-agreed examples kept. 100 hand-reviewed before inclusion. |
| Domains | UPI/digital payments, Indian e-commerce, customer-support routing |
| Known biases | Both models are Chinese LLMs trained on internet text. Indian Hinglish may reflect more formal register than real customer messages. UPI amounts, merchant names, and error codes are sampled from a fixed set of seeds. |
| Synthetic share | ~2,500 test examples across 5 slots |

---

## Collection Process

**How was the data collected?**
- MASSIVE: Official test split downloaded from the MASSIVE dataset card on HuggingFace.
- Banking77: English test set translated offline using NLLB on an H200 GPU.
- COMI-LINGUA: Official test split downloaded from HuggingFace.
- Synthetic: Generated via OpenRouter API using Qwen3-235B-A22B and DeepSeek-Chat-V3-0324.

**Who collected the data?**
Automated pipelines written by cmul8.com engineers. No crowd workers for the benchmark split itself.

**Over what timeframe?**
Phase 1–3 of the Nirṇaya project, September 2026.

---

## Preprocessing / Cleaning

- MASSIVE: no preprocessing beyond schema conversion.
- Banking77: ChrF quality filter (sacrebleu); reject if ChrF < 10 and length < 5 characters. 130 rows rejected from 311,454 total.
- COMI-LINGUA: schema conversion only.
- Synthetic: checker agreement required; 100 examples hand-reviewed.

---

## Uses

**Intended uses:**
- Benchmark evaluation of Indic decision AI models.
- Calibration research for structured prediction in Indian languages.
- Baseline comparison for new models.

**Out-of-scope uses:**
- Training data (test splits only; no train rows).
- Any use requiring text generation — this benchmark tests probability outputs only.

---

## Distribution

**How will the dataset be distributed?**
As `cmul8/IndicJevBench` on HuggingFace Datasets (release date: before Nirṇaya model).

**What license applies?**
The benchmark harness (this repo) is MIT. Dataset licenses per task:
- MASSIVE items: CC BY 4.0
- Banking77 items: CC BY 4.0
- COMI-LINGUA items: CC BY 4.0
- Synthetic items: CC BY 4.0

**Items NOT redistributed directly (download-script only):**
- Bitext customer support: CDLA-Sharing-1.0 (share-alike clause)
- IndicXNLI: CC BY-NC 4.0 (non-commercial)

---

## Maintenance

**Who maintains the dataset?**
cmul8.com. Contact: basab@lonere-labs.com.

**Will the dataset be updated?**
v2 planned post-release: replace all Bitext-derived tasks with synthetic CC BY 4.0 equivalents for a fully redistributable benchmark.

**Are there known errors or limitations?**
- mr-Deva and gu-Gujr absent (not in MASSIVE mirror used).
- Banking77 domain is UK retail banking, not India-specific.
- Synthetic Hinglish may not match real customer-support register.
- NLLB translation quality is approximate; not human-verified.
