# Ilera-ATLAS

**N-ATLaS fine-tuned for primary health care in Nigeria: Powered by Awarri.**
NAIC 2026 (National AI Innovation Challenge) · Academia & Research track · Problem Statement 3: Sectoral Fine-Tuning.

Ilera-ATLAS gives trained community health workers (CHEWs, JCHEWs, CHOs) decision support that follows
Nigeria's **2024 National Standing Orders** (CHPRBN, with FMoH and NPHCDA), in **English, Hausa and
Yoruba**. Every answer starts with a clear action and cites its source:

```
ACTION: TREAT | REFER | URGENT
WHAT TO DO: numbered steps
DANGER SIGNS: signs that mean urgent referral
SOURCE: document and section
```

The model is a QLoRA adapter on [NCAIR1/N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS) (Llama-3 8B).
No other foundation model is used anywhere in the pipeline: N-ATLaS also drafts and translates the
training data, which humans then review.

## Pipeline

| Step | Notebook | Runs on | Output (Hugging Face dataset `ilera-data`) |
|---|---|---|---|
| 1. Extract the Standing Orders | [`01_extract`](notebooks/01_extract.ipynb) | Kaggle CPU | `raw/chunks.jsonl`: 524 section-aware chunks with citations |
| 1b. Baseline: can raw N-ATLaS find the right reference? | [`01b_base_reference_eval`](notebooks/01b_base_reference_eval.ipynb) | Kaggle T4 x2 (raw float16 needs ~17 GB GPU) | `results/base_reference_summary_raw_fp16.csv`: closed-book, with-index and multiple-choice section accuracy of the unmodified model |
| 2. Draft Q&A and case vignettes | [`02_draft`](notebooks/02_draft.ipynb) | Kaggle T4 | `drafts/review_en_*.csv` for human review |
| 3. Human review | Google Sheets | team | `reviewed/review_en_final.csv` |
| 4. Translate to Hausa / Yoruba | [`03_translate`](notebooks/03_translate.ipynb) | Kaggle T4 | `translations/review_ha_yo_*.csv` with quality flags |
| 5. Freeze benchmark, build training files | [`03b_build`](notebooks/03b_build.ipynb) | Kaggle CPU | `bench/ilera_bench.jsonl` (+SHA-256), `train.jsonl`, `val.jsonl` |
| 6. Evaluate (baseline, run selection, final) | [`04_eval`](notebooks/04_eval.ipynb) | Kaggle T4 | `results/scores_*.csv`, paired-bootstrap comparison, blind-review sheet |
| 7. Train (QLoRA, Unsloth) | [`05_train`](notebooks/05_train.ipynb) | Kaggle T4 | `ilera-atlas-lora-run{A,B,C}` |
| 8. Export | [`06_export`](notebooks/06_export.ipynb) | Kaggle T4 | `ilera-atlas` (merged 16-bit), `ilera-atlas-GGUF` (Q4_K_M) |
| 9. Serve | [`serve/`](serve) | Modal + HF Space | OpenAI-compatible GPU API + demo with safety layer |

Every notebook has a smoke-test mode, resumes after a crash, and was tested offline before use
(GPU-only steps excepted). The full step-by-step guide is
[`docs/Ilera-ATLAS-Training-Runbook.docx`](docs/Ilera-ATLAS-Training-Runbook.docx) (its paths
`ilera-atlas/notebooks/...` are `notebooks/...` in this repo); the schedule and design are in
[`docs/Ilera-ATLAS-Execution-Plan.docx`](docs/Ilera-ATLAS-Execution-Plan.docx).

## Evaluation design

* **Ilera-Bench**: 600+ human-checked items (300 English, 150 Hausa, 150 Yoruba, parallel across
  languages) plus a ~60-item *unseen-section probe*, frozen with a SHA-256 hash **before** training.
* **External test**: 300 multiple-choice items from [AfriMed-QA](https://github.com/intron-innovation/AfriMed-QA)
  (test split, primary-care specialties, Nigerian items first).
* **Metrics**: correct action, **under-referral** (the dangerous error), over-referral, key-fact recall,
  correct source section, answer language (GlotLID), AfriMed-QA accuracy; paired bootstrap 95% CIs
  against base N-ATLaS; blind clinical review of 150 answer pairs.
* **Leak control**: no benchmark item, translation of one, probe-section text or near-duplicate
  question (RapidFuzz ≥ 85) enters training.

## Baseline: raw N-ATLaS before fine-tuning

`NCAIR1/N-ATLaS` loaded **as released** (float16, no quantisation, no fine-tuning) on a Kaggle
T4 x2, asked to find the Standing Orders reference for every part of the CHEW document
([`01b_base_reference_eval`](notebooks/01b_base_reference_eval.ipynb), 6 Oct 2026; data in
[`results/`](results)):

| Test | Accuracy [95% CI] | Chance |
|---|---|---|
| Closed book: section number for a topic (105 items) | **1.0%** [0.0, 2.9] | 0.7% |
| Excerpt + table of contents → section number (165 items) | **9.1%** [4.8, 13.3] (right title 17.0%) | 0.7% |
| Excerpt → right section among 4 options (165 items) | **75.2%** [68.5, 81.8] | 25% |

The base model understands Standing Orders content well, but cannot produce the correct reference
on its own, even with the table of contents in front of it. Ilera-ATLAS therefore (1) trains on
answers that always end with the exact `SOURCE:` section and (2) is evaluated with the cited section
as a metric (`source_ok`), with retrieval of the matching chunk as the planned stretch.

## Serving (free tiers)

* **GPU API**: [`serve/modal_app.py`](serve/modal_app.py), vLLM on one L4, scales to zero (Modal free credits).
* **Demo + safety layer**: [`serve/space/`](serve/space), Gradio on a free CPU Space; uses the GPU API
  when available, otherwise the GGUF on CPU. A red-flag list forces `ACTION: URGENT` for danger signs.
  Deploy with [`serve/deploy_space.py`](serve/deploy_space.py).
* **API docs**: [`docs/API.md`](docs/API.md).

## Data sources

* National Standing Orders 2024 for CHEW, JCHEW and CHO: CHPRBN with FMoH and NPHCDA, via the
  [PHC Workforce Hub](https://phcworkforcehub.org.ng/in-service/national-standing-order-revised-2024/).
* [Aya Dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) (Apache-2.0), ~10% general-language rows.
* [AfriMed-QA](https://github.com/intron-innovation/AfriMed-QA) (CC-BY-SA-4.0), evaluation only, by ID.

## Licence and responsible use

Model weights derived from N-ATLaS are released under the **N-ATLaS Terms of Use** (same licence for
derivatives; 1,000 active end-users per 30 days without a commercial licence). Attribution:
*Awarri Technologies and the Federal Ministry of Communications, Innovation and Digital Economy*.

Ilera-ATLAS is **decision support for trained health workers**. It is not a diagnosis tool and not for
patients. Medicines and doses must always be checked against the current Standing Orders.
