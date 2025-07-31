# LLM‑powered QA Pipeline for Vietnamese National High‑School Exams
*A complete workflow that converts scanned textbooks & past‑papers into validated multiple‑choice question–answer (QA) pairs.*

---

## 1 Project Overview
This repository contains an **end‑to‑end large‑language‑model (LLM) pipeline** for building structured QA datasets from Vietnamese educational content (THPT QG level).  
It automates:

1. **OCR** – recognise text from PDF/PNG pages.  
2. **Question & answer generation** – leverage open LLMs to suggest candidate MCQs.  
3. **Automatic verification** – filter out wrong or malformed pairs.  
4. **Dataset curation & lightweight fine‑tuning** – teach a base model to produce higher‑quality answers.

---

## 2 Repository Layout

```
.
├── Final/                   # Curated datasets, final notebooks & report
│   ├── *.json               # Final QA pairs
│   ├── final_finetune*.ipynb
│   ├── Report‑NLP.pdf
│   └── flow.txt             # High‑level pipeline diagram
│
├── OCR_LLM/                 # Stand‑alone OCR + QA generator demo
│   ├── ocr_material/        # Example raw text extracted by OCR
│   ├── qa_pairs_ocr/        # QA pairs generated from OCR output
│   ├── ocrLLM.py            # CLI: OCR → QA JSON
│   └── *.ipynb              # Colab experiments
│
├── qa_pairs_ocr/            # Additional generated QA datasets (grouped by subject)
│
├── ocr_material/            # Source PDFs used for local OCR tests
│
├── output/                  # Intermediate logs / plain‑text dumps
│
├── verification/            # Verified subsets saved by `verify.py`
│
├── testing/                 # Held‑out QA sets for evaluation
│
├── scripts & utilities
│   ├── Verification.py      # Batch‑level verifier
│   ├── verify.py            # Wrapper around verifier model
│   ├── verify‑test.py       # Quick smoke‑test script
│   ├── pdf2question.py      # Shortcut script: PDF → QA JSON
│   ├── combine_data.py      # Merge multiple QA JSONs
│   ├── extract_ocr.py       # Tesseract bulk OCR helper
│   └── ocr_tessa.py         # Minimal Tesseract OCR example
│
├── requirements.txt         # Python dependencies
└── README.md                # ← you are here
```


---

## 3 Installation

> **Prerequisites:** Python ≥ 3.10, Git, and a CUDA‑capable GPU if you want to run the LLM stages locally.

```bash
# 1) Create & activate virtual env
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 3) Install dependencies (PyTorch will auto‑detect GPU vs CPU)
pip install -r requirements.txt

# 4) (Optional) Install Tesseract for fastest local OCR
sudo apt-get update && sudo apt-get install tesseract-ocr
```

---

## 4 Quick Start

### 4.1 OCR → QA pairs in one shot

```bash
python OCR_LLM/ocrLLM.py \
       --pdf ocr_material/"Dia Ly 12.pdf" \
       --out qa_pairs_ocr/Lich_su_12/Lich_su_12_qa_pairs.json \
       --engine vintern            # or "tesseract"
```

### 4.2 Step‑by‑step pipeline

1. **Bulk OCR**

   ```bash
   python extract_ocr.py \
          --input-dir ocr_material \
          --output-dir output/ocr_json \
          --engine tesseract
   ```

2. **Question generation**

   ```bash
   python pdf2question.py \
          --ocr-json output/ocr_json/lichsu12_tessa.txt \
          --out qa_pairs_ocr/Lich_su_12/Lich_su_12_qa_pairs_1.json
   ```

3. **Verification**

   ```bash
   python verify.py \
          --qa-json qa_pairs_ocr/Lich_su_12/Lich_su_12_qa_pairs_1.json \
          --out verification/verified_qa_pairs_10.json
   ```

4. **(Optional) Fine‑tune base model**

   Open and run `Final/final_finetune.ipynb` – the notebook trains LoRA adapters for `LLaMA‑2‑7B‑chat`.

---

## 5 Configuration

Most scripts accept `--help` for all flags.  
YAML configs used by the Colab / Jupyter notebooks live in `Final/flow.txt` and inline notebook cells.

Key environment variables:

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | Hugging Face access token for gated LLM/OCR models |
| `CUDA_VISIBLE_DEVICES` | Select GPU(s) for generation / training |

---

## 6 Evaluation

Run:

```bash
python verify‑test.py
```

This script loads a random subset from `testing/qa_pairs_*.json` and reports:

* **Exact Match (EM)** and **F1**  
* **BLEU / ROUGE‑L** (for generative answers)  
* **Verifier pass‑rate** (LLM‑based judge)

Results are written to `output/eval_log.txt`.

---
---

## 🔧 AI Techniques Used

This project integrates multiple AI and NLP methods across the pipeline:

- **Optical Character Recognition (OCR)**  
  - Classical: `Tesseract-OCR` for lightweight, local character recognition.  
  - Advanced: `Vintern 1B v2` — a large language model for image-to-text OCR, combining:
    - `InternViT-300M` for visual recognition.
    - `Qwen2-0.5B-Instruct` for multilingual text interpretation.

- **Text Chunking & Preprocessing**  
  - Implemented both **semantic chunking** and **recursive chunking** to split scanned pages into meaning-preserving units.

- **Question Generation**  
  - Used `msmarco-vietnamese-mt5-base-v1`, a Vietnamese fine-tuned sequence-to-sequence model based on mT5, capable of generating contextually relevant questions from OCR'd text.

- **Answer Generation**  
  - Deployed `xlm-roberta-large-vi-qa` for extractive QA. The model handles both question–context alignment and answer span extraction from Vietnamese texts.

- **Answer Verification**  
  - Adopted `Meta-LLaMA-3-8B-Instruct` to filter invalid answers and ensure logical correctness via LLM-based feedback.

- **Fine-tuning & Training**  
  - Applied **quantization** via `bitsandbytes` to reduce memory use (4-bit & 8-bit support).
  - Used **LoRA (Low-Rank Adaptation)** from Hugging Face `PEFT` to perform **parameter-efficient fine-tuning** (PET) on top of `LLaMA-2-7B-chat`.
  - All training done with **HuggingFace Transformers**, `Accelerate`, and custom dataset pipelines.
