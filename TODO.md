# TODO.md — CulinaryAI end-to-end checklist (learning version)

This file is a **step-by-step checklist** for the entire CulinaryAI process:

- **Data ingestion** (CSV → `extracted_transcripts.json`)
- **Curation** (quality filtering/splitting → `curated_training.json`)
- **Training set creation** (turn examples into a format suitable for Llama fine-tuning)
- **Fine-tuning** (LoRA/QLoRA)
- **Evaluation + iteration**
- **Inference** (how you will use the model after training)

It also includes **everything we already did** in this project to wire up Hugging Face access and local testing.

---

## What we already did (completed steps in this repo)

### 0) Create a Python virtual environment for this repo
- [x] Create `.venv` in the project root (macOS default `python3`).
- [x] Install model/runtime dependencies into `.venv` (we installed `torch`, `transformers`, `accelerate`, `huggingface_hub`).

**Repro commands (from project root):**

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install "transformers>=4.44.0" "accelerate>=0.33.0" torch huggingface_hub
```

### 1) Store Hugging Face token in `.env` and load it in Python
- [x] Create `.env` in the repo root (contains `HF_TOKEN=...`).
- [x] Create `env.py` with `load_hf_token()` to load the token from `.env` / environment.
  - Note: `env.py` now has a fallback loader so it still works even if `python-dotenv` is missing.

### 2) Prevent secrets from being committed
- [x] Update `.gitignore` to ignore:
  - `.env`
  - `hftoken.txt`

### 3) Add test scripts for Llama access + inference
- [x] `quick_llama_check.py`
  - Fast: proves the token works + you have access to `meta-llama/Meta-Llama-3.1-8B-Instruct`
  - Downloads `config.json` as a lightweight access test
  - Attempts hosted inference (may fail depending on provider/model settings)
- [x] `test_llama.py`
  - Slow: loads the full model locally and runs `.generate()` (MPS on Apple Silicon)
  - We adjusted it to avoid disk offload and keep generation short

---

## Repo goal (what you’re building toward)

### End goal (high-level)
Train or fine-tune a Llama-family model so it can transform:

- **Input**: historical recipe transcript text (messy OCR, archaic spelling, inconsistent formatting)
- **Output**: consistent structured recipe data (likely **Schema.org `Recipe` JSON / JSON-LD**)

You already have the upstream pieces in place:

- `parse_culinary_csv.py` → extracts likely recipe pages
- `curate_training_data.py` / `create_training_sample.py` → selects higher-quality examples
- `validate_recipe_schema.py` → validates example `.jsonld` recipes against a Recipe schema

The main missing bridge is:

- **A generator that creates training examples** (instruction/input/output) from curated transcripts, plus human/automatic validation.

---

## Phase A — Data ingestion (CSV → extracted transcript dataset)

### A1) Put the raw Culinary Manuscripts CSV export in the repo
- [ ] Confirm the input CSV exists (example in repo): `culinary_collection7_20251218_141432.csv`
- [ ] If you use a newer export, keep it in the repo root (or document the path).

### A2) Run the parser to generate `extracted_transcripts.json`

- [ ] (Optional) Run analysis + print a few examples:

```bash
.venv/bin/python parse_culinary_csv.py --analyze --sample 5
```

- [ ] Generate transcripts:

```bash
.venv/bin/python parse_culinary_csv.py \
  --input culinary_collection7_20251218_141432.csv \
  --output extracted_transcripts.json
```

- [ ] Confirm output exists and looks sane:
  - [ ] `extracted_transcripts.json` exists
  - [ ] Each record has: `item_id`, `manuscript_title`, `page`, `transcript`, `detected_titles`, `char_count`, `confidence`

### A3) Sanity-check extraction quality (manual sampling)
- [ ] Open `extracted_transcripts.json` and spot-check 20–50 records:
  - [ ] “Cover/Index/Blank” pages are excluded
  - [ ] Real recipe procedure text is included
  - [ ] `detected_titles` look plausible when present

---

## Phase B — Curation (quality filtering & selection)

You have two scripts that operate on `extracted_transcripts.json`:

- `create_training_sample.py`: simpler sampler for quick experiments
- `curate_training_data.py`: stronger curation (splitting, completeness checks, dedupe, post-validation)

### B1) Run robust curation to create `curated_training.json`
- [ ] Create a curated set (start with 200):

```bash
.venv/bin/python curate_training_data.py --count 200 --output curated_training.json --analyze
```

- [ ] Inspect a handful of records in `curated_training.json`:
  - [ ] Endings are not obviously cut off
  - [ ] Damage markers (`[illegible]`, `[...]?`, etc.) are rare
  - [ ] Multi-recipe splits look correct (if split)

### B2) Create/refresh the smaller dev sample (`training_sample.json`)
- [ ] Generate:

```bash
.venv/bin/python create_training_sample.py --count 100 --output training_sample.json
```

- [ ] Use `training_sample.json` as your “fast iteration” dataset.

---

## Phase C — Define the task precisely (what should the model output?)

### C1) Choose output format (recommended: Schema.org Recipe JSON/JSON-LD)
- [ ] Decide whether your training target is:
  - [ ] JSON-LD (with `@context` and `@type: "Recipe"`)
  - [ ] or plain JSON that matches the same fields
- [ ] Decide **required** fields:
  - [ ] `name`
  - [ ] `recipeIngredient` (list of strings)
  - [ ] `recipeInstructions` (list of step objects or strings — pick one and stick to it)
- [ ] Decide **optional** fields you might add later:
  - [ ] `description`, `recipeYield`, `totalTime`, `author`, etc.

### C2) Decide what the model sees as input
- [ ] Choose input payload:
  - [ ] `transcript` only
  - [ ] transcript + `detected_titles`
  - [ ] transcript + manuscript metadata (`manuscript_title`, `page`)

### C3) Write the canonical instruction (prompt)
- [ ] Create a single instruction you’ll reuse across all training examples, e.g.:
  - “Convert the transcript into Schema.org Recipe JSON. Output JSON only.”
- [ ] Decide strictness:
  - [ ] “JSON only” (no prose)
  - [ ] No trailing commas
  - [ ] Must pass schema validation

---

## Phase D — Create supervised training examples (the missing bridge)

Right now, the repo doesn’t yet contain a script that **creates labeled examples** (input transcript → target recipe JSON).

### D1) Choose a training file format
- [ ] Pick one:
  - [ ] **JSONL** (recommended) — one example per line
  - [ ] Alpaca-style JSON: `{ "instruction", "input", "output" }`
  - [ ] Chat format: `{ "messages": [...] }`

### D2) Create a generator script (to be added)
- [ ] Create: `generate_training_examples.py`
  - [ ] Input: `curated_training.json`
  - [ ] Output: `training.jsonl` (or `training.json`)
  - [ ] For each record, produces one training example with your canonical instruction + transcript input

### D3) Decide how you will get the target outputs (“labels”)
- [ ] **Human labels (best quality):**
  - [ ] You manually create correct Schema.org `Recipe` JSON for a subset
- [ ] **Bootstrapped labels (faster):**
  - [ ] Use a strong model (e.g., Llama 3.1 Instruct) to draft outputs
  - [ ] Validate outputs automatically
  - [ ] Manually review and fix failures

### D4) Add an automatic validation script (to be added)
- [ ] Create: `validate_training_outputs.py`
  - [ ] Reads generated examples
  - [ ] Validates JSON parse
  - [ ] Validates schema (reuse/extend `validate_recipe_schema.py`)
  - [ ] Produces a report of failures and why

### D5) Add “golden examples” for learning + regression tests
- [ ] Create an `examples/` folder and add a few pairs:
  - [ ] `examples/receipt_001.input.txt`
  - [ ] `examples/receipt_001.output.jsonld`
- [ ] Use these to lock down what “good output” means before training.

---

## Phase E — Fine-tuning (Llama 3.1)

### E1) Choose fine-tuning approach + hardware
- [ ] Decide training method:
  - [ ] LoRA / QLoRA (recommended)
  - [ ] Full fine-tune (rarely needed)
- [ ] Decide where training runs:
  - [ ] Local CUDA GPU
  - [ ] Remote server / cloud GPU
  - [ ] (macOS MPS is usually not ideal for training)

### E2) Add fine-tuning dependencies (to be added to `requirements.txt`)
- [ ] Add packages you’ll need for training:
  - [ ] `datasets`
  - [ ] `peft`
  - [ ] `trl` (optional, common)
  - [ ] `evaluate` (optional)

### E3) Create a fine-tuning script (to be added)
- [ ] Create: `finetune_lora.py`
  - [ ] Loads base model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
  - [ ] Loads `training.jsonl`
  - [ ] Trains adapters
  - [ ] Saves adapters/checkpoints to `./checkpoints/`

### E4) Track experiments (so you can learn systematically)
- [ ] Create `experiments/` (folder) for:
  - [ ] training config
  - [ ] dataset version hash/notes
  - [ ] prompt version
  - [ ] evaluation reports

---

## Phase F — Evaluation (do not skip)

### F1) Create a held-out evaluation set
- [ ] Split your curated data into train/eval (and possibly test).

### F2) Define measurable metrics
- [ ] Automated:
  - [ ] JSON parse success rate
  - [ ] Schema validation pass rate
  - [ ] Required-field completeness
- [ ] Manual:
  - [ ] Ingredients are sensible
  - [ ] Steps are coherent
  - [ ] Output matches transcript intent

### F3) Create an evaluation script (to be added)
- [ ] Create: `eval_model.py`
  - [ ] Runs inference on eval transcripts
  - [ ] Validates outputs
  - [ ] Produces a summary report (JSON/CSV)

---

## Phase G — Inference (use the trained model)

### G1) Create an inference wrapper (to be added)
- [ ] Create: `infer_recipe.py`
  - [ ] Input: transcript text
  - [ ] Output: recipe JSON/JSON-LD
  - [ ] Deterministic decoding settings for structured output

### G2) Optional: CLI / API
- [ ] CLI:
  - [ ] `python infer_recipe.py --input-file mrs-roberts-recipe.txt --output out.jsonld`
- [ ] API (optional):
  - [ ] `api.py` using FastAPI for inference serving

---

## Phase H — Project hygiene & reproducibility

### H1) Update `requirements.txt` (important)
Right now `requirements.txt` only contains `jsonschema>=4.0.0`.

- [ ] Add at minimum what this repo uses today:
  - [ ] `python-dotenv`
  - [ ] `transformers`
  - [ ] `accelerate`
  - [ ] `huggingface_hub`
  - [ ] `torch`

### H2) Expand `.gitignore` (recommended)
- [ ] Add common ignores:
  - [ ] `.venv/`
  - [ ] `__pycache__/`
  - [ ] `.pytest_cache/`

### H3) Update `README.md` with the full pipeline
- [ ] Add explicit “end-to-end” commands:
  - [ ] Parse CSV → `extracted_transcripts.json`
  - [ ] Curate → `curated_training.json`
  - [ ] Generate training JSONL → `training.jsonl`
  - [ ] Fine-tune
  - [ ] Evaluate
  - [ ] Inference

---

## Quick commands (copy/paste)

### Quick “token + access” test (fast)

```bash
.venv/bin/python quick_llama_check.py
```

### Local generation test (slow; proves inference end-to-end)

```bash
.venv/bin/python test_llama.py
```

