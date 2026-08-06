# Reproducibility Guide

## Environment

The final experiments used Python 3.11.9, PyTorch with CUDA, and seed 42. Install the full environment from the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch/CUDA compatibility is hardware-specific. If the default `torch` wheel is unsuitable, install the correct build from the official PyTorch selector before installing the remaining requirements.

## Level 1: inspect the completed study

No computation is required. GitHub renders the executed notebooks, and the main tables and figures are committed under `results/`.

- `tm_tests_33.ipynb`: full study and decision log
- `tm_final_33.ipynb`: submitted single-model pipeline
- `notebooks/08_Agentic_Workflow.ipynb`: adaptive-routing demonstration
- `docs/report.pdf`: final report

## Level 2: validate the repository

Install only the lightweight dependencies and run the automated checks:

```bash
python -m pip install -r requirements-test.txt
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
```

The checks validate notebook-independent preprocessing, label/prediction synchronization, and the LightGBM agent interface without downloading large models.

## Level 3: recreate the submitted prediction file

Start Jupyter from the repository root:

```bash
python -m jupyter lab
```

Open `tm_final_33.ipynb`, restart the kernel, and run all cells. With the committed probability caches, this takes roughly two minutes and writes `pred_33.csv`.

If the expected caches are removed, the notebook trains the distilled student. The recorded 10-fold, 12-epoch run took approximately 47 minutes on an NVIDIA RTX 5070 with 12 GB VRAM.

## Level 4: rerun model training

The canonical command-line entry points are documented in [`../scripts/README.md`](../scripts/README.md). The final distilled configuration is:

```bash
python scripts/run_distill_cv.py \
  --model-name nickmuchi/finbert-tone-finetuned-fintwitter-classification \
  --tag finbert_distilled_12ep \
  --epochs 12 \
  --maxlen 128 \
  --lr 5e-6 \
  --batch-size 16 \
  --n-folds 10 \
  --alpha 0.5 \
  --temperature 2.0
```

This command expects teacher weights in `results/tables/ensemble_optimal_result.json` and the corresponding out-of-fold/test probability files in `results/predictions/`.

To fine-tune an individual encoder:

```bash
python scripts/run_transformer_cv_v2.py \
  --model-name nickmuchi/finbert-tone-finetuned-fintwitter-classification \
  --tag finbert_fintwitter_10ep_fixtext \
  --epochs 10 \
  --n-folds 10 \
  --maxlen 128 \
  --lr 5e-6 \
  --fix-text
```

Full-suite runtime is much longer than the distilled run because multiple large encoders are trained across repeated folds. Results depend on GPU model, mixed-precision support, upstream checkpoints, and package versions.

## Output conventions

- `results/tables/`: JSON/CSV metrics and experiment summaries
- `results/predictions/`: out-of-fold probabilities, test probabilities, and class predictions
- `results/figures/`: generated plots
- `data/processed/`: regenerable dense features, ignored by Git
- `results/models/`: the lightweight agent model is committed; other regenerable model weights are ignored by Git

Do not compare metrics across different fold definitions as if they were paired. The report explicitly identifies the lighter 5-fold experiments and the primary 10-fold transformer protocol.
