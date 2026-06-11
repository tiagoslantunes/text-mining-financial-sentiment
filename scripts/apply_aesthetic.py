# -*- coding: utf-8 -*-
"""Apply the group's notebook aesthetic (styled header card + green accents)
to the three delivered notebooks, and refresh the stale tm_tests summary."""

import json
import uuid
from pathlib import Path

GREEN = "#00D563"
MEMBERS = ("Alexandra Varela, 20250514<br>Francisca Fernandes, 20250406<br>"
           "Mariana Melo, 20250414<br>Tiago Antunes, 20250357")


def header(ghost, title_plain, title_accent, subtitle):
    return f"""<div style="font-family: 'Helvetica Neue', Arial, sans-serif; background:#fff; padding: 36px 40px; border-radius: 8px; margin-bottom: 4px; position:relative; overflow:hidden; border: 1.5px solid #e8e8e8;">

  <div style="font-size:130px; font-weight:900; color:rgba(0,0,0,0.04); position:absolute; top:-20px; right:30px; line-height:1; letter-spacing:-0.05em;">{ghost}</div>

  <div style="font-size:10px; color:{GREEN}; letter-spacing:0.25em; text-transform:uppercase; margin-bottom:16px;">NOVA IMS &middot; 2025/2026</div>
  <div style="font-size:36px; font-weight:800; color:#111; letter-spacing:-0.02em; line-height:1.1; margin-bottom:6px;">{title_plain} <span style="color:{GREEN};">{title_accent}</span></div>
  <div style="font-size:12px; color:{GREEN}; font-weight:500; margin-bottom:24px;">{subtitle}</div>

  <div style="display:flex; gap:48px;">
    <div>
      <div style="font-size:9px; color:{GREEN}; letter-spacing:0.2em; text-transform:uppercase; margin-bottom:6px;">Group 33</div>
      <div style="font-size:11px; color:#555; line-height:1.9;">{MEMBERS}</div>
    </div>
    <div>
      <div style="font-size:9px; color:{GREEN}; letter-spacing:0.2em; text-transform:uppercase; margin-bottom:6px;">Course</div>
      <div style="font-size:11px; color:#555; line-height:1.9;">Text Mining<br>MSc Data Science &amp; Advanced Analytics<br>NOVA Information Management School</div>
    </div>
  </div>
</div>"""


def md_cell(src):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": src.splitlines(keepends=True)}


# ════════════════════════════ tm_tests_33 ════════════════════════════
NB = Path("tm_tests_33.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

hdr = header("01", "Mining Sentiment", "from Financial Tweets.",
             "Notebook 1 &mdash; Experiments &amp; Evaluation")

summary = f"""**Task:** Multiclass classification of financial tweets &mdash; Bearish (0), Bullish (1), Neutral (2)
**Dataset:** 9,543 train | 2,388 test &middot; class imbalance 4.28:1
**Submitted model (single, Guidelines 5.2):** FinBERT student **distilled** from an 8-encoder ensemble teacher &mdash; OOF F1-macro **0.9139** (reproduced in `tm_final_33.ipynb`)
**Best ensemble (extra work):** weighted soft-vote of 8 fine-tuned encoders &mdash; OOF F1-macro **0.9201**
**Protocol:** 10-fold `StratifiedKFold(shuffle=True, random_state=42)` for all transformers; identical splits make every OOF score comparable

This notebook documents **all experiments and their evaluation**: EDA, preprocessing ablations over five dataset variants, eleven feature representations, twelve model families, a nine-config transformer backbone search, ensembling, distillation, and the statistical tests behind each decision (see the Decision Log before Section 9)."""

toc = f"""## <span style="color:{GREEN};"><b>Table of Contents</b></span>

- [Rubric Traceability Matrix](#Rubric-Traceability-Matrix)
- [1. Data Exploration](#1.-Data-Exploration-(2.00-pts))
- [2. Corpus Split and Validation Protocol](#2.-Corpus-Split-and-Validation-Protocol-(0.50-pts))
- [3. Data Preprocessing](#3.-Data-Preprocessing-(3.00-pts))
- [4. Feature Engineering](#4.-Feature-Engineering-(5.50-pts-base-+-1.00-extra))
- [5. Classification Models](#5.-Classification-Models-(4.50-pts-+-2.00-extra))
- [5b. Transformer Fine-tuning + Stacking Ensemble](#5b.-Transformer-Fine-tuning-+-Stacking-Ensemble-(the-decisive-levers))
- [6. Hyperparameter Tuning (Optuna)](#6.-Hyperparameter-Tuning-(Optuna-Bayesian-Optimisation))
- [7. Evaluation and Results](#7.-Evaluation-and-Results-(1.50-pts))
- [8. Extra Challenge 2: Agentic Workflow](#8.-Extra-Challenge-2:-Agentic-Workflow-(+1.50-pts))
- [Decision Log](#Decision-Log-—-how-each-experiment-shaped-the-final-solution)
- [9. Final Predictions](#9.-Final-Predictions)"""

nb["cells"][0] = md_cell(hdr)
nb["cells"].insert(1, md_cell(summary))
nb["cells"].insert(2, md_cell(toc))
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"tm_tests_33: header + summary novo + TOC ({len(nb['cells'])} cells)")

# ════════════════════════════ tm_final_33 ════════════════════════════
NB = Path("tm_final_33.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

hdr = header("02", "One Pipeline,", "One Model.",
             "Notebook 2 &mdash; Final Solution &middot; Restart &amp; Run All")

# keep the substantive content of the old header minus its three title lines
old = "".join(nb["cells"][0]["source"])
lines = old.splitlines()
body_start = next(i for i, l in enumerate(lines) if l.strip().startswith("**Submitted model"))
body = "\n".join(lines[body_start:])

nb["cells"][0] = md_cell(hdr)
nb["cells"].insert(1, md_cell(body))
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"tm_final_33: header + corpo preservado ({len(nb['cells'])} cells)")

# ════════════════════════ 08_Agentic_Workflow ════════════════════════
NB = Path("notebooks/08_Agentic_Workflow.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

hdr = header("EX", "An Agent that", "Knows Its Limits.",
             "Extra Challenge 2 &mdash; Agentic Workflow (+1.50 pts)")

old = "".join(nb["cells"][0]["source"])
lines = old.splitlines()
# drop the first two heading lines, keep the rest
body = "\n".join(l for l in lines if not l.startswith("# ") and not l.startswith("## Extra Challenge"))

nb["cells"][0] = md_cell(hdr)
nb["cells"].insert(1, md_cell(body.strip()))
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"08_Agentic_Workflow: header + corpo preservado ({len(nb['cells'])} cells)")
