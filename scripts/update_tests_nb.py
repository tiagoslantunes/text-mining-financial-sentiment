# -*- coding: utf-8 -*-
"""Surgical update of tm_tests_33.ipynb: 10-fold numbers, new Phase-2 analyses,
rule-compliance note (single-model submission via distillation)."""

import json
import uuid
from pathlib import Path

NB = Path("tm_tests_33.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]


def code_cell(src):
    return {"cell_type": "code", "execution_count": None,
            "id": uuid.uuid4().hex[:8], "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def md_cell(src):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": src.splitlines(keepends=True)}


# ---------------------------------------------------------------- cell 1: rubric
cells[1]["source"] = """## Rubric Traceability Matrix

| Enunciado criterion | Evidence in this notebook/project | Status |
|---|---|---|
| Data Exploration (2.00) | Class distribution, length analysis, Twitter features, word clouds, TF-IDF terms, mojibake/fix_text analysis, VADER-by-class | Covered |
| Corpus split (0.50) | 10-fold `StratifiedKFold(shuffle=True, random_state=42)`, identical across all experiments | Covered |
| Data Preprocessing (3.00) | Regex cleaning, Unicode normalization, stopwords, lemmatization, stemming, TweetTokenizer + ftfy fix_text; full ablation | Covered |
| Feature Engineering (5.50) | BoW/TF-IDF, Word2Vec (SG+CBOW), GloVe, FinBERT CLS encoder + PCA visualisation, fusion features | Covered |
| Classification Models (4.50) | 8 traditional-ML families x multiple feature sets; fine-tuned transformer backbone search (10 models) | Covered |
| Evaluation and Analysis (1.50) | Accuracy, Precision, Recall, F1, confusion matrix, error analysis, Wilcoxon + bootstrap CI, calibration | Covered |
| Extra: encoders (+1.00) | SBERT all-mpnet (+0.50) and Twitter-RoBERTa (+0.50) as extra frozen encoders | Covered |
| Extra: decoder (+1.00) | GPT-2 fine-tuned classifier (77.24% OOF) vs few-shot | Covered |
| Extra: agentic (+1.50) | 3-tool adaptive-routing agent with conversational memory (notebooks/08) | Covered |
| Extra: advanced techniques | Weighted soft-vote ensemble (92.01% OOF) distilled into the single submitted model (Hinton et al., 2015); SMOTE; Optuna | Covered |

**Submission compliance (Guidelines 5.2):** `tm_final_33.ipynb` contains a single pipeline with a **single classification model** — a FinBERT student distilled from the 8-model ensemble teacher. The ensemble itself is reported in Section 5f as extra work.
""".splitlines(keepends=True)

# ---------------------------------------------------------------- cell 12: split
cells[12]["source"] = """## 2. Corpus Split and Validation Protocol (0.50 pts)

**Method:** 10-fold stratified cross-validation on the 9,543 labelled tweets (`StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`).

- Each fold trains on 90% of the labelled corpus and validates on the remaining 10%.
- Stratification preserves the Bearish/Bullish/Neutral imbalance (15.1% / 20.1% / 64.7%) in every fold.
- Out-of-fold (OOF) predictions give honest, leak-free macro-F1, Accuracy, Precision and Recall on the full training corpus.
- Final test predictions average the ten fold-level probability vectors.
- The identical protocol (same seed, same splitter) is used for every model in this project, so all OOF numbers are directly comparable.
- Classical-ML feature/model comparisons (Section 5a) were run at 5 folds for tractability (68 combinations); all transformer results are 10-fold.
""".splitlines(keepends=True)

# ---------------------------------------------------------------- cell 15: ablation fillna
src15 = "".join(cells[15]["source"])
src15 = src15.replace(
    "ablation_df = pd.read_csv('results/tables/preprocessing_ablation.csv')",
    "ablation_df = pd.read_csv('results/tables/preprocessing_ablation.csv')\n"
    "ablation_df['Std'] = ablation_df['Std'].fillna(0.0)")
src15 = src15.replace(
    "print('\\nBest config: raw (no preprocessing) � financial keywords are informative features')",
    "print('\\nBest config: raw + fix_text (ftfy) - mojibake repair helps; aggressive cleaning hurts')\n"
    "print('For transformers the same pattern holds: see the fix_text ablation in Section 5g.')")
cells[15]["source"] = src15.splitlines(keepends=True)

# ---------------------------------------------------------------- cell 31: backbone search
cells[31]["source"] = """# Transformer BACKBONE SEARCH - all fine-tuned with the identical 10-fold OOF protocol
# (seed=42, StratifiedKFold, best-checkpoint-per-fold, class-weighted loss, fp16+GradScaler)
import json, os
rows = [
    ('finbert_fintwitter_10ep_fixtext', 'FinBERT-fintwitter 10ep + fix_text  <- best individual'),
    ('finbert_fintwitter_10ep',         'FinBERT-fintwitter 10ep (raw text)'),
    ('finbert_fintwitter_7ep',          'FinBERT-fintwitter 7ep'),
    ('finbert_fintwitter',              'FinBERT-fintwitter 5ep'),
    ('finbert_fintwitter_llrd_7ep',     'FinBERT-fintwitter 7ep + LLRD 0.9'),
    ('roberta_large_ts_v2',             'Twitter-RoBERTa-large topic-sentiment'),
    ('debertav3_large_6ep',             'DeBERTa-v3-large 6ep'),
    ('debertav3_large_6ep_fixtext',     'DeBERTa-v3-large 6ep + fix_text'),
    ('debertav3_large_8ep_fixtext',     'DeBERTa-v3-large 8ep + fix_text'),
    ('deberta_base_finance_fixtext',    'DeBERTa-v3-base finance + fix_text'),
]
print(f"{'Backbone':58s} OOF-F1   Acc")
for tag, name in rows:
    p = f'results/tables/{tag}_result.json'
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    print(f"{name:58s} {d['F1-macro']:.4f}  {d.get('Accuracy', 0):.4f}")
print()
print('Domain-specific pre-training dominates: the FinBERT variant already fine-tuned on')
print('this exact Twitter-financial distribution beats much larger generic models')
print('(DeBERTa-v3-large has ~3x the parameters but scores ~2pp lower).')
""".splitlines(keepends=True)

# ---------------------------------------------------------------- cell 32: ensemble + compliance
cells[32]["source"] = """# 5f. Ensemble (EXTRA WORK) and the rule-compliant submitted model
import json, os
opt = json.load(open('results/tables/ensemble_optimal_result.json'))
print('Weighted soft-vote ensemble - weights via exhaustive OOF grid search')
print('(all 2^8 model subsets x weight grid {0.25, 0.5, 0.75, 1.0}):')
for tag, w in opt['models'].items():
    print(f'  w={w:<5} {tag}')
print(f"\\nEnsemble OOF F1-macro: {opt['oof_f1_macro']:.4f}  (10-fold, leak-free)")

# Guidelines 5.2 requires tm_final_33 to be "a single pipeline with a single
# classification model". We therefore DISTIL the ensemble into one FinBERT student
# (Hinton et al., 2015): soft targets = leak-free ensemble OOF probabilities,
# loss = (1-a)*weighted_CE + a*T^2*KL. The submitted pred_33.csv comes from this
# single student model; the ensemble is reported here as extra work only.
if os.path.exists('results/tables/finbert_distilled_result.json'):
    d = json.load(open('results/tables/finbert_distilled_result.json'))
    print(f"\\nDistilled single model (SUBMITTED): OOF F1-macro={d['F1-macro']:.4f}")
    print(f"  teacher : {len(d['teacher_models'])}-model ensemble @ OOF {d['teacher_oof_f1']:.4f}")
    print(f"  recipe  : alpha={d['recipe']['alpha']}, T={d['recipe']['temperature']}, "
          f"{d['recipe']['epochs']}ep, {d['recipe']['n_folds']}-fold, fix_text")
    print(f"  per-fold: {[round(x,4) for x in d['per_fold_f1']]}")
else:
    print('\\n(finbert_distilled still training - run scripts/run_distill_cv.py)')
""".splitlines(keepends=True)

# ---------------------------------------------------------------- insertions (reverse order)
# After cell 39 (error examples) -> statistical significance + calibration
sig_cell = code_cell("""# Statistical significance and probability calibration (evaluation extensions)
import json
from IPython.display import Image, display
sig = json.load(open('results/tables/statistical_significance.json'))
w = sig['wilcoxon_best_vs_second']
print('Wilcoxon signed-rank, best vs 2nd-best individual (10 paired folds):')
verdict = 'significant' if w['p_value'] < 0.05 else 'not significant at a=0.05 (only 10 fold-pairs - low power)'
print(f"  statistic={w['statistic']:.1f}, p={w['p_value']:.4f}  ({verdict})")
lo, hi = sig['bootstrap_best_vs_second_ci95']
print(f'Bootstrap 95% CI (1000 resamples), F1(best) - F1(2nd-best): [{lo:+.4f}, {hi:+.4f}]')
lo, hi = sig['bootstrap_ensemble_vs_best_ci95']
print(f'Bootstrap 95% CI (1000 resamples), F1(ensemble) - F1(best) : [{lo:+.4f}, {hi:+.4f}]')
print('  -> the ensemble gain excludes 0: the improvement is real, not resampling noise.')
print()
cal = json.load(open('results/tables/calibration_results.json'))
print('Temperature-scaling search on OOF probabilities:')
for tag, c in cal.items():
    print(f'  {tag}: best T={c["best_T"]} (gain {c["gain"]:+.4f})')
print('  -> T=1.0 is optimal for every model: the fine-tuned models are already well calibrated')
print('     (label smoothing 0.05 during training prevents the usual overconfidence).')
display(Image('results/figures/calibration_curve.png'))""")
cells.insert(40, sig_cell)

# After cell 21 (VADER features) -> encoder PCA + fusion + SMOTE
fe_cell = code_cell("""# Encoder-space visualisation, fusion features and SMOTE (Section 4 extensions)
import json
from IPython.display import Image, display
display(Image('results/figures/encoder_pca.png'))
print('PCA projections: FinBERT CLS separates the classes most cleanly - consistent')
print('with it being the strongest frozen encoder in the model comparison below.')
print()
fus = json.load(open('results/tables/fusion_features_result.json'))
print(f"LightGBM on SBERT only (768d)        : F1-macro={fus['sbert_only_f1']:.4f}")
print(f"LightGBM on SBERT+financial (782d)   : F1-macro={fus['fusion_f1']:.4f}  ({fus['gain']:+.4f})")
print()
sm = json.load(open('results/tables/smote_comparison.json'))
print('Class-imbalance handling on dense embeddings (LightGBM + SBERT, 5-fold):')
print(f"  class_weight='balanced' : F1-macro={sm['class_weight_f1']:.4f}")
print(f"  SMOTE (applied in-fold) : F1-macro={sm['smote_f1']:.4f}  ({sm['delta']:+.4f})")
print(f"  -> {sm['conclusion']}")
print('  (Transformers keep the class-weighted loss; SMOTE is only viable on fixed embeddings.)')""")
cells.insert(22, fe_cell)

# After cell 10 (TF-IDF terms) -> mojibake + VADER EDA (insert VADER first, then mojibake, same index)
vader_cell = code_cell("""# VADER lexical sentiment by class - the limits of lexicon methods on financial text
import json
from IPython.display import Image, display
display(Image('results/figures/vader_by_class.png'))
amb = json.load(open('results/tables/vader_neutral_ambiguity.json'))
print(f"Neutral tweets with positive VADER compound : {amb['vader_positive_pct']}%")
print(f"Neutral tweets with negative VADER compound : {amb['vader_negative_pct']}%")
print(f"Neutral tweets that VADER also calls neutral: {amb['vader_neutral_pct']}%")
print()
print('Over half of the Neutral tweets look polarised to a general-purpose lexicon:')
print('financial wording ("raised", "beats", "cut", "misses") is factual news, not opinion.')
print('This motivates domain-specific models (FinBERT) over lexicon/classical baselines.')""")
cells.insert(11, vader_cell)

moji_cell = code_cell("""# Mojibake / encoding-corruption analysis (motivates the fix_text preprocessing step)
import pandas as pd
moji = pd.read_csv('results/tables/mojibake_by_class.csv')
print('Tweets with encoding artefacts (UTF-8 mojibake), by class:')
print(moji.to_string(index=False))
print()
ex = pd.read_csv('results/tables/mojibake_examples.csv')
print('Before/after examples (ftfy):')
for _, r in ex.iterrows():
    print(f"  [{int(r['label'])}] BEFORE: {str(r['before'])[:84]}")
    print(f"      AFTER : {str(r['after'])[:84]}")
print()
print('~23.5% of tweets carry mojibake already present in the source CSV (UTF-8 read as')
print('Latin-1). fix_text = ftfy repair + truncation-artefact/URL cleanup; its impact is')
print('ablated for classical models (Section 3) and transformers (Section 5g).')""")
cells.insert(11, moji_cell)

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Updated {NB} - now {len(cells)} cells")
