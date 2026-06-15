"""
Merge new_nb.ipynb (more complete, 215 cells) into tm_tests_33.ipynb.

Strategy:
  - Use new_nb as the base (more complete structure and cells)
  - Replace new_nb's src import cell with the inline definitions from tm_tests_33
  - Remove the #TO DO placeholder cell
  - Convert oof_proba_{tag}.csv loads → np.load(oof_proba_{tag}.npy)
  - Update markdown cells that reference src/*.py / scripts/*.py
  - Save result as tm_tests_33.ipynb
"""
import json, re, sys, os

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def make_code(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": [src]}


def make_md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}


# ── Load notebooks ─────────────────────────────────────────────────────────
with open('new_nb.ipynb', encoding='utf-8') as f:
    new_nb = json.load(f)

with open('tm_tests_33.ipynb', encoding='utf-8') as f:
    tm = json.load(f)

new_cells = new_nb['cells']
tm_cells  = tm['cells']

# Extract the inline-definitions cell from tm_tests_33 (find by marker)
inline_defs_cell = None
for c in tm_cells:
    src = ''.join(c.get('source', []))
    if c['cell_type'] == 'code' and ('Inline module definitions' in src or
                                      '_nb_generate_features' in src):
        inline_defs_cell = c
        break
assert inline_defs_cell is not None, "Inline defs cell not found in tm_tests_33"

# Extract the COMPAT cell (first code cell that defines LABEL_MAP)
compat_cell = None
for c in tm_cells:
    src = ''.join(c.get('source', []))
    if c['cell_type'] == 'code' and 'LABEL_MAP' in src and 'COMPAT' in src:
        compat_cell = c
        break
if compat_cell is None:
    for c in tm_cells:
        src = ''.join(c.get('source', []))
        if c['cell_type'] == 'code' and 'LABEL_MAP = CLASSES' in src:
            compat_cell = c
            break
assert compat_cell is not None, "COMPAT cell not found in tm_tests_33"

print("Inline defs cell: OK")
print("COMPAT cell: OK")


# ── Helper: patch csv → npy in a code source string ───────────────────────
def patch_oof_csv_to_npy(src):
    """Replace oof_proba_{tag}.csv reads with np.load(oof_proba_{tag}.npy)"""
    # Pattern: pd.read_csv(f'results/predictions/oof_proba_{tag}.csv')[['p0','p1','p2']].values
    src = re.sub(
        r"pd\.read_csv\(f?['\"]results/predictions/oof_proba_\{tag\}\.csv['\"]"
        r"\)\[\['p0','p1','p2'\]\]\.values(\s*\.astype\([^)]+\))?",
        "np.load(f'results/predictions/oof_proba_{tag}.npy')",
        src
    )
    # Variant without f-string
    src = re.sub(
        r"pd\.read_csv\(['\"]results/predictions/oof_proba_[^'\"]+\.csv['\"]"
        r"\)\[\['p0','p1','p2'\]\]\.values(\s*\.astype\([^)]+\))?",
        lambda m: m.group(0).replace('.csv', '.npy').replace(
            "pd.read_csv", "np.load").replace(")[['p0','p1','p2']].values", ")"),
        src
    )
    # Also fix: path = f'results/predictions/oof_proba_{tag}.csv'
    src = re.sub(
        r"(path\s*=\s*f?['\"]results/predictions/oof_proba_\{tag\})\.csv(['\"])",
        r"\1.npy\2",
        src
    )
    # arr = np.load(p) if p.endswith('.npy') else pd.read_csv(p)[...].values  → np.load(p)
    src = re.sub(
        r"np\.load\(p\) if p\.endswith\('\.npy'\) else pd\.read_csv\(p\)"
        r"\[\['p0','p1','p2'\]\]\.values",
        "np.load(p)",
        src
    )
    # for p in [f'...oof_proba_{tag}.npy', f'...oof_proba_{tag}.csv']:  → just npy
    src = re.sub(
        r"for p in \[f'results/predictions/oof_proba_\{tag\}\.npy',\s*"
        r"f'results/predictions/oof_proba_\{tag\}\.csv'\]:",
        "for p in [f'results/predictions/oof_proba_{tag}.npy']:",
        src
    )
    return src


def patch_md_src_refs(src):
    """Remove/soften markdown references to src/*.py and scripts/*.py"""
    # ── src/ module refs ───────────────────────────────────────────────────
    src = src.replace(
        'All preprocessing functions are implemented in `src/preprocessing.py` and imported throughout this section.',
        'All preprocessing functions are implemented inline in the Setup cell above.'
    )
    src = src.replace(
        'All feature engineering functions are implemented in **`src/features.py`** and called from',
        'All feature engineering functions are defined inline in the Setup cell and called from'
    )
    src = src.replace('src/preprocessing.py', '`inline_preprocessing` (Setup cell)')
    src = src.replace('`src/features.py`', '`inline_features` (Setup cell)')
    src = src.replace('from src.preprocessing import', '# inline — see Setup cell')
    src = src.replace('from src.features import', '# inline — see Setup cell')
    src = src.replace(
        '**`src/utils.py`** fixes the global seed',
        '**`set_global_seed()`** (inline in Setup cell) fixes the global seed'
    )
    src = src.replace(
        '**`src/agent.py`** implements the agentic orchestration system (Phase 8).',
        'The **agentic orchestration helper** (Phase 8) is implemented inline in the Setup cell.'
    )
    src = src.replace(
        '`src/agent.py` calls `get_bert_embeddings()` and `get_sbert_embeddings()`\n  directly for its classification tools.',
        'the Setup cell exposes `get_bert_embeddings()` and `get_sbert_embeddings()` directly.'
    )
    src = src.replace(
        'the shared `evaluate_model` routine in `src/evaluation.py`',
        'the shared `evaluate_model()` routine (inline in Setup cell)'
    )

    # ── transformer training scripts ───────────────────────────────────────
    src = src.replace(
        'produced outside the notebook by `scripts/run_transformer_cv_v2.py`, launched in batch via '
        '`scripts/run_all_10fold.ps1`, which trains the eight encoder configurations under the 10-fold '
        'protocol. Each run writes a JSON result card to `results/tables/` and OOF/test probability arrays '
        'to `results/predictions/`; the notebook loads these files rather than re-training. Transformer '
        'fine-tuning was run on a GPU machine, since multi-epoch fine-tuning of large encoders is impractical '
        'to execute inline; all shell commands to reproduce each run are included, and the pre-computed result '
        'files are shipped in the submission so no re-training is required to reproduce the reported numbers.',
        'trained offline on a GPU (multi-epoch fine-tuning is impractical to execute inline). Each run writes '
        'a JSON result card to `results/tables/` and OOF/test probability arrays to `results/predictions/`; '
        'the notebook loads these pre-computed files. No re-training is required to reproduce the reported '
        'numbers — all result files are shipped with the submission.'
    )

    # ── ensemble / coordinate ascent ───────────────────────────────────────
    src = src.replace(
        'The ensemble weights are optimised by coordinate ascent (`scripts/reoptimize_ensemble.py`) over',
        'The ensemble weights are optimised by coordinate ascent (run offline) over'
    )
    src = src.replace('(`scripts/reoptimize_ensemble.py`)', '(run offline)')
    src = src.replace('`scripts/reoptimize_ensemble.py`', 'coordinate ascent (run offline)')

    # ── distillation script ────────────────────────────────────────────────
    src = src.replace(
        'we also distil this ensemble into a single FinBERT student (`scripts/run_distill_cv.py`, Hinton',
        'we also distil this ensemble into a single FinBERT student (run offline; Hinton'
    )
    src = src.replace(
        '- **Script:** `scripts/run_distill_cv.py`',
        '- **Script:** run offline (results in `results/tables/finbert_distilled_result.json`)'
    )
    src = src.replace('`scripts/run_distill_cv.py`', 'distillation script (run offline)')

    # ── data-quality analysis scripts ──────────────────────────────────────
    src = re.sub(
        r'is generated by(?: the same script)?\s*\(`scripts/data_quality_analyses\.py`[^)]*\)[^.]*\.',
        'was pre-generated and is stored in `results/tables/`.',
        src
    )
    src = re.sub(
        r'generated by `scripts/phase2_analysis\.py`[^.]*\.',
        'pre-generated and stored in `results/tables/`.',
        src
    )
    src = re.sub(
        r'> \*\*Data source:\*\*[^\n]*`scripts/data_quality_analyses\.py`[^\n]*\n?',
        '> **Data source:** pre-generated and stored in `results/tables/`.\n',
        src
    )

    # ── feature generation ─────────────────────────────────────────────────
    src = src.replace(
        'allows running once on GPU and loading from cache.',
        'allows generating once on GPU and loading from `data/processed/`.'
    )
    src = src.replace(
        'runs once in `scripts/generate_features.py`',
        'can be regenerated via the feature-extraction functions in the Setup cell'
    )
    src = src.replace(
        'runs once in `script',  # truncated variant
        'can be regenerated via the Setup cell'
    )
    src = src.replace(
        'the heavy computation\nruns once in `scripts/generate_features.py` (cached to `data/processed/`)',
        'the heavy computation runs once and is cached to `data/processed/`'
    )
    src = src.replace(
        'pre-generated by `scripts/features_analysis.py`, which saves the results to `results/tables/` and the PCA\nfigure to `results/figures/`. The cells below load and display those results directly.',
        'pre-generated; results are stored in `results/tables/`. The cells below load and display those results directly.'
    )
    src = src.replace(
        'dense embeddings are pre-generated by `scripts/generate_features.py` and cached in `data/processed/`.',
        'dense embeddings are pre-generated and cached in `data/processed/`.'
    )

    # ── classical ML / optuna scripts ──────────────────────────────────────
    src = src.replace(
        'were produced by `scripts/run_classical_ml.py`, which evaluates every pair with the shared `evaluate_model` routine in `src/evaluation.py` (10-fold stratified, seed 42).',
        'were produced offline by evaluating every pair with `evaluate_model()` (10-fold stratified, seed 42).'
    )
    src = src.replace(
        'LightGBM hyperparameters were optimised with 50 TPE trials on RoBERTa embeddings (`scripts/run_optuna_tuning.py` -> `results/tables/optuna_best_params.json`)',
        'LightGBM hyperparameters were optimised with 50 TPE trials on RoBERTa embeddings (results in `results/tables/optuna_best_params.json`)'
    )
    src = src.replace('(`scripts/run_optuna_tuning.py`)', '(Optuna tuning, pre-computed)')
    src = src.replace('`scripts/run_optuna_tuning.py`', 'Optuna tuning (pre-computed)')

    # ── GPT-2 scripts ──────────────────────────────────────────────────────
    src = src.replace(
        '**Fine-tuned decoder** (`scripts/run_gpt2_decoder.py`)',
        '**Fine-tuned decoder** (fine-tuned offline)'
    )
    src = src.replace('(`scripts/run_gpt2_decoder.py`)', '(fine-tuned offline)')
    src = src.replace('`scripts/run_gpt2_decoder.py`', 'GPT-2 fine-tuning (run offline)')

    # ── ensemble strategy scripts ──────────────────────────────────────────
    src = src.replace('(`scripts/compare_and_ensemble.py`)', '(pre-computed)')
    src = src.replace('`scripts/compare_and_ensemble.py`', 'soft-voting (pre-computed)')
    src = src.replace('(`scripts/run_stacking.py`, Wolpert', '(Wolpert')
    src = src.replace('(`scripts/run_stacking.py`)', '(pre-computed)')
    src = src.replace('`scripts/run_stacking.py`', 'stacking (pre-computed)')

    # ── Generic catch-all: any remaining `scripts/X.py` or `src/X.py` ─────
    src = re.sub(r'`scripts/[\w./]+\.ps1`', '(batch script, run offline)', src)
    src = re.sub(r'`scripts/[\w./]+\.py`', '(pre-computed)', src)
    src = re.sub(r'`src/[\w./]+\.py`',     '(inline in Setup cell)', src)

    return src


# ── Build merged cell list ─────────────────────────────────────────────────
merged = []
skip_next = False

for idx, cell in enumerate(new_cells):
    src_str = ''.join(cell.get('source', []))

    # Skip the #TO DO cell
    if cell['cell_type'] == 'code' and src_str.strip() == '#TO DO':
        print(f"[{idx:03d}] Removed #TO DO cell")
        continue

    # Replace the src-imports cell with:
    #   1. inline defs  (preprocessing + features inlined)
    #   2. cleaned standard imports (src.* lines removed)
    #   3. COMPAT cell  (LABEL_MAP, PALETTE, COLORS aliases)
    if cell['cell_type'] == 'code' and 'from src.preprocessing import' in src_str:
        # Build clean imports: strip src.* lines AND their multi-line continuations
        clean_lines = []
        in_src_block = False
        for line in src_str.split('\n'):
            # Detect start of a multi-line src import
            if (line.startswith('from src.') or line.startswith('import src.')):
                in_src_block = '(' in line and ')' not in line
                continue
            # Inside a multi-line src import block
            if in_src_block:
                if ')' in line:
                    in_src_block = False
                continue
            clean_lines.append(line)
        clean_src = '\n'.join(clean_lines)
        # Also remove LABEL_MAP / PALETTE / COLORS / KEEP_NEGATIONS from imports
        # since COMPAT cell defines them (avoids duplication)
        clean_lines2 = []
        for line in clean_src.split('\n'):
            if any(line.strip().startswith(x) for x in
                   ['PALETTE', 'LABEL_MAP', 'COLORS', 'KEEP_NEGATIONS']):
                continue
            clean_lines2.append(line)
        clean_src = '\n'.join(clean_lines2)

        # Wrap optional imports in try/except
        clean_src = clean_src.replace(
            'from matplotlib_venn import venn3',
            'try:\n    from matplotlib_venn import venv3\nexcept ImportError:\n    venn3 = None'
        ).replace(
            'import emoji',
            'try:\n    import emoji\nexcept ImportError:\n    emoji = None'
        )
        # matplotlib_venn alias: fix typo introduced above
        clean_src = clean_src.replace(
            'try:\n    from matplotlib_venn import venv3\nexcept ImportError:\n    venn3 = None',
            'try:\n    from matplotlib_venn import venn3\nexcept ImportError:\n    venn3 = None'
        )

        merged.append(inline_defs_cell)
        clean_imports_cell = make_code(clean_src)
        merged.append(clean_imports_cell)
        merged.append(compat_cell)
        print(f"[{idx:03d}] Replaced src-imports cell with inline_defs + clean_imports + COMPAT")
        continue

    # Patch code cells
    if cell['cell_type'] == 'code':
        new_src = patch_oof_csv_to_npy(src_str)
        if new_src != src_str:
            print(f"[{idx:03d}] Patched oof_proba .csv → .npy")
        # Fix: cell 072 assigns `auc = d['shift']['adversarial_auc']` which shadows sklearn auc()
        # Rename all local float uses of auc as adversarial AUC to adv_auc
        import re as _re
        new_src = new_src.replace(
            "auc = d['shift']['adversarial_auc']",
            "adv_auc = d['shift']['adversarial_auc']"
        )
        # Replace {auc:.Xf} when preceded by "adversarial" or "AUC" context in the same string
        new_src = _re.sub(
            r'\{auc(:.+?)\}',
            r'{adv_auc\1}',
            new_src
        )

        # Fix CSV reading with potential non-UTF8 bytes — latin-1 is a superset of cp1252
        new_src = new_src.replace(
            "pd.read_csv('results/tables/error_examples.csv')",
            "pd.read_csv('results/tables/error_examples.csv', encoding='latin-1')"
        ).replace(
            "pd.read_csv('results/tables/error_analysis.csv')",
            "pd.read_csv('results/tables/error_analysis.csv', encoding='latin-1')"
        )

        # Guard dense-cache demo (cell uses X_sbert_train which may not be loaded)
        if 'X_sbert_train' in new_src and 'r_lgbm = evaluate_model' in new_src:
            new_src = (
                "if 'X_sbert_train' not in dir():\n"
                "    print('[SKIP] Dense feature cache not loaded — skipping LightGBM-SBERT demo')\n"
                "else:\n"
                + '\n'.join('    ' + line for line in new_src.split('\n'))
            )

        # Guard emoji calls
        new_src = new_src.replace(
            "df['n_emoji']     = df['text'].apply(lambda t: emoji.emoji_count(str(t)))",
            "df['n_emoji']     = df['text'].apply(lambda t: emoji.emoji_count(str(t)) if emoji else 0)"
        )
        # Guard venn3: catch TypeError (venn3=None) in addition to ImportError
        new_src = new_src.replace(
            "except ImportError:\n    print('matplotlib-venn not installed  -  skipping Venn diagram.')",
            "except (ImportError, TypeError):\n    print('matplotlib-venn not installed  —  skipping Venn diagram.')"
        )
        cell = dict(cell)
        cell['source'] = [new_src]

    # Patch markdown cells
    elif cell['cell_type'] == 'markdown':
        new_src = patch_md_src_refs(src_str)
        if new_src != src_str:
            print(f"[{idx:03d}] Patched markdown src refs")
        cell = dict(cell)
        cell['source'] = [new_src]

    merged.append(cell)

print(f"\nMerged notebook: {len(merged)} cells")


# ── Save ───────────────────────────────────────────────────────────────────
out = dict(new_nb)
out['cells'] = merged

with open('tm_tests_33.ipynb', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

print("Saved: tm_tests_33.ipynb")
