"""
Verify tm_tests_33.ipynb can Run All with the committed repo files.

File categories:
  REQUIRED  - data/raw/train.csv, test.csv (competition data, must be present, not in git)
  COMMITTED - results/tables/*.csv/.json, results/predictions/*.npy (in git, always available)
  OPTIONAL  - data/processed/*.npy (heavy cache, not in git, guarded cells)

Tests:
  1. No src/*.py code-cell imports  (new_nb fails this → NOT deliverable)
  2. All COMMITTED reads are actually committed to git
  3. OPTIONAL reads have os.path.exists / dir() guards
  4. Classical ML trains inline (evaluate_model / .fit calls outside definition)
  5. No syntax errors in any code cell
"""
import ast, json, os, re, subprocess, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

NB = 'tm_tests_33.ipynb'
OK   = '\033[92m PASS\033[0m'
BAD  = '\033[91m FAIL\033[0m'
WARN = '\033[93m WARN\033[0m'

# ── helpers ────────────────────────────────────────────────────────────────

def git_files():
    """Return set of paths committed to git, relative to deliverables/group_33/."""
    r = subprocess.run(['git', 'ls-files'], capture_output=True, text=True)
    raw = {p.replace('\\', '/') for p in r.stdout.splitlines()}
    # git ls-files from inside a subdir returns paths relative to cwd
    # but from repo root returns full paths — normalise both cases
    result = set()
    for p in raw:
        if 'deliverables/group_33/' in p:
            result.add(p.split('deliverables/group_33/')[-1])
        else:
            result.add(p)
    return result

committed = git_files()

# Paths that are expected external inputs (competition data — never in git)
REQUIRED_EXTERNAL = {'data/raw/train.csv', 'data/raw/test.csv'}

# Paths that MUST be in git to satisfy a fresh clone
EXPECTED_COMMITTED = {
    'results/tables/mojibake_by_class.csv', 'results/tables/mojibake_examples.csv',
    'results/tables/duplicates_cashtag_analysis.json',
    'results/tables/shift_and_noise_analysis.json',
    'results/tables/preprocessing_ablation.csv',
    'results/tables/fusion_features_result.json',
    'results/tables/optuna_best_params.json', 'results/tables/optuna_lr_best_params.json',
    'results/tables/statistical_significance.json',
    'results/tables/error_examples.csv', 'results/tables/error_analysis.csv',
    'results/tables/encoder_results.csv', 'results/tables/gpt2_results.csv',
    'results/tables/ensemble_results.csv', 'results/tables/kd_results.csv',
    'results/tables/classical_ml_full.csv', 'results/tables/ensemble_optimal_result.json',
}
for tag in ['deberta_base_finance_fixtext','debertav3_large_6ep','debertav3_large_6ep_fixtext',
            'finbert_fintwitter','finbert_fintwitter_10ep','finbert_fintwitter_10ep_fixtext',
            'finbert_fintwitter_7ep','roberta_large_ts_v2']:
    EXPECTED_COMMITTED.add(f'results/predictions/oof_proba_{tag}.npy')

# Paths that are OPTIONAL (heavy cache — guarded in notebook)
OPTIONAL_CACHE = {f'data/processed/{x}'
                  for x in ['X_finbert_train.npy','X_sbert_train.npy','X_roberta_train.npy',
                             'X_glove_train.npy','X_w2v_sg_train.npy','X_w2v_cbow_train.npy',
                             'X_fin_train.npy']}

with open(NB, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
code_cells = [(i, ''.join(c.get('source', []))) for i, c in enumerate(cells)
              if c['cell_type'] == 'code']

results = []

# ══ TEST 1 — no src/*.py code imports ════════════════════════════════════
print('\n─── TEST 1: No src/*.py code-cell imports ────────────────────────')
bad_imports = [(i, ln.strip()) for i, src in code_cells
               for ln in src.split('\n')
               if re.match(r'\s*(from src\.|import src\.)', ln)]
if not bad_imports:
    print(f'{OK}  tm_tests_33 has zero src.* imports in code cells')
    results.append(True)
else:
    for idx, ln in bad_imports:
        print(f'{BAD}  cell {idx:03d}: {ln}')
    results.append(False)

with open('new_nb.ipynb', encoding='utf-8') as f:
    nb2 = json.load(f)
bad2 = sum(1 for c in nb2['cells'] if c['cell_type'] == 'code'
           for ln in ''.join(c.get('source', [])).split('\n')
           if re.match(r'\s*(from src\.|import src\.)', ln))
print(f'  (new_nb has {bad2} src.* code import lines → NOT the deliverable)')

# ══ TEST 2 — committed files are actually in git ══════════════════════════
print('\n─── TEST 2: Pre-computed result files committed to git ───────────')
missing_from_git = [p for p in sorted(EXPECTED_COMMITTED) if p not in committed]
if not missing_from_git:
    print(f'{OK}  All {len(EXPECTED_COMMITTED)} expected result/prediction files are in git')
    results.append(True)
else:
    for p in missing_from_git:
        print(f'{BAD}  NOT in git: {p}')
    results.append(False)

# ══ TEST 3 — optional cache reads are guarded ════════════════════════════
print('\n─── TEST 3: Optional data/processed/ reads have guards ───────────')
FILE_PAT = re.compile(
    r"(?:pd\.read_csv|open|np\.load|json\.load\s*\(\s*open|plt\.imread|Image\.open)"
    r"\s*\(\s*f?['\"]([^'\"{}]+)['\"]"
)

guard_failures = []
for idx, src in code_cells:
    is_guarded = 'os.path.exists' in src or "not in dir()" in src or "try:" in src
    for m in FILE_PAT.finditer(src):
        path = m.group(1).lstrip('./')
        if '{' in path:
            continue
        if path in OPTIONAL_CACHE and not is_guarded:
            guard_failures.append((idx, path))

if not guard_failures:
    print(f'{OK}  All data/processed/ reads are in guarded cells')
    results.append(True)
else:
    for idx, path in guard_failures:
        print(f'{BAD}  cell {idx:03d} unguarded: {path}')
    results.append(False)

cache_on_disk = os.path.exists('data/processed/X_fin_train.npy')
print(f'  data/processed/ present locally: {cache_on_disk}  '
      f'({"will run fully" if cache_on_disk else "demo cell skipped — all else OK"})')

# ══ TEST 4 — classical ML trains inline ══════════════════════════════════
print('\n─── TEST 4: Classical ML trains inline ───────────────────────────')
# Exclude cell 005 (definition of evaluate_model itself)
train_calls = [(i, ln.strip()) for i, src in code_cells if i != 5
               for ln in src.split('\n')
               if re.search(r'evaluate_model\s*\(|(?:clf|svm|lr|lgbm|nb|mlp|rf|knn|xgb)'
                            r'[a-z_]*\s*=.*(?:Classifier|CV|SVC|SVR)\(', ln, re.I)]
inline_fits = [(i, ln.strip()) for i, src in code_cells if i != 5
               for ln in src.split('\n')
               if re.search(r'\.fit\s*\(', ln) and 'def ' not in ln]

print(f'  evaluate_model() calls outside definition: {len(train_calls)}')
for idx, ln in train_calls[:6]:
    print(f'    cell {idx:03d}: {ln[:90]}')
print(f'  .fit() calls outside definitions: {len(inline_fits)}')
ok = len(train_calls) >= 2 and len(inline_fits) >= 1
results.append(ok)
print(f'  {OK if ok else BAD}  Classical ML is trained inline in the notebook')

# ══ TEST 5 — syntax check all code cells ══════════════════════════════════
print('\n─── TEST 5: Syntax check every code cell ─────────────────────────')
syntax_errors = []
for idx, src in code_cells:
    try:
        ast.parse(src)
    except SyntaxError as e:
        syntax_errors.append((idx, e))
        print(f'{BAD}  cell {idx:03d}: SyntaxError line {e.lineno}: {e.msg}')
if not syntax_errors:
    print(f'{OK}  All {len(code_cells)} code cells parse without syntax errors')
results.append(len(syntax_errors) == 0)

# ══ SUMMARY ══════════════════════════════════════════════════════════════
print('\n══ SUMMARY ══════════════════════════════════════════════════════')
labels = [
    'No src.* code imports (self-contained)',
    'All pre-computed results committed to git',
    'Optional cache reads guarded',
    'Classical ML trains inline',
    'All code cells syntax-clean',
]
all_pass = all(results)
for label, res in zip(labels, results):
    print(f'  {OK if res else BAD}  {label}')

print()
if all_pass:
    print('✓  tm_tests_33.ipynb passes all checks.')
    print('   Anyone who clones the repo and has train.csv + test.csv can Run All.')
    print('   data/processed/ cache is optional — one demo cell is skipped if absent.')
else:
    print('✗  Some checks failed.')

print()
print('VERDICT')
print('  Deliver  → tm_tests_33.ipynb  (inline definitions, no external .py deps)')
print('  Discard  → new_nb.ipynb        (requires src/*.py at runtime)')
