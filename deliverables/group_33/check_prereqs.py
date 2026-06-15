"""
Run this before opening any notebook to verify all prerequisites are in place.
Usage: python check_prereqs.py  (from deliverables/group_33/)
"""
import sys, os, importlib

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

errors = []
warnings_list = []

def check(label, ok, msg=""):
    mark = "OK" if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {msg}" if msg else ""))
    if not ok:
        errors.append(label)

def warn(label, msg=""):
    print(f"  [WARN] {label}" + (f" — {msg}" if msg else ""))
    warnings_list.append(label)

print("\n=== 1. Python packages ===")
packages = [
    "numpy", "pandas", "sklearn", "matplotlib", "seaborn",
    "nltk", "gensim", "transformers", "torch", "sentence_transformers",
    "lightgbm", "xgboost", "optuna", "ftfy", "vaderSentiment",
    "IPython", "scipy", "joblib",
]
for pkg in packages:
    name = pkg if pkg != "sklearn" else "sklearn"
    try:
        importlib.import_module(name)
        check(pkg, True)
    except ImportError:
        check(pkg, False, "pip install " + pkg)

print("\n=== 2. Raw data ===")
check("data/raw/train.csv", os.path.exists("data/raw/train.csv"))
check("data/raw/test.csv",  os.path.exists("data/raw/test.csv"))

print("\n=== 3. Precomputed feature matrices (data/processed/) ===")
feature_files = [
    "X_fin_train.npy", "X_fin_test.npy",
    "X_finbert_train.npy", "X_finbert_test.npy",
    "X_sbert_train.npy", "X_sbert_test.npy",
    "X_glove_train.npy", "X_glove_test.npy",
    "X_w2v_sg_train.npy", "X_w2v_sg_test.npy",
    "X_w2v_cbow_train.npy", "X_w2v_cbow_test.npy",
    "X_roberta_train.npy", "X_roberta_test.npy",
    "train_processed.csv", "test_processed.csv",
]
missing_features = []
for f in feature_files:
    path = os.path.join("data/processed", f)
    if not os.path.exists(path):
        missing_features.append(f)
if missing_features:
    warn("data/processed/", f"{len(missing_features)} missing — notebook will regenerate them (slow, needs GPU for transformer features)")
    for f in missing_features:
        print(f"       missing: {f}")
else:
    check("data/processed/ (all 16 files)", True)

print("\n=== 4. Ensemble OOF + test probability caches (results/predictions/) ===")
npy_files = [
    "oof_proba_finbert_fintwitter.npy",      "test_proba_finbert_fintwitter.npy",
    "oof_proba_finbert_fintwitter_7ep.npy",  "test_proba_finbert_fintwitter_7ep.npy",
    "oof_proba_finbert_fintwitter_10ep.npy", "test_proba_finbert_fintwitter_10ep.npy",
    "oof_proba_finbert_fintwitter_10ep_fixtext.npy", "test_proba_finbert_fintwitter_10ep_fixtext.npy",
    "oof_proba_deberta_base_finance_fixtext.npy",    "test_proba_deberta_base_finance_fixtext.npy",
    "oof_proba_debertav3_large_6ep.npy",     "test_proba_debertav3_large_6ep.npy",
    "oof_proba_debertav3_large_6ep_fixtext.npy", "test_proba_debertav3_large_6ep_fixtext.npy",
    "oof_proba_roberta_large_ts_v2.npy",     "test_proba_roberta_large_ts_v2.npy",
]
for f in npy_files:
    check(f"results/predictions/{f}", os.path.exists(f"results/predictions/{f}"))

print("\n=== 5. Results tables (JSONs needed by tm_tests_33) ===")
json_files = [
    "results/tables/finbert_fintwitter_result.json",
    "results/tables/finbert_fintwitter_7ep_result.json",
    "results/tables/finbert_fintwitter_10ep_result.json",
    "results/tables/finbert_fintwitter_10ep_fixtext_result.json",
    "results/tables/deberta_base_finance_fixtext_result.json",
    "results/tables/debertav3_large_6ep_result.json",
    "results/tables/debertav3_large_6ep_fixtext_result.json",
    "results/tables/roberta_large_ts_v2_result.json",
    "results/tables/ensemble_optimal_result.json",
    "results/tables/optuna_best_params.json",
    "results/tables/phase7_summary.json",
    "results/tables/phase12_gpt2clf.json",
    "results/tables/gpt2_decoder_result.json",
    "results/tables/calibration_results.json",
    "results/tables/statistical_significance.json",
    "results/tables/stacking_summary.json",
    "results/tables/duplicates_cashtag_analysis.json",
    "results/tables/shift_and_noise_analysis.json",
    "results/tables/eda_summary.json",
    "results/tables/features_summary.json",
    "results/tables/smote_comparison.json",
    "results/tables/fusion_features_result.json",
    "results/tables/full_comparison_final.csv",
    "results/tables/error_examples.csv",
    "results/tables/error_analysis.csv",
]
for f in json_files:
    check(f, os.path.exists(f))

print("\n=== 6. Final prediction file ===")
check("pred_33.csv", os.path.exists("pred_33.csv"))

print("\n=== 7. GPU scripts (only needed if OOF caches above are missing) ===")
for f in ["scripts/run_transformer_cv.py", "scripts/reoptimize_ensemble.py"]:
    if not os.path.exists(f):
        warn(f, "not found — only needed if results/predictions/*.npy are missing")
    else:
        check(f, True)

print("\n" + "="*55)
if errors:
    print(f"FAIL: {len(errors)} issue(s) must be fixed before running notebooks:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
elif warnings_list:
    print(f"READY with {len(warnings_list)} warning(s) — notebooks will self-heal slow parts.")
else:
    print("ALL CHECKS PASSED — notebooks should run without errors.")
