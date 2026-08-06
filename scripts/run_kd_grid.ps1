Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..'))

# KD hyperparameter grid - 3-fold proxy (~12 min each)
# Baseline reference (a=0.5, T=2): folds 1-3 were [0.9101, 0.9078, 0.8999], mean 0.9060

Write-Host "=== KD GRID: a=0.7 T=2 ===" -ForegroundColor Cyan
python scripts\run_distill_cv.py --tag kdgrid_a07_t2 --alpha 0.7 --temperature 2.0 --epochs 10 --n-folds 10 --max-folds 3

Write-Host "=== KD GRID: a=0.9 T=2 ===" -ForegroundColor Cyan
python scripts\run_distill_cv.py --tag kdgrid_a09_t2 --alpha 0.9 --temperature 2.0 --epochs 10 --n-folds 10 --max-folds 3

Write-Host "=== KD GRID: a=0.7 T=3 ===" -ForegroundColor Cyan
python scripts\run_distill_cv.py --tag kdgrid_a07_t3 --alpha 0.7 --temperature 3.0 --epochs 10 --n-folds 10 --max-folds 3

Write-Host "=== KD GRID: a=0.5 T=3 ===" -ForegroundColor Cyan
python scripts\run_distill_cv.py --tag kdgrid_a05_t3 --alpha 0.5 --temperature 3.0 --epochs 10 --n-folds 10 --max-folds 3

Write-Host "=== KD GRID: a=1.0 T=2 (pure KD, no hard labels) ===" -ForegroundColor Cyan
python scripts\run_distill_cv.py --tag kdgrid_a10_t2 --alpha 1.0 --temperature 2.0 --epochs 10 --n-folds 10 --max-folds 3

Write-Host "KD GRID DONE" -ForegroundColor Yellow
