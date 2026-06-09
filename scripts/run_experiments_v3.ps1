$base = "C:\Users\tiago\OneDrive - NOVAIMS\Ambiente de Trabalho\textmining\group_33"
Set-Location $base

# EXP A: FinBERT 15ep fix-text, lr=2e-6 (lower LR for longer training)
Write-Host "=== EXP A: FinBERT 15ep fix-text lr=2e-6 ===" -ForegroundColor Cyan
python scripts\run_transformer_cv_v2.py `
    --model-name nickmuchi/finbert-tone-finetuned-fintwitter-classification `
    --tag finbert_fintwitter_15ep_fixtext `
    --epochs 15 --maxlen 128 --lr 2e-6 --weight-decay 0.01 `
    --batch-size 16 --label-smoothing 0.05 --warmup-ratio 0.06 `
    --schedule cosine --amp-dtype fp16 --fix-text
Write-Host "=== EXP A DONE ===" -ForegroundColor Green

# EXP B: ProsusAI/finbert (different pre-training: financial news, not tweets)
Write-Host "=== EXP B: ProsusAI/finbert 10ep fix-text ===" -ForegroundColor Cyan
python scripts\run_transformer_cv_v2.py `
    --model-name ProsusAI/finbert `
    --tag prosusai_finbert_10ep_fixtext `
    --epochs 10 --maxlen 128 --lr 5e-6 --weight-decay 0.01 `
    --batch-size 16 --label-smoothing 0.05 --warmup-ratio 0.06 `
    --schedule cosine --amp-dtype fp16 --fix-text
Write-Host "=== EXP B DONE ===" -ForegroundColor Green

# EXP C: DeBERTa-v3-large 6ep WITH fix-text (previously ran without fix-text)
Write-Host "=== EXP C: DeBERTa-v3-large 6ep fix-text ===" -ForegroundColor Cyan
python scripts\run_transformer_cv_v2.py `
    --model-name microsoft/deberta-v3-large `
    --tag debertav3_large_6ep_fixtext `
    --epochs 6 --maxlen 128 --lr 1e-5 --weight-decay 0.01 `
    --batch-size 8 --grad-accum 4 --label-smoothing 0.05 --warmup-ratio 0.1 `
    --schedule cosine --amp-dtype fp16 --fix-text
Write-Host "=== EXP C DONE ===" -ForegroundColor Green

Write-Host "ALL EXPERIMENTS DONE" -ForegroundColor Yellow
