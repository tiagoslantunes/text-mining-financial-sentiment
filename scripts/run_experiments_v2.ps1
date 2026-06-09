# Run experiments to beat 91% F1 macro
# Execute from group_33 directory: .\scripts\run_experiments_v2.ps1

$ErrorActionPreference = "Continue"
$script_path = "scripts\run_transformer_cv_v2.py"

Write-Host "============================================================"
Write-Host " EXPERIMENT QUEUE (target: beat 91% F1)"
Write-Host "============================================================"

# ─── Experiment 1: nickmuchi FinBERT-tone (already fine-tuned on FinTwit) ───
Write-Host "`n[EXP 1] nickmuchi/finbert-tone-finetuned-fintwit-classification"
Write-Host "       Pre-trained on financial Twitter sentiment - our exact task"
python $script_path `
    --model-name nickmuchi/finbert-tone-finetuned-fintwit-classification `
    --tag finbert_fintwit `
    --epochs 5 `
    --maxlen 128 `
    --lr 1e-5 `
    --label-smoothing 0.05 `
    --warmup-ratio 0.06 `
    --schedule cosine `
    --amp-dtype fp16

Write-Host "`n[EXP 2] twitter-roberta-large-topic-sentiment (improved recipe)"
Write-Host "       Current best model + best-checkpoint + cosine + label smoothing"
python $script_path `
    --model-name cardiffnlp/twitter-roberta-large-topic-sentiment-latest `
    --tag roberta_large_ts_v2 `
    --epochs 5 `
    --maxlen 128 `
    --lr 1e-5 `
    --label-smoothing 0.05 `
    --warmup-ratio 0.06 `
    --schedule cosine `
    --amp-dtype fp16

Write-Host "`n[EXP 3] DeBERTa-v3-large (fp16, grad_accum=4)"
Write-Host "       Most powerful discriminative model"
python $script_path `
    --model-name microsoft/deberta-v3-large `
    --tag debertav3_large `
    --epochs 4 `
    --maxlen 128 `
    --lr 5e-6 `
    --weight-decay 0.01 `
    --batch-size 8 `
    --grad-accum 4 `
    --label-smoothing 0.05 `
    --warmup-ratio 0.1 `
    --schedule cosine `
    --amp-dtype fp16

Write-Host "`n============================================================"
Write-Host " All experiments complete. Check results/tables/*.json"
Write-Host "============================================================"
