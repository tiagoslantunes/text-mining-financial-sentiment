$base = "C:\Users\tiago\OneDrive - NOVAIMS\Ambiente de Trabalho\textmining\group_33"
Set-Location $base

$FOLDS = 10

Write-Host "============================================" -ForegroundColor Yellow
Write-Host " Re-training all models with $FOLDS folds" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Yellow

# ── 1. FinBERT 5ep (raw) ────────────────────────────────────────────────────
Write-Host "`n=== [1/8] FinBERT 5ep raw ===" -ForegroundColor Cyan
python scripts\run_transformer_cv_v2.py --model-name nickmuchi/finbert-tone-finetuned-fintwitter-classification --tag finbert_fintwitter --epochs 5 --maxlen 128 --lr 1e-5 --weight-decay 0.01 --batch-size 16 --label-smoothing 0.05 --warmup-ratio 0.06 --schedule cosine --amp-dtype fp16 --n-folds $FOLDS
python -c "import numpy as np,pandas as pd; arr=np.load('results/predictions/test_proba_finbert_fintwitter.npy'); pd.DataFrame(arr,columns=['p0','p1','p2']).to_csv('results/predictions/prob_test_finbert_fintwitter.csv',index=False); print('Saved prob_test_finbert_fintwitter.csv')"
Write-Host "=== [1/8] DONE ===" -ForegroundColor Green

# ── 2. FinBERT 7ep (raw) ────────────────────────────────────────────────────
Write-Host "`n=== [2/8] FinBERT 7ep raw ===" -ForegroundColor Cyan
python scripts\run_transformer_cv_v2.py --model-name nickmuchi/finbert-tone-finetuned-fintwitter-classification --tag finbert_fintwitter_7ep --epochs 7 --maxlen 128 --lr 8e-6 --weight-decay 0.01 --batch-size 16 --label-smoothing 0.05 --warmup-ratio 0.06 --schedule cosine --amp-dtype fp16 --n-folds $FOLDS
python -c "import numpy as np,pandas as pd; arr=np.load('results/predictions/test_proba_finbert_fintwitter_7ep.npy'); pd.DataFrame(arr,columns=['p0','p1','p2']).to_csv('results/predictions/prob_test_finbert_fintwitter_7ep.csv',index=False); print('Saved prob_test_finbert_fintwitter_7ep.csv')"
Write-Host "=== [2/8] DONE ===" -ForegroundColor Green

# ── 3. FinBERT 10ep (raw) ───────────────────────────────────────────────────
Write-Host "`n=== [3/8] FinBERT 10ep raw ===" -ForegroundColor Cyan
python scripts\run_transformer_cv_v2.py --model-name nickmuchi/finbert-tone-finetuned-fintwitter-classification --tag finbert_fintwitter_10ep --epochs 10 --maxlen 128 --lr 5e-6 --weight-decay 0.01 --batch-size 16 --label-smoothing 0.05 --warmup-ratio 0.06 --schedule cosine --amp-dtype fp16 --n-folds $FOLDS
python -c "import numpy as np,pandas as pd; arr=np.load('results/predictions/test_proba_finbert_fintwitter_10ep.npy'); pd.DataFrame(arr,columns=['p0','p1','p2']).to_csv('results/predictions/prob_test_finbert_fintwitter_10ep.csv',index=False); print('Saved prob_test_finbert_fintwitter_10ep.csv')"
Write-Host "=== [3/8] DONE ===" -ForegroundColor Green

# ── 4. FinBERT 10ep + fix-text (PRIMARY NOTEBOOK MODEL) ────────────────────
Write-Host "`n=== [4/8] FinBERT 10ep fix-text (PRIMARY) ===" -ForegroundColor Cyan
python scripts\run_transformer_cv_v2.py --model-name nickmuchi/finbert-tone-finetuned-fintwitter-classification --tag finbert_fintwitter_10ep_fixtext --epochs 10 --maxlen 128 --lr 5e-6 --weight-decay 0.01 --batch-size 16 --label-smoothing 0.05 --warmup-ratio 0.06 --schedule cosine --amp-dtype fp16 --fix-text --n-folds $FOLDS
python -c "import numpy as np,pandas as pd; arr=np.load('results/predictions/test_proba_finbert_fintwitter_10ep_fixtext.npy'); pd.DataFrame(arr,columns=['p0','p1','p2']).to_csv('results/predictions/prob_test_finbert_fintwitter_10ep_fixtext.csv',index=False); print('Saved prob_test_finbert_fintwitter_10ep_fixtext.csv')"
Write-Host "=== [4/8] DONE ===" -ForegroundColor Green

# ── 5. RoBERTa-large ────────────────────────────────────────────────────────
Write-Host "`n=== [5/8] RoBERTa-large ===" -ForegroundColor Cyan
python scripts\run_transformer_cv_v2.py --model-name cardiffnlp/twitter-roberta-large-topic-sentiment-latest --tag roberta_large_ts_v2 --epochs 5 --maxlen 128 --lr 1e-5 --weight-decay 0.01 --batch-size 16 --label-smoothing 0.05 --warmup-ratio 0.06 --schedule cosine --amp-dtype fp16 --n-folds $FOLDS
python -c "import numpy as np,pandas as pd; arr=np.load('results/predictions/test_proba_roberta_large_ts_v2.npy'); pd.DataFrame(arr,columns=['p0','p1','p2']).to_csv('results/predictions/prob_test_roberta_large_ts_v2.csv',index=False); print('Saved prob_test_roberta_large_ts_v2.csv')"
Write-Host "=== [5/8] DONE ===" -ForegroundColor Green

# ── 6. DeBERTa-v3-base finance + fix-text ───────────────────────────────────
Write-Host "`n=== [6/8] DeBERTa-base finance fix-text ===" -ForegroundColor Cyan
python scripts\run_transformer_cv_v2.py --model-name nickmuchi/deberta-v3-base-finetuned-finance-text-classification --tag deberta_base_finance_fixtext --epochs 8 --maxlen 128 --lr 1e-5 --weight-decay 0.01 --batch-size 16 --grad-accum 2 --label-smoothing 0.05 --warmup-ratio 0.1 --schedule cosine --amp-dtype fp16 --fix-text --n-folds $FOLDS
python -c "import numpy as np,pandas as pd; arr=np.load('results/predictions/test_proba_deberta_base_finance_fixtext.npy'); pd.DataFrame(arr,columns=['p0','p1','p2']).to_csv('results/predictions/prob_test_deberta_base_finance_fixtext.csv',index=False); print('Saved prob_test_deberta_base_finance_fixtext.csv')"
Write-Host "=== [6/8] DONE ===" -ForegroundColor Green

# ── 7. DeBERTa-v3-large 6ep (raw) ───────────────────────────────────────────
Write-Host "`n=== [7/8] DeBERTa-v3-large 6ep raw ===" -ForegroundColor Cyan
python scripts\run_transformer_cv_v2.py --model-name microsoft/deberta-v3-large --tag debertav3_large_6ep --epochs 6 --maxlen 128 --lr 1e-5 --weight-decay 0.01 --batch-size 8 --grad-accum 4 --label-smoothing 0.05 --warmup-ratio 0.1 --schedule cosine --amp-dtype fp16 --n-folds $FOLDS
python -c "import numpy as np,pandas as pd; arr=np.load('results/predictions/test_proba_debertav3_large_6ep.npy'); pd.DataFrame(arr,columns=['p0','p1','p2']).to_csv('results/predictions/prob_test_debertav3_large_6ep.csv',index=False); print('Saved prob_test_debertav3_large_6ep.csv')"
Write-Host "=== [7/8] DONE ===" -ForegroundColor Green

# ── 8. DeBERTa-v3-large 6ep + fix-text ──────────────────────────────────────
Write-Host "`n=== [8/8] DeBERTa-v3-large 6ep fix-text ===" -ForegroundColor Cyan
python scripts\run_transformer_cv_v2.py --model-name microsoft/deberta-v3-large --tag debertav3_large_6ep_fixtext --epochs 6 --maxlen 128 --lr 1e-5 --weight-decay 0.01 --batch-size 8 --grad-accum 4 --label-smoothing 0.05 --warmup-ratio 0.1 --schedule cosine --amp-dtype fp16 --fix-text --n-folds $FOLDS
python -c "import numpy as np,pandas as pd; arr=np.load('results/predictions/test_proba_debertav3_large_6ep_fixtext.npy'); pd.DataFrame(arr,columns=['p0','p1','p2']).to_csv('results/predictions/prob_test_debertav3_large_6ep_fixtext.csv',index=False); print('Saved prob_test_debertav3_large_6ep_fixtext.csv')"
Write-Host "=== [8/8] DONE ===" -ForegroundColor Green

Write-Host "`n============================================" -ForegroundColor Yellow
Write-Host " ALL 8 MODELS DONE - 10-fold complete" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Yellow
