$base = "C:\Users\tiago\OneDrive - NOVAIMS\Ambiente de Trabalho\textmining\group_33"
Set-Location $base

# EXP P0-A: DeBERTa-v3-large 8ep fix-text, lr=8e-6 (was still improving at 6ep)
Write-Host "=== P0-A: DeBERTa-v3-large 8ep fix-text lr=8e-6 ===" -ForegroundColor Cyan
python scripts\run_transformer_cv_v2.py --model-name microsoft/deberta-v3-large --tag debertav3_large_8ep_fixtext --epochs 8 --maxlen 128 --lr 8e-6 --weight-decay 0.01 --batch-size 8 --grad-accum 4 --label-smoothing 0.05 --warmup-ratio 0.1 --schedule cosine --amp-dtype fp16 --fix-text --n-folds 10
python -c "import numpy as np,pandas as pd; arr=np.load('results/predictions/test_proba_debertav3_large_8ep_fixtext.npy'); pd.DataFrame(arr,columns=['p0','p1','p2']).to_csv('results/predictions/prob_test_debertav3_large_8ep_fixtext.csv',index=False); print('Saved prob_test_debertav3_large_8ep_fixtext.csv')"
Write-Host "=== P0-A DONE ===" -ForegroundColor Green

# EXP P0-B: FinBERT-fintwitter with LLRD 0.9
Write-Host "=== P0-B: FinBERT LLRD 7ep fix-text ===" -ForegroundColor Cyan
python scripts\run_transformer_cv_v2.py --model-name nickmuchi/finbert-tone-finetuned-fintwitter-classification --tag finbert_fintwitter_llrd_7ep --epochs 7 --maxlen 128 --lr 8e-6 --llrd 0.9 --weight-decay 0.01 --batch-size 16 --label-smoothing 0.05 --warmup-ratio 0.06 --schedule cosine --amp-dtype fp16 --fix-text --n-folds 10
python -c "import numpy as np,pandas as pd; arr=np.load('results/predictions/test_proba_finbert_fintwitter_llrd_7ep.npy'); pd.DataFrame(arr,columns=['p0','p1','p2']).to_csv('results/predictions/prob_test_finbert_fintwitter_llrd_7ep.csv',index=False); print('Saved prob_test_finbert_fintwitter_llrd_7ep.csv')"
Write-Host "=== P0-B DONE ===" -ForegroundColor Green

Write-Host "PHASE 0 EXPERIMENTS DONE" -ForegroundColor Yellow
