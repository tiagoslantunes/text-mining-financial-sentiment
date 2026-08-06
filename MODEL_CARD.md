# Model Card: Distilled FinBERT for Financial Tweet Sentiment

## Model summary

The submitted classifier is a 110M-parameter FinBERT student trained with knowledge distillation from an eight-encoder teacher ensemble. It maps short English financial social-media text to one of three labels:

| ID | Label |
|---:|---|
| 0 | Bearish |
| 1 | Bullish |
| 2 | Neutral |

The student starts from [`nickmuchi/finbert-tone-finetuned-fintwitter-classification`](https://huggingface.co/nickmuchi/finbert-tone-finetuned-fintwitter-classification). Its training objective combines class-weighted cross-entropy with temperature-scaled KL divergence from the teacher probabilities.

## Intended use

- Educational research on financial sentiment classification.
- Reproduction and comparison of text-mining methods.
- Offline experimentation on short, English, finance-related text.
- A component in human-reviewed research prototypes.

## Out-of-scope use

- Automated trading, investment advice, credit decisions, or risk limits.
- High-stakes decisions without domain-expert review.
- General-purpose sentiment analysis outside the financial-Twitter domain.
- Languages other than English or long-form financial documents.
- Claims about future market returns: the model classifies textual stance, not price direction.

## Training data

The project uses the Twitter Financial News Sentiment corpus with 9,543 labelled training examples. The three classes are imbalanced, with Neutral as the majority class. The project audits duplicate structure, text corruption, distribution shift, and a lower bound on annotation noise.

See [`data/README.md`](data/README.md) for provenance and schema.

## Evaluation

The submitted model is evaluated with 10-fold stratified out-of-fold predictions using seed 42.

| Metric | Score |
|---|---:|
| Macro-F1 | 0.9139 |
| Accuracy | 0.9334 |
| Macro precision | 0.9032 |
| Macro recall | 0.9258 |
| Bearish F1 | 0.8713 |
| Bullish F1 | 0.9165 |
| Neutral F1 | 0.9540 |

These are cross-validation estimates on the supplied training corpus, not results from a temporally separated live-market benchmark.

## Training configuration

- Base checkpoint: `nickmuchi/finbert-tone-finetuned-fintwitter-classification`
- Folds: 10, stratified
- Epochs: 12
- Maximum sequence length: 128
- Learning rate: `5e-6`
- Batch size: 16
- Distillation weight: 0.5
- Temperature: 2.0
- Preprocessing: conservative `ftfy` text repair
- Seed: 42

The recorded run completed in approximately 47 minutes on an NVIDIA RTX 5070 with 12 GB of VRAM. Runtime depends strongly on hardware and software versions.

## Limitations and risks

- **Domain shift:** language, tickers, market narratives, and platform conventions change over time.
- **Class imbalance:** Neutral dominates the corpus; macro-F1 is therefore the primary metric.
- **Annotation ambiguity:** analyst actions and weakly directional statements can sit on the Neutral/Bullish or Neutral/Bearish boundary.
- **No temporal holdout:** random stratified folds do not measure performance on future market regimes.
- **Social and sampling bias:** the corpus reflects English-language finance content collected from Twitter and is not representative of all investors or markets.
- **Confidence is not certainty:** predicted probabilities are not calibrated financial-risk estimates.
- **Upstream dependencies:** behaviour can change with tokenizer, checkpoint, CUDA, or library versions.

## Mitigations

- Report macro-averaged metrics and per-class results.
- Use out-of-fold predictions for model comparison and distillation.
- Preserve financial cues while limiting destructive preprocessing.
- Keep a human in the loop for any interpretation beyond academic benchmarking.
- Do not treat sentiment labels as buy/sell signals.

## Artifacts

- Final notebook: [`tm_final_33.ipynb`](tm_final_33.ipynb)
- Full experiments: [`tm_tests_33.ipynb`](tm_tests_33.ipynb)
- Metrics: [`results/tables/submission_summary.json`](results/tables/submission_summary.json)
- OOF confusion matrix: [`results/figures/confusion_matrix_distilled.png`](results/figures/confusion_matrix_distilled.png)
- Academic report: [`docs/report.pdf`](docs/report.pdf)
