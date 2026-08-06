# Data

## Provenance

The course files correspond to the [Twitter Financial News Sentiment dataset](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment), an English corpus of 11,931 finance-related tweets collected through the Twitter API.

The upstream dataset card identifies its license as MIT. That license applies to the dataset and does not automatically license the original code or written material in this repository.

## Local files

| File | Rows | Columns | Purpose |
|---|---:|---|---|
| `raw/train.csv` | 9,543 | `text`, `label` | Labelled modelling corpus |
| `raw/test.csv` | 2,388 | `id`, `text` | Course test split with labels withheld |

Label mapping:

| ID | Class |
|---:|---|
| 0 | Bearish |
| 1 | Bullish |
| 2 | Neutral |

The course copy contains encoding variants such as UTF-8 text decoded as Latin-1. They are intentionally preserved in `data/raw/` because detecting and repairing that corruption is part of the experiment. The selected transformer pipeline applies conservative `ftfy` repair instead of aggressive token deletion.

## Generated data

`data/processed/` is ignored by Git. Dense embeddings and other large derived features can be recreated from the raw CSVs with the scripts in [`../scripts/`](../scripts).

When replacing or extending the data, document the source, collection period, consent/privacy considerations, and redistribution license. Do not commit credentials, private text, or datasets whose redistribution terms are unclear.
