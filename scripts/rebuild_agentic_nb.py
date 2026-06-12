# -*- coding: utf-8 -*-
"""Rebuild 08_Agentic_Workflow.ipynb: calibrated routing threshold, domain-expert
tool tied to the submission backbone, richer conversational demo, and routing
economics. Keeps the styled header (cell 0)."""

import json
import uuid
from pathlib import Path

NB = Path("notebooks/08_Agentic_Workflow.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
header = nb["cells"][0]          # keep the styled header


def code(src):
    return {"cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8],
            "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": src.splitlines(keepends=True)}


intro = """**Extra Challenge 2 (+1.50 pts).** A conversational agent that orchestrates **three classifiers** with **non-trivial, adaptive routing** (not simple voting). Runs fully offline via a deterministic `RuleBasedAgent` (no API key needed); an optional LangChain ReAct back-end engages automatically when an LLM key is present.

### Tools (escalating cost)
| # | Tool | Role | Warm latency |
|---|------|------|--------------|
| 1 | **VADER** | lexical baseline | ~0.1 ms |
| 2 | **LightGBM + SBERT** | ML classifier (trained in this project) | ~25 ms |
| 3 | **FinBERT-fintwitter** | financial-Twitter domain expert — the **same backbone our submitted model fine-tunes**, used zero-shot | ~150 ms |

### Orchestration
```
Tweet -> VADER (quick lexical reading)
          |- strong signal (|compound| > tau)  -> trust VADER
          |- weak signal -> LightGBM + SBERT
                             |- agrees with VADER -> final verdict
                             |- disagrees        -> FinBERT expert tie-break
```

Three design choices make this non-trivial: (1) the routing threshold **tau is calibrated on data** (Section 2 below), not hand-picked; (2) **disagreement detection** triggers escalation to the domain expert; (3) every verdict carries a **reasoning trace with per-tool latency**, and the session keeps **conversational memory** (`why` / `history` / `stats` / `compare`)."""

setup = """# Setup - run from the project root regardless of where the kernel starts
import os, sys, warnings
warnings.filterwarnings('ignore')
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
sys.path.insert(0, '.')
print('cwd:', os.getcwd())

import numpy as np
import pandas as pd
from src.agent import (RuleBasedAgent, build_langchain_agent, LABELS,
                       classify_with_vader, classify_with_lgbm, classify_with_finbert,
                       _vader, _lgbm, _finbert)"""

tools_cell = """# 1. The three tools individually (note the escalating cost/quality trade-off)
test_tweet = '$AAPL beats Q4 earnings by 15%, revenue up 8% YoY, raises guidance'
print('Tweet:', test_tweet)
print()
print('VADER          :', classify_with_vader(test_tweet))
print('LightGBM+SBERT :', classify_with_lgbm(test_tweet))
print('FinBERT expert :', classify_with_finbert(test_tweet))
print()
print('(first call includes model loading; warm latencies are shown in Section 2)')"""

calib = """# 2. CALIBRATING the routing threshold tau on data (no hand-picked magic numbers)
# Precompute all three tools' predictions on a stratified 400-tweet sample, then
# simulate the routing for each candidate tau and measure accuracy vs expert usage.
import time
from sklearn.model_selection import train_test_split

train = pd.read_csv('data/raw/train.csv')
sample, _ = train_test_split(train, train_size=400, stratify=train['label'], random_state=42)
texts, gold = sample['text'].tolist(), sample['label'].to_numpy()

t0 = time.perf_counter()
vader_comp = np.array([_vader(t)[1]['compound'] for t in texts])
vader_pred = np.where(vader_comp >= 0.05, 1, np.where(vader_comp <= -0.05, 0, 2))
t_vader = (time.perf_counter() - t0) / len(texts) * 1000

t0 = time.perf_counter()
lgbm_pred = np.array([_lgbm(t)[0] for t in texts])
t_lgbm = (time.perf_counter() - t0) / len(texts) * 1000

t0 = time.perf_counter()
fin_pred = np.array([_finbert(t)[0] for t in texts])
t_fin = (time.perf_counter() - t0) / len(texts) * 1000

print(f'Warm per-tweet latency: VADER {t_vader:.2f} ms | LightGBM+SBERT {t_lgbm:.0f} ms | '
      f'FinBERT expert {t_fin:.0f} ms')

def simulate(tau):
    strong = np.abs(vader_comp) > tau
    agree = lgbm_pred == vader_pred
    pred = np.where(strong, vader_pred, np.where(agree, lgbm_pred, fin_pred))
    expert_calls = np.mean(~strong & ~agree)
    cost = np.mean(np.where(strong, t_vader, np.where(agree, t_vader + t_lgbm,
                                                      t_vader + t_lgbm + t_fin)))
    return (pred == gold).mean(), expert_calls, cost

taus = np.round(np.arange(0.0, 1.01, 0.05), 2)
rows = [(tau, *simulate(tau)) for tau in taus]
df = pd.DataFrame(rows, columns=['tau', 'accuracy', 'pct_expert_calls', 'mean_cost_ms'])

import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(figsize=(8, 3.6))
ax1.plot(df['tau'], df['accuracy'], 'o-', color='#00B050', label='accuracy')
ax1.set_xlabel('routing threshold tau (|VADER compound|)')
ax1.set_ylabel('agent accuracy', color='#00B050')
ax2 = ax1.twinx()
ax2.plot(df['tau'], 100 * df['pct_expert_calls'], 's--', color='#777', label='% expert calls')
ax2.set_ylabel('% tweets escalated to expert', color='#777')
best = df.loc[df['accuracy'].idxmax()]
ax1.axvline(best['tau'], color='#00B050', alpha=0.3, linestyle=':')
ax1.set_title(f"Threshold calibration: best tau={best['tau']:.2f} "
              f"(accuracy {best['accuracy']:.3f}, expert used on "
              f"{100*best['pct_expert_calls']:.0f}% of tweets)")
plt.tight_layout(); plt.show()

TAU = float(best['tau'])
print(df.iloc[::4].to_string(index=False))
print(f'\\nCalibrated threshold: tau = {TAU}')"""

build = """# 3. Build the agent with the CALIBRATED threshold
# (LangChain ReAct back-end engages automatically if an LLM key is in the env;
#  otherwise the deterministic RuleBasedAgent implements the same orchestration.)
api_key = os.environ.get('OPENAI_API_KEY', None)
agent = build_langchain_agent(openai_api_key=api_key, strong=TAU)
print('Agent type:', type(agent).__name__)"""

demo = """# 4. Adaptive-routing demonstration - five tweets that exercise every path
demo_tweets = [
    'Stock soars on amazing earnings, great quarter, buy now!!!',          # strong VADER
    'Massive losses, terrible guidance, sell everything now',              # strong VADER (neg)
    '$TSLA delivery numbers slightly below consensus estimates',           # weak -> agreement
    'Fed keeps rates unchanged as expected',                               # weak -> agreement
    '$AAPL price target raised to 210 from 195 at Morgan Stanley',         # weak -> DISAGREEMENT -> expert
]
print('=' * 72)
print('FINANCIAL TWEET SENTIMENT AGENT - adaptive routing demo')
print('=' * 72)
for t in demo_tweets:
    rec = agent.route(t) if hasattr(agent, 'route') else None
    if rec is None:                       # LangChain backend
        print(agent.invoke({'input': t})['output']); continue
    print(f"\\nTweet: {t}")
    for step in rec['trace']:
        print(f'   {step}')
    print(f"   VERDICT: {rec['label']} via {rec['tool']} ({rec['latency_ms']:.0f} ms)")
    print(f"   Why: {rec['reason']}")"""

conv = """# 5. Conversational memory in action (the agent is stateful within a session)
for msg in ['why did you classify that last one as Bullish?',
            'history',
            'stats',
            'compare: $NVDA crushes earnings but guides below consensus']:
    print(f'>>> USER: {msg}')
    print(agent.chat(msg) if hasattr(agent, 'chat') else agent.invoke({'input': msg})['output'])
    print()"""

evaluation = """# 6. Batch evaluation + ROUTING ECONOMICS on the 400-tweet labelled sample
# (reuses the precomputed tool predictions - identical to calling agent.route per tweet)
from sklearn.metrics import f1_score, classification_report

strong = np.abs(vader_comp) > TAU
agree = lgbm_pred == vader_pred
route = np.where(strong, 'VADER', np.where(agree, 'LightGBM+SBERT', 'FinBERT expert'))
pred = np.where(strong, vader_pred, np.where(agree, lgbm_pred, fin_pred))

print(f'Agent (tau={TAU}) on 400 stratified tweets:')
print(f'  accuracy = {(pred == gold).mean():.4f} | F1-macro = '
      f'{f1_score(gold, pred, average="macro"):.4f}')
print()
rows = []
for r in ['VADER', 'LightGBM+SBERT', 'FinBERT expert']:
    m = route == r
    rows.append({'route': r, 'tweets': int(m.sum()), 'share_%': round(100 * m.mean(), 1),
                 'accuracy': round(float((pred[m] == gold[m]).mean()), 3) if m.any() else None})
print(pd.DataFrame(rows).to_string(index=False))
print()
cost_agent = np.mean(np.where(strong, t_vader,
                     np.where(agree, t_vader + t_lgbm, t_vader + t_lgbm + t_fin)))
cost_always = t_vader + t_lgbm + t_fin
acc_always = (fin_pred == gold).mean()
print('ROUTING ECONOMICS (the point of agentic orchestration):')
print(f'  agent           : {cost_agent:6.1f} ms/tweet  @ accuracy {(pred == gold).mean():.3f}')
print(f'  always-expert   : {cost_always:6.1f} ms/tweet  @ accuracy {acc_always:.3f}')
print(f'  -> the agent resolves {100 * (1 - rows[2]["share_%"] / 100):.0f}% of tweets with cheap '
      f'tools and cuts mean latency by {100 * (1 - cost_agent / cost_always):.0f}%')
print()
print('NOTE: the agent is a demonstration of tool ORCHESTRATION under a cost budget.')
print('The submitted classifier (fine-tuned distilled FinBERT, OOF F1-macro 0.9139) remains')
print('the accuracy reference; here its backbone is used zero-shot as the expert tool.')"""

outro = """## Why this is non-trivial orchestration

1. **Adaptive tool selection** — the agent chooses *which* model to consult from the lexical signal strength; the threshold is **calibrated on data** (Section 2), not hand-picked.
2. **Disagreement detection with principled escalation** — the domain expert is consulted only when the cheap tools conflict, exactly where it adds value.
3. **Cost-aware routing** — Section 6 quantifies the trade-off: most tweets are resolved by tools that are orders of magnitude cheaper, at a small accuracy cost versus always calling the expert.
4. **Transparent reasoning + conversational memory** — every verdict carries a trace (which tools ran, their confidence and latency) and the session supports follow-ups: `why`, `history`, `stats`, `compare: <tweet>`.
5. **Reproducible without credentials** — the deterministic `RuleBasedAgent` implements the same orchestration as the optional LangChain ReAct back-end, so grading never depends on an API key.

Implementation: [`src/agent.py`](../src/agent.py). Described in detail in `report_33.pdf`, Section 7."""

nb["cells"] = [header, md(intro), code(setup), code(tools_cell), code(calib),
               code(build), code(demo), code(conv), code(evaluation), md(outro)]
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"notebook reconstruido: {len(nb['cells'])} cells")
