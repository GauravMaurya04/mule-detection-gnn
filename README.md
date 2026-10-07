# mule-detection-gnn

Money-mule detection on IBM's AMLworld HI-Small data (5,078,345 synthetic transactions,
515,088 accounts), framed the way RBIH's MuleHunter frames it: **score accounts**, not only
transactions. Built and evaluated in 9 days on one laptop (i5-13th gen, 16 GB, RTX 4050 6 GB).

Two tasks share one transaction graph:

- **Accounts (primary):** is this account a mule? A mule is an account that, inside a time
  window, sends on laundered money it received earlier ("rule B", defined below).
- **Transactions (benchmark):** is this transfer laundering? Directly comparable with IBM's
  published Multi-GNN results on the same split.

Both champions are **XGBoost + GraphSAGE ensembles**, chosen on validation before their single
test evaluation. Every number below can be rebuilt with `scripts/reproduce.sh`.

## Results (test window, Sep 9–18 2022)

### Accounts

| Model | PR-AUC (95% CI) | Without tail | R-precision | First-time mule recall | P@100 |
|---|---|---|---|---|---|
| Hand-written rules | 0.179 | 0.126 | 0.299 | 0.308 | 0.558 |
| XGBoost, 32 features (v1) | 0.675 | 0.581 | 0.659 | 0.662 | 0.953 |
| XGBoost + graph features | 0.748 | 0.638 | 0.705 | 0.703 | 0.957 |
| GraphSAGE alone | 0.662 | 0.549 | 0.647 | 0.659 | 0.930 |
| GraphSAGE, network structure only | 0.081 | 0.056 | 0.166 | 0.171 | 0.353 |
| **Champion: 0.7 XGBoost + 0.3 GraphSAGE** | **0.761** (0.721–0.798) | **0.657** | **0.713** | **0.710** | **0.960** |

Base rate: about 1 mule per 840 active accounts in the test window, so random PR-AUC ≈ 0.001.

### Transactions

| Model | PR-AUC | Minority F1 | Recall | F1 at 0.5 (IBM protocol) | F1 without tail |
|---|---|---|---|---|---|
| Hand-written rules | 0.048 | 0.099 | 0.145 | – | 0.040 |
| XGBoost (v1) | 0.720 | 0.722 | 0.645 | 0.506 | 0.579 |
| XGBoost + pattern features, tuned | 0.751 | 0.779 | 0.714 | 0.222 | 0.634 |
| **Champion: XGBoost (tuned) + GraphSAGE** | **0.773** | **0.786** | **0.729** | **0.764** | **0.654** |
| Multi-GIN+EU, IBM paper | – | – | – | 0.648 ± 0.012 | – |
| Multi-PNA+EU, IBM paper | – | – | – | 0.682 ± 0.026 | – |
| Multi-GIN+EU, our 2-epoch run (undertrained) | – | – | – | 0.508 | – |

Paper numbers: Egressy et al., *Provably Powerful Graph Neural Networks for Directed
Multigraphs* (AAAI 2024), AML Small HI. Same split and test transactions. Our pattern features
encode the simulator's laundering shapes, which IBM's models learn from raw transfers; see the
model card before reading this as a like-for-like win.

Full tables (validation, test, no-tail; every model and ablation): `reports/results_node.md`,
`reports/results_edge.md`.

## What the results say

- **Engineered behaviour beats raw structure.** GraphSAGE on the 55 features reaches 0.659
  validation PR-AUC; on network structure alone, 0.150. XGBoost on the same features: 0.704.
- **The GNN still earns its place.** Alone it is weaker on both tasks, but it catches
  different cases: the ensembles beat XGBoost on validation (paired bootstrap) and on test.
- **Edge direction did not help.** GINE with reverse message passing lost to direction-blind
  GraphSAGE on both tasks once neighbour behaviour was engineered into the features.
- **Pattern features drove the transaction gain** (validation PR-AUC 0.596 → 0.638), as the
  IBM paper's own graph-feature baseline suggested.
- **The remaining misses are lone laundering.** Pattern laundering is caught 79–100% of the
  time; laundering that belongs to no pattern (23% of test laundering), 0.8%.
- **The tail matters.** After Sep 10 the generator keeps laundering chains running while normal
  activity stops (59% of those transfers are laundering). Every test result is also reported
  without it; transaction F1 drops 0.786 → 0.654.
- **Robust to slower mules.** Delaying mules' laundering sends by 48 h moves the account
  champion's PR-AUC from 0.761 to 0.758 (the GNN member alone: 0.689 → 0.647).

## Design decisions

- **Split:** IBM's own day-based search, reproduced and confirmed against its log: train
  Sep 1–6, validation Sep 7–8, test Sep 9–18. Training rows come from three 2-day snapshots, the
  shape of the evaluation windows.
- **Mule label (rule B):** the send must fall inside the window; the receipt may come earlier.
  The stricter both-inside rule lost ~21% of mules at window boundaries.
- **Features:** 55 per account (volume, counterparties, pass-through, channels, cross-bank,
  graph, entity, laundering patterns), computed point-in-time over the window plus 72 hours.
  Tests check that nothing after a window's end and no label is ever read.
- **No absolute timestamps** in any model input: test timestamps lie beyond training, and the
  tail would reward exploiting that.
- **Model choice on validation only**, by paired bootstrap of the PR-AUC difference; if the
  95% interval includes zero, the model with fewer features stays. Test is scored once per
  model, behind a `--final` flag, and recorded results cannot be overwritten.
- **GNNs in plain PyTorch** with a custom neighbour sampler: 0.9 s per epoch on the laptop GPU
  (IBM's code: about 2.3 h). Messages are averaged, not summed, so sampled training and
  full-neighbourhood scoring see the same scale.
- **Explanations follow the score.** Each account's added risk is split between the two
  members; SHAP gives the reason codes, the GNN gives the evidence transfers and, when it
  supplies more than its 0.3 weight, code RC08.

## Explanations

```
$ mule why -a 12719:807214430
12719:807214430: score 0.9836, rank 1 of 354,590 in test
GNN share of added risk: 29%
Reason codes:
  RC07 ACH-heavy: Used ACH for 100% of transfers (typical: 4%)
  RC01 Pass-through: Passed on 100% of incoming transfers within 3 days
  RC04 Near-zero balance: Kept almost nothing: money in and out balance within 1%
Evidence transfers (from the network model):
  2022-09-10 10:37:00  out      9,044.70 USD  ACH          214529:80567CFE0
  2022-09-10 15:44:00  out      8,692.16 USD  ACH          25075:804E71900
  2022-09-08 19:15:00   in      9,300.86 USD  ACH          18067:80307A9E0
  ...
```

Checks on the top 200 validation accounts: removing the 5 evidence transfers lowers the GNN's
score about 10 times more than removing 5 random ones; for true mules, evidence transfers are
laundering 38% of the time against 20% in their neighbourhoods; two explainer runs share 37% of
evidence transfers; 87% of accounts get at least one catalog code. Test: `reports/explanations_test.md`.

## Quick start

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.lock --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu124 \
  -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
uv pip install -e .
pytest -q                                  # 81 tests (14 need PyTorch)

# put HI-Small_Trans.csv, HI-Small_accounts.csv, HI-Small_Patterns.txt in data/raw/
mule ingest && mule labels && mule features
mule baselines --seeds 1 -m xgb            # v1 baseline in ~5 min
mule why -a BANK:ACCOUNT                   # after the champions exist
scripts/reproduce.sh                       # everything, ~5–6 h
```

IBM Multi-GNN reference run (own conda env): `scripts/run_multignn.sh --n_epochs 2`, then
`mule import-multignn`. MLflow runs: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Layout

```
configs/      base.yaml (every setting), reason_codes.yaml
src/mule/     data/ labels/ features/ models/ (xgb, rules, gnn/) eval/ explain/ cli.py
tests/        81 tests: labels, leakage, features, patterns, models, GNNs, explanations
scripts/      reproduce.sh, run_multignn.sh
reports/      every results table and analysis
```

Limitations, disclosures and intended use: [MODEL_CARD.md](MODEL_CARD.md).
