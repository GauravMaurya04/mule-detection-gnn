# mule-detection-gnn

Money-mule account detection on IBM AMLworld (HI-Small) with graph neural networks.
Full design: see the project spec doc.

## Quick start

```bash
source .venv/bin/activate
uv pip install -e .        # registers the `mule` command
pytest -q                  # unit + smoke tests (CPU, seconds)
mule ingest                # data/raw CSVs -> data/interim Parquet
mule phase0                # dataset facts, mule counts, delays -> reports/
mule labels                # split manifest + account labels -> data/processed, reports/labels.md
mule features              # Tier 1 account features per window -> data/processed
mule baselines             # rule_ach, rules, xgb on both tasks -> runs/, reports/results_*.md (val)
mule evaluate -t node --final   # adds test + no-tail metrics (only when you mean it)
mule score --top 10        # highest-scoring accounts in the test window
mule score -a BANK:ACCOUNT # one account with its strongest SHAP evidence
```

Single model: `mule train -t edge -m xgb --seeds 3`. MLflow runs live in `mlflow.db`;
browse them with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

```text
v1 = ingest -> labels -> features -> baselines -> evaluate --final
```
