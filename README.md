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
```
