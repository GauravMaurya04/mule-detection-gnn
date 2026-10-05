#!/usr/bin/env bash
# Run IBM's Multi-GNN reference (Multi-GIN with edge updates, as in their README) in its own
# conda env, logging to runs/edge/multignn/. Extra arguments go straight to their main.py,
# e.g.  scripts/run_multignn.sh --n_epochs 2     (check names first: python main.py --help)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IBM="$ROOT/external/multignn"
DATA="$ROOT/data/multignn"
OUT="$ROOT/runs/edge/multignn"
CONDA="${CONDA_EXE:-$HOME/miniforge3/bin/conda}"
mkdir -p "$DATA/Small_HI" "$OUT"

# 1. IBM's own pre-processing, once. It may write next to its input, so look in both places.
if [ ! -f "$DATA/Small_HI/formatted_transactions.csv" ]; then
  (cd "$DATA/Small_HI" && "$CONDA" run -n multignn python "$IBM/format_kaggle_files.py" \
      "$ROOT/data/raw/HI-Small_Trans.csv")
  if [ -f "$ROOT/data/raw/formatted_transactions.csv" ]; then
    mv "$ROOT/data/raw/formatted_transactions.csv" "$DATA/Small_HI/"
  fi
fi
test -f "$DATA/Small_HI/formatted_transactions.csv" || {
  echo "formatted_transactions.csv not found; check where format_kaggle_files.py wrote it"; exit 1; }

# 2. Point their data_config.json at our folder (every "aml_data" key, wherever it sits).
"$CONDA" run -n multignn python - "$IBM/data_config.json" "$DATA" << 'PY'
import json, sys
path, data = sys.argv[1], sys.argv[2]
cfg = json.load(open(path))
def walk(d):
    for k, v in d.items():
        if k == "aml_data":
            d[k] = data
        elif isinstance(v, dict):
            walk(v)
walk(cfg)
json.dump(cfg, open(path, "w"), indent=4)
print("aml_data ->", data)
PY

# 3. Train. W&B is switched off; their per-epoch F1 lines land in train.log.
cd "$IBM"
start=$(date +%s)
WANDB_MODE=disabled "$CONDA" run --no-capture-output -n multignn \
  python main.py --data Small_HI --model gin --emlps --reverse_mp --ego --ports "$@" \
  2>&1 | tee "$OUT/train.log"
echo $(( $(date +%s) - start )) > "$OUT/seconds.txt"
echo "Done. Import with: mule import-multignn"
