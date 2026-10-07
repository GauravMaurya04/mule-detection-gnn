#!/usr/bin/env bash
# The full pipeline in the order the reported results were produced, from the raw Kaggle files
# in data/raw/ to every table in reports/. Times are for an i5-13th-gen laptop with an RTX 4050.
# Model choices between steps were made with `mule compare` on validation (see the spec/README);
# this script re-runs the chosen path rather than re-deciding it.
set -euo pipefail

mule ingest && mule labels && mule features                                # ~5 min
mule baselines --seeds 3 -m rule_ach -m rules -m xgb -m xgb_graph -m xgb_full \
  -m xgb_pattern -m xgb_pattern_full                                        # ~1 h

for m in sage gine gine_rmp; do mule gnn-tune -m "$m"; done                 # ~1.5 h
mule gnn-train -m sage --node-features minimal --params-from gnn_sage --seeds 3
mule gnn-edge-train -m sage --seeds 3
mule gnn-edge-train -m gine_rmp --seeds 3

mule tune-xgb -t node --base xgb_graph                                       # ~10 min
mule tune-xgb -t edge --base xgb_pattern                                     # ~30 min
mule stack                                                                   # ~15 min
mule ensemble -t node --a xgb_graph --b gnn_sage                             # account champion
mule ensemble -t node --a xgb_tuned --b gnn_sage
mule ensemble -t edge --a xgb_tuned --b gnn_sage                             # transaction champion
mule calibrate -t node -m ens_xgb_graph__gnn_sage
mule calibrate -t edge -m ens_xgb_tuned__gnn_sage
mule add-paper-rows

mule evaluate -t node --final
mule evaluate -t edge --final
mule explain -w val && mule explain -w test                                  # ~20 min
mule pattern-recall --final
mule stress --final
echo "Done: see reports/"
