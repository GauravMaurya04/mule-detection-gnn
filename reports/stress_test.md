# Stress test: mules' outgoing laundering delayed by 48 h (test)
- Mules in window: 421
- Laundering transfers delayed: 678

GNN member re-scored on the stressed graph.

| model | original | stressed | change |
|---|---|---|---|
| xgb_graph (XGBoost member) | 0.7524 | 0.7533 | 0.0008713 |
| gnn_sage (GNN member) | 0.6894 | 0.6465 | -0.04281 |
| ens_xgb_graph__gnn_sage (champion) | 0.761 | 0.7583 | -0.002705 |
