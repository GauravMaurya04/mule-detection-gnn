# Recall by laundering pattern

"none" = laundering that belongs to no pattern attempt (about 38% of all laundering).

## Transactions, ens_xgb_tuned__gnn_sage, window test (validation threshold)

| pattern | transactions | recall |
|---|---|---|
| BIPARTITE | 66 | 0.7879 |
| CYCLE | 99 | 0.9091 |
| FAN-IN | 123 | 0.8862 |
| FAN-OUT | 131 | 1 |
| GATHER-SCATTER | 381 | 0.9816 |
| RANDOM | 84 | 0.9048 |
| SCATTER-GATHER | 239 | 0.9833 |
| STACK | 117 | 0.8889 |
| none | 371 | 0.008086 |

## Accounts, ens_xgb_graph__gnn_sage, window test (recall inside the top R)

| pattern | mules | recall@R |
|---|---|---|
| BIPARTITE | 14 | 0.7143 |
| CYCLE | 91 | 0.6264 |
| FAN-IN | 13 | 0.4615 |
| FAN-OUT | 14 | 0.6429 |
| GATHER-SCATTER | 49 | 0.7959 |
| RANDOM | 76 | 0.8026 |
| SCATTER-GATHER | 148 | 0.7838 |
| STACK | 85 | 0.6353 |
| none | 10 | 0.2 |

## Transactions, ens_xgb_tuned__gnn_sage, window test_no_tail (validation threshold)

| pattern | transactions | recall |
|---|---|---|
| BIPARTITE | 44 | 0.8409 |
| CYCLE | 55 | 0.8545 |
| FAN-IN | 57 | 0.8596 |
| FAN-OUT | 65 | 0.9846 |
| GATHER-SCATTER | 127 | 0.937 |
| RANDOM | 43 | 0.8372 |
| SCATTER-GATHER | 110 | 0.9636 |
| STACK | 84 | 0.8333 |
| none | 371 | 0.005391 |

## Accounts, ens_xgb_graph__gnn_sage, window test_no_tail (recall inside the top R)

| pattern | mules | recall@R |
|---|---|---|
| BIPARTITE | 12 | 0.5833 |
| CYCLE | 47 | 0.5106 |
| FAN-IN | 8 | 0.5 |
| FAN-OUT | 10 | 0.7 |
| GATHER-SCATTER | 25 | 0.76 |
| RANDOM | 38 | 0.7368 |
| SCATTER-GATHER | 63 | 0.7778 |
| STACK | 60 | 0.55 |
| none | 8 | 0.125 |
