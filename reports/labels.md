# Labels report

## Split

- Method: IBM Multi-GNN day-based search
- Cut days: [6, 8]
- Actual shares of transactions: [0.6398, 0.1901, 0.1701]
- Training snapshots: train_s1, train_s2, train_s3
- Label rule: completed

## Windows

| window | from | to (excl.) | transactions | laundering_tx | active_accounts | mules | mules_rule_A | adjacent | repeat_mules | first_time_mules |
|---|---|---|---|---|---|---|---|---|---|---|
| train_s1 | 2022-09-01 | 2022-09-03 | 1,869,370 | 730 | 509,993 | 34 | 34 | 1,077 | 0 | 34 |
| train_s2 | 2022-09-03 | 2022-09-05 | 414,812 | 798 | 162,388 | 106 | 57 | 1,111 | 9 | 97 |
| train_s3 | 2022-09-05 | 2022-09-07 | 964,739 | 1,002 | 234,947 | 180 | 78 | 1,279 | 14 | 166 |
| val | 2022-09-07 | 2022-09-09 | 965,524 | 1,036 | 235,433 | 242 | 118 | 1,255 | 23 | 219 |
| test | 2022-09-09 | 2022-09-18 16:19:00 | 863,900 | 1,611 | 354,590 | 421 | 273 | 1,519 | 38 | 383 |
| test_no_tail | 2022-09-09 | 2022-09-11 | 862,792 | 956 | 354,447 | 222 | 90 | 1,175 | 31 | 191 |

- Training positive rows (all snapshots): 320
- Distinct mule accounts behind them: 297

mules_rule_A uses the stricter rule (receipt and send both inside the window). repeat_mules were also mules in an earlier window; test windows never count as earlier.

## K values per evaluation window

| window | K=100 | K=0.1% of active | R-precision |
|---|---|---|---|
| val | 100 | 235 | 242 |
| test | 100 | 355 | 421 |
| test_no_tail | 100 | 354 | 222 |
