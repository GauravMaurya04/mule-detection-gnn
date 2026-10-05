# Labels report

## Split

- Method: IBM Multi-GNN day-based search
- Cut days: [6, 8]
- Actual shares of transactions: [0.6398, 0.1901, 0.1701]
- Label rule: completed

## Windows

| window | from | to (excl.) | transactions | laundering_tx | active_accounts | mules | mules_rule_A | adjacent | unseen_accounts | unseen_mules |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 2022-09-01 | 2022-09-07 | 3,248,921 | 2,530 | 512,907 | 297 | 297 | 3,174 | - | - |
| val | 2022-09-07 | 2022-09-09 | 965,524 | 1,036 | 235,433 | 242 | 118 | 1,255 | 1,115 | 0 |
| test | 2022-09-09 | 2022-09-18 16:19:00 | 863,900 | 1,611 | 354,590 | 421 | 273 | 1,519 | 1,066 | 6 |
| test_no_tail | 2022-09-09 | 2022-09-11 | 862,792 | 956 | 354,447 | 222 | 90 | 1,175 | 1,056 | 1 |

mules_rule_A uses the stricter rule (receipt and send both inside the window).

## K values per evaluation window

| window | K=100 | K=0.1% of active | R-precision |
|---|---|---|---|
| val | 100 | 235 | 242 |
| test | 100 | 355 | 421 |
| test_no_tail | 100 | 354 | 222 |
