# Phase 0 report

## 1. Dataset facts

- Transactions: 5,078,345
- Time range: 2022-09-01 00:00:00 to 2022-09-18 16:18:00
- Calendar days with transactions: 18
- Accounts (bank, account pairs): 515,088
- Laundering transactions: 5,177 (1 per 981)
- Self-transfers: 591,212
- Cross-bank share: 86.4%

Transactions per day:

| date | transactions | laundering |
|---|---|---|
| 2022-09-01 | 1,114,921 | 322 |
| 2022-09-02 | 754,449 | 408 |
| 2022-09-03 | 207,382 | 391 |
| 2022-09-04 | 207,430 | 407 |
| 2022-09-05 | 482,650 | 471 |
| 2022-09-06 | 482,089 | 531 |
| 2022-09-07 | 482,751 | 497 |
| 2022-09-08 | 482,773 | 539 |
| 2022-09-09 | 654,467 | 514 |
| 2022-09-10 | 208,325 | 442 |
| 2022-09-11 | 396 | 232 |
| 2022-09-12 | 281 | 170 |
| 2022-09-13 | 184 | 106 |
| 2022-09-14 | 121 | 70 |
| 2022-09-15 | 46 | 28 |
| 2022-09-16 | 46 | 26 |
| 2022-09-17 | 23 | 15 |
| 2022-09-18 | 11 | 8 |

## 2. Payment formats

| payment_format | transactions | laundering | share | laundering_rate |
|---|---|---|---|---|
| Cheque | 1,864,331 | 324 | 36.7% | 1 per 5,754 |
| Credit Card | 1,323,324 | 206 | 26.1% | 1 per 6,424 |
| ACH | 600,797 | 4,483 | 11.8% | 1 per 134 |
| Cash | 490,891 | 108 | 9.7% | 1 per 4,545 |
| Reinvestment | 481,056 | 0 | 9.5% | none |
| Wire | 171,855 | 0 | 3.4% | none |
| Bitcoin | 146,091 | 56 | 2.9% | 1 per 2,609 |

## 3. Currencies and derived USD rates

| currency | usd_per_unit | paid_transactions |
|---|---|---|
| US Dollar | 1 | 1,895,172 |
| Euro | 1.172 | 1,168,297 |
| Swiss Franc | 1.093 | 234,860 |
| Yuan | 0.1493 | 213,752 |
| Shekel | 0.2961 | 192,184 |
| Rupee | 0.01362 | 190,202 |
| UK Pound | 1.292 | 180,738 |
| Yen | 0.009488 | 155,209 |
| Ruble | 0.01285 | 155,178 |
| Bitcoin | 11,874 | 146,066 |
| Canadian Dollar | 0.758 | 140,042 |
| Australian Dollar | 0.7078 | 136,769 |
| Mexican Peso | 0.0473 | 110,159 |
| Saudi Riyal | 0.2666 | 89,014 |
| Brazil Real | 0.1771 | 70,703 |

- Cross-currency transactions checked: 72,170
- USD paid / USD received, median: 1
- USD paid / USD received, 1st-99th percentile: 0.9825 to 1.005

A median near 1 with a narrow 1-99% range means the derived rates are consistent.

## 4. Accounts file (entities)

- Rows: 518,581
- Transaction accounts found in the accounts file: 100.0%
- Entities: 166,207
- Accounts per entity: mean / max: 3.12 / 7820
- Entities with more than one account: 63,518

Entity types parsed from entity names (top 10):

| entity_type | accounts |
|---|---|
| Partnership | 189,683 |
| Corporation | 172,351 |
| Sole Proprietorship | 149,048 |
| Country | 6,692 |
| Individual | 740 |
| Direct | 67 |

## 5. Laundering patterns file

| pattern | attempts | transactions |
|---|---|---|
| CYCLE | 54 | 287 |
| GATHER-SCATTER | 51 | 716 |
| BIPARTITE | 49 | 263 |
| FAN-OUT | 48 | 342 |
| SCATTER-GATHER | 44 | 626 |
| STACK | 43 | 466 |
| RANDOM | 41 | 191 |
| FAN-IN | 40 | 318 |

- Pattern rows matched to a transaction: 100.0%
- Pattern rows with more than one candidate match: 0
- Matched pattern rows labelled laundering: 100.0%
- Laundering transactions covered by the patterns file: 62.0%

## 6. Provisional time split and mule counts per window

Cut by transaction share 60% / 20% / 20%; Day 3 swaps in IBM's exact cutoffs.

| window | from | to | transactions | laundering_tx | active_accounts | mules | mule_rate | adjacent | unseen_accounts | unseen_mules |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 2022-09-01 00:00:00 | 2022-09-06 13:35:00 | 3,046,861 | 2,297 | 512,713 | 266 | 1 per 1,927 | 2,904 | - | - |
| val | 2022-09-06 13:36:00 | 2022-09-08 16:11:00 | 1,015,602 | 1,082 | 236,106 | 137 | 1 per 1,723 | 1,396 | 1,183 | 1 |
| test | 2022-09-08 16:12:00 | 2022-09-18 16:18:00 | 1,015,882 | 1,798 | 356,396 | 303 | 1 per 1,176 | 1,898 | 1,192 | 8 |

Candidate K values for precision@K on the test window:

| K | accounts | best_possible_precision | best_possible_recall |
|---|---|---|---|
| fixed 100 | 100 | 1 | 0.33 |
| fixed 500 | 500 | 0.606 | 1 |
| 0.1% of active | 356 | 0.8511 | 1 |
| 0.5% of active | 1,782 | 0.17 | 1 |
| 1.0% of active | 3,564 | 0.08502 | 1 |

Pick K values whose best possible precision and recall both stay meaningful.

## 7. Whole-period labels (reference only, never used for training)

- Active accounts: 515,088
- Mules: 899 (1 per 573)
- Laundering-adjacent: 5,458

## 8. Forwarding delays (sets X hours for the pass-through features)

Any incoming transfer to the account's next outgoing transfer, in minutes:

| incoming transfers into | count | never forwarded | p10 min | p25 min | p50 min | p75 min | p90 min |
|---|---|---|---|---|---|---|---|
| mule | 13,510 | 11.8% | 86 | 277 | 869 | 3,240 | 7,563 |
| non-mule | 4,473,623 | 52.7% | 78 | 243 | 627 | 1,358 | 3,551 |

Mules only: laundering receipt to the next laundering send:

| <= 1 h | <= 6 h | <= 24 h | <= 72 h |
|---|---|---|---|
| 2.6% | 17.1% | 45.0% | 88.9% |

Choose X where mules clearly separate from non-mules but before the curve flattens.
