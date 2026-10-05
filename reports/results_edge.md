# Results: transaction task

Mean ± std over seeds. Thresholds are chosen on validation and frozen for test. Test windows appear only for models evaluated with --final.

## Window: val

| model | seeds | PR-AUC | minority F1 | precision | recall | positives |
|---|---|---|---|---|---|---|
| rule_ach | 1 | 0.014 | 0.048 | 0.025 | 0.415 | 1,036 |
| rules | 1 | 0.027 | 0.088 | 0.073 | 0.111 | 1,036 |
| xgb | 3 | 0.590 ± 0.001 | 0.651 ± 0.003 | 0.886 ± 0.020 | 0.514 ± 0.010 | 1,036 |

## Window: test

| model | seeds | PR-AUC | minority F1 | precision | recall | positives |
|---|---|---|---|---|---|---|
| rule_ach | 1 | 0.016 | 0.041 | 0.022 | 0.452 | 1,611 |
| rules | 1 | 0.048 | 0.099 | 0.076 | 0.145 | 1,611 |
| xgb | 3 | 0.720 ± 0.002 | 0.722 ± 0.003 | 0.822 ± 0.024 | 0.645 ± 0.013 | 1,611 |

## Window: test_no_tail

| model | seeds | PR-AUC | minority F1 | precision | recall | positives |
|---|---|---|---|---|---|---|
| rule_ach | 1 | 0.008 | 0.022 | 0.011 | 0.383 | 956 |
| rules | 1 | 0.015 | 0.040 | 0.027 | 0.078 | 956 |
| xgb | 3 | 0.546 ± 0.002 | 0.579 ± 0.003 | 0.716 ± 0.038 | 0.487 ± 0.013 | 956 |
