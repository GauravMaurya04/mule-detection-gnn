# Results: transaction task

Mean ± std over seeds. Thresholds are chosen on validation and frozen for test. Test windows appear only for models evaluated with --final.

minority F1 uses a validation-chosen threshold. F1 @ 0.5 is IBM's protocol (argmax over two classes) and is the column comparable with Multi-GNN, which reports the test F1 at its best-validation epoch. Multi-GNN rows are imported from its own training log.

## Window: val

| model | seeds | PR-AUC | minority F1 | precision | recall | F1 @ 0.5 (IBM protocol) | positives |
|---|---|---|---|---|---|---|---|
| gnn_gine_rmp | 3 | 0.565 ± 0.004 | 0.614 ± 0.008 | 0.840 ± 0.009 | 0.484 ± 0.013 | 0.274 ± 0.028 | 1,036 |
| gnn_sage | 3 | 0.590 ± 0.002 | 0.635 ± 0.001 | 0.852 ± 0.011 | 0.507 ± 0.005 | 0.259 ± 0.013 | 1,036 |
| multignn | 1 | - | - | - | - | 0.331 | - |
| rule_ach | 1 | 0.014 | 0.048 | 0.025 | 0.415 | - | 1,036 |
| rules | 1 | 0.027 | 0.088 | 0.073 | 0.111 | - | 1,036 |
| xgb | 3 | 0.590 ± 0.001 | 0.651 ± 0.003 | 0.886 ± 0.020 | 0.514 ± 0.010 | 0.438 ± 0.020 | 1,036 |
| xgb_full | 3 | 0.596 ± 0.004 | 0.617 ± 0.004 | 0.902 ± 0.016 | 0.469 ± 0.004 | 0.409 ± 0.011 | 1,036 |
| xgb_graph | 3 | 0.590 ± 0.002 | 0.647 ± 0.004 | 0.887 ± 0.006 | 0.509 ± 0.005 | 0.407 ± 0.041 | 1,036 |
| xgb_pattern | 3 | 0.638 ± 0.003 | 0.689 ± 0.004 | 0.915 ± 0.010 | 0.553 ± 0.006 | 0.401 ± 0.010 | 1,036 |
| xgb_pattern_full | 3 | 0.634 ± 0.002 | 0.655 ± 0.002 | 0.902 ± 0.027 | 0.515 ± 0.011 | 0.381 ± 0.016 | 1,036 |

## Window: test

| model | seeds | PR-AUC | minority F1 | precision | recall | F1 @ 0.5 (IBM protocol) | positives |
|---|---|---|---|---|---|---|---|
| multignn | 1 | - | - | - | - | 0.508 | - |
| paper_multigin_eu | 5 | - | - | - | - | 0.648 ± 0.012 | - |
| paper_multipna_eu | 5 | - | - | - | - | 0.682 ± 0.026 | - |
| rule_ach | 1 | 0.016 | 0.041 | 0.022 | 0.452 | - | 1,611 |
| rules | 1 | 0.048 | 0.099 | 0.076 | 0.145 | - | 1,611 |
| xgb | 3 | 0.720 ± 0.002 | 0.722 ± 0.003 | 0.822 ± 0.024 | 0.645 ± 0.013 | 0.506 ± 0.030 | 1,611 |
| xgb_full | 3 | 0.736 ± 0.005 | 0.717 ± 0.001 | 0.887 ± 0.010 | 0.601 ± 0.005 | 0.498 ± 0.011 | 1,611 |
| xgb_graph | 3 | 0.706 ± 0.013 | 0.723 ± 0.009 | 0.819 ± 0.013 | 0.647 ± 0.007 | 0.402 ± 0.077 | 1,611 |
| xgb_pattern | 3 | 0.754 ± 0.002 | 0.766 ± 0.003 | 0.857 ± 0.013 | 0.694 ± 0.006 | 0.426 ± 0.012 | 1,611 |
| xgb_pattern_full | 3 | 0.761 ± 0.009 | 0.747 ± 0.006 | 0.879 ± 0.023 | 0.651 ± 0.008 | 0.474 ± 0.020 | 1,611 |

## Window: test_no_tail

| model | seeds | PR-AUC | minority F1 | precision | recall | F1 @ 0.5 (IBM protocol) | positives |
|---|---|---|---|---|---|---|---|
| rule_ach | 1 | 0.008 | 0.022 | 0.011 | 0.383 | - | 956 |
| rules | 1 | 0.015 | 0.040 | 0.027 | 0.078 | - | 956 |
| xgb | 3 | 0.546 ± 0.002 | 0.579 ± 0.003 | 0.716 ± 0.038 | 0.487 ± 0.013 | 0.338 ± 0.027 | 956 |
| xgb_full | 3 | 0.558 ± 0.006 | 0.576 ± 0.001 | 0.819 ± 0.021 | 0.445 ± 0.007 | 0.339 ± 0.009 | 956 |
| xgb_graph | 3 | 0.530 ± 0.013 | 0.580 ± 0.007 | 0.716 ± 0.018 | 0.488 ± 0.005 | 0.251 ± 0.062 | 956 |
| xgb_pattern | 3 | 0.582 ± 0.003 | 0.627 ± 0.003 | 0.775 ± 0.022 | 0.528 ± 0.006 | 0.272 ± 0.008 | 956 |
| xgb_pattern_full | 3 | 0.594 ± 0.009 | 0.611 ± 0.006 | 0.811 ± 0.041 | 0.491 ± 0.010 | 0.324 ± 0.017 | 956 |
