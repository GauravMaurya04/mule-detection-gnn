# Audit measurements (validation only)

Scores are seed-averaged. Every threshold and operating point is chosen on validation and applied unchanged to other windows. small_sample = fewer than 30 positives: treat those rows as anecdotes.

## 1. Dataset

- transactions: 5,078,345
- accounts: 515,088
- laundering: 5,177
- duplicate_rows_all_fields: 18
- duplicate_groups_with_conflicting_label: 0
- self_transfers: 591,212
- laundering_self_transfers: 11
- rows_sharing_minute_with_same_sender: 722,640
- reinvestment_self_share: 1
- missing values: none
| window | transactions | laundering_tx | tx_prevalence (= no-skill PR-AUC) | accounts | mules | adjacent | account_prevalence (= no-skill PR-AUC) | ach_share_of_traffic | ach_share_of_laundering |
|---|---|---|---|---|---|---|---|---|---|
| train_s1 | 1,869,370 | 730 | 0.0003905 | 509,993 | 34 | 1,077 | 6.667e-05 | 0.1006 | 0.7137 |
| train_s2 | 414,812 | 798 | 0.001924 | 162,388 | 106 | 1,111 | 0.0006528 | 0.1422 | 0.9085 |
| train_s3 | 964,739 | 1,002 | 0.001039 | 234,947 | 180 | 1,279 | 0.0007661 | 0.1208 | 0.8563 |
| val | 965,524 | 1,036 | 0.001073 | 235,433 | 242 | 1,255 | 0.001028 | 0.1211 | 0.8591 |

## 2. Lone vs patterned laundering (shares of transfers)

| window | kind | transfers | self_transfer | ACH | sender_is_mule | sender_is_adjacent | receiver_is_mule | sender_got_laundering_earlier | sender_got_laundering_within_72h |
|---|---|---|---|---|---|---|---|---|---|
| train_s1 | lone | 466 | 0 | 0.5515 | 0.004292 | 0.9957 | 0.004292 | 0.004292 | 0.004292 |
| train_s1 | patterned | 264 | 0.003788 | 1 | 0.1856 | 0.8144 | 0.2008 | 0.1705 | 0.1705 |
| train_s2 | lone | 347 | 0 | 0.7896 | 0.008646 | 0.9914 | 0.008646 | 0.008646 | 0.008646 |
| train_s2 | patterned | 451 | 0.006652 | 1 | 0.3038 | 0.6962 | 0.2195 | 0.2993 | 0.2661 |
| train_s3 | lone | 405 | 0 | 0.6444 | 0.004938 | 0.9951 | 0.002469 | 0.004938 | 0.002469 |
| train_s3 | patterned | 597 | 0.00335 | 1 | 0.402 | 0.598 | 0.2496 | 0.402 | 0.268 |
| val | lone | 379 | 0 | 0.6148 | 0.02902 | 0.971 | 0 | 0.02639 | 0.005277 |
| val | patterned | 657 | 0 | 1 | 0.4734 | 0.5266 | 0.2374 | 0.4734 | 0.3212 |

## 3. Accounts: metrics (seed-averaged; threshold = max F1 on val)

Window val

| metric | ens_xgb_graph__gnn_sage | xgb_graph | gnn_sage | xgb_pattern |
|---|---|---|---|---|
| rows | 235,433 | 235,433 | 235,433 | 235,433 |
| positives | 242 | 242 | 242 | 242 |
| prevalence | 0.001028 | 0.001028 | 0.001028 | 0.001028 |
| pr_auc | 0.7261 | 0.7066 | 0.6761 | 0.7115 |
| roc_auc | 0.9957 | 0.9957 | 0.9914 | 0.9961 |
| P@K=10 | 1 | 0.8 | 0.9 | 0.9 |
| R@K=10 | 0.04132 | 0.03306 | 0.03719 | 0.03719 |
| P@K=25 | 0.92 | 0.92 | 0.92 | 0.92 |
| R@K=25 | 0.09504 | 0.09504 | 0.09504 | 0.09504 |
| P@K=50 | 0.94 | 0.94 | 0.96 | 0.92 |
| R@K=50 | 0.1942 | 0.1942 | 0.1983 | 0.1901 |
| P@K=100 | 0.89 | 0.88 | 0.88 | 0.9 |
| R@K=100 | 0.3678 | 0.3636 | 0.3636 | 0.3719 |
| P@K=500 | 0.41 | 0.398 | 0.394 | 0.404 |
| R@K=500 | 0.8471 | 0.8223 | 0.814 | 0.8347 |
| P@K=1000 | 0.221 | 0.22 | 0.212 | 0.22 |
| R@K=1000 | 0.9132 | 0.9091 | 0.876 | 0.9091 |
| P@K=0.1% | 0.6936 | 0.6766 | 0.6596 | 0.6936 |
| R@K=0.1% | 0.6736 | 0.657 | 0.6405 | 0.6736 |
| P@R | 0.6818 | 0.6736 | 0.6488 | 0.6736 |
| R@R | 0.6818 | 0.6736 | 0.6488 | 0.6736 |
| K_R | 242 | 242 | 242 | 242 |
| thr_threshold | 0.3644 | 0.3199 | 0.3551 | 0.2823 |
| thr_flagged | 197 | 205 | 278 | 205 |
| thr_tp | 153 | 154 | 172 | 156 |
| thr_fp | 44 | 51 | 106 | 49 |
| thr_fn | 89 | 88 | 70 | 86 |
| thr_tn | 235,147 | 235,140 | 235,085 | 235,142 |
| thr_accuracy | 0.9994 | 0.9994 | 0.9993 | 0.9994 |
| thr_precision | 0.7766 | 0.7512 | 0.6187 | 0.761 |
| thr_recall | 0.6322 | 0.6364 | 0.7107 | 0.6446 |
| thr_f1 | 0.697 | 0.689 | 0.6615 | 0.698 |
| thr_fpr | 0.0001871 | 0.0002168 | 0.0004507 | 0.0002083 |
| thr_fnr | 0.3678 | 0.3636 | 0.2893 | 0.3554 |
| brier | 0.0004697 | 0.0004914 | 0.0005699 | 0.0004952 |
| ece | 0.0001227 | 0.0003193 | 0.0003657 | 0.0003467 |
| log_loss | 0.001976 | 0.00226 | 0.002443 | 0.002278 |
| brier_no_skill | 0.001027 | 0.001027 | 0.001027 | 0.001027 |
| cal_slope | 1.026 | 0.8007 | 0.7895 | 0.79 |
| cal_intercept | -0.079 | -0.03464 | -0.7479 | -0.03508 |
| ece_top | 0.0001969 | 0.00027 | 0.0003518 | 0.0002827 |

## 4. Accounts: operating points chosen on val (ens_xgb_graph__gnn_sage)

Window val

| operating_point | threshold | flagged | tp | fp | precision | recall | f1 |
|---|---|---|---|---|---|---|---|
| A max F1 | 0.3644 | 197 | 153 | 44 | 0.7766 | 0.6322 | 0.697 |
| B precision >= 0.80 | 0.4899 | 168 | 135 | 33 | 0.8036 | 0.5579 | 0.6585 |
| C precision >= 0.85 | 0.6362 | 138 | 118 | 20 | 0.8551 | 0.4876 | 0.6211 |
| D precision >= 0.90 | 0.8485 | 90 | 81 | 9 | 0.9 | 0.3347 | 0.488 |
| E recall >= 0.50 | 0.5613 | 149 | 125 | 24 | 0.8389 | 0.5165 | 0.6394 |
| F recall >= 0.60 | 0.4004 | 186 | 146 | 40 | 0.7849 | 0.6033 | 0.6822 |
| G recall >= 0.70 | 0.2313 | 253 | 170 | 83 | 0.6719 | 0.7025 | 0.6869 |

## 5. Accounts: 95% intervals for ens_xgb_graph__gnn_sage (row vs cluster bootstrap)

Window val

| model | metric | ci_low | ci_high | resampling |
|---|---|---|---|---|
| ens_xgb_graph__gnn_sage | pr_auc | 0.6665 | 0.7842 | row |
| ens_xgb_graph__gnn_sage | P@100 | 0.81 | 0.95 | row |
| ens_xgb_graph__gnn_sage | R-precision | 0.6348 | 0.732 | row |
| ens_xgb_graph__gnn_sage | precision | 0.7194 | 0.841 | row |
| ens_xgb_graph__gnn_sage | recall | 0.5708 | 0.6951 | row |
| ens_xgb_graph__gnn_sage | f1 | 0.6455 | 0.7459 | row |
| ens_xgb_graph__gnn_sage | pr_auc | 0.6472 | 0.7899 | cluster |
| ens_xgb_graph__gnn_sage | P@100 | 0.77 | 0.96 | cluster |
| ens_xgb_graph__gnn_sage | R-precision | 0.6161 | 0.7372 | cluster |
| ens_xgb_graph__gnn_sage | precision | 0.6899 | 0.8363 | cluster |
| ens_xgb_graph__gnn_sage | recall | 0.5488 | 0.7012 | cluster |
| ens_xgb_graph__gnn_sage | f1 | 0.6253 | 0.75 | cluster |

## 6. Accounts: reliability of ens_xgb_graph__gnn_sage (equal-mass bins over the top scores)

Window val

| bin | rows | positives | mean_predicted | observed_rate |
|---|---|---|---|---|
| top 1/10 | 242 | 165 | 0.6684 | 0.6818 |
| top 2/10 | 242 | 39 | 0.1088 | 0.1612 |
| top 3/10 | 242 | 14 | 0.0306 | 0.05785 |
| top 4/10 | 242 | 2 | 0.01666 | 0.008264 |
| top 5/10 | 242 | 3 | 0.01128 | 0.0124 |
| top 6/10 | 242 | 5 | 0.008512 | 0.02066 |
| top 7/10 | 242 | 2 | 0.006742 | 0.008264 |
| top 8/10 | 242 | 1 | 0.00546 | 0.004132 |
| top 9/10 | 242 | 2 | 0.004594 | 0.008264 |
| top 10/10 | 242 | 0 | 0.003906 | 0 |
| rest | 233,013 | 9 | 0.0001077 | 3.862e-05 |

## 7. Accounts: ensemble audit (ens_xgb_graph__gnn_sage, weight on xgb_graph = 0.7)

Window val

- PR-AUC (seed-averaged, Platt): {'xgb_graph': 0.7066, 'gnn_sage': 0.6761, 'ens_xgb_graph__gnn_sage': 0.7261}
- ensemble gain over best member: 0.0195
- gain 95% CI (paired): [0.0014, 0.0415]
- Spearman (all rows / positives): 0.789 / 0.833
| both_caught | only_xgb_graph | only_gnn_sage | both_missed | ensemble_caught |
|---|---|---|---|---|
| 144 | 10 | 28 | 60 | 153 |

| both_flagged | only_xgb_graph | only_gnn_sage | ensemble_flagged |
|---|---|---|---|
| 39 | 12 | 67 | 44 |

| jaccard_of_top_R | mules_in_both_top_R | mules_only_in_xgb_graph_top_R | mules_only_in_gnn_sage_top_R | mules_in_neither |
|---|---|---|---|---|
| 0.6463 | 141 | 22 | 16 | 63 |

## 8. Accounts: pattern-level recall (ens_xgb_graph__gnn_sage)

Window val

| pattern | positives | share_of_positives | recall@val_threshold | false_negatives | share_of_all_caught | small_sample | recall@R |
|---|---|---|---|---|---|---|---|
| BIPARTITE | 6 | 0.02479 | 0.8333 | 1 | 0.03268 | 1 | 0.8333 |
| CYCLE | 58 | 0.2397 | 0.6552 | 20 | 0.2484 | 0 | 0.7241 |
| FAN-IN | 5 | 0.02066 | 0.8 | 1 | 0.02614 | 1 | 0.8 |
| FAN-OUT | 12 | 0.04959 | 1 | 0 | 0.07843 | 1 | 1 |
| GATHER-SCATTER | 20 | 0.08264 | 0.95 | 1 | 0.1242 | 1 | 0.95 |
| RANDOM | 37 | 0.1529 | 0.6757 | 12 | 0.1634 | 0 | 0.7568 |
| SCATTER-GATHER | 57 | 0.2355 | 0.8246 | 10 | 0.3072 | 0 | 0.8596 |
| STACK | 76 | 0.314 | 0.5263 | 36 | 0.2614 | 0 | 0.5658 |
| none | 11 | 0.04545 | 0 | 11 | 0 | 1 | 0.09091 |
| lone only | 10 | 0.04132 | 0 | 10 | 0 | 1 | 0 |

## 9. Accounts: by payment channel (ens_xgb_graph__gnn_sage)

Window val

| group | rows | positives | flagged | precision | recall | f1 | pr_auc_within_group | small_sample |
|---|---|---|---|---|---|---|---|---|
| dominant ach | 39,902 | 114 | 122 | 0.8607 | 0.9211 | 0.8898 | 0.8994 | 0 |
| dominant bitcoin | 7,772 | 0 | 0 | - | - | - | - | 1 |
| dominant cash | 21,112 | 17 | 10 | 0.8 | 0.4706 | 0.5926 | - | 1 |
| dominant cheque | 132,708 | 99 | 58 | 0.569 | 0.3333 | 0.4204 | 0.4412 | 0 |
| dominant credit_card | 33,692 | 12 | 7 | 1 | 0.5833 | 0.7368 | - | 1 |
| dominant none | 111 | 0 | 0 | - | - | - | - | 1 |
| dominant wire | 136 | 0 | 0 | - | - | - | - | 1 |
| ACH share >= 0.5 | 24,164 | 106 | 106 | 0.9245 | 0.9245 | 0.9245 | 0.9131 | 0 |
| ACH share < 0.5 | 211,269 | 136 | 91 | 0.6044 | 0.4044 | 0.4846 | 0.465 | 0 |

## 10. Accounts: familiar vs unseen entities (ens_xgb_graph__gnn_sage)

Window val

| group | rows | positives | pr_auc_within_group | P@100_within_group | precision | recall | f1 | small_sample |
|---|---|---|---|---|---|---|---|---|
| active in training | 234,318 | 242 | 0.7262 | 0.89 | 0.7766 | 0.6322 | 0.697 | 0 |
| never active in training | 1,115 | 0 | - | 0 | - | - | - | 1 |
| repeat mule (prior window) + all non-mules | 235,214 | 23 | 0.5794 | 0.21 | 0.3231 | 0.913 | 0.4773 | 1 |
| first-time mules + all non-mules | 235,410 | 219 | 0.6934 | 0.83 | 0.75 | 0.6027 | 0.6684 | 0 |
| most counterparties unseen in training (<50% seen) | 468 | 0 | - | 0 | - | - | - | 1 |
| most counterparties seen in training | 234,840 | 242 | 0.7261 | 0.89 | 0.7766 | 0.6322 | 0.697 | 0 |

## 11. Accounts: error analysis (ens_xgb_graph__gnn_sage, val threshold)

Window val: {'TN': 235147, 'TP': 153, 'FN': 89, 'FP': 44}

- false_positives_adjacent: 9
- false_positives_uninvolved: 35

median_profile

| outcome | vol_active_days | vol_in_per_day | vol_out_per_day | cp_log_unique_senders | cp_log_unique_receivers | pt_fwd_24h_share | pt_fwd_72h_share | pt_median_delay_h | pt_net_flow | ch_share_ach | pa_log_fan_in_24h_max | gr_coreness |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FN | 5 | 2.2 | 2.25 | 1.099 | 1.099 | 0.8824 | 1 | 7.475 | -0.283 | 0.1522 | 1.792 | 2 |
| FP | 5 | 1.8 | 2.325 | 1.099 | 1.099 | 1 | 1 | 8.896 | -0.07426 | 0.271 | 1.609 | 2 |
| TN | 5 | 2 | 1 | 0.6931 | 0.6931 | 0 | 0 | 9.45 | 0.8412 | 0 | 1.609 | 2 |
| TP | 4 | 1 | 1 | 1.099 | 0.6931 | 0.4545 | 1 | 25.83 | 0.01089 | 0.7241 | 1.099 | 2 |

fn_vs_tp

| outcome | mules | first_laundering_receipt_before_window | ach_share_median | repeat_mule_share |
|---|---|---|---|---|
| FN | 89 | 0.5393 | 0.1522 | 0.02247 |
| TP | 153 | 0.6078 | 0.7241 | 0.1373 |

## 13. Accounts: reconciliation with stored metrics

| model | window | metric | audit (seed-averaged) | stored (mean of seeds) |
|---|---|---|---|---|
| ens_xgb_graph__gnn_sage | val | pr_auc | 0.7261 | 0.7261 |
| ens_xgb_graph__gnn_sage | val | P@K=100 | 0.89 | 0.89 |
| ens_xgb_graph__gnn_sage | val | P@R | 0.6818 | 0.6818 |
| ens_xgb_graph__gnn_sage | val | thr_f1 | 0.697 | 0.697 |
| xgb_graph | val | pr_auc | 0.7066 | 0.7044 |
| xgb_graph | val | P@K=100 | 0.88 | 0.88 |
| xgb_graph | val | P@R | 0.6736 | 0.6749 |
| xgb_graph | val | thr_f1 | 0.689 | 0.6855 |
| gnn_sage | val | pr_auc | 0.6761 | 0.6593 |
| gnn_sage | val | P@K=100 | 0.88 | 0.8767 |
| gnn_sage | val | P@R | 0.6488 | 0.635 |
| gnn_sage | val | thr_f1 | 0.6615 | 0.6435 |
| xgb_pattern | val | pr_auc | 0.7115 | 0.7075 |
| xgb_pattern | val | P@K=100 | 0.9 | 0.89 |
| xgb_pattern | val | P@R | 0.6736 | 0.6763 |
| xgb_pattern | val | thr_f1 | 0.698 | 0.6946 |

## 3. Transactions: metrics (seed-averaged; threshold = max F1 on val)

Window val

| metric | ens_xgb_tuned__gnn_sage | xgb_tuned | gnn_sage | xgb_pattern |
|---|---|---|---|---|
| rows | 965,524 | 965,524 | 965,524 | 965,524 |
| positives | 1,036 | 1,036 | 1,036 | 1,036 |
| prevalence | 0.001073 | 0.001073 | 0.001073 | 0.001073 |
| pr_auc | 0.667 | 0.652 | 0.6115 | 0.6413 |
| roc_auc | 0.9886 | 0.9874 | 0.9871 | 0.988 |
| P@K=10 | 1 | 1 | 1 | 1 |
| R@K=10 | 0.009653 | 0.009653 | 0.009653 | 0.009653 |
| P@K=25 | 1 | 1 | 1 | 1 |
| R@K=25 | 0.02413 | 0.02413 | 0.02413 | 0.02413 |
| P@K=50 | 1 | 1 | 1 | 1 |
| R@K=50 | 0.04826 | 0.04826 | 0.04826 | 0.04826 |
| P@K=100 | 1 | 1 | 1 | 1 |
| R@K=100 | 0.09653 | 0.09653 | 0.09653 | 0.09653 |
| P@K=500 | 0.974 | 0.984 | 0.934 | 0.966 |
| R@K=500 | 0.4701 | 0.4749 | 0.4508 | 0.4662 |
| P@K=1000 | 0.64 | 0.63 | 0.59 | 0.629 |
| R@K=1000 | 0.6178 | 0.6081 | 0.5695 | 0.6071 |
| P@R | 0.6245 | 0.6091 | 0.5724 | 0.6091 |
| R@R | 0.6245 | 0.6091 | 0.5724 | 0.6091 |
| K_R | 1,036 | 1,036 | 1,036 | 1,036 |
| thr_threshold | 0.2043 | 0.6603 | 0.9417 | 0.71 |
| thr_flagged | 667 | 675 | 599 | 649 |
| thr_tp | 603 | 599 | 534 | 582 |
| thr_fp | 64 | 76 | 65 | 67 |
| thr_fn | 433 | 437 | 502 | 454 |
| thr_tn | 964,424 | 964,412 | 964,423 | 964,421 |
| thr_accuracy | 0.9995 | 0.9995 | 0.9994 | 0.9995 |
| thr_precision | 0.904 | 0.8874 | 0.8915 | 0.8968 |
| thr_recall | 0.582 | 0.5782 | 0.5154 | 0.5618 |
| thr_f1 | 0.7082 | 0.7002 | 0.6532 | 0.6908 |
| thr_fpr | 6.636e-05 | 7.88e-05 | 6.739e-05 | 6.947e-05 |
| thr_fnr | 0.418 | 0.4218 | 0.4846 | 0.4382 |
| brier | 0.0005049 | 0.00201 | 0.004853 | 0.002063 |
| ece | 0.0001832 | 0.01308 | 0.0184 | 0.009195 |
| log_loss | 0.002739 | 0.01552 | 0.02337 | 0.01185 |
| brier_no_skill | 0.001072 | 0.001072 | 0.001072 | 0.001072 |
| cal_slope | 1.021 | 1.578 | 1.111 | 1.341 |
| cal_intercept | -0.07935 | -2.428 | -4.018 | -2.637 |
| ece_top | 0.0001863 | 0.01305 | 0.0184 | 0.009193 |

## 4. Transactions: operating points chosen on val (ens_xgb_tuned__gnn_sage)

Window val

| operating_point | threshold | flagged | tp | fp | precision | recall | f1 |
|---|---|---|---|---|---|---|---|
| A max F1 | 0.2043 | 667 | 603 | 64 | 0.904 | 0.582 | 0.7082 |
| B precision >= 0.80 | 0.1348 | 778 | 623 | 155 | 0.8008 | 0.6014 | 0.6869 |
| C precision >= 0.85 | 0.1569 | 722 | 614 | 108 | 0.8504 | 0.5927 | 0.6985 |
| D precision >= 0.90 | 0.2007 | 671 | 604 | 67 | 0.9001 | 0.583 | 0.7077 |
| E recall >= 0.50 | 0.5054 | 536 | 518 | 18 | 0.9664 | 0.5 | 0.659 |
| F recall >= 0.60 | 0.1348 | 777 | 623 | 154 | 0.8018 | 0.6014 | 0.6873 |
| G recall >= 0.70 | 0.0511 | 2,503 | 726 | 1,777 | 0.2901 | 0.7008 | 0.4103 |

## 5. Transactions: 95% intervals for ens_xgb_tuned__gnn_sage (row vs cluster bootstrap)

Window val

| model | metric | ci_low | ci_high | resampling |
|---|---|---|---|---|
| ens_xgb_tuned__gnn_sage | pr_auc | 0.6393 | 0.6946 | row |
| ens_xgb_tuned__gnn_sage | P@100 | 1 | 1 | row |
| ens_xgb_tuned__gnn_sage | R-precision | 0.5958 | 0.6519 | row |
| ens_xgb_tuned__gnn_sage | precision | 0.8857 | 0.9222 | row |
| ens_xgb_tuned__gnn_sage | recall | 0.5519 | 0.612 | row |
| ens_xgb_tuned__gnn_sage | f1 | 0.6833 | 0.7335 | row |
| ens_xgb_tuned__gnn_sage | pr_auc | 0.6123 | 0.7075 | cluster |
| ens_xgb_tuned__gnn_sage | P@100 | 1 | 1 | cluster |
| ens_xgb_tuned__gnn_sage | R-precision | 0.572 | 0.6608 | cluster |
| ens_xgb_tuned__gnn_sage | precision | 0.8714 | 0.9275 | cluster |
| ens_xgb_tuned__gnn_sage | recall | 0.5245 | 0.6301 | cluster |
| ens_xgb_tuned__gnn_sage | f1 | 0.66 | 0.7471 | cluster |

## 6. Transactions: reliability of ens_xgb_tuned__gnn_sage (equal-mass bins over the top scores)

Window val

| bin | rows | positives | mean_predicted | observed_rate |
|---|---|---|---|---|
| top 1/10 | 1,036 | 647 | 0.5366 | 0.6245 |
| top 2/10 | 1,036 | 54 | 0.07291 | 0.05212 |
| top 3/10 | 1,036 | 56 | 0.05014 | 0.05405 |
| top 4/10 | 1,036 | 25 | 0.03802 | 0.02413 |
| top 5/10 | 1,036 | 24 | 0.03014 | 0.02317 |
| top 6/10 | 1,036 | 20 | 0.02431 | 0.01931 |
| top 7/10 | 1,036 | 12 | 0.01967 | 0.01158 |
| top 8/10 | 1,036 | 11 | 0.01591 | 0.01062 |
| top 9/10 | 1,036 | 11 | 0.01285 | 0.01062 |
| top 10/10 | 1,036 | 6 | 0.01037 | 0.005792 |
| rest | 955,164 | 170 | 0.0001941 | 0.000178 |

## 7. Transactions: ensemble audit (ens_xgb_tuned__gnn_sage, weight on xgb_tuned = 0.6)

Window val

- PR-AUC (seed-averaged, Platt): {'xgb_tuned': 0.652, 'gnn_sage': 0.6115, 'ens_xgb_tuned__gnn_sage': 0.667}
- ensemble gain over best member: 0.015
- gain 95% CI (paired): [0.0081, 0.0232]
- Spearman (all rows / positives): 0.613 / 0.867
| both_caught | only_xgb_tuned | only_gnn_sage | both_missed | ensemble_caught |
|---|---|---|---|---|
| 516 | 83 | 18 | 419 | 603 |

| both_flagged | only_xgb_tuned | only_gnn_sage | ensemble_flagged |
|---|---|---|---|
| 19 | 57 | 46 | 64 |

| jaccard_of_top_R | mules_in_both_top_R | mules_only_in_xgb_tuned_top_R | mules_only_in_gnn_sage_top_R | mules_in_neither |
|---|---|---|---|---|
| 0.4339 | 557 | 74 | 36 | 369 |

## 8. Transactions: pattern-level recall (ens_xgb_tuned__gnn_sage)

Window val

| pattern | positives | share_of_positives | recall@val_threshold | false_negatives | share_of_all_caught | small_sample |
|---|---|---|---|---|---|---|
| BIPARTITE | 49 | 0.0473 | 0.7959 | 10 | 0.06468 | 0 |
| CYCLE | 66 | 0.06371 | 0.9394 | 4 | 0.1028 | 0 |
| FAN-IN | 60 | 0.05792 | 0.8667 | 8 | 0.08624 | 0 |
| FAN-OUT | 64 | 0.06178 | 0.9844 | 1 | 0.1045 | 0 |
| GATHER-SCATTER | 96 | 0.09266 | 0.9479 | 5 | 0.1509 | 0 |
| RANDOM | 38 | 0.03668 | 0.8947 | 4 | 0.05638 | 0 |
| SCATTER-GATHER | 134 | 0.1293 | 0.9925 | 1 | 0.2206 | 0 |
| STACK | 150 | 0.1448 | 0.8533 | 22 | 0.2123 | 0 |
| none | 379 | 0.3658 | 0.002639 | 378 | 0.001658 | 0 |

## 9. Transactions: by payment channel (ens_xgb_tuned__gnn_sage)

Window val

| group | rows | positives | flagged | precision | recall | f1 | pr_auc_within_group | small_sample |
|---|---|---|---|---|---|---|---|---|
| ACH | 116,910 | 890 | 667 | 0.904 | 0.6775 | 0.7746 | 0.774 | 0 |
| Bitcoin | 28,643 | 12 | 0 | - | 0 | 0 | - | 1 |
| Cash | 103,357 | 25 | 0 | - | 0 | 0 | 0.001095 | 1 |
| Cheque | 397,108 | 59 | 0 | - | 0 | 0 | 0.001383 | 0 |
| Credit Card | 282,817 | 50 | 0 | - | 0 | 0 | 0.0016 | 0 |
| Wire | 36,689 | 0 | 0 | - | - | - | - | 1 |

## 10. Transactions: familiar vs unseen entities (ens_xgb_tuned__gnn_sage)

Window val

| group | rows | positives | pr_auc_within_group | P@100_within_group | precision | recall | f1 | small_sample |
|---|---|---|---|---|---|---|---|---|
| sender and receiver seen in training | 963,680 | 1,001 | 0.6667 | 1 | 0.9028 | 0.5844 | 0.7095 | 0 |
| sender or receiver new | 1,844 | 35 | 0.6861 | 0.28 | 0.9474 | 0.5143 | 0.6667 | 0 |
| both endpoints new | 53 | 0 | - | 0 | - | - | - | 1 |
| pair traded in training | 926,843 | 210 | 0.2252 | 0.44 | 0.8696 | 0.1905 | 0.3125 | 0 |
| new pair | 38,681 | 826 | 0.7777 | 1 | 0.9066 | 0.6816 | 0.7782 | 0 |

## 11. Transactions: error analysis (ens_xgb_tuned__gnn_sage, val threshold)

Window val: {'TN': 964424, 'TP': 603, 'FN': 433, 'FP': 64}

fn_clusters

| pattern | format | sender | receiver | false_negatives |
|---|---|---|---|---|
| none | ACH | adjacent | adjacent | 222 |
| none | Cheque | adjacent | adjacent | 59 |
| none | Credit Card | adjacent | adjacent | 50 |
| none | Cash | adjacent | adjacent | 25 |
| none | Bitcoin | adjacent | adjacent | 12 |
| STACK | ACH | adjacent | adjacent | 10 |
| none | ACH | mule | adjacent | 10 |
| BIPARTITE | ACH | adjacent | adjacent | 9 |
| STACK | ACH | adjacent | mule | 9 |
| FAN-IN | ACH | adjacent | adjacent | 8 |
| GATHER-SCATTER | ACH | adjacent | adjacent | 5 |
| STACK | ACH | mule | adjacent | 3 |
| RANDOM | ACH | adjacent | adjacent | 2 |
| RANDOM | ACH | mule | mule | 2 |
| CYCLE | ACH | mule | mule | 2 |

positives_by_outcome

| outcome | transfers | ach | self_transfer | median_log10_usd | sender_is_mule | receiver_is_mule |
|---|---|---|---|---|---|---|
| FN | 433 | 0.6628 | 0 | 3.42 | 0.04157 | 0.03233 |
| TP | 603 | 1 | 0 | 4.048 | 0.5041 | 0.2355 |

fp_clusters

| format | sender | receiver | false_positives |
|---|---|---|---|
| ACH | mule | adjacent | 25 |
| ACH | other | other | 22 |
| ACH | mule | mule | 8 |
| ACH | adjacent | other | 6 |
| ACH | adjacent | adjacent | 1 |
| ACH | mule | other | 1 |
| ACH | other | mule | 1 |

## 12. Transactions: by calendar day (frozen val threshold)

Window val

| day | transactions | laundering | pr_auc | precision | recall | f1 | small_sample |
|---|---|---|---|---|---|---|---|
| 2022-09-07 | 482,751 | 497 | 0.692 | 0.9164 | 0.6177 | 0.738 | 0 |
| 2022-09-08 | 482,773 | 539 | 0.6445 | 0.8916 | 0.5492 | 0.6797 | 0 |

## 13. Transactions: reconciliation with stored metrics

| model | window | metric | audit (seed-averaged) | stored (mean of seeds) |
|---|---|---|---|---|
| ens_xgb_tuned__gnn_sage | val | pr_auc | 0.667 | 0.667 |
| ens_xgb_tuned__gnn_sage | val | thr_f1 | 0.7082 | 0.7082 |
| ens_xgb_tuned__gnn_sage | val | thr_precision | 0.904 | 0.904 |
| ens_xgb_tuned__gnn_sage | val | thr_recall | 0.582 | 0.582 |
| xgb_tuned | val | pr_auc | 0.652 | 0.651 |
| xgb_tuned | val | thr_f1 | 0.7002 | 0.6988 |
| xgb_tuned | val | thr_precision | 0.8874 | 0.9108 |
| xgb_tuned | val | thr_recall | 0.5782 | 0.5672 |
| gnn_sage | val | pr_auc | 0.6115 | 0.6014 |
| gnn_sage | val | thr_f1 | 0.6532 | 0.6406 |
| gnn_sage | val | thr_precision | 0.8915 | 0.8431 |
| gnn_sage | val | thr_recall | 0.5154 | 0.5167 |
| xgb_pattern | val | pr_auc | 0.6413 | 0.6385 |
| xgb_pattern | val | thr_f1 | 0.6908 | 0.6891 |
| xgb_pattern | val | thr_precision | 0.8968 | 0.915 |
| xgb_pattern | val | thr_recall | 0.5618 | 0.5528 |

## 14. Feature importance: node xgb_graph

- largest single-feature gain share: 0.283
- top-3 gain share: 0.46
- val PR-AUC before permutation: 0.7066
| group | features | pr_auc_drop_mean | pr_auc_drop_std |
|---|---|---|---|
| ch | 9 | 0.5478 | 0.02598 |
| cp | 6 | 0.3773 | 0.001196 |
| gr | 8 | 0.232 | 0.01477 |
| pt | 6 | 0.5124 | 0.02649 |
| vol | 9 | 0.09974 | 0.01012 |
| xb | 2 | 0.001606 | 0.0004413 |

| feature | gain_share | split_share | cover_share | shap_mean_abs |
|---|---|---|---|---|
| ch_n_currencies | 0.283 | 0.03662 | 0.1276 | 0.4551 |
| ch_share_ach | 0.09863 | 0.05206 | 0.05241 | 0.8578 |
| gr_out_nbr_ach_mean | 0.07801 | 0.05605 | 0.0788 | 1.06 |
| gr_in_nbr_ach_mean | 0.06763 | 0.05293 | 0.06954 | 0.5257 |
| pt_fwd_ever_share | 0.0674 | 0.02056 | 0.1004 | 1.218 |
| cp_repeat_share | 0.04209 | 0.03673 | 0.02298 | 0.7383 |
| cp_reciprocity | 0.03899 | 0.01741 | 0.09168 | 0.06514 |
| vol_log_usd_in_per_day | 0.02998 | 0.03518 | 0.01428 | 0.3904 |
| pt_median_delay_h | 0.02925 | 0.04744 | 0.1338 | 0.5371 |
| gr_log_pagerank_rev | 0.02096 | 0.04133 | 0.08492 | 0.1412 |
| cp_log_unique_senders | 0.01943 | 0.01978 | 0.01241 | 0.2911 |
| vol_self_share | 0.01842 | 0.02853 | 0.006521 | 0.09038 |
| pt_out_in_ratio | 0.0173 | 0.02641 | 0.02027 | 0.298 |
| gr_log_pagerank | 0.01429 | 0.04896 | 0.01967 | 0.1959 |
| vol_log_usd_out_per_day | 0.01394 | 0.03353 | 0.006592 | 0.2818 |
| gr_in_nbr_fwd72_mean | 0.01354 | 0.02905 | 0.007676 | 0.2061 |
| cp_fan_out_ratio | 0.01283 | 0.02375 | 0.008598 | 0.581 |
| vol_log_amount_std | 0.0124 | 0.04109 | 0.01085 | 0.2243 |
| ch_share_cheque | 0.0117 | 0.03144 | 0.008207 | 0.3157 |
| ch_share_credit_card | 0.01134 | 0.02607 | 0.00755 | 0.2606 |
| vol_active_days | 0.01075 | 0.01648 | 0.01015 | 0.2019 |
| gr_out_nbr_fwd72_mean | 0.01011 | 0.02636 | 0.006107 | 0.1815 |
| vol_out_per_day | 0.009197 | 0.0191 | 0.008133 | 0.07579 |
| pt_net_flow | 0.008938 | 0.0194 | 0.007256 | 0.2477 |
| vol_log_amount_max | 0.008651 | 0.03232 | 0.008563 | 0.1546 |

## 14. Feature importance: edge xgb_tuned

- largest single-feature gain share: 0.259
- top-3 gain share: 0.584
| feature | gain_share | split_share | cover_share |
|---|---|---|---|
| e_fmt_ach | 0.2591 | 0.01092 | 0.08512 |
| e_pa_log_pair_count | 0.2256 | 0.04448 | 0.1562 |
| e_src_gap_h | 0.0991 | 0.04667 | 0.07614 |
| src_cp_log_unique_receivers | 0.05038 | 0.009982 | 0.07245 |
| dst_cp_reciprocity | 0.02492 | 0.01095 | 0.07271 |
| dst_gr_in_nbr_ach_mean | 0.02392 | 0.02352 | 0.01804 |
| e_log_usd | 0.02244 | 0.05796 | 0.02368 |
| src_ch_share_ach | 0.0188 | 0.01478 | 0.03979 |
| src_cp_repeat_share | 0.01757 | 0.01092 | 0.01258 |
| src_ch_n_currencies | 0.01741 | 0.008885 | 0.007946 |
| src_cp_reciprocity | 0.01191 | 0.009674 | 0.02013 |
| dst_vol_active_days | 0.01176 | 0.01401 | 0.00563 |
| src_pa_log_fan_out_24h_max | 0.0102 | 0.004047 | 0.01797 |
| src_gr_log_pagerank | 0.009591 | 0.01077 | 0.009775 |
| dst_ch_share_ach | 0.009009 | 0.01918 | 0.04467 |
| src_vol_log_amount_max | 0.008567 | 0.01918 | 0.006402 |
| src_gr_out_nbr_ach_mean | 0.008552 | 0.01168 | 0.01448 |
| src_pa_log_sg_source | 0.007886 | 0.002108 | 0.01374 |
| dst_pa_log_fan_in_24h_max | 0.006901 | 0.01368 | 0.004236 |
| dst_cp_repeat_share | 0.005798 | 0.01254 | 0.005704 |
| dst_vol_log_amount_std | 0.005622 | 0.02984 | 0.007308 |
| src_cp_fan_out_ratio | 0.005489 | 0.007953 | 0.00328 |
| dst_gr_log_pagerank | 0.005396 | 0.02341 | 0.01719 |
| dst_cp_fan_in_ratio | 0.00513 | 0.01707 | 0.01278 |
| dst_vol_log_amount_max | 0.005126 | 0.02133 | 0.008082 |

## 15. Train vs validation fit: node xgb_graph (in-sample PR-AUC)

| seed | best_iteration | train_pr_auc | val_pr_auc | train_s1_pr_auc | train_s2_pr_auc | train_s3_pr_auc |
|---|---|---|---|---|---|---|
| seed0 | 310 | 0.9983 | 0.7031 | 1 | 0.9998 | 0.9978 |
| seed1 | 257 | 0.988 | 0.7048 | 0.9775 | 0.9916 | 0.9875 |
| seed2 | 318 | 0.9991 | 0.7052 | 1 | 0.9996 | 0.9988 |
