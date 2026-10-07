# Model card: money-mule detection on AMLworld HI-Small

## Model details

- **Account champion:** `ens_xgb_graph__gnn_sage`. A weighted average (0.7 / 0.3) of a
  Platt-calibrated XGBoost on 40 account features (behaviour + graph) and a Platt-calibrated
  2-layer GraphSAGE on 48 account features and 16 transaction features. Weight chosen on
  validation.
- **Transaction champion:** `ens_xgb_tuned__gnn_sage`. A tuned XGBoost on the transaction's own
  features, pattern features and both endpoints' account features, averaged with a GraphSAGE
  edge model (sender embedding, receiver embedding, transaction features).
- Built October 2026 as a portfolio project; not deployed anywhere.

## Intended use

- Ranking accounts and transfers for **review by a human analyst**, with reason codes and
  evidence transfers attached. Scores are calibrated probabilities (Brier ≈ 5×10⁻⁴).
- A worked, reproducible example of mule detection with honest evaluation: temporal split,
  leakage tests, validation-only model choice, a single guarded test evaluation.

**Out of scope:** automated blocking or account closure; use on real bank or UPI data without
retraining and re-validation; any decision about a person without human review.

## Data

IBM AMLworld HI-Small (Altman et al., NeurIPS 2023): 5,078,345 synthetic transactions between
515,088 accounts, 5,177 laundering (1 per 981), Sep 1–18 2022. Synthetic: no real people.

- **Split** (IBM's own algorithm): train Sep 1–6, validation Sep 7–8, test Sep 9–18.
- **Mule label** (ours, not in the dataset): an account that, inside a window, sends a laundering
  transfer after receiving one at any earlier time (rule B). Train snapshots hold 320 positive
  rows (297 accounts); validation 242 mules; test 421; test without tail 222.
- **The tail:** after Sep 10, 1,108 transactions remain, 655 of them laundering (59%). They make
  up 41% of test laundering and inflate test scores, so every test result is also reported on a
  window that stops at Sep 10.

## Metrics

PR-AUC (headline), precision and recall at K (100, 0.1% of active accounts, R-precision),
minority-class F1 at a validation-chosen threshold and at 0.5 (IBM's protocol), recall for
first-time and repeat mules, Brier score and calibration error. Seeds: 3 per model; a 1,000-
resample bootstrap interval for account PR-AUC.

## Results (test)

| | PR-AUC | Without tail | Other |
|---|---|---|---|
| Account champion | 0.761 (0.721–0.798) | 0.657 | P@100 0.960, R-precision 0.713 |
| Transaction champion | 0.773 | 0.603 | F1 0.786 (0.654 without tail); F1 at 0.5: 0.764 |

Against IBM's paper (F1 at 0.5, same test transactions): Multi-GIN+EU 0.648 ± 0.012,
Multi-PNA+EU 0.682 ± 0.026. See the caveats below before comparing.

## Explanations

Reason codes come from exact SHAP values of the XGBoost member (a code fires only when the value
is extreme against the training population **and** its feature group drove the score). Evidence
transfers come from a GNNExplainer-style mask on the GraphSAGE member, averaged over 3 runs.
Code RC08 appears when the GNN supplies more of an account's added risk than its 0.3 weight.

Validation, top 200 accounts: 87% receive a catalog code; removing the 5 evidence transfers
lowers the GNN score by 0.067 against 0.007 for 5 random transfers; for true mules, 38% of
evidence transfers are laundering against 20% of their neighbourhoods; two independent runs
share 37% of their evidence. Test: `reports/explanations_test.md`.

## Limitations and failure modes

- **Synthetic data.** AMLworld mules are slow (median forwarding delay about a day; 45% of
  laundered receipts move on within 24 h). Real UPI mules often move money within minutes.
  Nothing here has been checked on real transactions.
- **Pattern features know the generator.** They count the exact laundering shapes the
  simulator plants (fan-in/out, scatter-gather, cycles). This explains much of the transaction
  gain and would not transfer as cleanly to real laundering.
- **Lone laundering is nearly invisible.** Laundering that belongs to no pattern (23% of test
  laundering transactions) is caught 0.8% of the time.
- **Fan-in mules and cycle members are caught least often** among account patterns (recall 0.46
  and 0.63 inside the top R, on small counts).
- **Small validation set.** 242 mules: differences below about 0.01 PR-AUC cannot be separated
  from noise, which is why the paired bootstrap decides between close models.
- **Stress test is narrow.** Delaying mules' laundering sends by 48 h barely moved the champion
  (0.761 → 0.758) but cost the GNN member 0.043; delays were capped at the 2-day window, so
  slower laundering over longer periods is untested.
- **Explanations are approximate.** SHAP codes are exact for the XGBoost member, but evidence
  transfers come from an optimised mask and only 37% of them recur between runs.
- **Calibration.** Isotonic recalibration of the ensembles lowered PR-AUC (account test 0.761 →
  0.723) without improving Brier score; Platt-scaling the members already calibrates them.

## Disclosures

1. **Test results seen before the improvement stage.** On Day 5, test results were displayed
   for the three GNNs and both pattern models. Every later choice used validation only, through
   rules written down before the comparisons ran.
2. **The account champion's win is narrow.** Validation PR-AUC +0.019 over XGBoost, interval
   0.00004 to 0.039, with the ensemble weight chosen on the same validation set. It held on
   test (+0.013), but the best single test number belongs to an ablation (XGBoost with pattern
   features, 0.773), which validation could not separate from the chosen model.
3. **Explanation settings changed after a test run.** RC08's threshold (20% → the GNN's weight)
   and single → averaged masks followed a first test run that showed RC08 on 192 of 200 accounts
   and 25% stability. Both changes were checked on validation before test was rerun.
4. **The Multi-GNN comparison is against one paper.** Newer work on HI-Small was not checked.
   Our own Multi-GNN run is 2 epochs (about 2.3 h each on this laptop) and undertrained.
5. **A display bug hid small values** (FPR, Brier) as 0.000 in results tables until Day 7.
   Stored metrics were always correct.

## Ethical considerations

Mule accounts are often held by people who were recruited, coerced or deceived. A flag should
start a review, not a punishment. False positives fall on ordinary customers: at the account
champion's validation-chosen threshold the test false-positive rate is 2.7×10⁻⁴, which is still
around 100 of the 354,590 active accounts in the test window. About a third of the champion's false positives
in its top 0.1% are laundering sources or cash-out accounts rather than uninvolved customers.
