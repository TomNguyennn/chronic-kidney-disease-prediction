# Controlled experiment protocol

Implementation entry point: model_exeperimental_notebook.ipynb. Treatment definitions are in configs/experiments.json. All learned preprocessing and models use PyTorch; no real-data experiments have been run during implementation.

## Fixed controls

Use the existing CSV hash and saved row-level split/CV manifests. CKD=1. Fit transformations and selectors only within each training partition. CV uses threshold 0.5. Validation thresholds maximize F2 subject to precision >=0.80, with higher precision then higher threshold as ties. An infeasible threshold makes a configuration ineligible. Compare NN seeds 42/43/44 separately and use their averaged probabilities for finalist decisions. Test exactly one frozen selected bundle; do not refit on validation.

## Schedule

| Stage | Treatment |
|---|---|
| E00 | Most-frequent and always-CKD dummies |
| E01 | Median/mode, full features, ordered numerical values, appropriate scaling; LR/RF/NN |
| E02 | Add per-feature missingness indicators |
| E03 | Scale-aware KNN continuous imputation |
| E04 | Iterative ridge continuous imputation |
| E05 | Training-only feature drop above 30% or 50% missingness |
| E06 | Separate history-only, redundancy filtering, model selection, ordered-one-hot and explicit-Unknown treatments |
| E07 | Robust or MinMax versus E01 standard scaling for LR/NN |
| E08 | Append log1p(sc), log1p(bu), log1p(bgr) |
| E09 | User-selected preprocessing parents, fixed training-CV parameter searches |
| E10 | Fixed NN parent and seeds: BCE / weighted BCE / probability MSE / focal |
| E11 | User-selected eligible finalist, frozen validation threshold, one-shot held-out test |

Every first-pass variant starts from E01. Full history plus labs is E01 and is not duplicated. All-missing retained predictors and nonconverged LR fits fail explicitly. Do not carry consistently inferior variants into tuning simply to increase the search size. No automated stacking of treatments.

## Model defaults

LR: L2 C=1, zero initialization, float64 LBFGS with strong-Wolfe search, 1,000 iterations maximum, unpenalized intercept. RF: 200 bootstrap CART trees, weighted Gini, sqrt feature sampling, unlimited depth, minimum leaf size 1. NN: [16] ReLU, Adam 1e-3, batch size 16, weight decay 0, 200 epochs maximum and internal training-only early stopping with patience 20. Every NN fit then retrains a fresh pipeline on the permitted training partition for its selected epoch count.

KNN uses five donors and uniform means with training-mean fallback. Iterative ridge uses penalty 1 and ten stored rounds. Redundancy filtering uses training-only absolute Spearman >0.95. Model-based selection retains half the encoded features by a fixed Torch forest's impurity importance. None of these thresholds are clinical rules.

## Reporting and acceptance

Record configuration, environment, code and dataset hashes, source snapshot, split, seed, fitted state, fold predictions/metrics, validation probabilities and threshold. Include confusion counts, specificity and always-CKD performance alongside F2; display PR, ROC, threshold, calibration, learning, importance and paired-fold plots. Report training-fold variation separately from NN seed variation.

Final test intervals use 2,000 stratified bootstrap resamples and do not account for training or selection uncertainty. The report must disclose prior whole-dataset EDA, source repairs, unavailable patient/time identifiers and unknown prediction-time feature availability. No clinical diagnostic claim is supported.

Software acceptance: compilation, notebook structure/code-cell checks, synthetic class tests, leakage guards, deterministic reruns, serialization and guarded end-to-end workflow. Unit tests are in tests/ and run with unittest or pytest. Pytest installation was unavailable due to network restrictions, so unittest was used for verification.
