# Controlled CKD experiments in PyTorch

Start with **model_exeperimental_notebook.ipynb**. It provides the complete workflow: frozen splits, E00/E01 references, E02–E08 ablations, E09 searches, E10 losses, comparison plots, finalist freezing, and one-shot E11 evaluation. All training and evaluation gates default to `False`.

No real CKD model has been fitted or evaluated during implementation. Synthetic tests exercise the entire workflow in temporary directories.

## Environment and checks

Select the project `.venv` as your notebook kernel. The verified package versions are in `requirements.txt`; `environment.lock.txt` records the full installed environment. Python 3.14 was used locally. Install the optional test runner with `requirements-dev.txt` when network access is available.

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src tests
# Equivalent discovery with pytest installed:
.venv/bin/python -m pytest tests/
```

Tests use standard-library unittest assertions and are collected by pytest. Pytest could not be installed in the implementation environment because PyPI DNS resolution was unavailable; unittest is the verified runner.

## Protocol

- Input: existing `Chronic_Kidney_Disease/chronic_kidney_disease_full.csv`. Never rerun the converter as part of experiments.
- `class`: CKD=1, non-CKD=0. Fixed schema mappings preserve predictor NaNs.
- Frozen 240/80/80 stratified split, seed 42; exact memberships and five training-only CV folds live in `data/splits/`.
- Every fold gets fresh preprocessing, feature selection and models. NN epoch selection fits preprocessing on an internal training subset, then refits a fresh pipeline on the entire permitted training partition.
- CV metrics use threshold 0.5. Validation selects maximum F2 with precision ≥0.80; ties prefer precision then the larger threshold. No positive predictions or no qualifying cutoff means ineligible.
- NN experiments use seeds 42, 43 and 44. Finalist decisions use their averaged probabilities; seed and fold variability remain separate in artifacts.
- You choose configurations for tuning and a finalist using development evidence. There is no automatic feature stacking or automatic selection from test performance.
- Failed runs retain a failure summary and traceback; do not silently exclude them from your analysis.

## Models and preprocessing

All fitting uses Torch. Logistic regression uses full-batch L2-regularized LBFGS with an unpenalized intercept. The random forest implements bootstrap CART with weighted Gini splits, random feature subsets, leaf constraints and averaged leaf probabilities. It is a CPU research implementation, not an optimized GPU forest or a promise of bitwise parity with other libraries.

The shallow NN supports BCE, weighted BCE, probability MSE and focal loss. All use the same optimizer and stopping policy; numerical losses are not comparable between different objectives. Class weights are estimated from the relevant training partition only.

The schema-aware `TabularPreprocessor` handles median/mode, scaled-distance KNN, iterative ridge imputation, missingness filtering, optional indicators/Unknown levels, ordered encoding, scaling, log features, clinical subsets, redundancy filtering and model-based selection. Ridge imputation is explicitly a ridge-based procedure, not a reproduction of another library's Bayesian iterative imputer. All-missing retained features fail with their names. All source nominal categories are binary, so baseline 0/1 encoding is lossless; one-hot encoding is used for explicit Unknown and the ordered-one-hot treatment.

`CKDPreprocessor` is the smaller two-tensor median/mode class discussed in the conversation. The experiment runner uses `TabularPreprocessor` so schema, encoding, filtering and selection are preserved with the fitted state.

## Running individual experiments

```python
import json
from src.train import run_experiment
configs = json.load(open('configs/experiments.json'))
config = next(c for c in configs if c['experiment_id'] == 'E01-lr')
run_dir = run_experiment(config)
```

Each first-pass configuration differs from E01 by one treatment. Standard scaling is E01 for LR/NN; RF uses no scaling. Full history plus labs is the full-feature E01 reference, so it is not rerun under another ID. The history subset excludes anemia because its ascertainment is uncertain.

E09 enumerates eight LR settings, samples 24 RF settings with seed 42, or enumerates eight NN settings. E10 takes one fixed NN parent configuration and varies loss only. The custom forest and repeated CV can be slow; run references before the entire ablation menu.

## Artifacts

Each `results/<run_id>/` contains config, provenance, a source snapshot, fitted pipeline bundle, feature lists, NN histories, per-fold metrics/predictions, per-seed validation predictions, ensemble validation predictions, convergence diagnostics and summary. `results/runs.csv` contains one row per seed/configuration. An infeasible threshold leaves development metrics at 0.5 for context and is explicitly marked ineligible.

`compare_runs` produces shared PR/ROC plots, confusion matrices, threshold curves, five-bin calibration, NN learning curves and CV score distributions. Paired fold deltas compare runs with the same saved split. Permutation importance uses validation rows and the frozen validation threshold; because validation also informs selection, importance is descriptive rather than an independent estimate.

## Final test protection

`freeze_finalist` copies the selected fitted bundle into `results/final_test/`, records its hash and threshold, and refuses to replace an existing selection. `evaluate_final` checks dataset/split/source/model hashes and creates an exclusive test-start marker before accessing test rows. It never retrains or tunes thresholds.

A second invocation is rejected. If inference succeeded but reporting failed, `report_saved_test` rebuilds metrics and intervals from cached predictions. If failure occurred before predictions were saved, the start marker remains: inspect and document the failure manually rather than silently repeating test access. These guards prevent accidental reuse, not deliberate filesystem tampering.

The bootstrap uses 2,000 stratified resamples, seed 42. Intervals condition on the selected pipeline and observed class counts, and omit training/model-selection uncertainty. Use `report_template.md` for your final discussion.

## Dataset limitations

See `data/inspection_handoff.md`, `data/schema.json` and `data/validation_rules.json`. The existing EDA examined every row by class. Three ARFF repairs lack clinical-record verification. Patient independence and feature timing cannot be established. Unusual values remain present; young CKD cases are not discarded. This is a classification benchmark, not a clinical diagnostic product.

Torch bundles contain Python objects: only deserialize artifacts you created or trust. CPU deterministic execution and pinned versions improve repeatability; identical results across different hardware/software versions are not guaranteed.

## Final single-seed model

`model.py` trains a single E09-style neural network using seed 44, hidden layers `[32, 16]`, batch size 32, weight decay 0, and a 200-epoch maximum. It uses BCEWithLogitsLoss (`loss='bce'`), KNN numeric imputation, standard scaling, and ordered one-hot encoding for specific gravity (`sg`), albumin (`al`), and sugar (`su`). Early stopping uses a stratified internal holdout and validation-loss patience 20; then a fresh pipeline is refit on the complete training partition for the selected epoch count. The classification threshold is selected on the validation partition using maximum F2 subject to precision ≥ 0.80. The script creates a separate seed-43 stratified split in `data/splits_model/` and writes the fitted model and validation/test report to `results/final_model/`. Run with `.venv/bin/python model.py`. The new test membership differs from the prior split, but it is a reshuffled holdout from previously examined data, not an independent evaluation.
