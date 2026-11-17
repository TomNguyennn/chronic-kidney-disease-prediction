# CKD controlled experiment report

## Protocol and provenance
Record dataset/split/source hashes, environment, seeds, prior EDA exposure, validation threshold policy and the final selected run. Explain any failed or excluded configurations. Describe the 240/80/80 split and internal early stopping.

## Paired ablations
Compare each treatment with E01 within each model family on identical folds. Report F2 mean ± sample SD, precision, recall, specificity, AP, ROC-AUC and Brier score. Separate NN seed variation from fold variation. Include paired fold differences and avoid treating small differences as conclusive.

## Models and losses
Compare the best development-selected LR, RF and NN configurations, including both dummy baselines. Compare NN losses with identical architecture, features, seeds and stopping policy. Explain the benefit or cost of laboratory features versus history/examination alone.

## Final selected pipeline
Record the frozen validation threshold and its precision constraint. Report held-out confusion counts, metrics, confidence intervals and plots once. Do not change the pipeline because of test results.

## What worked, what did not, and limitations
Address missingness signal, transform/selection complexity, NN stability, source repairs, unusual observations, unknown predictor timing/patient identity, small sample uncertainty and prior whole-dataset EDA. Distinguish dataset-label prediction from clinical diagnosis. Bootstrap intervals omit training and selection uncertainty.
