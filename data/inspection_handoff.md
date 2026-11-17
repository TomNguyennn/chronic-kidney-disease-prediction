# Inspection hand-off

Source: existing data_exploration_notbook.ipynb, cells 2–8 and 10–23, and local ARFF metadata.

- Dataset SHA-256: `c0eaa5003c90a82d9254be8bc2f9d30701956322a3ff9e47bcfbfaf3fe96b5fd`; 400 rows, 25 columns.
- Target: class; CKD=1 (250), non-CKD=0 (150).
- Numerical predictors: age, bp, bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc.
- Ordered predictors: sg, al, su. Other categories are binary with mappings in schema.json.
- Existing inspection found no exact or predictor-only duplicates. No patient IDs or timestamps are present; repeated-patient independence cannot be verified.
- Existing missingness: rbc 38%, rbcc 32.75%, wbcc 26.5%, pot 22%, sod 21.75%; full per-feature counts are in validation_rules.json.
- Source converter removed one extra empty field on three ARFF rows. These repairs were not checked against clinical records; experiments do not rerun the converter.
- Existing review flags include potassium up to 47, serum creatinine up to 76, and sodium down to 4.5. These are retained, not automatically clipped or corrected.
- No invalid predictor tokens remain after the current CSV parser's fixed schema checks. Missing values remain NaN until training-fitted imputation.
- Prior EDA examined every row by class: the test partition is held out from model development but not entirely untouched by inspection.
- No documented outcome-derived feature is present. Timing/ascertainment of predictors is unknown, especially ane. The full set is a benchmark set, not a certified leakage-free deployment set.
- History/examination subset excludes ane. Young CKD or hypertensive patients are not removed.

The existing EDA was reused; only schema, token and hash checks were performed for the hand-off.
