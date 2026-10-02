# Credit Risk Modeling: PD / LGD / EAD / Expected Loss

A Basel-style credit-risk engine in Python on real lending data. It builds a probability of default (PD) scorecard, a two-stage loss given default (LGD) model, an exposure at default (EAD) model via a credit conversion factor, and combines them into loan-level expected loss and an approve / reprice / reject rule. Built for the Rutgers MQF course "OOPs II" (object-oriented programming), where the focus was extending scikit-learn with custom classes.

**Note:** this was a course group submission. The feature pipeline, modeling and OOP design in these notebooks were written by me. The report documents are not included here.

## Data
LendingClub public loan data, 2007 to 2014: 466,285 loans, 75 raw columns expanding to 200+ after one-hot encoding, roughly a 9 to 10% bad rate. 43,236 charged-off loans are carved out for the LGD and EAD models. The ~1.5 GB of raw and processed CSVs are not in this repo. Download the LendingClub dataset and run the Preparation notebook to regenerate them.

## Files
- `Credit Risk Modeling - Preparation.ipynb`: feature engineering and WoE/IV pipeline (202 cells).
- `Credit Risk Modeling - Final.ipynb`: PD, LGD, EAD, expected loss and decision engine (171 cells).

## Pipeline
Raw CSV -> preparation notebook -> preprocessed CSV -> defaults-only file for LGD/EAD, plus a decoupled PD train/test split written to four CSVs. Fitted models are persisted with `pickle`.

## Feature engineering (Preparation notebook)
- Parsed `issue_d` and `earliest_cr_line` into months-since features, fixing negative months caused by two-digit year parsing.
- Cleaned `emp_length` and `term` to numeric values.
- Missing values: `annual_inc` to the mean, credit and account counts to 0, `total_rev_hi_lim` to `funded_amnt`.
- One-hot encoded grade, sub-grade, home ownership, verification status, purpose, state and list status.
- **Weight of Evidence and Information Value** for every candidate predictor through reusable `woe_discrete` and `woe_ordered_continuous` functions. Continuous variables are fine-classed into 50 to 100 bins, then coarse-classed by merging bins with similar WoE, with WoE plots used to check monotonicity. Unordered variables such as state are grouped into about 10 WoE-ordered classes with a reference category. Example IVs: grade 0.30, purpose 0.045, home ownership 0.023, verification status 0.023.
- Target `good_bad`: bad is Charged Off, Default, Late (31 to 120 days) or In Grace Period; everything else is good.

## Modeling (Final notebook)
- **`LogisticRegression_with_p_values`**: a custom class around scikit-learn logistic regression that adds what sklearn omits: standard errors from the design-matrix covariance, z-scores, two-tailed p-values and a `summary()` table, the kind of coefficient-significance evidence an audit or validation review expects.
- **`LinearRegression` subclass**: extends sklearn by inheritance, overriding `fit()` via `super().fit()` to add residual standard errors, t-values and p-values. Used for LGD stage 2 and EAD.
- **`align_columns`**: forces the dummy-variable structure at predict time to match training, so the model does not break on unseen category levels.
- **PD**: logistic regression on WoE-binned features, `predict_proba` as probability of default.
- **LGD, two stage**: stage 1 is a logistic model of whether any recovery occurs (test AUROC 0.647); stage 2 is a linear regression of the recovery amount given recovery. Combined as `LGD = 1 - (stage 1 probability x stage 2 amount)`, clipped to [0, 1].
- **EAD**: linear regression on the credit conversion factor `CCF = (funded_amnt - total_rec_prncp) / funded_amnt` (mean about 0.74), test correlation of actual and predicted 0.53, with `EAD = CCF_hat x funded_amnt`.
- **Expected loss**: `EL = PD x LGD x EAD` per loan, summed to portfolio EL and expressed as a percentage of funded amount.
- **Decision engine**: approve immediately (low PD and low EL), approve at a higher rate (medium PD, low LGD and EAD), or reject (high PD or high EL).

## Known limitations
- The LGD stage 1 AUROC of 0.647 is modest, and the EAD correlation of 0.53 is moderate.
- Pandas chained-assignment warnings remain in places.
- No scorecard scaling, KS statistic, calibration curve or population-stability check. A separate validation project on the same data covers those: see [credit-risk-model-validation-lendingclub](https://github.com/CrunckCode/credit-risk-model-validation-lendingclub).

## Skills shown
Basel PD/LGD/EAD/EL framework, WoE/IV scorecard methodology, logistic scorecard modeling, two-stage LGD, EAD/CCF modeling, ROC/AUROC validation, extending scikit-learn through inheritance and composition, modular file-based pipeline design, large-data handling.
