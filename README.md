# bcsc-fp-prediction

R code for "Performance of Statistical and Machine Learning Risk Prediction Models for Surveillance Mammography False Positives in Breast Cancer Survivors" (manuscript under review).

We compared seven models for predicting false-positive recall and false-positive biopsy recommendation after surveillance mammography, using data from the Breast Cancer Surveillance Consortium (BCSC).

The data are not included. Access requires approval from the BCSC; see the data availability statement in the paper.

## Files

Set the data and results paths in `R/config.R`, then run scripts from the repo root.

- `01_table1.Rmd`: descriptive table
- `02_fit_models.R`: fits all models on each imputed dataset
- `03_evaluate_models.R`: discrimination, calibration, and F1, pooled across imputations
- `04_evaluate_by_race.R`: performance by race and ethnicity
- `05_figure1.R`: ROC curves (Figure 1)
- `06_variable_importance.R`: variable importance
- `07_smote_fit.R`, `08_smote_evaluate.R`: models fit with SMOTE to address class imbalance

I'm still cleaning up this code, so structure may change.
