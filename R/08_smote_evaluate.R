# =============================================================
# Evaluation of SMOTE-based risk prediction models
# Author: Tiffany Hsieh
# Date: 2025-04-16
# =============================================================

source("R/config.R")

library(pROC)
library(MLmetrics)
library(precrec)
library(broom)
library(ggplot2)

# Output directory for evaluation plots
output_dir <- file.path(results_dir, "eval_SMOTE_Feb2026")
dir.create(output_dir, showWarnings = FALSE)

# =============================================================
# Load model predictions saved from SMOTE modeling pipeline
# =============================================================

load(file.path(results_dir, "FP_recall_human_glm_SMOTE.RData"))
load(file.path(results_dir, "FP_biop_human_glm_SMOTE.RData"))
# Load saved predictions
load(file.path(results_dir, "FP_recall_glm_SMOTE.RData"))
risk_pr_recall_glm <- preds  # extract from loaded file

load(file.path(results_dir, "FP_biop_glm_SMOTE.RData"))
risk_pr_biop_glm <- preds

load(file.path(results_dir, "FP_recall_lasso_SMOTE.RData"))
load(file.path(results_dir, "FP_biop_lasso_SMOTE.RData"))

load(file.path(results_dir, "FP_recall_elasticnet_SMOTE.RData"))
load(file.path(results_dir, "FP_biop_elasticnet_SMOTE.RData"))

load(file.path(results_dir, "FP_recall_rf_SMOTE.RData"))
load(file.path(results_dir, "FP_biop_rf_SMOTE.RData"))

load(file.path(results_dir, "FP_recall_gbm_SMOTE.RData"))
load(file.path(results_dir, "FP_biop_gbm_SMOTE.RData"))

load(file.path(results_dir, "FP_recall_nnet_SMOTE.RData"))
load(file.path(results_dir, "FP_biop_nnet_SMOTE.RData"))


# check prediction
model_preds <- list(
  FP_recall_glm_human = risk_pr_recall_human,
  FP_biop_glm_human   = risk_pr_biop_human,
  FP_recall_lasso     = risk_pr_recall_lasso,
  FP_biop_lasso       = risk_pr_biop_lasso,
  FP_recall_elastic   = risk_pr_recall_enet,
  FP_biop_elastic     = risk_pr_biop_enet,
  FP_recall_rf        = rf_recall$predictions,
  FP_biop_rf          = rf_biop$predictions,
  FP_recall_gbm       = gbm_recall$predictions,
  FP_biop_gbm         = gbm_biop$predictions,
  FP_recall_nnet      = as.vector(nnet_recall$predictions),
  FP_biop_nnet        = as.vector(nnet_biop$predictions)
)

outcomes <- list(
  FP_recall = test_recall$FP_recall,
  FP_biop   = test_biop$FP_biop
)

# Function to run checks
check_preds <- function(preds, outcome_name, outcome_vector) {
  cat("\nChecking:", outcome_name, "\n")
  cat("Range:", summary(preds), "\n")
  cat("Any NA?:", anyNA(preds), "\n")
  cat("Out of bounds?:", any(preds < 0 | preds > 1), "\n")
  cat("Length match:", length(preds) == length(outcome_vector), "\n")
  cat("Outcome table:\n"); print(table(outcome_vector))
  cat("Predicted classes:\n"); print(table(ifelse(preds >= 0.5, 1, 0)))
}

# Loop over predictions
for (name in names(model_preds)) {
  outcome_var <- if (grepl("biop", name)) "FP_biop" else "FP_recall"
  check_preds(model_preds[[name]], name, outcomes[[outcome_var]])
}

evaluate_model <- function(predicted_probs, response_var, model_name, test_data, train_data) {
  library(pROC)
  library(MLmetrics)
  library(precrec)
  library(ggplot2)
  library(brglm2)
  
  test_data[[response_var]] <- as.numeric(as.character(test_data[[response_var]]))
  train_data[[response_var]] <- as.numeric(as.character(train_data[[response_var]]))
  
  # --- AUC ---
  roc_curve <- pROC::roc(test_data[[response_var]], predicted_probs, quiet = TRUE)
  auc_value <- as.numeric(pROC::auc(roc_curve))
  auc_ci <- pROC::ci.auc(roc_curve, method = "delong")
  auc_ci_fmt <- sprintf("%.3f (%.3f, %.3f)", auc_value, auc_ci[1], auc_ci[3])
  
  # ============================================================
  # Prevalence-based threshold (from TRAINING data)
  # ============================================================
  prevalence_threshold <- mean(train_data[[response_var]])
  
  # --- F1 Score ---
  f1_score <- F1_Score(y_pred = ifelse(predicted_probs >= prevalence_threshold, 1, 0),
                       y_true = test_data[[response_var]], positive = "1")
  boot_f1 <- replicate(1000, {
    idx <- sample(seq_len(nrow(test_data)), replace = TRUE)
    tryCatch({
      F1_Score(
        y_pred = ifelse(predicted_probs[idx] >= prevalence_threshold, 1, 0),
        y_true = test_data[[response_var]][idx], positive = "1"
      )
    }, error = function(e) NA)
  })
  f1_ci <- quantile(boot_f1, probs = c(0.025, 0.975), na.rm = TRUE)
  f1_ci_fmt <- sprintf("%.3f (%.3f, %.3f)", f1_score, f1_ci[1], f1_ci[2])
  
  # ============================================================
  
  # --- Expected / Observed Ratio ---
  expected <- sum(predicted_probs)
  observed <- sum(test_data[[response_var]])
  eo_ratio <- expected / observed
  eo_ci_lb <- eo_ratio * exp(-1.96 * sqrt(1 / observed))
  eo_ci_ub <- eo_ratio * exp(1.96 * sqrt(1 / observed))
  eo_ci_fmt <- sprintf("%.3f (%.3f, %.3f)", eo_ratio, eo_ci_lb, eo_ci_ub)
  
  # --- Calibration Data ---
  calib_data <- data.frame(predicted = predicted_probs, observed = test_data[[response_var]])
  calib_data <- calib_data[!is.na(calib_data$predicted), ]
  calib_data$predicted <- pmin(pmax(calib_data$predicted, 0.001), 0.999)
  calib_data$logit_pred <- log(calib_data$predicted / (1 - calib_data$predicted))
  
  # --- Cox calibration (bias-reduced logistic regression) ---
  safe_brglm <- function(formula, data) {
    tryCatch({
      glm(formula, data = data, family = binomial(), method = "brglmFit",
          control = brglmControl(maxit = 100))
    }, error = function(e) {
      warning("Falling back to glm due to brglmFit failure.")
      glm(formula, data = data, family = binomial())
    })
  }
  
  intercept_model <- safe_brglm(observed ~ 1 + offset(logit_pred), calib_data)
  slope_model <- safe_brglm(observed ~ logit_pred, calib_data)
  
  intercept_summary <- summary(intercept_model)$coefficients
  slope_summary <- summary(slope_model)$coefficients
  
  intercept_ci <- sprintf("%.3f (%.3f, %.3f)",
                          intercept_summary[1, "Estimate"],
                          intercept_summary[1, "Estimate"] - 1.96 * intercept_summary[1, "Std. Error"],
                          intercept_summary[1, "Estimate"] + 1.96 * intercept_summary[1, "Std. Error"])
  
  slope_ci <- sprintf("%.3f (%.3f, %.3f)",
                      slope_summary[2, "Estimate"],
                      slope_summary[2, "Estimate"] - 1.96 * slope_summary[2, "Std. Error"],
                      slope_summary[2, "Estimate"] + 1.96 * slope_summary[2, "Std. Error"])
  
  # --- ROC Plot ---
  roc_data <- data.frame(tpr = rev(roc_curve$sensitivities),
                         fpr = rev(1 - roc_curve$specificities))
  ggsave(
    file.path(output_dir, paste0(model_name, "_", response_var, "_ROC_SMOTE.png")),
    ggplot(roc_data, aes(x = fpr, y = tpr)) +
      geom_line(color = "blue", size = 1) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
      theme_minimal() +
      ggtitle(paste(model_name, "-", response_var, "AUC:", round(auc_value, 3))),
    width = 8, height = 6
  )
  
  # --- Calibration Plot ---
  breaks <- unique(quantile(calib_data$predicted, probs = seq(0, 1, 0.1)))
  if (length(breaks) < 3) {
    breaks <- seq(min(calib_data$predicted), max(calib_data$predicted), length.out = 11)
  }
  calib_data$bin <- cut(calib_data$predicted, breaks = breaks, include.lowest = TRUE)
  calib_summary <- aggregate(observed ~ bin, data = calib_data, mean)
  calib_summary$predicted <- aggregate(predicted ~ bin, data = calib_data, mean)$predicted
  ggsave(
    file.path(output_dir, paste0(model_name, "_", response_var, "_Calibration_SMOTE.png")),
    ggplot(calib_summary, aes(x = predicted, y = observed)) +
      geom_point() +
      geom_smooth(method = "loess", se = FALSE) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
      theme_minimal() +
      ggtitle(paste(model_name, "-", response_var, "Calibration")),
    width = 8, height = 6
  )
  
  # --- PR Curve ---
  pr_obj <- evalmod(scores = predicted_probs, labels = test_data[[response_var]])
  ggsave(
    file.path(output_dir, paste0(model_name, "_", response_var, "_PR_SMOTE.png")),
    autoplot(pr_obj),
    width = 8, height = 6
  )
  
  return(list(
    auc = auc_ci_fmt,
    f1  = f1_ci_fmt,
    f1_threshold = prevalence_threshold,
    eo  = eo_ci_fmt,
    cox_intercept = intercept_ci,
    cox_slope = slope_ci
  ))
}




# Evaluate using ORIGINAL test and train data
results_smote <- list()

results_smote[["FP_recall_glm_human"]] <- evaluate_model(
  risk_pr_recall_human, "FP_recall", "GLM_Human", 
  test_recall, train_recall)

results_smote[["FP_biop_glm_human"]] <- evaluate_model(
  risk_pr_biop_human, "FP_biop", "GLM_Human", 
  test_biop, train_biop)

results_smote[["FP_recall_glm"]] <- evaluate_model(
  risk_pr_recall_glm, "FP_recall", "GLM_General", 
  test_recall, train_recall)

results_smote[["FP_biop_glm"]] <- evaluate_model(
  risk_pr_biop_glm, "FP_biop", "GLM_General", 
  test_biop, train_biop)

# LASSO
results_smote[["FP_recall_lasso"]] <- evaluate_model(
  risk_pr_recall_lasso, "FP_recall", "LASSO", 
  test_recall, train_recall)

results_smote[["FP_biop_lasso"]] <- evaluate_model(
  risk_pr_biop_lasso, "FP_biop", "LASSO", 
  test_biop, train_biop)

# Elastic Net
results_smote[["FP_recall_elasticnet"]] <- evaluate_model(
  risk_pr_recall_enet, "FP_recall", "ElasticNet", 
  test_recall, train_recall)

results_smote[["FP_biop_elasticnet"]] <- evaluate_model(
  risk_pr_biop_enet, "FP_biop", "ElasticNet", 
  test_biop, train_biop)

# Random Forest
results_smote[["FP_recall_rf"]] <- evaluate_model(
  rf_recall$predictions, "FP_recall", "RandomForest", 
  test_recall, train_recall)

results_smote[["FP_biop_rf"]] <- evaluate_model(
  rf_biop$predictions, "FP_biop", "RandomForest", 
  test_biop, train_biop)

# Gradient Boosting
results_smote[["FP_recall_gbm"]] <- evaluate_model(
  gbm_recall$predictions, "FP_recall", "GradientBoosting", 
  test_recall, train_recall)

results_smote[["FP_biop_gbm"]] <- evaluate_model(
  gbm_biop$predictions, "FP_biop", "GradientBoosting", 
  test_biop, train_biop)

# Neural Net
results_smote[["FP_recall_nnet"]] <- evaluate_model(
  as.vector(nnet_recall$predictions), "FP_recall", "NeuralNet", 
  test_recall, train_recall)

results_smote[["FP_biop_nnet"]] <- evaluate_model(
  as.vector(nnet_biop$predictions), "FP_biop", "NeuralNet", 
  test_biop, train_biop)
# Save with new filename
save(results_smote, file = file.path(results_dir, "eval_SMOTE_Feb2026/fp_models_evaluation_SMOTE_prevalence_f1.RData"))
# Save to disk
# save(results_smote, file = file.path(results_dir, "fp_models_evaluation_SMOTE.RData"))
