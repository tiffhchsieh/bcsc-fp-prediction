################################################################################
## Evaluate risk prediction models across 10 imputed datasets
## Date: 01/12/2026
## Programmer: Tiffany Hsieh 
## Modification: Maximize F1 score per RH's comment during the LWG meeting
################################################################################
source("R/config.R")

library(pROC)
library(MLmetrics)
library(precrec)
library(ggplot2)
library(dplyr)
library(scales)
# Set directories
load(file.path(data_dir, "AB468b_data_noPHI_shared_randomForests.RData"))
base_dir <- file.path(results_dir, "all_imputations_nov_2025")
output_dir <- results_dir
dir.create(output_dir, showWarnings = FALSE)

# Define models and outcomes
model_list <- c("human_glm", "gen_glm", "lasso", "en", "rf", "gbm_model", "nnet")
outcomes <- c("FP_recall", "FP_biop")

## evaluate_model: F1 uses the outcome prevalence in the training data as the threshold
evaluate_model <- function(predicted_probs, response_var, model_name, test_data, train_data) {
  
  test_data[[response_var]] <- as.numeric(as.character(test_data[[response_var]]))
  train_data[[response_var]] <- as.numeric(as.character(train_data[[response_var]]))
  
  # AUC and CI
  roc_curve <- pROC::roc(test_data[[response_var]], predicted_probs, quiet = TRUE)
  auc_value <- as.numeric(pROC::auc(roc_curve))
  auc_ci <- ci.auc(roc_curve, method = "delong")
  auc_ci_fmt <- sprintf("%.3f (%.3f, %.3f)", auc_value, auc_ci[1], auc_ci[3])
  
  # ============================================================
  # Prevalence-based threshold (from TRAINING data)
  # ============================================================
  prevalence_threshold <- mean(train_data[[response_var]])
  
  # F1 Score using prevalence as threshold
  f1_score <- F1_Score(
    y_pred = ifelse(predicted_probs >= prevalence_threshold, 1, 0),
    y_true = test_data[[response_var]], positive = "1"
  )
  
  # F1 Score with bootstrapping
  boot_f1 <- replicate(1000, {
    idx <- sample(seq_len(nrow(test_data)), replace = TRUE)
    tryCatch({
      F1_Score(
        y_pred = ifelse(predicted_probs[idx] >= prevalence_threshold, 1, 0),
        y_true = test_data[[response_var]][idx], positive = "1"
      )
    }, error = function(e) NA)
  })
  f1_ci <- quantile(boot_f1, c(0.025, 0.975), na.rm = TRUE)
  f1_ci_fmt <- sprintf("%.3f (%.3f, %.3f)", f1_score, f1_ci[1], f1_ci[2])
  
  # ============================================================
  
  # Expected/Observed
  expected <- sum(predicted_probs)
  observed <- sum(test_data[[response_var]])
  eo <- expected / observed
  eo_ci <- eo * exp(c(-1.96, 1.96) * sqrt(1 / observed))
  eo_ci_fmt <- sprintf("%.3f (%.3f, %.3f)", eo, eo_ci[1], eo_ci[2])
  
  # Cox calibration
  df <- data.frame(predicted = predicted_probs, observed = test_data[[response_var]])
  df <- df[df$predicted > 0 & df$predicted < 1, ]
  df$logit_pred <- log(df$predicted / (1 - df$predicted))
  
  intercept_model <- glm(observed ~ 1 + offset(logit_pred), data = df, family = binomial)
  slope_model <- glm(observed ~ logit_pred, data = df, family = binomial)
  intercept <- coef(intercept_model)[1]
  slope <- coef(slope_model)[2]
  intercept_ci <- sprintf("%.3f (%.3f, %.3f)", intercept, intercept - 1.96 * summary(intercept_model)$coefficients[1,2], intercept + 1.96 * summary(intercept_model)$coefficients[1,2])
  slope_ci <- sprintf("%.3f (%.3f, %.3f)", slope, slope - 1.96 * summary(slope_model)$coefficients[2,2], slope + 1.96 * summary(slope_model)$coefficients[2,2])
  
  return(list(
    auc = auc_value,
    auc_ci = auc_ci_fmt,
    f1 = f1_score,
    f1_ci = f1_ci_fmt,
    f1_threshold = prevalence_threshold,
    eo = eo,
    eo_ci = eo_ci_fmt,
    cox_intercept = intercept_ci,
    cox_slope = slope_ci
  ))
}

evaluate_all_models_by_imputation <- function(base_dir, imputations = 1:2) {
  model_list <- c("human_glm", "gen_glm", "lasso", "en", "rf", "gbm_model", "nnet")
  outcomes <- c("FP_recall", "FP_biop")
  results_all <- list()
  
  for (i in imputations) {
    cat("\n=== Evaluating Models for Imputation", i, "===\n")
    imp_dir <- file.path(base_dir, sprintf("imp%02d", i))
    
    # Load split data
    load(file.path(base_dir, sprintf("imp%02d_split.RData", i)))
    
    # Subset test AND training sets
    test_recall  <- dat_m[dat_m$split_recall == "testing", ]
    train_recall <- dat_m[dat_m$split_recall == "training", ]
    test_biop    <- dat_m[dat_m$split_biop == "testing", ]
    train_biop   <- dat_m[dat_m$split_biop == "training", ]
    
    for (resp in outcomes) {
      test_data  <- if (resp == "FP_recall") test_recall else test_biop
      train_data <- if (resp == "FP_recall") train_recall else train_biop
      
      for (model in model_list) {
        pred_file <- file.path(imp_dir, paste0(resp, "_", model, ".RData"))
        if (!file.exists(pred_file)) {
          warning("Missing file: ", pred_file)
          next
        }
        
        load(pred_file)
        preds <- switch(model,
                        "human_glm" = if (exists("pr_human")) as.numeric(pr_human) else NULL,
                        "gen_glm"   = if (exists("pr_general")) as.numeric(pr_general) else NULL,
                        "lasso"     = if (exists("pr_lasso")) as.numeric(pr_lasso) else NULL,
                        "en"        = if (exists("pr_en")) as.numeric(pr_en) else NULL,
                        "rf"        = if (exists("pr_rf")) as.numeric(pr_rf) else NULL,
                        "nnet"      = if (exists("pr_nn")) as.numeric(pr_nn) else NULL,
                        "gbm_model" = if (exists("gbm")) gbm$predictions else NULL
        )
        
        if (is.null(preds)) {
          warning("No predictions found for ", model, " | ", resp, " | imp ", i)
          next
        }
        
        result <- tryCatch({
          evaluate_model(preds, resp, model, test_data, train_data)
        }, error = function(e) {
          warning("Evaluation error for ", model, " | ", resp, " | imp ", i, ": ", e$message)
          return(NULL)
        })
        
        if (!is.null(result)) {
          key <- paste0("imp", i, "_", resp, "_", model)
          results_all[[key]] <- result
        }
      }
    }
  }
  
  return(results_all)
}


### Rubin
pool_metrics_rubin <- function(results_all, imputations = 2) {
  keys <- unique(gsub("^imp[0-9]+_", "", names(results_all)))
  pooled_results <- list()
  
  for (key in keys) {
    matched <- grep(paste0("_", key, "$"), names(results_all), value = TRUE)
    if (length(matched) < imputations) next
    
    extract_numeric <- function(val) as.numeric(gsub(" .*", "", val))  # Extract number before space in CI string
    
    auc_vals <- sapply(results_all[matched], function(x) x$auc)
    f1_vals <- sapply(results_all[matched], function(x) x$f1)
    eo_vals <- sapply(results_all[matched], function(x) x$eo)
    intercept_vals <- sapply(results_all[matched], function(x) extract_numeric(x$cox_intercept))
    slope_vals <- sapply(results_all[matched], function(x) extract_numeric(x$cox_slope))
    
    # Rubin's Rules pooling
    pool_metric <- function(estimates) {
      q_bar <- mean(estimates)
      u_bar <- var(estimates) / length(estimates)
      b_m <- var(estimates)
      t_var <- u_bar + (1 + 1 / length(estimates)) * b_m
      ci <- q_bar + c(-1.96, 1.96) * sqrt(t_var)
      list(mean = q_bar, ci = ci)
    }
    
    pooled_auc <- pool_metric(auc_vals)
    pooled_f1 <- pool_metric(f1_vals)
    pooled_eo <- pool_metric(eo_vals)
    pooled_intercept <- pool_metric(intercept_vals)
    pooled_slope <- pool_metric(slope_vals)
    
    # Add threshold extraction
    threshold_vals <- sapply(results_all[matched], function(x) x$f1_threshold)
    pooled_threshold <- pool_metric(threshold_vals)
    
    
    pooled_results[[key]] <- list(
      auc = sprintf("%.3f (%.3f, %.3f)", pooled_auc$mean, pooled_auc$ci[1], pooled_auc$ci[2]),
      f1 = sprintf("%.3f (%.3f, %.3f)", pooled_f1$mean, pooled_f1$ci[1], pooled_f1$ci[2]),
      f1_threshold = sprintf("%.3f (%.3f, %.3f)", pooled_threshold$mean, pooled_threshold$ci[1], pooled_threshold$ci[2]),
      eo = sprintf("%.3f (%.3f, %.3f)", pooled_eo$mean, pooled_eo$ci[1], pooled_eo$ci[2]),
      cox_intercept = sprintf("%.3f (%.3f, %.3f)", pooled_intercept$mean, pooled_intercept$ci[1], pooled_intercept$ci[2]),
      cox_slope = sprintf("%.3f (%.3f, %.3f)", pooled_slope$mean, pooled_slope$ci[1], pooled_slope$ci[2])
    )
  }
  
  return(pooled_results)
}

results_all <- evaluate_all_models_by_imputation(
  base_dir = file.path(results_dir, "all_imputations_nov_2025"),
  imputations = 1:2
)

# Run the updated pooling
pooled_summary <- pool_metrics_rubin(results_all, imputations = 2)


results_additional <- evaluate_all_models_by_imputation(
  base_dir = file.path(results_dir, "all_imputations_nov_2025"),
  imputations = 3:10
)
results_combined <- c(results_all, results_additional)
pooled_summary_10imp <- pool_metrics_rubin(results_combined, imputations = 10)
# ==============================================================================
# SAVE RESULTS - UPDATED FILENAME TO REFLECT PREVALENCE-BASED THRESHOLD
# ==============================================================================
save(
  pooled_summary,
  pooled_summary_10imp,
  results_all,
  results_additional,
  results_combined,
  file = file.path(results_dir, "eval_results_prevalence_f1.RData")  
)