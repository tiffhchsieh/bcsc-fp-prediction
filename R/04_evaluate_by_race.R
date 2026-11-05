# ------------------------------------------------------------
# Title: Model Evaluation Stratified by Race/Ethnicity
# Author: Tiffany Hsieh 
# Date: 2025-05-21 (updated)
# ------------------------------------------------------------

# DESCRIPTION -------------------------------------------------
# Replacement for the earlier evaluation pipeline that adds
# stratification by a grouping variable (here: "racenci_new").
# It computes the same performance metrics (AUC, F1, Expected/Observed,
# Cox calibration intercept & slope) *within* each subgroup for every
# model, outcome, and imputed dataset, then pools across imputations
# using Rubin's rules.
# ------------------------------------------------------------

source("R/config.R")

# Load required libraries ---------------------------------------------------
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

# ------------------------------------------------------------
# 1. Metric computation for a single (model x outcome x group).
# ------------------------------------------------------------

evaluate_model <- function(predicted_probs, response_var, test_data) {
  test_data[[response_var]] <- as.numeric(as.character(test_data[[response_var]]))
  
  # --- AUC (Delong CI) ---
  roc_curve <- pROC::roc(test_data[[response_var]], predicted_probs, quiet = TRUE)
  auc_value <- as.numeric(pROC::auc(roc_curve))
  auc_ci    <- ci.auc(roc_curve, method = "delong")
  auc_ci_fmt <- sprintf("%.3f (%.3f, %.3f)", auc_value, auc_ci[1], auc_ci[3])
  
  # --- F1 (bootstrap 1000x) ---
  f1_score <- F1_Score(y_pred = ifelse(predicted_probs >= 0.5, 1, 0),
                       y_true = test_data[[response_var]], positive = "1")
  boot_f1 <- replicate(1000, {
    idx <- sample(seq_len(nrow(test_data)), replace = TRUE)
    F1_Score(y_pred = ifelse(predicted_probs[idx] >= 0.5, 1, 0),
             y_true = test_data[[response_var]][idx], positive = "1")
  })
  f1_ci <- quantile(boot_f1, c(0.025, 0.975), na.rm = TRUE)
  f1_ci_fmt <- sprintf("%.3f (%.3f, %.3f)", f1_score, f1_ci[1], f1_ci[2])
  
  # --- Expected / Observed ---
  expected <- sum(predicted_probs)
  observed <- sum(test_data[[response_var]])
  eo <- expected / observed
  eo_ci <- eo * exp(c(-1.96, 1.96) * sqrt(1 / observed))
  eo_ci_fmt <- sprintf("%.3f (%.3f, %.3f)", eo, eo_ci[1], eo_ci[2])
  
  # --- Cox calibration (intercept & slope) ---
  df <- data.frame(predicted = predicted_probs,
                   observed  = test_data[[response_var]]) %>%
    filter(predicted > 0 & predicted < 1)
  df$logit_pred <- log(df$predicted / (1 - df$predicted))
  
  intercept_model <- glm(observed ~ 1 + offset(logit_pred), data = df, family = binomial)
  slope_model     <- glm(observed ~ logit_pred,             data = df, family = binomial)
  
  intercept <- coef(intercept_model)[1]
  slope     <- coef(slope_model)[2]
  intercept_ci <- sprintf("%.3f (%.3f, %.3f)",
                          intercept,
                          intercept - 1.96 * summary(intercept_model)$coefficients[1,2],
                          intercept + 1.96 * summary(intercept_model)$coefficients[1,2])
  slope_ci     <- sprintf("%.3f (%.3f, %.3f)",
                          slope,
                          slope - 1.96 * summary(slope_model)$coefficients[2,2],
                          slope + 1.96 * summary(slope_model)$coefficients[2,2])
  
  list(auc = auc_value, auc_ci = auc_ci_fmt,
       f1  = f1_score,  f1_ci  = f1_ci_fmt,
       eo  = eo,        eo_ci  = eo_ci_fmt,
       cox_intercept = intercept_ci,
       cox_slope     = slope_ci)
}

# ------------------------------------------------------------
# 2. Evaluate all models, outcomes, imputations *and* groups.
# ------------------------------------------------------------

evaluate_all_models_by_imputation_grouped <- function(base_dir,
                                                      imputations = 1:10,
                                                      group_var    = "racenci_new",
                                                      group_levels = c("White","Black","Asian","Hispanic","Others"),
                                                      min_events   = 10) {
  model_list <- c("human_glm", "gen_glm", "lasso", "en", "rf", "gbm_model", "nnet")
  outcomes   <- c("FP_recall", "FP_biop")
  results_all <- list()
  
  for (i in imputations) {
    cat("\n=== Evaluating Imputation", i, "===\n")
    imp_dir <- file.path(base_dir, sprintf("imp%02d", i))
    
    # Load split data created during model-fitting
    load(file.path(base_dir, sprintf("imp%02d_split.RData", i)))
    
    # Subset test sets (retain ALL columns for grouping var)
    test_recall <- dat_m[dat_m$split_recall == "testing", ]
    test_biop   <- dat_m[dat_m$split_biop   == "testing", ]
    
    for (resp in outcomes) {
      test_data <- if (resp == "FP_recall") test_recall else test_biop
      
      # Determine groups to iterate over (ordered)
      groups <- if (is.null(group_levels)) {
        na.omit(unique(test_data[[group_var]]))
      } else {
        group_levels[group_levels %in% unique(test_data[[group_var]])]
      }
      
      for (grp in groups) {
        idx_grp <- which(test_data[[group_var]] == grp & !is.na(test_data[[group_var]]))
        grp_data <- test_data[idx_grp, ]
        
        # Skip tiny subgroups (insufficient positives for CIs)
        if (sum(grp_data[[resp]]) < min_events) {
          warning("Skipping group '", grp, "' in imp", i,
                  " (", sum(grp_data[[resp]]), " events <", min_events, ")")
          next
        }
        
        for (model in model_list) {
          pred_file <- file.path(imp_dir, paste0(resp, "_", model, ".RData"))
          if (!file.exists(pred_file)) {
            warning("Missing file: ", pred_file)
            next
          }
          
          load(pred_file)
          preds <- switch(model,
                          human_glm = if (exists("pr_human"))   as.numeric(pr_human)   else NULL,
                          gen_glm   = if (exists("pr_general")) as.numeric(pr_general) else NULL,
                          lasso     = if (exists("pr_lasso"))   as.numeric(pr_lasso)   else NULL,
                          en        = if (exists("pr_en"))      as.numeric(pr_en)      else NULL,
                          rf        = if (exists("pr_rf"))      as.numeric(pr_rf)      else NULL,
                          nnet      = if (exists("pr_nn"))      as.numeric(pr_nn)      else NULL,
                          gbm_model = if (exists("gbm")) gbm$predictions else NULL)
          
          if (is.null(preds)) {
            warning("No predictions for ", model, " | ", resp, " | imp", i)
            next
          }
          
          preds_grp <- preds[idx_grp]
          if (length(preds_grp) != nrow(grp_data)) {
            warning("Prediction/Test set length mismatch for ", model, " - ", grp,
                    " (imp", i, ")")
            next
          }
          
          result <- tryCatch({
            evaluate_model(preds_grp, resp, grp_data)
          }, error = function(e) {
            warning("Evaluation error (", model, ", ", grp, ", imp", i, "): ", e$message)
            return(NULL)
          })
          
          if (!is.null(result)) {
            key <- paste0("imp", i, "_", resp, "_", model, "_", grp)
            results_all[[key]] <- result
          }
        }
      }
    }
  }
  results_all
}

# ------------------------------------------------------------
# 3. Pool the metrics across imputations using Rubin's rules.
# ------------------------------------------------------------

pool_metrics_rubin_grouped <- function(results_all, imputations = 10) {
  # Remove the leading "impNN_" to identify unique (outcome_model_group) keys
  keys <- unique(sub("^imp[0-9]+_", "", names(results_all)))
  pooled_results <- list()
  
  extract_numeric <- function(val) as.numeric(sub(" .*", "", val))
  
  pool_metric <- function(estimates) {
    m <- length(estimates)
    q_bar <- mean(estimates)
    u_bar <- var(estimates) / m
    b_m   <- var(estimates)
    t_var <- u_bar + (1 + 1/m) * b_m
    ci    <- q_bar + c(-1.96, 1.96) * sqrt(t_var)
    list(mean = q_bar, ci = ci)
  }
  
  for (key in keys) {
    matched <- grep(paste0("_", key, "$"), names(results_all), value = TRUE)
    if (length(matched) < imputations) next  # skip if incomplete
    
    auc_vals       <- sapply(results_all[matched], function(x) x$auc)
    f1_vals        <- sapply(results_all[matched], function(x) x$f1)
    eo_vals        <- sapply(results_all[matched], function(x) x$eo)
    intercept_vals <- sapply(results_all[matched], function(x) extract_numeric(x$cox_intercept))
    slope_vals     <- sapply(results_all[matched], function(x) extract_numeric(x$cox_slope))
    
    pooled_auc       <- pool_metric(auc_vals)
    pooled_f1        <- pool_metric(f1_vals)
    pooled_eo        <- pool_metric(eo_vals)
    pooled_intercept <- pool_metric(intercept_vals)
    pooled_slope     <- pool_metric(slope_vals)
    
    pooled_results[[key]] <- list(
      auc           = sprintf("%.3f (%.3f, %.3f)", pooled_auc$mean,       pooled_auc$ci[1],       pooled_auc$ci[2]),
      f1            = sprintf("%.3f (%.3f, %.3f)", pooled_f1$mean,        pooled_f1$ci[1],        pooled_f1$ci[2]),
      eo            = sprintf("%.3f (%.3f, %.3f)", pooled_eo$mean,        pooled_eo$ci[1],        pooled_eo$ci[2]),
      cox_intercept = sprintf("%.3f (%.3f, %.3f)", pooled_intercept$mean, pooled_intercept$ci[1], pooled_intercept$ci[2]),
      cox_slope     = sprintf("%.3f (%.3f, %.3f)", pooled_slope$mean,     pooled_slope$ci[1],     pooled_slope$ci[2])
    )
  }
  pooled_results
}

# ------------------------------------------------------------
# 4. EXAMPLE USAGE -------------------------------------------
# ------------------------------------------------------------

 results_by_race <- evaluate_all_models_by_imputation_grouped(
   base_dir    = file.path(results_dir, "all_imputations_nov_2025"),
   imputations = 1:10,
   group_var   = "racenci_new"
 )
#
pooled_by_race <- pool_metrics_rubin_grouped(results_by_race, imputations = 10)
#
save(pooled_by_race, results_by_race,
      file = file.path(results_dir, "eval_by_race.RData"))

# ------------------------------------------------------------
# END OF FILE ------------------------------------------------
