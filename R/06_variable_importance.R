################################################################################
## Variable Importance Analysis for Well-Calibrated Models
## Date: 01/20/2026 Revised per LWG 
## Purpose: Extract variable selection frequency and effect sizes
##          for LASSO, Elastic Net, and Gradient Boosting
################################################################################

source("R/config.R")

library(dplyr)
library(ggplot2)
library(tidyr)
library(stringr)
library(tibble)  # for tribble

## ------------------------------- 
## Directories
## ------------------------------- 
base_dir <- file.path(results_dir, "all_imputations_nov_2025")
output_dir <- file.path(results_dir, "variable_importance_feb_2026")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

var_label_lookup <- tribble(
  ~base_variable,                              ~label,
  # Adjuvant therapy
  "adjthrpy_1both",                            "Adjuvant therapy: both chemo and hormonal",
  "adjthrpy_1chemoOnly",                       "Adjuvant therapy: chemo only",
  "adjthrpy_1hormonalOnly",                    "Adjuvant therapy: hormonal only",
  
  # Age at exam (splines)
  "age_ns.1",                                  "Age at exam: spline 1",
  "age_ns.2",                                  "Age at exam: spline 2",
  "age_ns.3",                                  "Age at exam: spline 3",
  "age_ns.4",                                  "Age at exam: spline 4",
  
  # BMI (splines)
  "bmi_ns.1",                                  "BMI: spline 1",
  "bmi_ns.2",                                  "BMI: spline 2",
  "bmi_ns.3",                                  "BMI: spline 3",
  "bmi_ns.4",                                  "BMI: spline 4",
  
  # Breast density main effects
  "density_c2",                                "Breast density: B",
  "density_c3",                                "Breast density: C",
  "density_c4",                                "Breast density: D",
  
  # Breast density * BMI interactions
  "density_c2:bmi_ns.1",                       "Breast density B * BMI, spline 1",
  "density_c2:bmi_ns.2",                       "Breast density B * BMI, spline 2",
  "density_c2:bmi_ns.3",                       "Breast density B * BMI, spline 3",
  "density_c2:bmi_ns.4",                       "Breast density B * BMI, spline 4",
  "density_c3:bmi_ns.1",                       "Breast density C * BMI, spline 1",
  "density_c3:bmi_ns.2",                       "Breast density C * BMI, spline 2",
  "density_c3:bmi_ns.3",                       "Breast density C * BMI, spline 3",
  "density_c3:bmi_ns.4",                       "Breast density C * BMI, spline 4",
  "density_c4:bmi_ns.1",                       "Breast density D * BMI, spline 1",
  "density_c4:bmi_ns.2",                       "Breast density D * BMI, spline 2",
  "density_c4:bmi_ns.3",                       "Breast density D * BMI, spline 3",
  "density_c4:bmi_ns.4",                       "Breast density D * BMI, spline 4",
  
  # Age at diagnosis (splines)
  "dxage_ns.1",                                "Age at diagnosis: spline 1",
  "dxage_ns.2",                                "Age at diagnosis: spline 2",
  "dxage_ns.3",                                "Age at diagnosis: spline 3",
  "dxage_ns.4",                                "Age at diagnosis: spline 4",
  
  # Diagnosis year (splines)
  "dxyear_ns.1",                               "Diagnosis year: spline 1",
  "dxyear_ns.2",                               "Diagnosis year: spline 2",
  "dxyear_ns.3",                               "Diagnosis year: spline 3",
  "dxyear_ns.4",                               "Diagnosis year: spline 4",
  
  # ER/PR status
  "erplus1ERnPRp",                             "ER/PR status: -/+",
  "erplus1ERpPRn",                             "ER/PR status: +/-",
  "erplus1ERpPRp",                             "ER/PR status: +/+",
  
  # Family history
  "famhx_cyes",                                "Family history: yes",
  
  # Grade
  "grade1_c2",                                 "Grade: 2",
  "grade1_c3",                                 "Grade: 3",
  
  # Histology
  "hist1_newnon-ductal",                       "Histology: non-ductal",
  
  # Month since previous mammography (splines)
  "log_prvmos_ns.1",                           "Month since previous mammography: spline 1",
  "log_prvmos_ns.2",                           "Month since previous mammography: spline 2",
  "log_prvmos_ns.3",                           "Month since previous mammography: spline 3",
  
  # Mammography modality
  "mammtype_newfilm",                          "Mammography modality: film mammography",
  "mammtype_newDBT",                           "Mammography modality: digital breast tomosynthesis",
  
  # Menopausal status
  "menopht_2catPre",                           "Peri/pre-menopausal",
  
  # Mode of detection
  "mod1_newclinical",                          "Mode of detection: clinical",
  "mod1_newinterval",                          "Mode of detection: interval",
  
  # Month since previous surveillance mammography
  "mossinceprvsurvmm_cat1st surveillance",     "Month since previous surveillance mammography: 1st surveillance mammogram",
  "mossinceprvsurvmm_cat3-8",                  "Month since previous surveillance mammography: 3-8 months",
  "mossinceprvsurvmm_cat15-23",                "Month since previous surveillance mammography: 15-23 months",
  "mossinceprvsurvmm_cat24+",                  "Month since previous surveillance mammography: ≥24 months",
  
  # Previous false positives
  "prv_FP_biop",                               "Previous false positive biopsy recommendation",
  "prv_FP_recall",                             "Previous false positive recall",
  
  # Race/ethnicity main effects
  "racenci_newAsian",                          "Race and ethnicity: Non-Hispanic Asian/PI",
  "racenci_newBlack",                          "Race and ethnicity: Non-Hispanic Black",
  "racenci_newHispanic",                       "Race and ethnicity: Hispanic",
  "racenci_newOthers",                         "Race and ethnicity: Non-Hispanic Others",
  
  # Race * BMI interactions
  "racenci_newAsian:bmi_ns.1",                 "Non-Hispanic Asian/PI * BMI, spline 1",
  "racenci_newAsian:bmi_ns.2",                 "Non-Hispanic Asian/PI * BMI, spline 2",
  "racenci_newAsian:bmi_ns.3",                 "Non-Hispanic Asian/PI * BMI, spline 3",
  "racenci_newAsian:bmi_ns.4",                 "Non-Hispanic Asian/PI * BMI, spline 4",
  "racenci_newBlack:bmi_ns.1",                 "Non-Hispanic Black * BMI, spline 1",
  "racenci_newBlack:bmi_ns.2",                 "Non-Hispanic Black * BMI, spline 2",
  "racenci_newBlack:bmi_ns.3",                 "Non-Hispanic Black * BMI, spline 3",
  "racenci_newBlack:bmi_ns.4",                 "Non-Hispanic Black * BMI, spline 4",
  "racenci_newHispanic:bmi_ns.1",              "Hispanic * BMI, spline 1",
  "racenci_newHispanic:bmi_ns.2",              "Hispanic * BMI, spline 2",
  "racenci_newHispanic:bmi_ns.3",              "Hispanic * BMI, spline 3",
  "racenci_newHispanic:bmi_ns.4",              "Hispanic * BMI, spline 4",
  "racenci_newOthers:bmi_ns.1",                "Non-Hispanic Others * BMI, spline 1",
  "racenci_newOthers:bmi_ns.2",                "Non-Hispanic Others * BMI, spline 2",
  "racenci_newOthers:bmi_ns.3",                "Non-Hispanic Others * BMI, spline 3",
  "racenci_newOthers:bmi_ns.4",                "Non-Hispanic Others * BMI, spline 4",
  
  # Race * breast density interactions
  "racenci_newAsian:density_c2",               "Non-Hispanic Asian/PI * Breast density B",
  "racenci_newAsian:density_c3",               "Non-Hispanic Asian/PI * Breast density C",
  "racenci_newAsian:density_c4",               "Non-Hispanic Asian/PI * Breast density D",
  "racenci_newBlack:density_c2",               "Non-Hispanic Black * Breast density B",
  "racenci_newBlack:density_c3",               "Non-Hispanic Black * Breast density C",
  "racenci_newBlack:density_c4",               "Non-Hispanic Black * Breast density D",
  "racenci_newHispanic:density_c2",            "Hispanic * Breast density B",
  "racenci_newHispanic:density_c3",            "Hispanic * Breast density C",
  "racenci_newHispanic:density_c4",            "Hispanic * Breast density D",
  "racenci_newOthers:density_c2",              "Non-Hispanic Others * Breast density B",
  "racenci_newOthers:density_c3",              "Non-Hispanic Others * Breast density C",
  "racenci_newOthers:density_c4",              "Non-Hispanic Others * Breast density D",
  
  # Radiation treatment
  "rt_newwithout RT",                          "Radiation treatment: no",
  
  # Surgery
  "srg1_newBCS",                               "Surgery treatment: Breast conserving surgery",
  
  # AJCC stage
  "stage1DCIS",                                "AJCC stage: I",
  "stage1II NOS and IIA",                      "AJCC stage: IIA or II not specified",
  "stage1IIB+",                                "AJCC stage: IIB+"
)

add_labels <- function(df) {
  df %>%
    left_join(var_label_lookup, by = "base_variable") %>%
    mutate(
      label = if_else(is.na(label), base_variable, label)
    )
}

## ------------------------------- 
## Helper: Extract base variable name from spline terms
## ------------------------------- 
extract_base_variable <- function(var_name) {
  # Remove spline suffixes like ": spline 1", ": spline 2", etc.
  base <- gsub(": spline [0-9]+$", "", var_name)
  # Also handle other common patterns
  base <- gsub("_spline[0-9]+$", "", base)
  base <- gsub("\\.spline[0-9]+$", "", base)
  return(base)
}

## ------------------------------- 
## Function 1: Extract coefficients from LASSO/EN models
## ------------------------------- 
extract_lasso_en_coefficients <- function(base_dir, imputations = 1:10) {
  
  all_coefs <- list()
  
  for (i in imputations) {
    cat("Processing imputation", i, "\n")
    imp_dir <- file.path(base_dir, sprintf("imp%02d", i))
    
    for (outcome in c("FP_recall", "FP_biop")) {
      for (model in c("lasso", "en")) {
        
        # Load the model file
        model_file <- file.path(imp_dir, paste0(outcome, "_", model, ".RData"))
        
        if (!file.exists(model_file)) {
          warning("File not found: ", model_file)
          next
        }
        
        # Load the file - need to check what objects are saved
        env <- new.env()
        load(model_file, envir = env)
        
        # Try to find the fitted model object
        # Models are saved as cv_lasso, cv_en
        model_obj <- NULL
        possible_names <- c(
          paste0("cv_", model),      # This is the actual name: cv_lasso or cv_en
          paste0("fit_", model), 
          paste0(model, "_model"),
          paste0(model, "_fit"),
          "fit", "model", "mod",
          model                       # "lasso" or "en"
        )
        
        for (name in possible_names) {
          if (exists(name, envir = env)) {
            obj <- get(name, envir = env)
            # Check if it's a valid model object
            if (any(c("glmnet", "cv.glmnet") %in% class(obj))) {
              model_obj <- obj
              cat("  Found model object '", name, "' for", outcome, "-", model, "\n")
              break
            }
          }
        }
        
        if (is.null(model_obj)) {
          warning("Could not find model object in ", model_file)
          warning("Available objects: ", paste(ls(envir = env), collapse = ", "))
          warning("Looking for cv.glmnet or glmnet object")
          next
        }
        
        # Extract coefficients (for glmnet models)
        tryCatch({
          if ("cv.glmnet" %in% class(model_obj)) {
            # For cv.glmnet
            coefs <- coef(model_obj, s = "lambda.min")
            coef_df <- data.frame(
              variable = rownames(coefs),
              coefficient = as.numeric(coefs),
              stringsAsFactors = FALSE
            )
          } else if ("glmnet" %in% class(model_obj)) {
            # Use lambda.min; fall back to the first lambda if missing
            if (!is.null(model_obj$lambda.min)) {
              coefs <- coef(model_obj, s = model_obj$lambda.min)
            } else {
              # Use the first lambda if lambda.min is not available
              coefs <- coef(model_obj, s = model_obj$lambda[1])
            }
            coef_df <- data.frame(
              variable = rownames(coefs),
              coefficient = as.numeric(coefs),
              stringsAsFactors = FALSE
            )
          } else {
            warning("Unknown model class: ", paste(class(model_obj), collapse = ", "))
            next
          }
          
          # Add metadata
          coef_df$imputation <- i
          coef_df$outcome <- outcome
          coef_df$model <- model
          coef_df$base_variable <- sapply(coef_df$variable, extract_base_variable)
          coef_df$selected <- abs(coef_df$coefficient) > 1e-10  # Non-zero coefficient
          
          all_coefs[[length(all_coefs) + 1]] <- coef_df
          
        }, error = function(e) {
          warning("Error extracting coefficients from ", model_file, ": ", e$message)
        })
      }
    }
  }
  
  # Combine all results
  if (length(all_coefs) > 0) {
    all_coefs_df <- bind_rows(all_coefs)
    return(all_coefs_df)
  } else {
    warning("No coefficients extracted!")
    return(NULL)
  }
}

## ------------------------------- 
## Function 2: Calculate selection frequency
## ------------------------------- 
calculate_selection_frequency <- function(coef_df, imputations = 10) {
  
  # For each base variable, check if it was selected (any spline term non-zero)
  selection_by_imp <- coef_df %>%
    filter(variable != "(Intercept)") %>%
    group_by(imputation, outcome, model, base_variable) %>%
    summarise(
      selected = any(selected),
      .groups = "drop"
    )
  
  # Calculate frequency across imputations
  freq_summary <- selection_by_imp %>%
    group_by(outcome, model, base_variable) %>%
    summarise(
      selection_count = sum(selected),
      selection_frequency = sum(selected) / imputations,
      .groups = "drop"
    ) %>%
    arrange(outcome, model, desc(selection_frequency))
  
  return(list(
    by_imputation = selection_by_imp,
    frequency_summary = freq_summary
  ))
}

## ------------------------------- 
## Function 3: Calculate average effect sizes
## ------------------------------- 
calculate_effect_sizes <- function(coef_df) {
  
  # Calculate effect size for each variable (averaging across imputations)
  effect_sizes <- coef_df %>%
    filter(variable != "(Intercept)", selected) %>%
    group_by(outcome, model, variable, base_variable) %>%
    summarise(
      mean_coefficient = mean(coefficient),
      abs_mean_coefficient = mean(abs(coefficient)),
      n_imputations = n(),
      .groups = "drop"
    )
  
  # For variables with spline terms, find the maximum effect size
  effect_sizes_grouped <- effect_sizes %>%
    group_by(outcome, model, base_variable) %>%
    summarise(
      max_abs_coefficient = max(abs_mean_coefficient),
      primary_term = variable[which.max(abs_mean_coefficient)],
      all_terms = paste(variable, collapse = "; "),
      mean_coefficient = mean_coefficient[which.max(abs_mean_coefficient)],
      .groups = "drop"
    ) %>%
    arrange(outcome, model, desc(max_abs_coefficient))
  
  return(list(
    by_term = effect_sizes,
    by_variable = effect_sizes_grouped
  ))
}

## ------------------------------- 
## Function 4: Extract GBM variable importance
## ------------------------------- 
extract_gbm_importance <- function(base_dir, imputations = 1:10) {
  
  # Load required packages
  if (!require(data.table)) {
    install.packages("data.table")
    library(data.table)
  }
  
  all_importance <- list()
  
  for (i in imputations) {
    imp_dir <- file.path(base_dir, sprintf("imp%02d", i))
    
    for (outcome in c("FP_recall", "FP_biop")) {
      
      model_file <- file.path(imp_dir, paste0(outcome, "_gbm_model.RData"))
      
      if (!file.exists(model_file)) {
        warning("File not found: ", model_file)
        next
      }
      
      # Load the file
      env <- new.env()
      load(model_file, envir = env)
      
      # Models are saved as "gbm"
      model_obj <- NULL
      if (exists("gbm", envir = env)) {
        model_obj <- get("gbm", envir = env)
        cat("  Found model 'gbm' for", outcome, "- imputation", i, "\n")
      } else {
        # Try other object names
        for (name in c("gbm_model", "model", "fit")) {
          if (exists(name, envir = env)) {
            model_obj <- get(name, envir = env)
            cat("  Found model '", name, "' for", outcome, "- imputation", i, "\n")
            break
          }
        }
      }
      
      if (is.null(model_obj)) {
        warning("Could not find model object in ", model_file)
        warning("Available objects: ", paste(ls(envir = env), collapse = ", "))
        next
      }
      
      # Extract importance
      imp_df <- NULL
      
      # Importance is stored in $variable.importance (data.table or list)
      if (!is.null(model_obj$variable.importance)) {
        tryCatch({
          vi <- model_obj$importance
          
          # Convert data.table to data.frame if needed
          if (inherits(vi, "data.table") || is.data.frame(vi)) {
            vi_df <- as.data.frame(vi)
            
            # Look for columns with variable names and importance values
            var_col <- NULL
            imp_col <- NULL
            
            # Common column names for variables
            for (col in names(vi_df)) {
              if (tolower(col) %in% c("feature", "variable", "var", "name", "predictor")) {
                var_col <- col
                break
              }
            }
            
            # Common column names for importance
            for (col in names(vi_df)) {
              if (tolower(col) %in% c("gain", "importance", "weight", "cover", "frequency", "total_gain")) {
                imp_col <- col
                break
              }
            }
            
            # If columns can't be identified by name, use position
            if (is.null(var_col) && ncol(vi_df) >= 1) {
              var_col <- names(vi_df)[1]
            }
            if (is.null(imp_col) && ncol(vi_df) >= 2) {
              imp_col <- names(vi_df)[2]
            }
            
            if (!is.null(var_col) && !is.null(imp_col)) {
              imp_df <- data.frame(
                variable = as.character(vi_df[[var_col]]),
                importance = as.numeric(vi_df[[imp_col]]),
                stringsAsFactors = FALSE
              )
              cat("    Extracted", nrow(imp_df), "variables from $variable.importance\n")
            } else {
              warning("Could not identify variable and importance columns in data.table")
              cat("    Available columns:", paste(names(vi_df), collapse = ", "), "\n")
            }
          }
          # Handle if it's a list structure
          else if (is.list(vi)) {
            # Try common list element names
            if (!is.null(vi$Feature) && !is.null(vi$Gain)) {
              imp_df <- data.frame(
                variable = as.character(vi$Feature),
                importance = as.numeric(vi$Gain),
                stringsAsFactors = FALSE
              )
            } else if (!is.null(vi$Feature) && !is.null(vi$Importance)) {
              imp_df <- data.frame(
                variable = as.character(vi$Feature),
                importance = as.numeric(vi$Importance),
                stringsAsFactors = FALSE
              )
            } else if (length(vi) >= 2) {
              # Use first two elements if structure is unclear
              imp_df <- data.frame(
                variable = as.character(vi[[1]]),
                importance = as.numeric(vi[[2]]),
                stringsAsFactors = FALSE
              )
            }
            if (!is.null(imp_df)) {
              cat("    Extracted", nrow(imp_df), "variables from list structure\n")
            }
          }
        }, error = function(e) {
          warning("Error extracting from $variable.importance: ", e$message)
        })
      }
      
      # Alternative: check $importance field
      if (is.null(imp_df) && !is.null(model_obj$importance)) {
        tryCatch({
          imp <- model_obj$importance
          
          if (is.data.frame(imp) || inherits(imp, "data.table")) {
            imp_df <- as.data.frame(imp)
            if (ncol(imp_df) >= 2) {
              # First column: variables; second: importance
              imp_df <- data.frame(
                variable = as.character(imp_df[,1]),
                importance = as.numeric(imp_df[,2]),
                stringsAsFactors = FALSE
              )
              cat("    Extracted", nrow(imp_df), "variables from $importance\n")
            }
          }
        }, error = function(e) {
          warning("Error extracting from $importance: ", e$message)
        })
      }
      
      if (is.null(imp_df)) {
        warning("Could not extract importance from model in ", model_file)
        next
      }
      
      # Clean and validate
      imp_df <- imp_df[!is.na(imp_df$importance) & imp_df$importance > 0, ]
      
      if (nrow(imp_df) == 0) {
        warning("No valid importance values found for ", model_file)
        next
      }
      
      # Normalize to sum to 100
      imp_df$importance <- 100 * imp_df$importance / sum(imp_df$importance)
      
      # Add metadata
      imp_df$imputation <- i
      imp_df$outcome <- outcome
      imp_df$base_variable <- sapply(imp_df$variable, extract_base_variable)
      
      all_importance[[length(all_importance) + 1]] <- imp_df
    }
  }
  
  if (length(all_importance) > 0) {
    all_importance_df <- bind_rows(all_importance)
    return(all_importance_df)
  } else {
    return(NULL)
  }
}

## ------------------------------- 
## Function 5: Calculate GBM average importance and rank
## ------------------------------- 
calculate_gbm_importance_summary <- function(importance_df) {
  
  # Average importance across imputations, grouping by base variable
  importance_summary <- importance_df %>%
    group_by(outcome, base_variable) %>%
    summarise(
      mean_importance = mean(importance),
      sd_importance = sd(importance),
      n_imputations = n(),
      .groups = "drop"
    ) %>%
    group_by(outcome) %>%
    arrange(outcome, desc(mean_importance)) %>%
    mutate(rank = row_number()) %>%
    ungroup()
  
  return(importance_summary)
}

## ------------------------------- 
## Function 6: Create selection frequency plot
## ------------------------------- 
plot_selection_frequency <- function(freq_summary, outcome_filter = NULL, top_n = 30) {
  
  plot_data <- freq_summary
  if (!is.null(outcome_filter)) {
    plot_data <- plot_data %>% filter(outcome == outcome_filter)
  }
  
  # Keep only top N variables per model/outcome combination
  plot_data <- plot_data %>%
    group_by(outcome, model) %>%
    top_n(n = top_n, wt = selection_frequency) %>%
    ungroup()
  
  # Create separate panels for each outcome
  p <- ggplot(plot_data, aes(x = reorder(base_variable, selection_frequency), 
                              y = selection_frequency, 
                              fill = selection_frequency)) +
    geom_bar(stat = "identity") +
    facet_grid(outcome ~ model, scales = "free_y", space = "free_y") +
    coord_flip() +
    scale_fill_gradient2(low = "white", mid = "lightblue", high = "darkblue",
                         midpoint = 0.5, limits = c(0, 1),
                         name = "Frequency") +
    labs(
      title = "Variable Selection Frequency Across Imputations",
      subtitle = paste0("LASSO and Elastic Net models (Top ", top_n, " variables)"),
      x = "Variable",
      y = "Selection Frequency"
    ) +
    theme_minimal() +
    theme(
      axis.text.y = element_text(size = 8),
      strip.text = element_text(face = "bold"),
      legend.position = "right"
    )
  
  return(p)
}

## ------------------------------- 
## Function 7: Create combined results table
## ------------------------------- 
create_combined_results_table <- function(lasso_effects, en_effects, gbm_importance) {
  
  # Prepare LASSO results
  lasso_table <- lasso_effects$by_variable %>%
    dplyr::select(outcome, base_variable, mean_coefficient, max_abs_coefficient) %>%
    rename(
      lasso_coefficient = mean_coefficient,
      lasso_abs_coefficient = max_abs_coefficient
    )
  
  # Prepare Elastic Net results
  en_table <- en_effects$by_variable %>%
    dplyr::select(outcome, base_variable, mean_coefficient, max_abs_coefficient) %>%
    rename(
      en_coefficient = mean_coefficient,
      en_abs_coefficient = max_abs_coefficient
    )
  
  # Prepare GBM results
  gbm_table <- gbm_importance %>%
    dplyr::select(outcome, base_variable, mean_importance, rank) %>%
    rename(
      gbm_importance = mean_importance,
      gbm_rank = rank
    )
  
  # Combine all
  combined <- full_join(lasso_table, en_table, 
                        by = c("outcome", "base_variable")) %>%
    full_join(gbm_table, by = c("outcome", "base_variable")) %>%
    arrange(outcome, desc(pmax(lasso_abs_coefficient, en_abs_coefficient, na.rm = TRUE)))
  
  return(combined)
}

## ------------------------------- 
## MAIN EXECUTION
## ------------------------------- 

cat("\n=== Starting Variable Importance Analysis ===\n")
cat("Base directory:", base_dir, "\n")
cat("Output directory:", output_dir, "\n\n")

cat("=== Extracting LASSO and Elastic Net Coefficients ===\n")
lasso_en_coefs <- extract_lasso_en_coefficients(base_dir, imputations = 1:10)

if (!is.null(lasso_en_coefs)) {
  
  # Check extracted results
  cat("\nExtracted coefficients summary:\n")
  cat("Total rows:", nrow(lasso_en_coefs), "\n")
  cat("Unique outcomes:", unique(lasso_en_coefs$outcome), "\n")
  cat("Unique models:", unique(lasso_en_coefs$model), "\n")
  cat("Imputations:", sort(unique(lasso_en_coefs$imputation)), "\n")
  
  cat("\n=== Calculating Selection Frequencies ===\n")
  selection_freq <- calculate_selection_frequency(lasso_en_coefs, imputations = 10)
  
  cat("\n=== Calculating Effect Sizes ===\n")
  lasso_effects <- calculate_effect_sizes(
    lasso_en_coefs %>% filter(model == "lasso")
  )
  en_effects <- calculate_effect_sizes(
    lasso_en_coefs %>% filter(model == "en")
  )
  
  cat("\n=== Creating Selection Frequency Plot ===\n")
  freq_plot <- plot_selection_frequency(selection_freq$frequency_summary)
  
  ggsave(
    file.path(output_dir, "selection_frequency_plot.png"),
    freq_plot,
    width = 12, height = 10, dpi = 300
  )
  cat("Plot saved to:", file.path(output_dir, "selection_frequency_plot.png"), "\n")
  
  # Save frequency table
  write.csv(
    selection_freq$frequency_summary,
    file.path(output_dir, "selection_frequency_table.csv"),
    row.names = FALSE
  )
  
  # Save effect size tables
  write.csv(
    lasso_effects$by_variable,
    file.path(output_dir, "lasso_effect_sizes.csv"),
    row.names = FALSE
  )
  
  write.csv(
    en_effects$by_variable,
    file.path(output_dir, "en_effect_sizes.csv"),
    row.names = FALSE
  )
  
  cat("\nTop 10 most frequently selected variables (LASSO):\n")
  print(selection_freq$frequency_summary %>% 
          filter(model == "lasso") %>% 
          group_by(outcome) %>% 
          top_n(10, selection_frequency) %>%
          arrange(outcome, desc(selection_frequency)))
  
} else {
  cat("\n!!! WARNING: No LASSO/EN coefficients were extracted !!!\n")
}

cat("\n=== Extracting GBM Variable Importance ===\n")
gbm_importance <- extract_gbm_importance(base_dir, imputations = 1:10)

if (!is.null(gbm_importance)) {
  gbm_summary <- calculate_gbm_importance_summary(gbm_importance)
  
  write.csv(
    gbm_summary,
    file.path(output_dir, "gbm_importance_ranks.csv"),
    row.names = FALSE
  )
  
  cat("\nTop 10 most important variables (GBM):\n")
  print(gbm_summary %>% 
          group_by(outcome) %>% 
          top_n(10, mean_importance) %>%
          arrange(outcome, desc(mean_importance)))
  
} else {
  cat("\n!!! WARNING: No GBM importance values were extracted !!!\n")
}

cat("\n=== Creating Combined Results Table ===\n")
if (!is.null(lasso_en_coefs) && !is.null(gbm_importance)) {
  combined_table <- create_combined_results_table(
    lasso_effects, en_effects, gbm_summary
  )
  
  write.csv(
    combined_table,
    file.path(output_dir, "combined_variable_importance.csv"),
    row.names = FALSE
  )
  
  cat("\nCombined table preview (top 20 variables):\n")
  print(head(combined_table, 20))
} else {
  cat("\nCannot create combined table - missing either LASSO/EN or GBM results\n")
}

if (!is.null(lasso_en_coefs)) {
  freq_table_word <- selection_freq$frequency_summary %>%
    add_labels() %>%
    filter(base_variable != "(Intercept)") %>%
    dplyr::select(
      Outcome  = outcome,
      Model    = model,
      Predictor = label,
      `Selection count` = selection_count,
      `Selection frequency` = selection_frequency
    ) %>%
    mutate(
      `Selection frequency` = round(`Selection frequency`, 2)
    ) %>%
    arrange(Outcome, Model, desc(`Selection frequency`))
  
  write.csv(
    freq_table_word,
    file.path(output_dir, "selection_frequency_for_word.csv"),
    row.names = FALSE
  )
}
if (!is.null(lasso_en_coefs)) {
  
  lasso_table_word <- lasso_effects$by_variable %>%
    add_labels() %>%
    filter(base_variable != "(Intercept)") %>%
    dplyr::select(
      Outcome = outcome,
      Predictor = label,
      `Mean coefficient` = mean_coefficient,
      `Abs mean coefficient` = max_abs_coefficient
    ) %>%
    mutate(
      `Mean coefficient` = round(`Mean coefficient`, 3),
      `Abs mean coefficient` = round(`Abs mean coefficient`, 3)
    ) %>%
    arrange(Outcome, desc(`Abs mean coefficient`))
  
  write.csv(
    lasso_table_word,
    file.path(output_dir, "lasso_effect_sizes_for_word.csv"),
    row.names = FALSE
  )
  
  en_table_word <- en_effects$by_variable %>%
    add_labels() %>%
    filter(base_variable != "(Intercept)") %>%
    dplyr::select(
      Outcome = outcome,
      Predictor = label,
      `Mean coefficient` = mean_coefficient,
      `Abs mean coefficient` = max_abs_coefficient
    ) %>%
    mutate(
      `Mean coefficient` = round(`Mean coefficient`, 3),
      `Abs mean coefficient` = round(`Abs mean coefficient`, 3)
    ) %>%
    arrange(Outcome, desc(`Abs mean coefficient`))
  
  write.csv(
    en_table_word,
    file.path(output_dir, "en_effect_sizes_for_word.csv"),
    row.names = FALSE
  )
}
if (!is.null(gbm_importance)) {
  
  gbm_table_word <- gbm_summary %>%
    add_labels() %>%
    dplyr::select(
      Outcome = outcome,
      Predictor = label,
      `Mean importance` = mean_importance,
      `SD importance` = sd_importance,
      Rank = rank
    ) %>%
    mutate(
      `Mean importance` = round(`Mean importance`, 2),
      `SD importance`   = round(`SD importance`, 2)
    ) %>%
    arrange(Outcome, Rank)
  
  write.csv(
    gbm_table_word,
    file.path(output_dir, "gbm_importance_for_word.csv"),
    row.names = FALSE
  )
}

if (!is.null(lasso_en_coefs) && !is.null(gbm_importance)) {
  combined_table_word <- combined_table %>%
    add_labels() %>%
    filter(base_variable != "(Intercept)") %>%
    dplyr::select(
      Outcome = outcome,
      Predictor = label,
      `LASSO coef` = lasso_coefficient,
      `LASSO |coef|` = lasso_abs_coefficient,
      `EN coef` = en_coefficient,
      `EN |coef|` = en_abs_coefficient,
      `GBM mean importance` = gbm_importance,
      `GBM rank` = gbm_rank
    ) %>%
    mutate(
      `LASSO coef`          = round(`LASSO coef`, 3),
      `LASSO |coef|`        = round(`LASSO |coef|`, 3),
      `EN coef`             = round(`EN coef`, 3),
      `EN |coef|`           = round(`EN |coef|`, 3),
      `GBM mean importance` = round(`GBM mean importance`, 2)
    ) %>%
    arrange(Outcome, `GBM rank`)
  
  write.csv(
    combined_table_word,
    file.path(output_dir, "combined_variable_importance_for_word.csv"),
    row.names = FALSE
  )
}

## ------------------------------- 
## Function: Extract coefficients from GLM models
## ------------------------------- 
extract_glm_coefficients <- function(base_dir, imputations = 1:10) {
  
  all_coefs <- list()
  
  for (i in imputations) {
    cat("Processing imputation", i, "\n")
    imp_dir <- file.path(base_dir, sprintf("imp%02d", i))
    
    for (outcome in c("FP_recall", "FP_biop")) {
      for (model in c("human_glm", "gen_glm")) {
        
        # Load the model file
        model_file <- file.path(imp_dir, paste0(outcome, "_", model, ".RData"))
        
        if (!file.exists(model_file)) {
          warning("File not found: ", model_file)
          next
        }
        
        # Load into separate environment
        env <- new.env()
        load(model_file, envir = env)
        
        # Find the GLM object
        model_obj <- NULL
        possible_names <- c(
          "fit_human", "fit_general", "human_glm", "gen_glm",
          "glm_human", "glm_general", "fit", "model", "mod"
        )
        
        for (name in possible_names) {
          if (exists(name, envir = env)) {
            obj <- get(name, envir = env)
            if ("glm" %in% class(obj)) {
              model_obj <- obj
              cat("  Found GLM object '", name, "' for", outcome, "-", model, "\n")
              break
            }
          }
        }
        
        if (is.null(model_obj)) {
          # Try to find any glm object
          for (name in ls(envir = env)) {
            obj <- get(name, envir = env)
            if ("glm" %in% class(obj)) {
              model_obj <- obj
              cat("  Found GLM object '", name, "' for", outcome, "-", model, "\n")
              break
            }
          }
        }
        
        if (is.null(model_obj)) {
          warning("Could not find GLM object in ", model_file)
          warning("Available objects: ", paste(ls(envir = env), collapse = ", "))
          next
        }
        
        # Extract coefficients
        tryCatch({
          coefs <- coef(model_obj)
          coef_df <- data.frame(
            variable = names(coefs),
            coefficient = as.numeric(coefs),
            stringsAsFactors = FALSE
          )
          
          # Add metadata
          coef_df$imputation <- i
          coef_df$outcome <- outcome
          coef_df$model <- model
          coef_df$base_variable <- coef_df$variable
          
          all_coefs[[length(all_coefs) + 1]] <- coef_df
          
        }, error = function(e) {
          warning("Error extracting coefficients from ", model_file, ": ", e$message)
        })
      }
    }
  }
  
  if (length(all_coefs) > 0) {
    all_coefs_df <- bind_rows(all_coefs)
    return(all_coefs_df)
  } else {
    warning("No GLM coefficients extracted!")
    return(NULL)
  }
}

## ------------------------------- 
## Function: Calculate average GLM effect sizes
## ------------------------------- 
calculate_glm_effect_sizes <- function(coef_df) {
  
  effect_sizes <- coef_df %>%
    filter(variable != "(Intercept)") %>%
    group_by(outcome, model, variable, base_variable) %>%
    summarise(
      mean_coefficient = mean(coefficient),
      sd_coefficient = sd(coefficient),
      n_imputations = n(),
      .groups = "drop"
    ) %>%
    arrange(outcome, model, desc(abs(mean_coefficient)))
  
  return(effect_sizes)
}

## ------------------------------- 
## MAIN: Extract and combine with existing results
## ------------------------------- 

cat("\n=== Extracting GLM Coefficients ===\n")
glm_coefs <- extract_glm_coefficients(base_dir, imputations = 1:10)

if (!is.null(glm_coefs)) {
  
  # Calculate effect sizes for each GLM
  glm_effects <- calculate_glm_effect_sizes(glm_coefs)
  
  # Pivot to wide format
  glm_wide <- glm_effects %>%
    select(outcome, variable, model, mean_coefficient) %>%
    pivot_wider(
      names_from = model,
      values_from = mean_coefficient,
      names_prefix = "coef_"
    ) %>%
    rename(
      base_variable = variable,
      expert_glm_coef = coef_human_glm,
      full_glm_coef = coef_gen_glm
    )
  
  # Combine with LASSO/EN effect sizes (lasso_effects, en_effects from above)
  
  combined_effect_table <- glm_wide %>%
    full_join(
      lasso_effects$by_term %>%
        filter(variable != "(Intercept)") %>%
        select(outcome, variable, mean_coefficient) %>%
        rename(base_variable = variable, lasso_coef = mean_coefficient),
      by = c("outcome", "base_variable")
    ) %>%
    full_join(
      en_effects$by_term %>%
        filter(variable != "(Intercept)") %>%
        select(outcome, variable, mean_coefficient) %>%
        rename(base_variable = variable, en_coef = mean_coefficient),
      by = c("outcome", "base_variable")
    ) %>%
    # Add labels
    add_labels() %>%
    select(
      Outcome = outcome,
      Variables = label,
      `Expert GLM` = expert_glm_coef,
      `Full GLM` = full_glm_coef,
      LASSO = lasso_coef,
      `Elastic net` = en_coef
    ) %>%
    mutate(
      Outcome = recode(Outcome,
                       "FP_recall" = "False-positive recall",
                       "FP_biop" = "False-positive biopsy recommendation"),
      `Expert GLM` = round(`Expert GLM`, 2),
      `Full GLM` = round(`Full GLM`, 2),
      LASSO = round(LASSO, 2),
      `Elastic net` = round(`Elastic net`, 2)
    ) %>%
    arrange(Outcome, Variables)
  
  # Save to CSV
  write.csv(
    combined_effect_table,
    file.path(output_dir, "all_model_effect_sizes.csv"),
    row.names = FALSE
  )
  
  cat("\nCombined effect sizes table preview:\n")
  print(head(combined_effect_table, 20))
}


# Create wide format table (both outcomes side by side)
if (exists("combined_effect_table")) {
  
  effect_wide <- combined_effect_table %>%
    pivot_wider(
      names_from = Outcome,
      values_from = c(`Expert GLM`, `Full GLM`, LASSO, `Elastic net`),
      names_glue = "{Outcome}_{.value}"
    )
  
  # Rename for flextable
  tab_eff <- effect_wide %>%
    rename(
      FPrecall_Expert = `False-positive recall_Expert GLM`,
      FPrecall_Full = `False-positive recall_Full GLM`,
      FPrecall_LASSO = `False-positive recall_LASSO`,
      FPrecall_EN = `False-positive recall_Elastic net`,
      FPbiopsy_Expert = `False-positive biopsy recommendation_Expert GLM`,
      FPbiopsy_Full = `False-positive biopsy recommendation_Full GLM`,
      FPbiopsy_LASSO = `False-positive biopsy recommendation_LASSO`,
      FPbiopsy_EN = `False-positive biopsy recommendation_Elastic net`
    )
  
  ft_eff <- flextable(tab_eff)
  
  # Bottom header row (model names)
  ft_eff <- set_header_labels(
    ft_eff,
    Variables       = "Variables",
    FPrecall_Expert = "Expert\nGLM",
    FPrecall_Full   = "Full\nGLM",
    FPrecall_LASSO  = "LASSO",
    FPrecall_EN     = "Elastic\nnet",
    FPbiopsy_Expert = "Expert\nGLM",
    FPbiopsy_Full   = "Full\nGLM",
    FPbiopsy_LASSO  = "LASSO",
    FPbiopsy_EN     = "Elastic\nnet"
  )
  
  # Top header row (outcome groups)
  ft_eff <- add_header_row(
    ft_eff,
    values    = c("Variables", "False-positive recall", "False-positive biopsy recommendation"),
    colwidths = c(1, 4, 4)
  )
  
  # Styling
  ft_eff <- merge_h(ft_eff, part = "header") %>%
    align(part = "header", align = "center") %>%
    valign(part = "header", valign = "center") %>%
    bold(part = "header") %>%
    set_table_properties(layout = "fixed") %>%
    autofit() %>%
    width(j = 1, width = 2.5) %>%
    width(j = 2:9, width = 0.65) %>%
    fontsize(part = "all", size = 9) %>%
    align(j = 1, align = "left", part = "body") %>%
    align(j = 2:9, align = "center", part = "body")
  
  ft_eff
}
