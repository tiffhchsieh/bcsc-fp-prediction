################################################################################
## Develop Risk Prediction Models on ALL Imputed Datasets 
## Date: 11/08/2025
## Programmer: Tiffany Hsieh
################################################################################

source("R/config.R")

# Load libraries
library(mice)
library(glmnet)
library(randomForest)
library(xgboost)
library(Matrix)
library(nnet)

## -------------------------------
## Local results directory
## -------------------------------
output_dir <- file.path(results_dir, "all_imputations_nov_2025")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)


## -------------------------------
## Predictor lists 
## -------------------------------
var_list <- c("age_c","racenci_new","menopht_2cat","famhx_c","bmi_c",
              "mammtype_new","density_c","prvmos_c","mossinceprvsurvmm_cat",
              "mod1_new","dxage","dxyear",
              "hist1_new","stage1","grade1_c","erplus1",
              "srg1_new","rt_new","adjthrpy_1")

cont_var_list <- c("age_c","bmi_c","prvmos_c","dxage","dxyear")

var_list_human_model <- c(
  "age_ns.1","age_ns.2","age_ns.3","age_ns.4",
  "bmi_ns.1","bmi_ns.2","bmi_ns.3","bmi_ns.4",
  "famhx_c","racenci_new","mod1_new","density_c","menopht_2cat",
  "log_prvmos_ns.1","log_prvmos_ns.2","log_prvmos_ns.3",
  "prv_FP_recall","prv_FP_biop"
)

predictor_vars <- c(setdiff(colnames(imp_data_1),
                            c("FP_recall","FP_biop","eventtype_trc",
                              "bmi_c","age_c","yrsincdx_cat_new","log_prvmos",
                              "surv_round")))

## -------------------------------
## Interaction term specification
## -------------------------------
make_interaction_terms <- function() {
  bmi_splines <- c("bmi_ns.1","bmi_ns.2","bmi_ns.3","bmi_ns.4")
  c(
    paste0("racenci_new:", bmi_splines),    # race x BMI splines
    paste0("density_c:",  bmi_splines),     # density x BMI splines
    "density_c:racenci_new"                  # density x race
  )
}

## A formula (R "~" syntax) that includes main effects + interactions
interaction_terms <- make_interaction_terms()

# Formula string with main effects and interactions
full_formula_with_interactions <- function(response) {
  paste(
    response, "~",
    paste(c(predictor_vars, interaction_terms), collapse = " + ")
  )
}


## -------------------------------
## Split function
## -------------------------------
make_split <- function(dat, prop_test = 0.25, seed = 123) {
  set.seed(seed)
  n <- nrow(dat)
  test_idx_recall <- sample(seq_len(n), size = ceiling(prop_test * n))
  test_idx_biop   <- sample(seq_len(n), size = ceiling(prop_test * n))
  dat$split_recall <- "training"
  dat$split_biop   <- "training"
  dat$split_recall[test_idx_recall] <- "testing"
  dat$split_biop[test_idx_biop]     <- "testing"
  dat
}

## -------------------------------
## Seed for each imputation, outcome, and model
## -------------------------------
seed_for <- function(imp, outcome, model) {
  base <- 123
  o <- match(outcome, c("FP_recall","FP_biop"))
  m <- match(model,   c("glm_human","glm_general","lasso","enet","rf","gbm","nnet"))
  base + 1000*imp + 10*o + m
}


## -------------------------------
## GBM with CV + early stopping
## -------------------------------
fit_gbm_model <- function(train_data, test_data, response_var,
                          seed = 123,
                          nfold = 5,
                          early_stopping_rounds = 50,
                          max_nrounds = 5000,
                          params = list(
                            objective = "binary:logistic",
                            eval_metric = c("logloss", "aucpr"),
                            eta = 0.05,
                            max_depth = 4,
                            subsample = 0.8,
                            colsample_bytree = 0.8,
                            min_child_weight = 1
                          ),
                          use_scale_pos_weight = TRUE) {
  
  dtrain <- prepare_xgb_matrix(train_data, response_var)
  dtest  <- prepare_xgb_matrix(test_data,  response_var)
  
  # Optional class-imbalance handling
  if (use_scale_pos_weight) {
    y <- xgboost::getinfo(dtrain, "label")
    pos <- sum(y == 1); neg <- sum(y == 0)
    if (pos > 0) params$scale_pos_weight <- neg / pos
  }
  
  set.seed(seed)
  cv <- xgboost::xgb.cv(
    params = params,
    data = dtrain,
    nrounds = max_nrounds,
    nfold = nfold,
    early_stopping_rounds = early_stopping_rounds,
    verbose = 0,
    maximize = FALSE
  )
  
  best_nrounds <- cv$best_iteration
  best_score   <- cv$evaluation_log$test_logloss_mean[cv$best_iteration]
  
  gbm_model <- xgboost::xgb.train(
    params  = params,
    data    = dtrain,
    nrounds = best_nrounds,
    watchlist = list(train = dtrain),
    verbose = 0
  )
  
  preds <- predict(gbm_model, dtest)
  
  list(
    model = gbm_model,
    predictions = preds,
    best_nrounds = best_nrounds,
    cv_best_logloss = best_score,
    params_used = params,
    importance = tryCatch(xgboost::xgb.importance(model = gbm_model), error = function(e) NULL)
  )
}

## -------------------------------
## Main model fitting function
## -------------------------------
run_all_models <- function(dat, imp_id) {
  imp_dir <- file.path(output_dir, sprintf("imp%02d", imp_id))
  dir.create(imp_dir, showWarnings = FALSE)
  
  for (resp in c("FP_recall", "FP_biop")) {
    split_flag <- paste0("split_", tolower(gsub("FP_", "", resp)))
    tr <- dat[dat[[split_flag]] == "training", , drop = FALSE]
    te <- dat[dat[[split_flag]] == "testing",  , drop = FALSE]
    
    ## ---- Human GLM
    set.seed(seed_for(imp_id, resp, "glm_human"))
    form_human <- as.formula(paste(resp, "~", paste(var_list_human_model, collapse = "+")))
    glm_human <- glm(form_human, data = tr, family = binomial)
    pr_human  <- predict(glm_human, newdata = te, type="response")
    save(glm_human, pr_human, file = file.path(imp_dir, paste0(resp, "_human_glm.RData")))
    
    ## ---- General GLM (with interactions)
    set.seed(seed_for(imp_id, resp, "glm_general"))
    form_general <- as.formula(full_formula_with_interactions(resp))
    glm_general <- glm(form_general, data = tr, family = binomial)
    pr_general  <- predict(glm_general, newdata = te, type="response")
    save(glm_general, pr_general, file = file.path(imp_dir, paste0(resp, "_gen_glm.RData")))
    
    ## ---- LASSO / EN (with interactions)
    fml <- as.formula(full_formula_with_interactions(resp))
    
    # Build Terms from TRAIN
    tr_terms <- stats::terms(fml, data = tr)   # NOT terms(formula = ...)
    
    # TRAIN matrix
    mm_tr <- model.matrix(tr_terms, data = tr)
    
    # Drop intercept by name (safer than [, -1])
    int_col <- "(Intercept)"
    Xtr <- if (int_col %in% colnames(mm_tr)) {
      mm_tr[, setdiff(colnames(mm_tr), int_col), drop = FALSE]
    } else {
      mm_tr
    }
    
    # TEST matrix using SAME terms
    mm_te <- model.matrix(tr_terms, data = te)
    Xte <- if (int_col %in% colnames(mm_te)) {
      mm_te[, setdiff(colnames(mm_te), int_col), drop = FALSE]
    } else {
      mm_te
    }
    
    # Align columns when factor levels are missing in the test set
    missing_in_te <- setdiff(colnames(Xtr), colnames(Xte))
    if (length(missing_in_te)) {
      Xte <- cbind(
        Xte,
        matrix(0, nrow = nrow(Xte), ncol = length(missing_in_te),
               dimnames = list(NULL, missing_in_te))
      )
    }
    Xte <- Xte[, colnames(Xtr), drop = FALSE]
    
    # Outcome must be numeric 0/1
    ytr <- as.numeric(as.character(tr[[resp]]))
    
    # LASSO CV
    set.seed(seed_for(imp_id, resp, "lasso"))
    cv_lasso <- glmnet::cv.glmnet(Xtr, ytr, family = "binomial",
                                  alpha = 1, nfolds = 5, type.measure = "auc")
    pr_lasso <- as.numeric(predict(cv_lasso, Xte, s = "lambda.min", type = "response"))
    save(cv_lasso, pr_lasso, file = file.path(imp_dir, paste0(resp, "_lasso.RData")))
    
    # Elastic Net
    set.seed(seed_for(imp_id, resp, "enet"))
    cv_en <- glmnet::cv.glmnet(Xtr, ytr, family = "binomial",
                               alpha = 0.5, nfolds = 5, type.measure = "auc")
    pr_en <- as.numeric(predict(cv_en, Xte, s = "lambda.min", type = "response"))
    save(cv_en, pr_en, file = file.path(imp_dir, paste0(resp, "_en.RData")))
    
    
    ## ---- Random Forest on engineered design (interactions materialized)
    set.seed(seed_for(imp_id, resp, "rf"))
    
    # Build the same engineered matrices as above, then convert to data.frame
    rf_Xtr <- as.data.frame(Xtr)
    rf_Xte <- as.data.frame(Xte)
    rf_tr <- cbind(resp = factor(tr[[resp]]), rf_Xtr)
    rf_te <- cbind(resp = factor(te[[resp]]), rf_Xte)
    
    ## rf_Xtr and rf_Xte: design matrices from model.matrix
    ## make sure names are syntactic
    colnames(rf_Xtr) <- make.names(colnames(rf_Xtr))
    colnames(rf_Xte) <- make.names(colnames(rf_Xte))
    
    y_rf_tr <- factor(tr[[resp]])
    
    set.seed(seed_for(imp_id, resp, "rf"))
    mtry_val <- max(1, floor(sqrt(ncol(rf_Xtr))))
    rf <- randomForest::randomForest(
      x = rf_Xtr,
      y = y_rf_tr,
      ntree = 500,
      importance = TRUE,
      mtry = mtry_val
    )
    
    pr_rf <- as.numeric(predict(rf, newdata = rf_Xte, type = "prob")[, 2])
    save(rf, pr_rf, file = file.path(imp_dir, paste0(resp, "_rf.RData")))
    
    
    
    ## ---- Gradient Boosting with CV using engineered design
    set.seed(seed_for(imp_id, resp, "gbm"))
    
    # Use the same Xtr/Xte built above
    dtrain <- xgboost::xgb.DMatrix(data = as.matrix(Xtr), label = ytr)
    yte    <- as.numeric(as.character(te[[resp]]))
    dtest  <- xgboost::xgb.DMatrix(data = as.matrix(Xte), label = yte)
    
    params <- list(
      objective = "binary:logistic",
      eval_metric = c("logloss","aucpr"),
      eta = 0.05, max_depth = 4, subsample = 0.8, colsample_bytree = 0.8, min_child_weight = 1
    )
    
    # Optional class imbalance
    #pos <- sum(ytr == 1); neg <- sum(ytr == 0)
    # if (pos > 0) params$scale_pos_weight <- neg / pos
    
    cv <- xgboost::xgb.cv(
      params = params, data = dtrain, nrounds = 5000, nfold = 5,
      early_stopping_rounds = 50, verbose = 0, maximize = FALSE
    )
    best_nrounds <- cv$best_iteration
    gbm_model <- xgboost::xgb.train(params = params, data = dtrain,
                                    nrounds = best_nrounds, verbose = 0)
    preds <- predict(gbm_model, newdata = dtest)
    
    gbm <- list(
      model = gbm_model,
      predictions = preds,
      best_nrounds = best_nrounds,
      cv_best_logloss = cv$evaluation_log$test_logloss_mean[best_nrounds],
      params_used = params,
      importance = tryCatch(xgboost::xgb.importance(model = gbm_model), error = function(e) NULL)
    )
    save(gbm, file = file.path(imp_dir, paste0(resp, "_gbm_model.RData")))
    
    ## ---- Neural Net on engineered design
    set.seed(seed_for(imp_id, resp, "nnet"))
    nn_tr <- data.frame(y = ytr, Xtr)
    nn_te <- data.frame(Xte)
    nn <- nnet::nnet(
      y ~ .,
      data = nn_tr, size = 5, decay = 0.1, maxit = 200, trace = FALSE
    )
    pr_nn <- as.numeric(predict(nn, newdata = nn_te, type = "raw"))
    save(nn, pr_nn, file = file.path(imp_dir, paste0(resp, "_nnet.RData")))
    
  }
}

## -------------------------------
## Build list of imputed datasets
## -------------------------------
# Imputations 1-10 (imp_data_1 ... imp_data_10)
imputed_list <- lapply(1:2, function(m) get(paste0("imp_data_", m)))

## -------------------------------
## MAIN LOOP
## -------------------------------
for (m in seq_along(imputed_list)) {
  message("=== Running models for Imputation ", m, " ===")
  dat_m <- make_split(imputed_list[[m]], seed = 123 + m)
  save(dat_m, file = file.path(output_dir, sprintf("imp%02d_split.RData", m)))
  run_all_models(dat_m, imp_id = m)
}

# Run imputations 3 through 10
for (m in 3:10) {
  msg <- paste0("=== Running models for Imputation ", sprintf("%02d", m), " ===")
  message(msg)
  
  # make sure the imputed dataset exists
  obj <- paste0("imp_data_", m)
  if (!exists(obj)) stop("Missing object: ", obj, " (load it before running)")
  
  dat_m <- make_split(get(obj), seed = 123 + m)
  
  # save split and run models
  imp_dir <- file.path(output_dir, sprintf("imp%02d", m))
  dir.create(imp_dir, recursive = TRUE, showWarnings = FALSE)
  save(dat_m, file = file.path(imp_dir, "split.RData"))
  
  run_all_models(dat_m, imp_id = m)
}

