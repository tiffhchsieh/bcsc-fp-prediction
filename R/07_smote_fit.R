# ==============================================================
#  BCSC - Risk-modeling pipeline WITH SMOTE (tidymodels/themis)
#  Author: Tiffany Hsieh
#  Date  : 2025-04-16
# ==============================================================

# ----------------------------
# 0.  House-keeping
# ----------------------------
source("R/config.R")

load(file.path(data_dir, "AB468b_data_noPHI_shared_randomForests.RData"))

# Required libraries ---------------------------------------------------------
library(tidymodels)   # broom, parsnip, recipes, rsample, etc.
library(themis)       # step_smote()
library(glmnet)       # still used for glmnet models
library(randomForest)
library(xgboost)
library(Matrix)
library(nnet)

# ----------------------------
# 1. Predictor sets and interaction terms
# ----------------------------
predictor_vars <- colnames(imp_data_1)[-c(1, 2, 3, 6, 8, 22, 35, 36, 37, 42, 43)]
interaction_terms <- c(
  paste("racenci_new", c("bmi_ns.1", "bmi_ns.2", "bmi_ns.3", "bmi_ns.4"), sep = ":"),
  paste("density_c", c("bmi_ns.1", "bmi_ns.2", "bmi_ns.3", "bmi_ns.4"), sep = ":"),
  "density_c:racenci_new"
)
var_list_human_model <- c("age_ns.1","age_ns.2","age_ns.3","age_ns.4",
                          "bmi_ns.1","bmi_ns.2","bmi_ns.3","bmi_ns.4",
                          "famhx_c","racenci_new","mod1_new","density_c",
                          "menopht_2cat","log_prvmos_ns.1","log_prvmos_ns.2","log_prvmos_ns.3")
all_terms <- c(predictor_vars, interaction_terms)
formula_smote <- as.formula(paste("~", paste(all_terms, collapse = " + ")))


# ----------------------------
# 2. Train / test split
# ----------------------------
set.seed(123)

split_data <- function(data, outcome) {
  test_case   <- sample(which(data[[outcome]] == 1), ceiling(sum(data[[outcome]] == 1)/4))
  test_control<- sample(which(data[[outcome]] == 0), ceiling(sum(data[[outcome]] == 0)/4))
  data[[paste0("split_", outcome)]] <- "training"
  data[c(test_case, test_control), paste0("split_", outcome)] <- "testing"
  data
}

imp_data_1 <- split_data(imp_data_1, "FP_recall")
imp_data_1 <- split_data(imp_data_1, "FP_biop")


# ----------------------------
# 3.  SMOTE using tidymodels/themis
# ----------------------------
make_smote_recipe <- function(df, outcome) {
  df[[outcome]] <- factor(df[[outcome]], levels = c(0, 1))
  df <- df |> dplyr::select(all_of(c(outcome, predictor_vars)))  # Keep only relevant predictors
  recipe(as.formula(paste(outcome, "~ .")), data = df) %>%
    step_zv(all_predictors()) %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
    step_normalize(all_numeric_predictors()) %>%
    step_smote(all_outcomes(), over_ratio = 0.5)
}

# Generate SMOTEd datasets
train_recall  <- subset(imp_data_1, split_FP_recall == "training")
train_biop    <- subset(imp_data_1, split_FP_biop == "training")
test_recall   <- subset(imp_data_1, split_FP_recall == "testing")
test_biop     <- subset(imp_data_1, split_FP_biop == "testing")

rec_recall <- make_smote_recipe(train_recall, "FP_recall")
rec_biop   <- make_smote_recipe(train_biop,   "FP_biop")

prepped_recall <- prep(rec_recall)
prepped_biop   <- prep(rec_biop)

train_recall_sm <- bake(prepped_recall, new_data = NULL)
#test_recall_sm  <- bake(prepped_recall, new_data = test_recall)
train_biop_sm   <- bake(prepped_biop, new_data = NULL)
#test_biop_sm    <- bake(prepped_biop, new_data = test_biop)

# ----------------------------
# 4.1  GLM - Human model (SMOTEd)
# ----------------------------
# Keep all the columns that were originally part of var_list_human_model but now dummy-encoded
human_vars <- grep(paste(var_list_human_model, collapse = "|"), names(train_recall_sm), value = TRUE)

# Build formula dynamically
formula_glm_human <- as.formula(paste("FP_recall ~", paste(human_vars, collapse = " + ")))
glm_fit_recall_human <- glm(formula_glm_human, data = train_recall_sm, family = binomial())

risk_pr_recall_human <- predict(glm_fit_recall_human, newdata = test_recall_sm, type = "response")

save(list = c("risk_pr_recall_human", "glm_fit_recall_human"),
     file = file.path(results_dir, "FP_recall_human_glm_SMOTE.RData"))

# FP_biop --------------------------------------------------------------------
formula_glm_human_biop <- as.formula(paste("FP_biop ~", paste(human_vars, collapse = " + ")))

glm_fit_biop_human <- glm(formula_glm_human_biop, data = train_biop_sm, family = binomial())
risk_pr_biop_human <- predict(glm_fit_biop_human, newdata = test_biop_sm, type = "response")

save(list = c("risk_pr_biop_human", "glm_fit_biop_human"),
     file = file.path(results_dir, "FP_biop_human_glm_SMOTE.RData"))

# ----------------------------
# 4. General GLM using same predictors as non-SMOTE
# ----------------------------
fit_general_glm <- function(train_df, test_df, outcome, save_path) {
  # Outcome as numeric
  train_df[[outcome]] <- as.numeric(as.character(train_df[[outcome]]))
  test_df[[outcome]]  <- as.numeric(as.character(test_df[[outcome]]))
  
  # Fit the model using formula interface
  formula <- as.formula(paste(outcome, "~ ."))
  glm_fit <- glm(formula, data = train_df, family = binomial())
  
  # Predict using the fitted model
  preds <- predict(glm_fit, newdata = test_df, type = "response")
  
  # Save model and predictions
  save(list = c("glm_fit", "preds"), file = save_path)
}

fit_general_glm(train_recall_sm, test_recall_sm, "FP_recall", file.path(results_dir, "FP_recall_glm_SMOTE.RData"))
fit_general_glm(train_biop_sm,   test_biop_sm,   "FP_biop",   file.path(results_dir, "FP_biop_glm_SMOTE.RData"))


# ----------------------------
# 5.  LASSO & Elastic-Net (SMOTEd)
# ----------------------------
# Build model matrices after the recipe so dummy-coded columns match.

build_mm <- function(df, outcome) {
  model.matrix(~ ., data = dplyr::select(df, -all_of(outcome)))[,-1]
}

mm_recall_sm   <- build_mm(train_recall_sm, "FP_recall")
mm_recall_test <- build_mm(test_recall_sm,  "FP_recall")

y_train_sm <- train_recall_sm$FP_recall
dim(mm_recall_sm)
length(y_train_sm)         # should match exactly

glmnet_cv_recall <- cv.glmnet(
  mm_recall_sm,
  y = train_recall_sm$FP_recall,
  family = "binomial"
)

risk_pr_recall_lasso <- predict(glmnet_cv_recall, mm_recall_test, s = "lambda.min", type = "response")[,1]

save(list = c("glmnet_cv_recall", "risk_pr_recall_lasso"),
     file = file.path(results_dir, "FP_recall_lasso_SMOTE.RData"))

# FP_biop --------------------------------------------------------------------
mm_biop_sm   <- build_mm(train_biop_sm, "FP_biop")
mm_biop_test <- build_mm(test_biop_sm,   "FP_biop")

y_train_sm <- train_biop_sm$FP_biop
dim(mm_biop_sm)
length(y_train_sm) 

glmnet_cv_biop <- cv.glmnet(
  mm_biop_sm,
  y = train_biop_sm$FP_biop,
  family = "binomial"
)

risk_pr_biop_lasso <- predict(glmnet_cv_biop, mm_biop_test, s = "lambda.min", type = "response")[,1]

save(list = c("glmnet_cv_biop", "risk_pr_biop_lasso"),
     file = file.path(results_dir, "FP_biop_lasso_SMOTE.RData"))

# Elastic-Net: alpha = 0.5 ----------------------------------------------------
# (repeat same pattern, not shown)
# ----------------------------
# Elastic-Net Models (SMOTEd)
# ----------------------------

# FP_recall (Elastic-Net, alpha = 0.5)
set.seed(123)

cv_enet_recall <- cv.glmnet(
  mm_recall_sm,
  y = train_recall_sm$FP_recall,
  family = "binomial",
  alpha = 0.5
)

risk_pr_recall_enet <- predict(cv_enet_recall, mm_recall_test, s = "lambda.min", type = "response")[,1]

save(list = c("cv_enet_recall", "risk_pr_recall_enet"),
     file = file.path(results_dir, "FP_recall_elasticnet_SMOTE.RData"))


# FP_biop (Elastic-Net, alpha = 0.5)
set.seed(123)

cv_enet_biop <- cv.glmnet(
  mm_biop_sm,
  y = train_biop_sm$FP_biop,
  family = "binomial",
  alpha = 0.5
)

risk_pr_biop_enet <- predict(cv_enet_biop, mm_biop_test, s = "lambda.min", type = "response")[,1]

save(list = c("cv_enet_biop", "risk_pr_biop_enet"),
     file = file.path(results_dir, "FP_biop_elasticnet_SMOTE.RData"))

# ----------------------------
# 6.  Random Forest (SMOTEd)
# ----------------------------
fit_rf_model <- function(train_data, test_data, outcome) {
  # Make sure outcome is a factor
  train_data[[outcome]] <- factor(train_data[[outcome]], levels = c(0, 1))
  test_data[[outcome]]  <- factor(test_data[[outcome]],  levels = c(0, 1))
  
  # Extract predictors
  predictors <- setdiff(names(train_data), outcome)
  
  # Fit Random Forest
  rf_model <- randomForest(
    formula = as.formula(paste(outcome, "~ .")),
    data = train_data,
    ntree = 500,
    importance = TRUE
  )
  
  # Predict probabilities
  probs <- predict(rf_model, newdata = test_data, type = "prob")[, 2]
  
  return(list(model = rf_model, predictions = probs))
}


rf_recall <- fit_rf_model(train_recall_sm, test_recall_sm, "FP_recall")

rf_biop   <- fit_rf_model(train_biop_sm,   test_biop_sm,   "FP_biop")

save(rf_recall, file = file.path(results_dir, "FP_recall_rf_SMOTE.RData"))
save(rf_biop,   file = file.path(results_dir, "FP_biop_rf_SMOTE.RData"))

# ----------------------------
# 7.  Gradient Boosting (SMOTEd)
# ----------------------------
prepare_xgb_matrix <- function(data, response_var) {
  # Drop response and convert rest to matrix
  x <- data |> dplyr::select(-all_of(response_var)) |> as.matrix()
  
  # Convert outcome to numeric 0/1
  y <- as.numeric(as.character(data[[response_var]]))
  
  # Sanity check
  if (!all(y %in% c(0, 1))) {
    stop("Outcome must be strictly 0 or 1")
  }
  
  xgb.DMatrix(data = x, label = y)
}

fit_gbm <- function(train_df, test_df, response_var) {
  dtrain <- prepare_xgb_matrix(train_df, response_var)
  dtest  <- prepare_xgb_matrix(test_df,  response_var)
  params <- list(objective = "binary:logistic", eval_metric = "logloss", eta = 0.05, max_depth = 4)
  cv <- xgb.cv(params = params, data = dtrain, nrounds = 5000, nfold = 5,
               early_stopping_rounds = 200, verbose = 0)
  mod <- xgboost(params = params, data = dtrain, nrounds = cv$best_iteration, verbose = 0)
  pred <- predict(mod, dtest)
  list(model = mod, predictions = pred)
}

gbm_recall <- fit_gbm(train_recall_sm, test_recall_sm, "FP_recall")
gbm_biop   <- fit_gbm(train_biop_sm,   test_biop_sm,   "FP_biop")

save(gbm_recall, file = file.path(results_dir, "FP_recall_gbm_SMOTE.RData"))
save(gbm_biop,   file = file.path(results_dir, "FP_biop_gbm_SMOTE.RData"))

# ----------------------------
# 8.  Feed-Forward Neural Network (SMOTEd)
# ----------------------------
fit_nnet <- function(train_df, test_df, response_var, size = 5, decay = 0.1, maxit = 300) {
  train_df[[response_var]] <- as.factor(train_df[[response_var]])
  test_df[[response_var]]  <- as.factor(test_df[[response_var]])
  
  mod <- nnet(as.formula(paste(response_var, "~ .")), data = train_df,
              size = size, decay = decay, maxit = maxit, trace = FALSE)
  
  pred <- predict(mod, newdata = test_df, type = "raw")
  
  # FIX HERE: only subset if it's a matrix
  if (is.matrix(pred) && ncol(pred) == 2) {
    pred <- pred[, 2]
  }
  
  list(model = mod, predictions = as.vector(pred))
}


nnet_recall <- fit_nnet(train_recall_sm, test_recall_sm, "FP_recall")
nnet_biop   <- fit_nnet(train_biop_sm,   test_biop_sm,   "FP_biop")

save(nnet_recall, file = file.path(results_dir, "FP_recall_nnet_SMOTE.RData"))
save(nnet_biop,   file = file.path(results_dir, "FP_biop_nnet_SMOTE.RData"))

# ----------------------------
# 9.  Done!  SMOTE applied via themis; test sets remain untouched.
# ==============================================================


