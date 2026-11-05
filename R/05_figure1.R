source("R/config.R")

library(pROC)
library(ggplot2)
library(dplyr)
library(purrr)

# --- Settings ---
base_dir <- file.path(results_dir, "all_imputations_nov_2025")
imps     <- 1:10
models   <- c("human_glm","gen_glm","lasso","en","rf","gbm_model","nnet")
outcomes <- c("FP_recall","FP_biop")

# --- Find split file ---
get_split_file <- function(base_dir, imp_num) {
  # Try base directory first (imp01, imp02 pattern)
  f1 <- file.path(base_dir, sprintf("imp%02d_split.RData", imp_num))
  if (file.exists(f1)) return(f1)
  
  # Try inside folder (imp03-imp10 pattern)
  f2 <- file.path(base_dir, sprintf("imp%02d", imp_num), "split.RData")
  if (file.exists(f2)) return(f2)
  
  stop("Cannot find split file for imputation ", imp_num)
}

# --- Load predictions ---
load_preds <- function(imp_dir, outcome, model) {
  f <- file.path(imp_dir, sprintf("%s_%s.RData", outcome, model))
  
  if (!file.exists(f)) return(NULL)
  
  tryCatch({
    env <- new.env()
    load(f, envir = env)
    
    result <- switch(model,
                     "human_glm" = if (exists("pr_human", envir = env))   as.numeric(get("pr_human", envir = env)),
                     "gen_glm"   = if (exists("pr_general", envir = env)) as.numeric(get("pr_general", envir = env)),
                     "lasso"     = if (exists("pr_lasso", envir = env))   as.numeric(get("pr_lasso", envir = env)),
                     "en"        = if (exists("pr_en", envir = env))      as.numeric(get("pr_en", envir = env)),
                     "rf"        = if (exists("pr_rf", envir = env))      as.numeric(get("pr_rf", envir = env)),
                     "nnet"      = if (exists("pr_nn", envir = env))      as.numeric(get("pr_nn", envir = env)),
                     "gbm_model" = if (exists("gbm", envir = env))        as.numeric(get("gbm", envir = env)$predictions),
                     NULL
    )
    return(result)
    
  }, error = function(e) {
    return(NULL)
  })
}

# --- Reference test sets from imp01 ---
load(get_split_file(base_dir, 1))

test_sets <- list(
  FP_recall = dat_m %>% filter(split_recall == "testing") %>%
    mutate(y = as.numeric(as.character(FP_recall))),
  FP_biop   = dat_m %>% filter(split_biop   == "testing") %>%
    mutate(y = as.numeric(as.character(FP_biop)))
)

for (nm in names(test_sets)) {
  if (!"id" %in% names(test_sets[[nm]])) {
    test_sets[[nm]]$id <- seq_len(nrow(test_sets[[nm]]))
  }
}

# --- Mean predictions across imputations---
mean_pred <- list()
for (outcome in outcomes) {
  ref <- test_sets[[outcome]]
  y_ref  <- ref$y
  id_ref <- ref$id
  
  for (m in models) {
    pm <- matrix(NA_real_, nrow = nrow(ref), ncol = length(imps))
    for (j in seq_along(imps)) {
      i <- imps[j]
      
      load(get_split_file(base_dir, i))
      
      test_i <- if (outcome == "FP_recall") {
        dat_m %>% filter(split_recall == "testing")
      } else {
        dat_m %>% filter(split_biop == "testing")
      }
      
      if (!"id" %in% names(test_i)) test_i$id <- seq_len(nrow(test_i))
      if (nrow(test_i) != nrow(ref)) next
      
      pr <- load_preds(file.path(base_dir, sprintf("imp%02d", i)), outcome, m)
      if (is.null(pr) || length(pr) != nrow(test_i)) next
      
      if (!identical(test_i$id, id_ref)) {
        ord <- match(id_ref, test_i$id)
        if (!anyNA(ord)) pr <- pr[ord]
      }
      pm[, j] <- pr
    }
    pr_mean <- rowMeans(pm, na.rm = TRUE)
    if (!all(is.na(pr_mean))) {
      mean_pred[[paste(outcome, m, sep = "__")]] <- list(
        outcome = outcome, model = m, y = y_ref, pr = pr_mean
      )
    }
  }
}

# Build ROC points
roc_df <- purrr::map_dfr(mean_pred, function(x) {
  
  r <- pROC::roc(x$y, x$pr, quiet = TRUE, direction = "<")
  data.frame(
    outcome = factor(x$outcome, levels = c("FP_recall","FP_biop"),
                     labels = c("FP recall","FP biopsy")),
    model = x$model,
    fpr = 1 - r$specificities,
    tpr = r$sensitivities
  )
})

# Recode model names
model_labels <- c(
  "human_glm" = "Expert GLM",
  "gen_glm"   = "General GLM",
  "lasso"     = "LASSO",
  "en"        = "Elastic net",
  "rf"        = "Random forest",
  "gbm_model" = "Gradient boosting",
  "nnet"      = "Neural network"
)

roc_df <- roc_df %>%
  mutate(
    model = factor(model, 
                   levels = names(model_labels),
                   labels = model_labels)
  )

# Plot
p <- ggplot(roc_df, aes(x = fpr, y = tpr, color = model)) +
  geom_abline(intercept = 0, slope = 1, linetype = 3, linewidth = 0.8) +
  geom_line(linewidth = 1.2) +
  coord_equal() +
  facet_wrap(~ outcome, ncol = 2) +
  labs(
    x = "False positive rate (1 − Specificity)",
    y = "True positive rate (Sensitivity)",
    title = "ROC Curves by Outcome (Mean Prediction Across 10 Imputations)",
    color = "Model"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    strip.text = element_text(size = 16, face = "bold"),
    legend.position = "bottom",
    legend.title = element_text(size = 14, face = "bold"),
    legend.text = element_text(size = 12),
    panel.spacing = unit(1.5, "lines")
  ) +
  guides(color = guide_legend(nrow = 2))

print(p)
