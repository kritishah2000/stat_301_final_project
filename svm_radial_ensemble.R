# load required objects ----
library(tidyverse)
library(skimr)
library(tidymodels)
library(janitor)
library(vcd)
library(corrplot)
library(corrr)
# library(stacks)
# library(textrecipes)

set.seed(3013)

# load required objects ----
load("data/setup.rda")

# Define model ----
svm_model <- svm_rbf(
  mode = "classification",
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab")

# # check tuning parameters
# parameters(svm_model)

# set-up tuning grid ----
svm_params <- parameters(svm_model)

# define grid
svm_grid <- grid_regular(svm_params, levels = 5)

# workflow ----
svm_workflow_2 <- workflow() %>%
  add_model(svm_model) %>%
  add_recipe(patients_recipe)

# control settings ----

# ctrl_grid <- control_stack_grid()

# Tuning/fitting ----
svm_res_2 <- svm_workflow_2 %>%
  tune_grid(
    resamples = patients_folds,
    grid = svm_grid
    # control = ctrl_grid
  )

save(svm_res_2, file = "data/svm_results_ensemble.rda")

load("data/svm_results_ensemble.rda")

autoplot(svm_res_2, metric = "roc_auc")
select_best(svm_res_2, metric = "roc_auc")
show_best(svm_res_2, metric = "roc_auc")

svm_workflow_tuned <- svm_workflow_2 %>%
finalize_workflow(select_best(svm_res_2, metric = "roc_auc"))

svm_results <- fit(svm_workflow_tuned, patients_train)

patients_predict <- predict(svm_results, patients_testing, type = "prob") %>%
  bind_cols(patients_testing %>% select(stay))

roc_auc(patients_predict, truth = patients_testing$stay, `.pred_0-10`, `.pred_11-20`, `.pred_21-30`, `.pred_31-40`)

roc_curve(patients_predict, truth = patients_testing$stay, `.pred_0-10`, `.pred_11-20`, `.pred_21-30`, `.pred_31-40`) %>%
  autoplot()

