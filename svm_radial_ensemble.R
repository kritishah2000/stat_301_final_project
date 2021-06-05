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


#show_best(svm_res, metric = "accuracy")

#svm_res_2_workflow_tuned <- svm_workflow %>% 
  #finalize_workflow(select_best(svm_res_2, metric = "accuracy"))

#svm_res_2_results <- fit(svm_res_2_workflow_tuned, patients_train)

#patients_metrics <- metric_set(roc_auc, accuracy)

#predict(rf_results_knn, new_data = titanic_test, type = "prob") %>%
  #bind_cols(titanic_test %>% select(survived)) %>%
  #roc_auc(survived, .pred_Yes) 
