#loading packages
library(tidyverse)
library(skimr)
library(tidymodels)
library(janitor)
library(vcd)
library(corrplot)
library(corrr)
library(stacks)
 
set.seed(3013)

# load required objects ----
patients_data <- read_csv("data/unprocessed/train_data.csv") %>% 
  clean_names()

patients_data <- patients_data %>% 
  filter(stay == "0-10" | stay == "11-20" | stay == "21-30" | stay == "31-40")

patients_split <- initial_split(data = patients_data, prop = 0.01, strata = stay)
patients_train <- training(patients_split)
patients_testing <- testing(patients_split)

#ggplot(patients_data) +
  #geom_bar(mapping = aes(stay))

#dim(patients_train)

patients_folds <- vfold_cv(patients_train, v = 5, repeats = 3, strata = stay)


# hospital_code + hospital_type_code + available_extra_rooms_in_hospital + department + ward_type + ward_facility_code + severity_of_illness + visitors_with_patient + age + admission_deposit
patients_recipe_2 <- recipe(stay ~ 
                            department + ward_type + ward_facility_code + severity_of_illness + visitors_with_patient + age + admission_deposit,
                          data = patients_train) %>% 
  step_other(all_nominal(), -all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_interact(~ starts_with("severity_of_illness"):visitors_with_patient) %>% 
  step_normalize(all_numeric())

prep(patients_recipe_2) %>% 
  bake(new_data = NULL)

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
  add_recipe(patients_recipe_2)

# control settings ----

ctrl_grid <- control_stack_grid()

# Tuning/fitting ----
svm_res_2 <- svm_workflow_2 %>%
  tune_grid(
    resamples = patients_folds,
    grid = svm_grid,
    control = ctrl_grid
  )

save(svm_res_2, file = "svm_results.rda")


#show_best(svm_res, metric = "accuracy")

#svm_res_2_workflow_tuned <- svm_workflow %>% 
  #finalize_workflow(select_best(svm_res_2, metric = "accuracy"))

#svm_res_2_results <- fit(svm_res_2_workflow_tuned, patients_train)

#patients_metrics <- metric_set(roc_auc, accuracy)

#predict(rf_results_knn, new_data = titanic_test, type = "prob") %>%
  #bind_cols(titanic_test %>% select(survived)) %>%
  #roc_auc(survived, .pred_Yes) 
