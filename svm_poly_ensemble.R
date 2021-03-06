#loading packages
library(tidyverse)
library(skimr)
library(tidymodels)
library(janitor)
library(vcd)
library(corrplot)
# library(corrr)
# library(stacks)
# library(textrecipes)


set.seed(3013)

# load required objects ----
load("data/setup.rda")
# 
# patients_data <- read_csv("data/unprocessed/train_data.csv") %>% 
#   clean_names()
# 
# patients_data <- patients_data %>% 
#   filter(stay == "0-10" | stay == "11-20" | stay == "21-30" | stay == "31-40")
# 
# patients_split <- initial_split(data = patients_data, prop = 0.01, strata = stay)
# patients_train <- training(patients_split)
# patients_testing <- testing(patients_split)
# 
# dim(patients_train)
# 
# patients_folds <- vfold_cv(patients_train, v = 5, repeats = 3, strata = stay)
# 
# 
# patients_recipe <- recipe(stay ~ 
#                             hospital_code + hospital_type_code + available_extra_rooms_in_hospital + department + ward_type + ward_facility_code + severity_of_illness + visitors_with_patient + age + admission_deposit,
#                           data = patients_train) %>% 
#   step_other(all_nominal(), -all_outcomes()) %>% 
#   step_dummy(all_nominal(), -all_outcomes(), one_hot = T) %>% 
#   step_normalize(all_numeric())
# 
# prep(patients_recipe) %>% 
#   bake(new_data = NULL)

# Define model ----
svm_poly_model <- svm_poly(
  mode = "classification",
  cost = tune()
) %>%
  set_engine("kernlab")

# # check tuning parameters
# parameters(svm_model)

# set-up tuning grid ----
svm_params <- parameters(svm_poly_model)

# define grid
svm_grid <- grid_regular(svm_params, levels = 3)

# workflow ----
svm_workflow_poly <- workflow() %>%
  add_model(svm_poly_model) %>%
  add_recipe(patients_recipe)

# Control Settings ----
# use inside tune grid
# ctrl_grid <- control_stack_grid() 
# # use inside fit resamples
# ctrl_res <- control_stack_resamples()

# Tuning/fitting ----
svm_poly_res <- svm_workflow_poly %>%
  tune_grid(
    resamples = patients_folds,
    grid = svm_grid
    # control = ctrl_grid
  )

save(svm_poly_res, file = "data/svm_poly_ensemble.rda")