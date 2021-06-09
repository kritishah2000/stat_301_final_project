# Kriti - Mars

#loading packages
library(tidyverse)
library(skimr)
library(tidymodels)
library(janitor)
library(vcd)
library(corrplot)
library(corrr)


set.seed(3013)

# load required objects ----
patients_data <- read_csv("data/unprocessed/train_data.csv") %>% 
  clean_names()

patients_data <- patients_data %>% 
  filter(stay == "0-10" | stay == "11-20" | stay == "21-30" | stay == "31-40")
  
# ggplot(aes(stay)) +
  # geom_bar()

patients_split <- initial_split(data = patients_data, prop = 0.01, strata = stay)
patients_train <- training(patients_split)
patients_testing <- testing(patients_split)

dim(patients_train)

patients_train <- patients_train %>% 
  mutate(
    stay = as.factor(stay),
    hospital_code = as.integer(hospital_code),
    hospital_type_code = as.factor(hospital_type_code),
    hospital_region_code = as.factor(hospital_region_code),
    available_extra_rooms_in_hospital = as.integer(available_extra_rooms_in_hospital),
    department = as.factor(department),
    ward_type = as.factor(ward_type),
    bed_grade = as.factor(bed_grade),
    severity_of_illness = as.factor(severity_of_illness),
    visitors_with_patient = as.integer(visitors_with_patient),
    age = as.factor(age),
    type_of_admission = as.factor(type_of_admission)
  )

patients_testing <- patients_testing %>% 
  mutate(
    hospital_code = as.integer(hospital_code),
    hospital_type_code = as.factor(hospital_type_code),
    hospital_region_code = as.factor(hospital_region_code),
    available_extra_rooms_in_hospital = as.integer(available_extra_rooms_in_hospital),
    department = as.factor(department),
    ward_type = as.factor(ward_type),
    bed_grade = as.factor(bed_grade),
    severity_of_illness = as.factor(severity_of_illness),
    visitors_with_patient = as.integer(visitors_with_patient),
    age = as.factor(age),
    type_of_admission = as.factor(type_of_admission)
  )

patients_folds <- vfold_cv(patients_train, v = 5, repeats = 3, strata = stay)

# got rid of bed_grade
# get rid of hospital_code, hospital_region_code, department, visitors_with_patient
# patients_recipe <- recipe(stay ~ hospital_code + hospital_type_code + hospital_region_code + available_extra_rooms_in_hospital +
#          department + ward_type + severity_of_illness + visitors_with_patient + age + type_of_admission,
#          data = patients_train) %>% 
#   step_other(all_nominal(), -all_outcomes()) %>% 
#   step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
#   step_normalize(all_numeric())

# patients_recipe <- recipe(stay ~ hospital_type_code + available_extra_rooms_in_hospital + 
#                             ward_type + severity_of_illness + age + type_of_admission,
#                   data = patients_train) %>%
#            step_other(all_nominal(), -all_outcomes()) %>%
#            step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>%
#            step_normalize(all_numeric())


# hospital_code + hospital_type_code + available_extra_rooms_in_hospital +
# keep ward_type, ward_facility_code + 
patients_recipe <- recipe(stay ~  
                           department + ward_type +  ward_facility_code + severity_of_illness + 
                            visitors_with_patient + age + admission_deposit + bed_grade,
                          data = patients_train) %>%
  step_other(all_nominal(), -all_outcomes()) %>%
  # step_dummy(all_nominal(), -all_outcomes(), -severity_of_illness, one_hot = TRUE) %>%
  # step_dummy(severity_of_illness) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_interact(~ starts_with("severity_of_illness"):visitors_with_patient) %>%
  step_scale(all_numeric()) %>% 
  step_normalize(all_numeric())

prep(patients_recipe) %>% 
  bake(new_data = NULL)


ggplot(patients_train, aes(bed_grade)) +
  geom_bar()

patients_train %>% 
  select(stay)

# Define model

slnn_model <- mlp(hidden_units = tune(), penalty = tune()) %>% 
  set_engine("nnet") %>% 
  set_mode("classification")

# set-up tuning grid ----
slnn_params <- parameters(slnn_model)

# define tuning grid
slnn_grid <- grid_regular(slnn_params, levels = 5)

# workflow ----
slnn_workflow <- workflow() %>% 
  add_model(slnn_model) %>% 
  add_recipe(patients_recipe)

# Tuning/fitting ----

slnn_tune <- slnn_workflow %>% 
  tune_grid(
    resamples = patients_folds, 
    grid = slnn_grid
  )

save(slnn_tune, slnn_workflow, file = "data/slnn_tune.rda")

load("data/slnn_tune.rda")

autoplot(slnn_tune, metric = "accuracy")
select_best(slnn_tune, metric = "accuracy")
show_best(slnn_tune, metric = "accuracy")

slnn_workflow_tuned <- slnn_workflow %>% 
  finalize_workflow(select_best(slnn_tune, metric = "accuracy"))

# slnn_results <- fit(slnn_workflow_tuned, loan_train)
# 
# # loan_metric <- metric_set(accuracy)
# 
# loan_predict <- predict(slnn_results, loan_test) %>%
#   bind_cols(loan_test %>% select(id))

# # how to save in the proper file. submit the submission file to Kaggle
# submission_format <- loan_predict %>%
#   mutate(Category = .pred_class) %>%
#   select(id, Category) %>%
#   arrange(id)
# 
# write_csv(submission_format, "data/slnn_submission_v2")