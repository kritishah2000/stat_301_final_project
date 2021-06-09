#loading packages
library(tidyverse)
library(skimr)
library(tidymodels)
library(janitor)
library(vcd)
library(corrplot)
library(corrr)


set.seed(3013)
#instead of accuracy, use F measure, auc, balance accuracy, average precision
#binary (less than 50 days or more than 50 days)
#confusion matrix (systematic over estimating and underestimating)

# load required objects ----
patients_data <- read_csv("data/unprocessed/train_data.csv") %>% 
  clean_names() %>% 
  view()

patietns_data <- patients_data %>% 
  mutate(stay = if_else(stay == "0-10" | stay == "11-20" | stay == "21-30" | stay == "31-40" |stay == "41-50", "0-50", "> 50")) %>% 
  select(stay)


patients_split <- initial_split(data = patients_data, prop = 0.5, strata = stay)
patients_train <- training(patients_split)
patients_testing <- testing(patients_split)


#cleaning up the data
patients_train <- patients_train %>% 
  mutate(
    stay = as.factor(stay),
    hospital_code = as.integer(hospital_code),
    hospital_type_code = as.factor(hospital_type_code),
    hospital_region_code = as.factor(hospital_region_code),
    available_extra_rooms_in_hospital = as.integer(available_extra_rooms_in_hospital),
    department = as.factor(department),
    ward_type = as.factor(ward_type),
    bed_grade = as.integer(bed_grade),
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
    bed_grade = as.integer(bed_grade),
    severity_of_illness = as.factor(severity_of_illness),
    visitors_with_patient = as.integer(visitors_with_patient),
    age = as.factor(age),
    type_of_admission = as.factor(type_of_admission)
  )


#creating folds

dim(patients_train)

patients_folds <- vfold_cv(patients_train, v = 5, repeats = 1, strata = stay)

#creating recipe
# patients_recipe <- recipe(stay ~ department + ward_type + ward_facility_code +
#                           severity_of_illness + visitors_with_patient + age + admission_deposit +
#                            bed_grade,
#                           data = patients_train) %>% 
#   step_other(all_nominal(), -all_outcomes(), threshold = 0.1) %>%  #increase threshold values
#   step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
#   step_interact(~starts_with("severity_of_illness"):visitors_with_patient) %>% 
#   step_scale(all_numeric()) %>% 
#   step_normalize(all_numeric())

patients_recipe <- recipe(stay ~ department + ward_type + ward_facility_code + severity_of_illness + age + visitors_with_patient + admission_deposit,
                          data = patients_train) %>% 
  step_other(all_nominal(), -all_outcomes(), threshold = 0.1) %>%  #increase threshold values
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
  step_interact(~starts_with("severity_of_illness"):visitors_with_patient) %>% 
  step_scale(all_numeric()) %>% 
  step_normalize(all_numeric())

prep(patients_recipe) %>% 
  bake(new_data = NULL)



# Define model ----

elastic_model <-  multinom_reg(mode = "classification", penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")



# set-up tuning grid ----
elastic_params <- parameters(elastic_model)


# define tuning gridgrid
elastic_grid <- grid_regular(elastic_params, levels = 5)

# workflow ----
elastic_workflow <- workflow() %>% 
  add_model(elastic_model) %>% 
  add_recipe(patients_recipe)

# Tuning/fitting ----
elastic_tune <- elastic_workflow %>% 
  tune_grid(
    resamples = patients_folds,
    grid = elastic_grid
  )


save(elastic_tune, elastic_workflow, file = "data/elastic_tune.rda")


#Results
# autoplot(elastic_tune, metric = "accuracy")
# select_best(elastic_tune, metric = "accuracy")
 show_best(elastic_tune, metric = "accuracy")

# elastic_workflow_tuned <- elastic_workflow %>%
#   finalize_workflow(select_best(elastic_tune, metric = "accuracy"))
# 
# 
# elastic_results <- fit(elastic_workflow_tuned, training_data)
# 
# data_metric_elastic <- metric_set(accuracy)
# 
# data_predict_elastic <- predict(elastic_results, new_data = testing_data) %>%
#   bind_cols(testing_data %>% select(id))
# 










