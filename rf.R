#loading packages
library(tidyverse)
library(skimr)
library(tidymodels)
library(janitor)




set.seed(3013)

# load required objects ----
patients_data <- read_csv("data/unprocessed/train_data.csv") %>% 
  clean_names() 

#patients_data <- patients_data %>% 
#  mutate(stay = if_else(stay == "0-10" | stay == "11-20" | stay == "21-30" | stay == "31-40" |stay == "41-50", "0-50", "> 50")) %>% 
#  select(stay)

patients_data <- patients_data %>% 
  filter(stay == "0-10" | stay == "11-20" | stay == "21-30" | stay == "31-40")

patients_split <- initial_split(data = patients_data, prop = 0.1, strata = stay)
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
# patients_recipe <- recipe(stay ~ hospital_type_code + hospital_code + available_extra_rooms_in_hospital +
#                             department + ward_type + ward_facility_code + severity_of_illness + age + visitors_with_patient + admission_deposit,
#                           data = patients_train) %>% 
#   step_other(all_nominal(), -all_outcomes(), threshold = 0.1) %>%  #increase threshold values
#   step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
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

rf_model <-  rand_forest(mode = "classification", mtry = tune(), min_n = tune()) %>% 
  set_engine("ranger")

# set-up tuning grid ----
rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(1,20)))


# define tuning grid

rf_grid <- grid_regular(rf_params, levels = 3)

# workflow ----
rf_workflow3 <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(patients_recipe)


# tuning
rf_tune3 <- rf_workflow3 %>% 
  tune_grid(
    resamples = patients_folds,
    grid = rf_grid
  )


save(rf_tune3, rf_workflow3, file = "data/rf_tune3.rda")



# #results
# autoplot(rf_tune2, metric = "accuracy")
# select_best(rf_tune2, metric = "accuracy")
show_best(rf_tune3, metric = "accuracy")
#accuracy = 0.454

# 
# rf_workflow_tuned <- rf_workflow %>%
#   finalize_workflow(select_best(rf_tune, metric = "accuracy"))
# 
# 
# rf_results <- fit(rf_workflow_tuned, training_data)
# 
# data_metric_rf <- metric_set(accuracy)
# 
# data_predict_rf <- predict(rf_results, new_data = testing_data) %>%
#   bind_cols(testing_data %>% select(id))









