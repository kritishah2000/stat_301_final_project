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

patients_split <- initial_split(data = patients_data, prop = 0.7, strata = stay)
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
patients_recipe <- recipe(stay ~  
                            department + ward_type +  ward_facility_code + severity_of_illness + 
                            visitors_with_patient + age + admission_deposit,
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

patients_train %>% 
  select(stay)

# Define model ----

mars_model <- mars(num_terms = tune(), prod_degree = tune()) %>% 
  set_engine("earth") %>% 
  set_mode("classification")


# set-up tuning grid ----
mars_params <- parameters(mars_model)
# update(num_terms = num_terms(range = c(1, 250)))


# define tuning grid
mars_grid <- grid_regular(mars_params, levels = 5)

# workflow ----
mars_workflow <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(patients_recipe)

# Tuning/fitting ----
# Place tuning code in here
mars_tune <- mars_workflow %>% 
  tune_grid(
    resamples = patients_folds, 
    grid = mars_grid
  )


save(mars_tune, mars_workflow, file = "data/mars_tune.rda")

load("data/mars_tune.rda")

autoplot(mars_tune, metric = "accuracy")
select_best(mars_tune, metric = "accuracy")
show_best(mars_tune, metric = "accuracy")

mars_workflow_tuned <- mars_workflow %>%
  finalize_workflow(select_best(mars_tune, metric = "accuracy"))

mars_results <- fit(mars_workflow_tuned, patients_train)

money_predict <- predict(mars_results, patients_test) %>%
  bind_cols(money_test %>% select(id))

# using metric accuracy and roc_auc as our judge
