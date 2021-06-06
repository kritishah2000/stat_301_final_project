#loading packages
library(tidyverse)
library(skimr)
library(tidymodels)
library(janitor)
library(xgboost)

library(doParallel)
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

set.seed(3013)

# load required objects ----
patients_data <- read_csv("data/unprocessed/train_data.csv") %>% 
  clean_names()

patients_data <- patients_data %>% 
  filter(stay == "0-10" | stay == "11-20" | stay == "21-30" | stay == "31-40")

patients_split <- initial_split(data = patients_data, prop = 0.7, strata = stay)
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

#dim(patients_train)

#patients_folds <- vfold_cv(patients_train, v = 5, repeats = 3, strata = stay)


#patients_recipe <- recipe(stay ~ department + ward_type + ward_facility_code + severity_of_illness + age + visitors_with_patient + admission_deposit,
#                          data = patients_train) %>% 
#  step_other(all_nominal(), -all_outcomes(), threshold = 0.1) %>%  #increase threshold values
#  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
#  step_interact(~starts_with("severity_of_illness"):visitors_with_patient) %>% 
#  step_scale(all_numeric()) %>% 
#  step_normalize(all_numeric())

prep(patients_recipe) %>% 
  bake(new_data = NULL)

# Define model ----

boosted_model <-  boost_tree(mode = "classification",
                        mtry = tune(),
                        min_n = tune()) %>%
  set_engine("xgboost")

# set-up tuning grid ----
boosted_params <- parameters(boosted_model) %>%
  update(mtry = mtry(range = c(1, 8)),
         min_n = min_n(range = c(1, 4)))


# define tuning grid

boosted_grid <- grid_regular(boosted_params, levels = 5)

# workflow ----
boosted_workflow <- workflow() %>% 
  add_model(boosted_model) %>% 
  add_recipe(patients_recipe)


# tuning
boosted_tune <- boosted_workflow %>% 
  tune_grid(
    resamples = patients_folds,
    grid = boosted_grid
  )


save(boosted_tune, boosted_workflow, file = "data/boosted_tune.rda")


show_best(boosted_tune, metric = "accuracy")

stopCluster(cl)