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

patients_folds <- vfold_cv(patients_train, v = 5, repeats = 3, strata = stay)

patients_recipe <- recipe(stay ~ hospital_code + hospital_type_code + hospital_region_code + available_extra_rooms_in_hospital +
         department + ward_type + bed_grade + severity_of_illness + visitors_with_patient + age + type_of_admission,
         data = patients_train) %>% 
  step_other(all_nominal(), -all_outcomes()) %>% 
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>% 
  step_normalize(all_numeric())

prep(patients_recipe) %>% 
  bake(new_data = NULL)

# text in the group if:
# adding/removing a predictor
# if we transformed or wanted to add another step to the recipe
# any changes to the recipe in general

# elastic net - Enat
# random forest - Enat
# nearest neighbors - Kranti
# boosted tree - Kranti
# mars - Kriti
# slnn - Kriti
# svm polynomial - Angie
# svm radial - Angie

# using metric accuracy and roc_auc as our judge
