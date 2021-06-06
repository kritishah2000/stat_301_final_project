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




load("data/setup.rda")

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

stopCluster(cl)

save(boosted_tune, boosted_workflow, file = "data/boosted_tune.rda")

autoplot(boosted_tune, metric = "roc_auc")
select_best(boosted_tune, metric = "roc_auc")
show_best(boosted_tune, metric = "roc_auc")

boosted_workflow_tuned <- boosted_workflow %>% 
  finalize_workflow(select_best(boosted_tune, metric = "roc_auc"))
                    
boosted_results <- fit(boosted_workflow_tuned, patients_train)

patients_predict <- predict(boosted_results, patients_testing, type = "prob") %>%
  bind_cols(patients_testing %>% select(stay))

roc_auc(patients_predict, truth = patients_testing$stay, `.pred_0-10`, `.pred_11-20`, `.pred_21-30`, `.pred_31-40`)

roc_curve(patients_predict, truth = patients_testing$stay, `.pred_0-10`, `.pred_11-20`, `.pred_21-30`, `.pred_31-40`) %>% 
  autoplot()


