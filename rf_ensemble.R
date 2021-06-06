#loading packages
library(tidyverse)
library(skimr)
library(tidymodels)
library(janitor)
#library(stacks)
#library(conflicted)
#library(textrecipes)

# Handle common conflicts
#tidymodels_prefer()


set.seed(3013)

load("data/setup.rda")

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
rf_workflow4 <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(patients_recipe)


#control settings
# ctrl_grid <- control_stack_grid()
# ctrl_res <- control_stack_resamples()

# tuning
rf_tune4 <- rf_workflow4 %>% 
  tune_grid(
    resamples = patients_folds,
    grid = rf_grid
    # control = ctrl_grid
  )


rf_tune5 <- rf_tune4

# save(rf_tune4, rf_workflow4, file = "data/rf_tune4.rda")
save(rf_tune5, file = "data/rf_tune5.rda")




load("data/rf_tune5.rda")
# 
autoplot(rf_tune5, metric = "roc_auc")
select_best(rf_tune5, metric = "roc_auc")
show_best(rf_tune5, metric = "roc_auc")

# 
rf_workflow_tuned <- rf_workflow4 %>%
  finalize_workflow(select_best(rf_tune5, metric = "roc_auc"))

rf_results <- fit(rf_workflow_tuned, patients_train)
# 
# # loan_metric <- metric_set(accuracy)
# 
patients_predict <- predict(rf_results, patients_testing, type = "prob") %>%
  bind_cols(patients_testing %>% select(stay))

roc_auc(patients_predict, truth = patients_testing$stay, `.pred_0-10`, `.pred_11-20`, `.pred_21-30`, `.pred_31-40`)

roc_curve(patients_predict, truth = patients_testing$stay, `.pred_0-10`, `.pred_11-20`, `.pred_21-30`, `.pred_31-40`) %>% 
  autoplot()



