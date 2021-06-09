# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(janitor)
library(stacks)
library(textrecipes)

# Handle common conflicts
# tidymodels_prefer()

# Load candidate model info ----
# load("model_info/knn_res.rda")
# load("model_info/svm_res.rda")
# load("model_info/lin_reg_res.rda")
load("data/slnn_ensemble.rda")
load("data/rf_tune5.rda")
load("data/svm_poly_ensemble.rda")
load("data/svm_results_ensemble.rda")

# Load split data object & get testing data
load("data/setup.rda")

# Create data stack ----
patients_st <- stacks() %>% 
  add_candidates(rf_tune5) %>% 
  add_candidates(slnn_tune) %>% 
  add_candidates(svm_poly_res) %>% 
  add_candidates(svm_res_2)
  # add_candidates(lin_reg_res) %>% 
  # add_candidates(svm_res)

as_tibble(patients_st)


# Fit the stack ----
# penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)

# Blend predictions using penalty defined above (tuning step, set seed)
set.seed(9876)

patients_st_tuned <- patients_st %>% 
  blend_predictions(penalty = blend_penalty)

# Save blended model stack for reproducibility & easy reference (Rmd report)

save(patients_st, patients_st_tuned, file = "tuned_stack.rda")
# Explore the blended model stack
autoplot(patients_st_tuned)
autoplot(patients_st_tuned, type = "members")
autoplot(patients_st_tuned, type = "weight")

# fit to ensemble to entire training set ----

patients_st_final <- patients_st_tuned  %>% 
  fit_members(required_pkgs())

# Save trained ensemble model for reproducibility & easy reference (Rmd report)
save(patients_st_final, file = "trained_ensemble.rda")

# Explore and assess trained ensemble model
patients_test_fitted <- patients_testing %>% 
  bind_cols(predict(patients_st_final, .))
patients_test_fitted

ggplot(patients_test_fitted, aes(stay, .pred)) +
  geom_point()

member_preds <- 
  patients_test_fitted %>%
  select(stay) %>%
  bind_cols(predict(patients_st_final, patients_test_fitted, members = TRUE))

map_dfr(member_preds, rmse, truth = stay, data = member_preds) %>%
  mutate(member = colnames(member_preds))
