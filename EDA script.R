#loading packages
library(tidyverse)
library(skimr)
library(tidymodels)

#loading data
patients_data <- read_csv("data/unprocessed/train_data.csv")

#EDA
skim_without_charts(patients_data)