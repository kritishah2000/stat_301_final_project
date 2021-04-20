#loading packages
library(tidyverse)
library(skimr)
library(tidymodels)

#loading data
patients_data <- read_csv("data/unprocessed/train_data.csv")

#EDA
skim_without_charts(patients_data)

#Initial overview of data:
  ##source(s) and any relevant information concerning how the data was collected/formed
   ##Number of observations (n), number of features (p), analysis of missingness (amount and patterns)

#Essential Findings (Important or Interesting)
  ##Thorough univariate investigation of response variable(s)

ggplot(patients_data) +
  geom_histogram(mapping = aes(Hospital_code))

ggplot(patients_data) +
  geom_bar(mapping = aes(Hospital_type_code))

ggplot(patients_data) +
  geom_bar(mapping = aes(Hospital_region_code))

ggplot(patients_data) +
  geom_bar(mapping = aes(Stay))

ggplot(patients_data) +
  geom_bar(mapping = aes(Stay))

ggplot(patients_data) +
  geom_bar(mapping = aes(Stay))

ggplot(patients_data) +
  geom_bar(mapping = aes(Stay))

ggplot(patients_data) +
  geom_bar(mapping = aes(Stay))
  
  ##Thorough univariate investigation of important predictor variable(s) - ones either believed to be important (domain knowledge) or hypothesized to be important. 


  ##Interesting/important relationships between response variable(s) and predictor variables.
#Hospital_type_Code, Hospital_region_code, `Available Extra Rooms in Hospital`, Department, Ward_Type**
#


   ##Interesting/important relationships among predictor variables.






#Secondary Findings

  ##Standard variable explorations for the domain area that are unsurprising and mainly conducted out of convention. 



  ##Findings that don't seem interesting or important, but show some potential. 