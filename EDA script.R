#loading packages
library(tidyverse)
library(skimr)
library(tidymodels)
library(janitor)
library(vcd)
library(corrplot)
library(corrr)


#loading data
patients_data <- read_csv("data/unprocessed/train_data.csv") %>% 
  clean_names()



#EDA
skim_without_charts(patients_data)


#Initial overview of data:
  ##source(s) and any relevant information concerning how the data was collected/formed
   ##Number of observations (n), number of features (p), analysis of missingness (amount and patterns)

Our data was taken from a Kaggle dataset called [AV: Healthcare Analytics II](https://www.kaggle.com/nehaprabhavalkar/av-healthcare-analytics-ii) published by Neha Prabhavlakar in 2020. The data was compiled by Analytics Vidhya for a Kaggle competition exploring the theme of healthcare analytics and the application of machine learning on healthcare case studies. 

In general, the dataset contains information on various factors that may influence the length of stay for a patient during the COVID-19 pandemic. Ultimately, using these variables, we were intending to train models to predict the length of stay by relating to the response variable (`stay` -- number of stay days by the patient) with the following 13 predictor variables:
* `hospital_code`: unique code for hospital
* `hospital_type_code`: unique code for type of hospital
* `hospital_region_code`: region code of hospital
* `available_extra_rooms_in_hospital`: number of extra rooms available in the hospital
* `department`: department overlooking the case
* `ward_type`: code for the ward type
* `ward_facility_code`: code for the ward facility
* `bed_grade`: condition of bed in the ward
* `visitors_with_patient`: number of visitors with patient
* `age`: age of the patient
* `severity_of_illness`: severity of illness recorded at the time of admission
* `type_of_admission`: admission type registered by the hospital
* `admission_deposit`: deposit at the admission time

In total, there are 318438 rows of data. Upon reading in the data, we can see that there are only two variables with missing data: `bed_grade` and `city_code_patient`. Because `city_code_patient` is a unique identifier for each patient that would not reveal interesting relationships, we decided to exclude this variable from our analysis. However, we decided to keep `bed_grade`, so we planned to filter for  rows of data that were not missing `bed_grade` (excluding the 113 values of missing data).

Lastly, for our models, we also decided to exclude `case_id` and `patientid` because these were unique identifiers for each patient that would not show important relationships in the data.

#Essential Findings (Important or Interesting)
  ##Thorough univariate investigation of response variable(s)

ggplot(patients_data) +
  geom_bar(mapping = aes(stay))

  
  ##Thorough univariate investigation of important predictor variable(s) - ones either believed to be important (domain knowledge) or hypothesized to be important. 

ggplot(patients_data) +
  geom_histogram(mapping = aes(Hospital_code))

ggplot(patients_data) +
  geom_bar(mapping = aes(Hospital_type_code))

ggplot(patients_data) +
  geom_bar(mapping = aes(Hospital_region_code))

ggplot(patients_data) +
  geom_bar(mapping = aes(available_extra_rooms_in_hospital))

ggplot(patients_data) +
  geom_bar(mapping = aes(department))

ggplot(patients_data) +
  geom_bar(mapping = aes(ward_type))

ggplot(patients_data) +
  geom_bar(mapping = aes(bed_grade))

ggplot(patients_data) +
  geom_bar(mapping = aes(severity_of_illness))

ggplot(patients_data) +
  geom_bar(mapping = aes(visitors_with_patient))

ggplot(patients_data) +
  geom_bar(mapping = aes(age))

ggplot(patients_data) +
  geom_bar(mapping = aes(severity_of_illness))

ggplot(patients_data) +
  geom_bar(mapping = aes(type_of_admission))

  ##Interesting/important relationships between response variable(s) and predictor variables.


# mosaic(stay ~ age + bed_grade, data = patients_data)
# 
# png(file = "data/mosaic.png", width = 9, height = 9, units = "in", res = 140)
# mosaic(stay ~ age, patients_data, rot_labels=c(90,90,0,0)
# )
# dev.off()


# IMPORTANT: As age increases, length of stay increases
ggplot(patients_data, aes(x = age, fill = stay)) +
  geom_bar(position = "fill")

# Important: There is a difference in length of stay across departments; surgery has the longest
ggplot(patients_data, aes(x = department, fill = stay)) +
  geom_bar(position = "fill") +
  coord_flip()

# Trauma patients have the longest stay
ggplot(patients_data, aes(x = type_of_admission, fill = stay)) +
  geom_bar(position = "fill")

# Extreme illness has longest stay
ggplot(patients_data, aes(x = severity_of_illness, fill = stay)) +
  geom_bar(position = "fill")

# Important: The longer you stay, the more visitors you have
ggplot(patients_data, mapping = aes(x = stay, y = visitors_with_patient)) +
  geom_boxplot()

#    ##Interesting/important relationships among predictor variables.

ggplot(patients_data, mapping = aes(visitors_with_patient, available_extra_rooms_in_hospital)) +
  geom_point() 




#Secondary Findings

  ##Standard variable explorations for the domain area that are unsurprising and mainly conducted out of convention. 



  ##Findings that don't seem interesting or important, but show some potential. 


# Important: There is no significant difference in length of stay between regions
ggplot(patients_data, aes(x = hospital_region_code, fill = stay)) +
  geom_bar(position = "fill")


# Important: There is a difference in length of stay across ward type
ggplot(patients_data, aes(x = ward_type, fill = stay)) +
  geom_bar(position = "fill")