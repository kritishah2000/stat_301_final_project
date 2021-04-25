#loading packages
library(tidyverse)
library(skimr)
library(tidymodels)
library(janitor)
library(vcd)
library(corrplot)


#loading data
patients_data <- read_csv("data/unprocessed/train_data.csv") %>% 
  clean_names()



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
  geom_bar(mapping = aes(stay))
  
  ##Thorough univariate investigation of important predictor variable(s) - ones either believed to be important (domain knowledge) or hypothesized to be important. 


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


# mosaic(stay ~ hospital_type_code, patients_data)
# 
# ggplot(patients_data, aes(x = hospital_type_code, fill = stay)) +
#   geom_bar(position = "fill")

# mosaic(stay ~ city_code_hospital, patients_data)
# 
# ggplot(patients_data, aes(x = city_code_hospital, fill = stay)) +
#   geom_bar(position = "fill")

# mosaic(stay ~ hospital_region_code, patients_data)



# mosaic(stay ~ department, patients_data)

# Important: There is a difference in length of stay across departments; surgery has the longest
ggplot(patients_data, aes(x = department, fill = stay)) +
  geom_bar(position = "fill") +
  coord_flip()

# mosaic(stay ~ ward_type, patients_data)


# mosaic(stay ~ ward_facility_code, patients_data)
# 
# ggplot(patients_data, aes(x = ward_facility_code, fill = stay)) +
#   geom_bar(position = "fill")

# mosaic(stay ~ bed_grade, patients_data)
# 
# ggplot(patients_data, aes(x = bed_grade, fill = stay)) +
#   geom_bar(position = "fill")

# mosaic(stay ~ type_of_admission, patients_data)

# Trauma patients have the longest stay
ggplot(patients_data, aes(x = type_of_admission, fill = stay)) +
  geom_bar(position = "fill")
# 
# mosaic(stay ~ severity_of_illness, patients_data)

# Extreme illness has longest stay
ggplot(patients_data, aes(x = severity_of_illness, fill = stay)) +
  geom_bar(position = "fill")

# mosaic(stay ~ hospital_code, patients_data)
# 
# ggplot(patients_data, aes(x = hospital_code, fill = stay)) +
#   geom_bar(position = "fill")

# ggplot(patients_data, mapping = aes(x = stay, y = available_extra_rooms_in_hospital)) +
#   geom_boxplot()

# ggplot(patients_data, mapping = aes(x = stay, y = admission_deposit)) +
#   geom_boxplot()

# Important: The longer you stay, the more visitors you have
ggplot(patients_data, mapping = aes(x = stay, y = visitors_with_patient)) +
  geom_boxplot()

# ggplot(patients_data, mapping = aes(x = stay, y = city_code_patient)) +
#   geom_boxplot()


#Hospital_type_Code, Hospital_region_code, `Available Extra Rooms in Hospital`, Department, Ward_Type**
#



#    ##Interesting/important relationships among predictor variables.
# 
# mosaic( ~ hospital_code + hospital_type_code, patients_data)

ggplot(patients_data, mapping = aes(visitors_with_patient, available_extra_rooms_in_hospital)) +
  geom_point() 



# ggplot(patients_data, mapping = aes(admission_deposit, available_extra_rooms_in_hospital)) +
#   geom_point()



#Secondary Findings

  ##Standard variable explorations for the domain area that are unsurprising and mainly conducted out of convention. 



  ##Findings that don't seem interesting or important, but show some potential. 


# Important: There is no significant difference in length of stay between regions
ggplot(patients_data, aes(x = hospital_region_code, fill = stay)) +
  geom_bar(position = "fill")


# Important: There is a difference in length of stay across ward type
ggplot(patients_data, aes(x = ward_type, fill = stay)) +
  geom_bar(position = "fill")