library(dplyr)
library(readr)
read_csv("../1-data/put_your_data_files_here.csv")
### save combined file into 1-data directory

install.packages("tidyverse")
library(tidyverse)

setwd("C:/Users/adminas/Desktop/Daugela/KTU-DVDA-PROJECT-JUDITA/project/")

data <- read_csv("1-data/1-sample_data.csv")
data2 <- read_csv("1-data/2-additional_data.csv")
data_additional_features <- read_csv("1-data/3-additional_features.csv")

dim(data)
dim(data2)
dim(data_additional_features)

combined_data <- rbind(data, data2)
head(combined_data)
dim(combined_data)

joined_data <- full_join(combined_data, data_additional_features, by = "id")

write_csv(joined_data, "1-data/train_data.csv")

head(joined_data)
dim(joined_data)
