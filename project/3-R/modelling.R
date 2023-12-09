install.packages("h2o")
library(h2o)
library(tidyverse)
#update.packages()
h2o.init(max_mem_size = "6g")
h2o.init()
h2o.algorithm("xgboost")


# Check if XGBoost is available
if ("xgboost" %in% rownames(h2o.list_core_extensions())) {
  print("XGBoost is available")
} else {
  print("XGBoost is not available")
}

# Check the H2O version
packageVersion("h2o")


setwd("C:/Users/adminas/Desktop/Daugela/KTU-DVDA-PROJECT-JUDITA/project/3-R")

print(getwd())

df <- h2o.importFile("../../project/1-data/train_data.csv")
test_data <- h2o.importFile("../../project/1-data/test_data.csv")

get_unique_values <- function(data, column_name) {
  unique_values <- h2o.unique(data[[column_name]])
  return(as.data.frame(unique_values))
}

unique_values <- get_unique_values(df, "credit_problems")

print(unique_values)

# Assuming 'y' is the name of your column in 'df'
y_col <- as.vector(df$y)  # Convert the H2OFrame column to a regular R vector

# Now, use the table function to get the distribution
class_distribution <- table(y_col)

# Convert the distribution to percentages
class_distribution_percent <- prop.table(class_distribution) * 100

# Print the distribution in percentage
print(class_distribution_percent)


head(df)
summary(df)
str(df)
class(df)
class(test_data)
summary(test_data)


y <- "y"
x <- setdiff(names(df), c(y, "id"))
df$y <- as.factor(df$y)
summary(df)

splits <- h2o.splitFrame(df, c(0.6,0.2), seed=123)
train  <- h2o.assign(splits[[1]], "train") # 60%
valid  <- h2o.assign(splits[[2]], "valid") # 20%
test   <- h2o.assign(splits[[3]], "test")  # 20%

aml <- h2o.automl(x = x,
                  y = y,
                  training_frame = train,
                  validation_frame = valid,
                  max_runtime_secs = 5000)


aml@leaderboard

model <- aml@leader
model <- h2o.getModel("StackedEnsemble_AllModels_2_AutoML_1_20231203_202933")

h2o.performance(model, train = TRUE)
h2o.performance(model, valid = TRUE)
perf <- h2o.performance(model, newdata = test)
perf

h2o.auc(perf)
plot(perf, type = "roc")

#Spėjimas

#h2o.performance(model, newdata = test_data)

predictions <- h2o.predict(model, test_data)

predictions

predictions %>%
  as_tibble() %>%
  mutate(id = row_number(), y = p0) %>%
  select(id, y) %>%
  write_csv("../5-predictions/predictions12_03_(0_82).csv")

### ID, Y

h2o.saveModel(model, "../4-model/", filename = "my_best_automlmodel_12_03(0_82)")

model <- h2o.loadModel("../4-model/my_best_automlmodel_12_03(0_82)")
h2o.varimp_plot(model)


###LightGBM model
# AR PATARTUMET BANDYTI SUTVARKYT INA AR VISGI IGNOROTI???

install.packages("lightgbm")
install.packages("data.table")
install.packages("readxl")

# Load packages
library(lightgbm)
library(data.table)
library(readxl)

# Load your data
data <- read_csv("../../project/1-data/train_data.csv")

# Convert to data.table
df1 <- as.data.table(data)

# Prepare the data
# Assuming 'y' is your target variable and 'id' is a non-predictive field
y <- "y"
x <- setdiff(names(df1), c(y, "id"))
df1$y <- as.factor(df1$y)


# Set seed for reproducibility
set.seed(123)

# Calculate the number of rows to sample for training data (60% of the data)
train_size <- floor(nrow(df1) * 0.6)

# Sample indices for training data
train_indices <- sample(nrow(df1), train_size)

# Split the data into training and validation sets
train_data <- df1[train_indices, ]
valid_data <- df1[-train_indices, ]

# and y is your target variable

# Convert training and validation data to matrix
train_matrix <- as.matrix(train_data[, -y, with = FALSE])
valid_matrix <- as.matrix(valid_data[, -y, with = FALSE])

# Extract the target variable
train_label <- train_data[[y]]
valid_label <- valid_data[[y]]

# Prepare datasets for LightGBM
dtrain <- lgb.Dataset(data = train_matrix, label = train_label)
dvalid <- lgb.Dataset(data = valid_matrix, label = valid_label)


# Train the model
params <- list(
  objective = "binary", 
  metric = "binary_error", 
  num_leaves = 31,
  learning_rate = 0.05,
  n_estimators = 100
)
model <- lgb.train(
  params,
  dtrain,
  valids = list(val = dvalid),
  nrounds = 1000, # Correct parameter name
  early_stopping_rounds = 10
)

# Evaluate the model
predictions <- predict(model, valid_data[, -y, with = FALSE])
predictions

# Install and load the pROC package
install.packages("pROC")
library(pROC)

# Make predictions on the validation set
# Ensure that 'predictions' are probabilities
predictions <- predict(model, valid_matrix)

# Assuming the actual labels are binary (0 or 1) and are stored in 'valid_label'
roc_curve <- roc(valid_label, predictions)

# Print the AUC
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))

###GBM model


gbm_model <- h2o.gbm(x,
                     y,
                     training_frame = train,
                     validation_frame = valid,
                     ntrees = 40,
                     max_depth = 100,
                     stopping_metric = "AUC",
                     seed = 1234)
h2o.auc(gbm_model)
h2o.auc(h2o.performance(gbm_model, valid = TRUE))
h2o.auc(h2o.performance(gbm_model, newdata = test))





##-----------GBM auto--------------------

# Hyperparameter tuning setup
hyper_params <- list(
  learn_rate = c(0.01, 0.1),
  max_depth = c(3, 5, 9),
  sample_rate = c(0.8, 1.0),
  col_sample_rate = c(0.2, 0.5, 1.0),
  ntrees = 100
)

search_criteria <- list(
  strategy = "RandomDiscrete",
  max_models = 20,
  max_runtime_secs = 3600,
  seed = 123
)

# Use balance_classes for class imbalance (optional)
gbm_grid <- h2o.grid(
  "gbm",
  x = x, 
  y = y,
  training_frame = train,
  validation_frame = valid,
  grid_id = "gbm_grid",
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  balance_classes = TRUE,
  stopping_metric = "AUC",
  stopping_rounds = 2,
  seed = 123
)

# Retrieve the grid object
gbm_grid <- h2o.getGrid("gbm_grid")

# Summary of the grid search including stack traces
summary(gbm_grid, show_stack_traces = TRUE)

# Get the grid object
gbm_grid <- h2o.getGrid("gbm_grid")

# Get the sorted model IDs based on AUC
sorted_model_ids <- h2o.getGrid("gbm_grid", sort_by = "auc", decreasing = TRUE)@model_ids

# Check if there are any models
if (length(sorted_model_ids) > 0) {
  # Get the best model ID
  best_model_id <- sorted_model_ids[[1]]
  
  # Fetch the best model
  best_model <- h2o.getModel(best_model_id)
  
  # Print or use the best model
  print(best_model)
} else {
  print("No models were successfully built in the grid search.")
}


h2o.auc(best_model)
h2o.auc(h2o.performance(best_model, valid = TRUE))
h2o.auc(h2o.performance(best_model, newdata = test))

#------------------RF

rf_model <- h2o.randomForest(x,
                             y,
                             training_frame = train,
                             validation_frame = valid,
                             ntrees = 100,
                             max_depth = 10,
                             stopping_metric = "AUC",
                             seed = 123)
rf_model
h2o.auc(rf_model)
h2o.auc(h2o.performance(rf_model, valid = TRUE))
h2o.auc(h2o.performance(rf_model, newdata = test))

h2o.saveModel(rf_model, "../4-model/", filename = "rf_model1")

# Define the hyperparameters to search over
hyper_params <- list(
  ntrees = seq(10, 300, 5),           # Number of trees (e.g., from 10 to 100 in steps of 10)
  max_depth = seq(1, 30, 1),           # Maximum depth (e.g., from 5 to 20 in steps of 5)
  sample_rate = seq(0.6, 1, 0.1),       # Sample rate (e.g., from 0.6 to 1.0 in steps of 0.1)
  mtries = seq(1,20,1),
  min_rows = seq(1,10,1)
  )

# Define the search criteria
search_criteria <- list(
  strategy = "RandomDiscrete",         # Random search
  #max_models = 20,                     # Maximum number of models to build
  max_runtime_secs = 700,             # Maximum total runtime
  seed = 1234                          # Seed for reproducibility
)

# Perform the grid search
grid <- h2o.grid(
  algorithm = "randomForest",
  grid_id = "rf_grid",
  x = x,
  y = y,
  training_frame = train,
  validation_frame = valid,
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  stopping_metric = "AUC",
  seed = 1234
)

# Get the sorted model IDs based on AUC
sorted_model_ids <- h2o.getGrid("rf_grid", sort_by = "auc", decreasing = TRUE)@model_ids

# Check if there are any models
if (length(sorted_model_ids) > 0) {
  # Get the best model ID
  best_model_id <- sorted_model_ids[[1]]
  
  # Fetch the best model
  best_model_rf <- h2o.getModel(best_model_id)
  
  # Print or use the best model
  print(best_model)
} else {
  print("No models were successfully built in the grid search.")
}


h2o.auc(best_model_rf)
h2o.auc(h2o.performance(best_model_rf, valid = TRUE))
h2o.auc(h2o.performance(best_model_rf, newdata = test))
