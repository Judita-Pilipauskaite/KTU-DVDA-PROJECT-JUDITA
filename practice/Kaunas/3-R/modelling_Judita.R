install.packages('h2o')
library(h2o)
library(tidyverse)
h2o.init(max_mem_size = "8g")

df <- h2o.importFile("../../../project/1-data/train_data.csv")
test_data <- h2o.importFile("../../../project/1-data/test_data.csv")
df
class(df)
summary(df)

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
                  max_runtime_secs = 60)

aml@leaderboard

model <- aml@leader


model <- h2o.getModel("GBM_1_AutoML_2_20231027_195712")

h2o.performance(model, train = TRUE)
h2o.performance(model, valid = TRUE)
perf <- h2o.performance(model, newdata = test)

h2o.auc(perf)
plot(perf, type = "roc")

#h2o.performance(model, newdata = test_data)

predictions <- h2o.predict(model, test_data)

predictions

predictions %>%
  as_tibble() %>%
  mutate(id = row_number(), y = p0) %>%
  select(id, y) %>%
  write_csv("../5-predictions/predictions1.csv")

### ID, Y

h2o.saveModel(model, "../4-model/", filename = "my_best_automlmodel")

model <- h2o.loadModel("../4-model/my_best_automlmodel")
h2o.varimp_plot(model)


# 2023.11.10

rf_model <- h2o.randomForest(x,
                             y,
                             training_frame = train,
                             validation_frame = valid,
                             ntrees = 20,
                             max_depth = 10,
                             stopping_metric = "AUC",
                             seed = 1234)
rf_model
h2o.auc(rf_model)
h2o.auc(h2o.performance(rf_model, valid = TRUE))
h2o.auc(h2o.performance(rf_model, newdata = test))

h2o.saveModel(rf_model, "../4-model/", filename = "rf_model1")

# Write GBM?


gbm_model <- h2o.gbm(x,
                     y,
                     training_frame = train,
                     validation_frame = valid,
                     ntrees = 20,
                     max_depth = 10,
                     stopping_metric = "AUC",
                     seed = 1234)
h2o.auc(gbm_model)
h2o.auc(h2o.performance(gbm_model, valid = TRUE))
h2o.auc(h2o.performance(gbm_model, newdata = test))

# deep learning


dl_model <- h2o.deeplearning(
  model_id="dl_model",
  activation =  "Tanh",
  training_frame=train, 
  validation_frame=valid, 
  x=x, 
  y=y, 
  overwrite_with_best_model=F,    ## Return the final model after 10 epochs, even if not the best
  hidden=c(32,16,32),           ## more hidden layers -> more complex interactions
  epochs=5,                      ## to keep it short enough
  score_validation_samples=10000, ## down sample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  adaptive_rate=F,                ## manually tuned learning rate
  rate=0.01, 
  rate_annealing=2e-6,            
  momentum_start=0.2,             ## manually tuned momentum
  momentum_stable=0.4, 
  momentum_ramp=1e7, 
  l1=1e-5,                        ## add some L1/L2 regularization
  l2=1e-5,
  seed = 1234
) 

# model performance
summary(dl_model)
h2o.auc(dl_model)
h2o.auc(h2o.performance(dl_model, valid = TRUE))
h2o.auc(h2o.performance(dl_model, newdata = test))

# Grid search


dl_params <- list(hidden = list(10, c(10, 10), c(10,10,10)))

dl_grid <- h2o.grid(algorithm = "deeplearning",
                    grid_id = "ktu_grid",
                    x,
                    y,
                    training_frame = train,
                    validation_frame = valid,
                    epochs = 1,
                    stopping_metric = "AUC",
                    hyper_params = dl_params)


h2o.getGrid(dl_grid@grid_id, sort_by = "auc")

best_grid <- h2o.getModel(dl_grid@model_ids[[3]])
