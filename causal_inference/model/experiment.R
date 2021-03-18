# Title     : Experiment
# Objective : Run Repeated Experiments with BART
# Created by: adam
# Created on: 2/23/21


####################################
### Load Packages and Functions  ###
####################################

options(java.parameters = "-Xmx50g")

library("bartMachine")
set_bart_machine_num_cores(16)
library("tidyr")

load_data <- function(file, outcome) {
  df <- read.csv(file)
  df <- df %>% drop_na(outcome)
  outcomes = c("pf_ratio_2h_8h_outcome",
               "pf_ratio_2h_8h_manual_outcome",
               "pf_ratio_12h_24h_outcome",
               "pf_ratio_12h_24h_manual_outcome")
  df <- df[, -which(names(df) %in% setdiff(outcomes, outcome))]
  return(df)
}
train_test_split <- function(df, outcome, train_size = 0.8) {
  df_treated <- df[df[, "treated"] == "True", -which(names(df) %in% "treated")]
  df_control <- df[df[, "treated"] == "False", -which(names(df) %in% "treated")]

  sample_treated <- sample.int(n = nrow(df_treated), size = floor(train_size*nrow(df_treated)), replace = F)
  sample_control <- sample.int(n = nrow(df_control), size = floor(train_size*nrow(df_control)), replace = F)

  X_train_treated <- df_treated[sample_treated, -which(names(df_treated) %in% outcome)]
  X_train_control <- df_control[sample_control, -which(names(df_control) %in% outcome)]
  y_train_treated <- df_treated[sample_treated, outcome]
  y_train_control <- df_control[sample_control, outcome]

  X_test_treated <- df_treated[-sample_treated, -which(names(df_treated) %in% outcome)]
  X_test_control <- df_control[-sample_control, -which(names(df_control) %in% outcome)]
  y_test_treated <- df_treated[-sample_treated, outcome]
  y_test_control <- df_control[-sample_control, outcome]

  return(list(X_train_treated,
              X_train_control,
              X_test_treated,
              X_test_control,
              y_train_treated,
              y_train_control,
              y_test_treated,
              y_test_control))
}
s_learner_predict_ate <- function(model, data) {
  # Predict outcomes, if everybody in the test set would have been treated
  data$treated <- TRUE
  m_1 <- predict(model, data)
  # Predict outcomes, if everybody in the control set would have been control
  data$treated <- FALSE
  m_0 <- predict(model, data)
  # Calculate the mean difference
  cate <- mean(m_1) - mean(m_0)
  # Take confidence intervals
  ci_low <- cate - 1.96 * sqrt(var(m_1)/length(m_1) + var(m_0)/length(m_0))
  ci_high <- cate + 1.96 * sqrt(var(m_1)/length(m_1) + var(m_0)/length(m_0))
  return(list(cate, ci_low, ci_high))
}
prepare_test_data <- function(data) {
  X_test_treated <- data[[3]]
  X_test_control <- data[[4]]
  y_test_treated <- data[[7]]
  y_test_control <- data[[8]]

  X_test_treated$treated <- TRUE
  X_test_control$treated <- FALSE

  X_test <- rbind(X_test_treated, X_test_control)
  y_test <- c(y_test_treated, y_test_control)

  return(list(X_test, y_test))
}
run_training_loop <- function(data) {
  X_train_treated <- data[[1]]
  X_train_control <- data[[2]]
  y_train_treated <- data[[5]]
  y_train_control <- data[[6]]

  # Bootstrap
  sample_treated <- sample.int(n = nrow(X_train_treated), size = floor(0.95*nrow(X_train_treated)), replace = T)
  X_train_treated <- X_train_treated[sample_treated, ]
  y_train_treated <- y_train_treated[sample_treated]

  sample_control <- sample.int(n = nrow(X_train_control), size = floor(0.95*nrow(X_train_control)), replace = T)
  X_train_control <- X_train_control[sample_control, ]
  y_train_control <- y_train_control[sample_control]

  X_train_treated$treated <- TRUE
  X_train_control$treated <- FALSE

  X_train <- rbind(X_train_treated, X_train_control)
  y_train <- c(y_train_treated, y_train_control)

  # shuffle
  shuffle <- sample(nrow(X_train))
  X_train <- X_train[shuffle, ]
  y_train <- y_train[shuffle]

  bart_machine <- bartMachine(X_train,
                              y_train,
                              use_missing_data = TRUE,
                              use_missing_data_dummies_as_covars = FALSE,
                              replace_missing_data_with_x_j_bar = TRUE,
                              num_trees = 50,
                              mem_cache_for_speed=TRUE,
                              alpha = 0.95,
                              beta = 2,
                              k = 2,
                              q = 0.9,
                              nu = 3,
                              num_burn_in = 500,
                              num_iterations_after_burn_in = 1000)

  return(bart_machine)
}
test_training_loop <- function(model, test_data) {
  oos_perf = bart_predict_for_test_data(model, test_data[[1]], test_data[[2]])
  rmse <- oos_perf$rmse
  r2 <- 1 - ((oos_perf$L2) / sum((test_data[[2]] - mean(test_data[[2]]))^2))
  ate <- s_learner_predict_ate(model, test_data[[1]])[[1]]

  return(list(ate, rmse, r2))
}
run_experiment <- function(data, test_data, n_of_experiments) {
  ate_vector <- 0
  rmse_vector <- 0
  r2_vector <- 0

  for (i in 1:n_of_experiments){
    print(i)
    model <- run_training_loop(data)
    results <- test_training_loop(model, test_data)
    ate_vector <- c(ate_vector, results[[1]])
    rmse_vector <- c(rmse_vector, results[[2]])
    r2_vector <- c(r2_vector, results[[3]])
  }
  return(list(ate_vector[-1], rmse_vector[-1], r2_vector[-1]))
}
save_results <- function(results, path) {
  ate <- results[[1]]
  rmse <- results[[2]]
  r2 <- results[[3]]
  df_results <- data.frame(ate, rmse, r2)
  write.csv(df_results, path, row.names = TRUE)
}
save_summary <- function(results, path){
  ate <- c(mean(results[[1]]),
           quantile(results[[1]], probs = c(0.025, 0.975))[[1]],
           quantile(results[[1]], probs = c(0.025, 0.975))[[2]])
  rmse <- c(mean(results[[2]]),
            quantile(results[[2]], probs = c(0.025, 0.975))[[1]],
            quantile(results[[2]], probs = c(0.025, 0.975))[[2]])
  r2 <- c(mean(results[[3]]),
          quantile(results[[3]], probs = c(0.025, 0.975))[[1]],
          quantile(results[[3]], probs = c(0.025, 0.975))[[2]])

  df_summary <- t(data.frame(ate, rmse, r2, row.names = c("mean", "0.025_percentile", "0.975_percentile")))
  write.csv(df_summary, path, row.names = TRUE)
}

############################################
### Set Seed and Define Global Variables ###
############################################


set.seed(12345) # seed
path <- "/home/adam/adam/data/19012021/" #data folder
setwd(path)
outcome <- "pf_ratio_2h_8h_manual_outcome"
n_of_experiments <- 100


#########################
### Run the Experiment ###
#########################

df <- load_data('data_guerin_rct_fixed_prone.csv', outcome)
data <- train_test_split(df, outcome, train_size = 0.8)
test_data <- prepare_test_data(data)
results <- run_experiment(data, test_data, n_of_experiments)

#########################
### Print the Results ###
#########################

print(mean(results[[1]])) # mean ATE
print(mean(results[[2]])) # mean RMSE
print(mean(results[[3]])) # mean R2

quantile(results[[1]], probs = c(0.025, 0.975))
quantile(results[[2]], probs = c(0.025, 0.975))
quantile(results[[3]], probs = c(0.025, 0.975))

####################
### Save Results ###
####################

setwd("/home/adam/adam/data/results_fix/")

#path_predictions <- sprintf("results_BART_%s.csv", outcome) not yet implemented
#save_predictions(results, path_predictions)

path_results <- "results_BART_pf_ratio_2h_8h_manual_outcome.csv"
save_results(results, path_results)

path_summary <- "summary_BART_pf_ratio_2h_8h_manual_outcome.csv"
save_summary(results, path_summary)

# training with obesity
# should add training data results
# should incorporate loading of two separate files training and test .csv
# I am not dropping any features