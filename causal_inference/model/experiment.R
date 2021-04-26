# Title     : BART
# Objective : Run the experiment with BART on bootsrapped data.
# Created by: adam
# Created on: 2/23/21

########################
### Define Constants ###
########################

SEED <- 1234
ENV_NAME <- 'bart'
N_OF_ITERATIONS <- 100
PATH_TRAIN_DATA <- '/home/adam/adam/data/causal_inference/data/processed/guerin_2_8_train.npz'
PATH_TEST_DATA <- '/home/adam/adam/data/causal_inference/data/processed/guerin_2_8_test.npz'
PATH_RESULTS <- ''
PATH_SUMMARY <- ''

OLD_TRAIN <- '/home/adam/adam/cfrnet/data/bfpguerin_2_8.train.npz'
OLD_TEST <- '/home/adam/adam/cfrnet/data/bfpguerin_2_8.test.npz'
###########################
### Set Hyperparameters ###
###########################

N_OF_TREES <- 50
ALPHA <- 0.95
BETA <- 2
K <- 2
Q <- 0.9
NU <- 3

###################
### Set Options ###
###################

options(java.parameters = "-Xmx50g")
set.seed(SEED)

#########################
### Import Libraries  ###
#########################

library("bartMachine")
set_bart_machine_num_cores(16)
library("tidyr")
library("reticulate")

#########################
### Define Functions  ###
#########################

load_data <- function (path, np) {
  #' Loads bootsrapped data for the purpose of the experiment.
  #'
  #' @param path Path to load the data
  #'
  #' @return Returns a list list(y, t, X) of training/test data.

  # Load the data
  data <- np$load(path)

  return(list(y=data$f[["yf"]], t=data$f[["t"]], X=data$f[["x"]]))
}

train <- function (y, t, X) {
  #' Train a BART model.
  #'
  #' @param y training labels
  #' @param t training treatment indicator
  #' @param X training covariates matrix
  #'
  #' @return BART model trained on input data

  X <- as.data.frame.matrix(cbind(t, X))
  colnames <- colnames(X)

  bart_machine <- bartMachine(X,
                              y,
                              use_missing_data = TRUE,
                              use_missing_data_dummies_as_covars = FALSE,
                              replace_missing_data_with_x_j_bar = TRUE,
                              num_trees = N_OF_TREES,
                              mem_cache_for_speed=FALSE,
                              alpha = ALPHA,
                              beta = BETA,
                              k = K,
                              q = Q,
                              nu = NU,
                              num_burn_in = 500,
                              num_iterations_after_burn_in = 1000,
                              run_in_sample = FALSE)

  return(list(model=bart_machine, colnames=colnames))
}

evaluate <- function(model, y, t, X) {
  #' Evaluate a BART model.
  #'
  #' @param y training/test labels
  #' @param t training/test treatment indicator
  #' @param X training/test covariates matrix
  #'
  #' @return A list of list(rmse, r2, ate) model's evaluation.

  X <- as.data.frame.matrix(cbind(t, X))
  colnames(X) <- model$colnames

  oos_perf <- bart_predict_for_test_data(bart_machine = model$model,
                                         Xtest = X,
                                         ytest = y)
  rmse <- oos_perf$rmse
  r2 <- 1 - ((oos_perf$L2) / sum((y - mean(y))^2))
  ate <- s_learner_ate(model, X)

  return(list(rmse=rmse, r2=r2, ate=ate))
}

run_experiment <- function(path_train_data, path_test_data, n_of_iterations, env_name) {
  #' Runs the experiment
  #'
  #' @param train_data
  #' @param test_data
  #' @param n_of_iterations
  #'
  #' @return

  np <- load_enviroment(env_name)

  # Initialize result vectors
  results <- 0

  for (iteration in 1:n_of_iterations) {

    train_data <- load_data(path_train_data, np)

    model <- train(y = train_data$y[ ,iteration],
                   t = train_data$t[ ,iteration],
                   X = train_data$X[ , ,iteration])

    result_train <- evaluate(model,
                             y = train_data$y[ ,iteration],
                             t = train_data$t[ ,iteration],
                             X = train_data$X[ , ,iteration])

    test_data <- load_data(path_test_data, np)

    result_test <- evaluate(model,
                            y = test_data$y[ ,iteration],
                            t = test_data$t[ ,iteration],
                            X = test_data$X[ , ,iteration])

    result <- list(rmse_train = result_train$rmse,
                   r2_train = result_train$r2,
                   ate_train = result_train$ate,
                   rmse_test = result_test$rmse,
                   r2_test = result_test$r2,
                   ate_test = result_test$ate)

  if ('numeric' %in% class(0)) {results <- result} else {results <- rbind(results, result)}
  }
  return(results)
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

###########################
### Auxiliary Functions ###
###########################

load_enviroment <- function (env_name) {
  myenvs <- conda_list()
  envname <- myenvs[myenvs$name == env_name, 'name']
  use_condaenv(envname, required = TRUE)
  np <- import("numpy")
  return(np)
}

s_learner_ate <- function (model, X) {
  #' Calculate ATE with an S-learner.
  #'
  #' @param model model to be used as an S-learner
  #' @param X covariates matrix (without the treatment indicator!)
  #'
  #' @return ATE

  # Load data as a data frame
  #X <- as.data.frame.matrix(cbind(integer(length = nrow(X)), X))
  #colnames(X) <- model$colnames

  # Obtain treated prediction
  X[ ,"t"] <- 1
  m_1 <- predict(model$model, X)

  # Obtain control prediction
  X[ ,"t"] <- 0
  m_0 <- predict(model$model, X)

  # Obtain an ATE estimate
  ate <- mean(m_1 - m_0)

  return(ate)
}

##########################
### Run the Experiment ###
##########################

run_experiment(OLD_TRAIN, OLD_TEST, N_OF_ITERATIONS, ENV_NAME)

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

setwd("/home/adam/adam/data/causal_inference/results/pf_ratio_2h_8h_manual_outcome/")

#path_predictions <- sprintf("results_BART_%s.csv", outcome) not yet implemented
#save_predictions(results, path_predictions)

path_results <- "results_BART_pf_ratio_2h_8h_manual_outcome.csv"
save_results(results, path_results)

path_summary <- "summary_BART_pf_ratio_2h_8h_manual_outcome.csv"
save_summary(results, path_summary)

# training with obesity
# should add training data results
# should incorporate loading of two separate files training and test .csv
# I am not dropping any as.data.frame.matrix(cbind(rep(True, nrow(X), X)))features