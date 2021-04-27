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
PATH_TRAIN_DATA <- '/home/adam/adam/data/causal_inference/data/processed/guerin_12_24_train.npz'
PATH_TEST_DATA <- '/home/adam/adam/data/causal_inference/data/processed/guerin_12_24_test.npz'
PATH_SAVE_RESULTS <- '/home/adam/adam/data/causal_inference/results/old/12h_24h/results_old_BART.csv'
PATH_SUMMARY <- ''

OLD_TRAIN <- '/home/adam/adam/cfrnet/data/bfpguerin_12_24.train.npz'
OLD_TEST <- '/home/adam/adam/cfrnet/data/bfpguerin_12_24.test.npz'
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

run_experiment <- function(path_train_data, path_test_data, n_of_iterations, env_name, path_save_results) {
  #' Runs the experiment.
  #'
  #' @param path_train_data A path to the bootsrapped training data.
  #' @param test_data A path to the bootsrapped test data.
  #' @param n_of_iterations The number of bootsrapped samples to be used.
  #' @param env_name The name of the conda enviroment with NumPy library installed.
  #' @param path_save_results A path to save the results of the experiment
  #'
  #' @return Results of the experiment.

  # Initialize the results vectors
  results <- 0

  for (iteration in 1:n_of_iterations) {

    print(iteration)

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

    if ('numeric' %in% class(results)) {results <- result} else {results <- rbind(results, result)}

  }

  # Reset index and write to a file.
  rownames(results) <- NULL
  write.csv(results, path_save_results)

  return(results)
}

###########################
### Auxiliary Functions ###
###########################

load_data <- function (path, env_name) {
  #' Loads bootsrapped data for the purpose of the experiment.
  #'
  #' @param path Path to load the data
  #' @param env_name The name of the conda enviroment with NumPy library installed.
  #'
  #' @return Returns a list list(y, t, X) of training/test data.

  # Load NumPy
  np <- load_enviroment(env_name)
  # Load the data
  data <- np$load(path)

  return(list(y=data$f[["yf"]], t=data$f[["t"]], X=data$f[["x"]]))
}

load_enviroment <- function (env_name) {
  #' Loads the enviroment with NumPy library.
  #'
  #' @param env_name The name of the conda enviroment with NumPy library installed.
  #'
  #' @return Imported NumPy library.

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

  # Obtain treated prediction
  X[ ,"t"] <- 1
  m_1 <- predict(model$model, X)

  # Obtain control prediction
  X[ ,"t"] <- 0
  m_0 <- predict(model$model, X)

  # Obtain an ATE estimate (eqvivalent to mean(m_1 - m_0))
  ate <- mean(m_1 - m_0)

  return(ate)
}

##########################
### Run the Experiment ###
##########################

run_experiment(OLD_TRAIN, OLD_TEST, N_OF_ITERATIONS, ENV_NAME, PATH_SAVE_RESULTS)
