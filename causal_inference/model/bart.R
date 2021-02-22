# Title     : bart
# Objective : run bart on proning data
# Created by: adam
# Created on: 1/13/21
options(java.parameters = "-Xmx5g")

library("bartCause")
library("bartMachine")
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

s_learner_predict_att <- function(model, data) {
  data <- data[data$treated == "TRUE",]
  data$treated <- TRUE
  m_1 <- predict(model, data)
  data$treated <- FALSE
  m_0 <- predict(model, data)
  cate <- mean(m_1 - m_0)
  ci_low <- cate - 1.96 * sqrt(var(m_1)/length(m_1) + var(m_0)/length(m_0))
  ci_high <- cate + 1.96 * sqrt(var(m_1)/length(m_1) + var(m_0)/length(m_0))
  return(list(cate, ci_low, ci_high))
}

set.seed(12345)
set_bart_machine_num_cores(4)

PATH = "/home/adam/adam/data/19012021/"
setwd(PATH)

outcome = "pf_ratio_12h_24h_manual_outcome"
df <- load_data('data_guerin_rct.csv', outcome)
colnames(df)

data <- train_test_split(df, outcome, train_size = 0.8)

X_train_treated <- data[[1]]
X_train_control <- data[[2]]
X_test_treated <- data[[3]]
X_test_control <- data[[4]]
y_train_treated <- data[[5]]
y_train_control <- data[[6]]
y_test_treated <- data[[7]]
y_test_control <- data[[8]]


X_train_treated$treated <- TRUE
X_train_control$treated <- FALSE
X_test_treated$treated <- TRUE
X_test_control$treated <- FALSE

X_train <- rbind(X_train_treated, X_train_control)
X_test <- rbind(X_test_treated, X_test_control)
y_train <- c(y_train_treated, y_train_control)
y_test <- c(y_test_treated, y_test_control)

### Fit an S-learner

bart_machine_treated_cv = bartMachineCV(X_train,
                                        y_train,
                                        use_missing_data = TRUE,
                                        num_tree_cvs = 200,
                                        k_cvs = c(2, 3, 5),
                                        mem_cache_for_speed=FALSE,
                                        k_folds = 5)

bart_machine_treated_cv
print(bart_machine_treated_cv)

bart_machine_treated_cv$cv_stats[1,]

bart_machine <- bartMachine(X_train,
                            y_train,
                            use_missing_data = TRUE,
                            use_missing_data_dummies_as_covars = FALSE,
                            replace_missing_data_with_x_j_bar = TRUE,
                            num_trees = 200,
                            mem_cache_for_speed=FALSE,
                            alpha = 0.95,
                            beta = 2,
                            k = 2,
                            q = 0.9,
                            nu = 3,
                            num_burn_in = 1000,
                            num_iterations_after_burn_in = 2500)

bart_machine

oos_perf = bart_predict_for_test_data(bart_machine,
                                      X_test,
                                      y_test)

print(oos_perf$rmse)
print(1 - ((oos_perf$L2) / sum((y_test - mean(y_test))^2)))

plot_y_vs_yhat(bart_machine,
               prediction_intervals = TRUE,
               Xtest=X_test,
               ytest=y_test)


rmse_by_num_trees(bart_machine_treated, tree_list=c(seq(100, 500, by=100)), num_replicates=5)

# test whether variables influenced the model; p<.05 is significant

cov_importance_test(bart_machine, covariates = "treated",
num_permutation_samples = 25, plot = TRUE)

estimates <- s_learner_predict_ate(bart_machine, X_test)
ate <- estimates[[1]]
ci_low <- estimates[[2]]
ci_high <- estimates[[3]]
print(ate)
print(ci_low)
print(ci_high)

estimates_att <- s_learner_predict_att(bart_machine, X_test_treated)

att <- estimates[[1]]
att_ci_low <- estimates[[2]]
att_ci_high <- estimates[[3]]
print(att)
print(att_ci_low)
print(att_ci_high)

X_test_1 <- X_test
X_test_1$treated <- TRUE

X_test_0 <- X_test
X_test_1$treated <- FALSE

posterior_treated = bart_machine_get_posterior(bart_machine, X_test_1)
posterior_control = bart_machine_get_posterior(bart_machine, X_test_0)

mean(posterior_treated$y_hat - posterior_control$y_hat)

# 3. cv to choose hyperparameters for treated model

bart_machine_treated <- bartMachine(X_train_treated,
                                    y_train_treated,
                                    use_missing_data = TRUE,
                                    use_missing_data_dummies_as_covars = FALSE,
                                     replace_missing_data_with_x_j_bar = TRUE,
                                    num_trees = 200,
                                    mem_cache_for_speed=FALSE,
                                    alpha = 0.75,
                                    beta = 2, #0.95 10
                                    k = 2,
                                    q = 0.75,
                                    nu = 10,
                                    num_burn_in = 1000,
                                    num_iterations_after_burn_in = 2500)
bart_machine_treated

oos_perf = bart_predict_for_test_data(bart_machine_treated,
                                      X_test_treated,
                                      y_test_treated)

print(oos_perf$rmse)

rmse_by_num_trees(bart_machine_treated, tree_list=c(seq(100, 500, by=100)), num_replicates=5)


bart_machine_treated_cv = bartMachineCV(X_train_treated,
                                        y_train_treated,
                                        use_missing_data = TRUE,
                                        num_tree_cvs = 200,
                                        k_cvs = c(2, 3, 5),
                                        mem_cache_for_speed=FALSE,
                                        k_folds = 10)
bart_machine_treated_cv
print(bart_machine_treated_cv)

bart_machine_treated_cv$cv_stats[1,]

# Serialize?
# Select number of trees
rmse_by_num_trees(bart_machine_treated, num_replicates = 20)

# 4. Diagnostics for the treated model

#Just as when interpreting the results from a linear model, non-normality implies we should
#be circumspect concerning bartMachine output that relies on this distributional assumption
#such as the credible and prediction intervals of Section 4.4.

check_bart_error_assumptions(bart_machine)
plot_convergence_diagnostics(bart_machine)
# explanation:
# https://robjhyndman.com/hyndsight/intervals/
plot_y_vs_yhat(bart_machine_treated,
               credible_intervals = TRUE,
               Xtest=X_test_treated,
               ytest=y_test_treated)
plot_y_vs_yhat(bart_machine_treated,
               prediction_intervals = TRUE,
               Xtest=X_test_treated,
               ytest=y_test_treated) # I think we should look at this as we predict

# 5. Check variable importance

# which variables are split on

investigate_var_importance(bart_machine, num_replicates_for_avg = 10)

# test whether variables influenced the model; p<.05 is significant

cov_importance_test(bart_machine, covariates = "treated",
num_permutation_samples = 10, plot = TRUE)


# Perform variable selection

var_sel_cv = var_selection_by_permute_cv(bart_machine_treated, k_folds = 3)

var_sel_cv$best_method

var_sel_cv$important_vars_cv

# Additional important

var_selection_by_permute(bart_machine,
num_reps_for_avg = 5, num_permute_samples = 25,
num_trees_for_permute = 5, alpha = 0.05,
plot = TRUE, num_var_plot = Inf, bottom_margin = 5)

# interaction

interaction_investigator(bart_machine)

# 6. Do the same for treated

# 7. Error on test set

# 8. Calculate CATE


bart_machine_control_cv = bartMachineCV(X_control_train,
                                        y_control_train,
                                        use_missing_data = TRUE,
                                        num_tree_cvs = c(50, 100, 150),
                                        k_cvs = c(2, 3, 5, 7),
                                        mem_cache_for_speed=FALSE,
                                        k_folds = 10)




bart_machine_control <- bartMachine(X_train_control,
                            y_train_control,
                            use_missing_data = TRUE,
                            use_missing_data_dummies_as_covars = FALSE,
                            num_trees = 50,
                            mem_cache_for_speed=FALSE)
bart_machine_control

plot_y_vs_yhat(bart_machine_control, prediction_intervals = TRUE, Xtest=X_test, ytest=y_test)


posterior_treated = bart_machine_get_posterior(bart_machine_treated, X_test)
posterior_control = bart_machine_get_posterior(bart_machine_control, X_test)


sqrt(mean((posterior_treated$y_hat - y_treated_test)^2))
sqrt(mean((posterior_control$y_hat - y_control_test)^2))

#calculate adjusted R^2

print(mean(posterior_treated$y_hat))
print(hist(posterior_treated$y_hat))
print(mean(posterior_control$y_hat))
print(hist(posterior_control$y_hat))

# bart: E(Y | X = x, T=t) = f(x, t) + \epsilon , where \epsilon ~ N(0, \sigma^2)
# cate  = 1/n \sum_{i=1}^n f(x_i, 1) - f(x_i, 1)
cate = posterior_treated$y_hat - posterior_control$y_hat
print(mean(cate))
hist(cate, breaks = 50)


pd_plot(bart_machine_treated, "pf_ratio")
pd_plot(bart_machine_control, "pf_ratio")

pd_plot(bart_machine_treated, "po2")
pd_plot(bart_machine_control, "po2")

pd_plot(bart_machine_treated, "sofa_score")
pd_plot(bart_machine_control, "sofa_score")

pd_plot(bart_machine_treated, "tidal_volume")
pd_plot(bart_machine_control, "tidal_volume")

cred_int = calc_credible_intervals(bart_machine_treated, X_test)
print(head(cred_int))

pred_int = calc_prediction_intervals(bart_machine_treated, X_test)
print(mean(pred_int[1]))


plot_convergence_diagnostics(bart_machine_treated)
check_bart_error_assumptions(bart_machine_treated)
interaction_investigator(bart_machine_treated)
var_selection_by_permute(bart_machine, num_reps_for_avg=20)

vs <- var_selection_by_permute(bart_machine_treated,
                               bottom_margin = 10,
                               num_permute_samples = 10)
vs$important_vars_local_names

vs$important_vars_global_max_names

vs$important_vars_global_se_names


cov_importance_test(bart_machine_treated_cv, covariates = NULL,
num_permutation_samples = 100, plot = TRUE)



