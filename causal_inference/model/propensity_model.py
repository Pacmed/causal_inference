from causalinference import CausalModel
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score

SEED = 1234

class PropensityModel(object):

    def __init__(self,
                 outcome,
                 treatment,
                 covariates,
                 outcome_name=None,
                 treatment_name=None,
                 covariates_name=None):

        self.causal_model = CausalModel(outcome, treatment, covariates)
        self.covariates = covariates
        self.treatment = treatment
        self.outcome = outcome
        self.outcome_name = outcome_name
        self.treatment_name = treatment_name
        self.covariates_name = covariates_name

    def summary_stats(self):
        print(self.causal_model.summary_stats)

    def get_imbalanced_covariates(self, thresh = 0.1):

        dict = {k: round(v, 2) for k, v in zip(self.covariates_name, self.causal_model.summary_stats['ndiff'])}
        cols = [k for (k, v) in dict.items() if abs(v) > thresh]

        return cols

    def est_propensity(self, X, t, method = 'default', pscore_flag = False, pscore_model = None):

        if method == 'default':
            self.causal_model.est_propensity()

        if method == 'data-driven':
            self.causal_model.est_propensity_s()

        if method == 'balanced':
            if not 'pscore' in self.causal_model.raw_data._dict.keys():
                self.causal_model.est_propensity()

            clf = LogisticRegression(random_state=SEED,
                                     class_weight='balanced',
                                     penalty='none',
                                     max_iter=10000).fit(X, t)

            self.causal_model.raw_data._dict['pscore'] = clf.predict_proba(X)[:, 1]

            if pscore_flag:
                self.causal_model.raw_data._dict['pscore'] = pscore_model.predict_proba(X)[:, 1]
                print(self.causal_model.raw_data._dict['pscore'])
            #print("roc_auc:",roc_auc_score(t, clf.predict_proba(X)[:, 1]))

            #coef = clf.coef_.round(2).T.tolist()
            #print(coef)

        if method == 'polynomial':
            # Select variables with highest ndiff that have big covariates in pscore model
            return

        print(self.causal_model.propensity)

    def show_propensity(self):

        sns.distplot(self.causal_model.raw_data['pscore'][self.treatment],
                     hist=True,
                     #bins=10,
                     kde=True,
                     label='Prone',
                     norm_hist=True)

        sns.distplot(self.causal_model.raw_data['pscore'][~self.treatment],
                     hist=True,
                     #bins=10,
                     kde=True,
                     label='Supine',
                     norm_hist=True)

        # Plot formatting
        plt.legend(prop={'size': 12})
        plt.title('Estimated propensity score of being turned to prone position.')
        plt.xlabel('Propensity score')
        plt.ylabel('Density')
        plt.show()

    def plot_propensity(self):
        treated_pscore = self.causal_model.raw_data['pscore'][self.treatment]
        treated = {'Propensity_score': treated_pscore, 'Group': np.ones(treated_pscore.shape)}
        df_trated = pd.DataFrame(treated)

        control_pscore = self.causal_model.raw_data['pscore'][~self.treatment]
        control = {'Propensity_score': control_pscore, 'Group': np.zeros(control_pscore.shape)}
        df_control = pd.DataFrame(control)

        df_plot = pd.concat([df_trated, df_control])
        df_plot.loc[df_plot.Group == 1, 'Group'] = 'Treated'
        df_plot.loc[df_plot.Group == 0, 'Group'] = 'Control'

        sns.displot(df_plot, x="Propensity_score", hue="Group", stat="probability")

    def trim(self):

        self.causal_model.trim_s()

        cutoff = self.causal_model.cutoff
        print('Observations with propensity score lower than {} were dropped.'.format(cutoff))


    def access_balance(self, method='default'):
        if method == 'default':
            self.causal_model.stratify()
            print(self.causal_model.strata)

        if method == 'data-driven':
            self.causal_model.stratify_s()
            print(self.causal_model.strata)

        #propensity_model.causal_model.summary_stats['N']

        for stratum in self.causal_model.strata:
            print("The within strata balances are given by:", np.mean(abs(stratum.summary_stats['ndiff'])))

        balance = [np.mean(abs(stratum.summary_stats['ndiff'])) for stratum in self.causal_model.strata]
        weights = [stratum.summary_stats['N'] for stratum in self.causal_model.strata]
        print("The overall balance is:", np.average(balance, weights=weights))

    def access_stability_via_matching(self):
        for stratum in self.causal_model.strata:
            stratum.est_via_matching(bias_adj=True)

        ate_per_stratum = [stratum.estimates['matching']['ate'] for stratum in self.causal_model.strata]

        return ate_per_stratum

    def access_stability_via_ols(self):
        for stratum in self.causal_model.strata:
            stratum.est_via_ols()

        ate_per_stratum = [stratum.estimates['ols']['ate'] for stratum in self.causal_model.strata]

        return ate_per_stratum

    def est_treatment_effect(self, matching_only=False):

        self.causal_model.est_via_matching(bias_adj=True, weights='maha')

        if not matching_only:
            self.causal_model.est_via_ols(adj=1)
            self.causal_model.est_via_weighting()
            self.causal_model.est_via_blocking()
            print(self.causal_model.estimates)


    def print_models(self, raw_effect=None, true_effect=None):

        y = [22.19, 22.24, 18.97]
        yerr = [0.23, 0.21, 1.3]

        #y = [8.27, 16.5, 13.77]
        #yerr = [0.20, 0.20, 1.2]

        #y = [13.77]
        #yerr = [1.2]

        x_label = ["CFR Wass", "TARNet", "BART"]
        #x_label = ["BART"]

        for method, result in dict(self.causal_model.estimates).items():
            y.append(result["ate"])
            yerr.append(1.96*result["ate_se"])
            x_label.append(method)

        if raw_effect:
            y.append(raw_effect)
            yerr.append(0)
            x_label.append("raw_effect")

        x = np.arange(len(y))
        plt.figure(figsize=(8, 4))
        plt.errorbar(x=x, y=y, yerr=yerr, linestyle="none", capsize=5, marker="o")
        plt.xticks(x, x_label)
        #plt.title("Estimated ATE of proning", fontsize=18)
        plt.ylabel('Estimated ATE with 95%CI')
        if true_effect:
            plt.hlines(y=true_effect, xmin=-0.5, xmax=7.5, linestyles="dashed")
        # plt.xlim(-0.5,3.5);

def run_matching_experiment(y, t, x, n_of_experiments):

    ate = []
    rmse = []
    r2 = []

    for i in range(n_of_experiments):
        #y, t, x = y[:, i].reshape(len(y[:, i], )), t[:, i].reshape(len(t[:, i], )), x[:, :, i].reshape(x[:, :, i].shape[0], x[:, :, i].shape[1]))
        print(i)
        y_i, t_i, x_i = y[:, i], t[:, i], x[:, :, i]
        propensity_model = PropensityModel(outcome=y_i, treatment=t_i, covariates=x_i)
        propensity_model.est_propensity(X=x_i, t=t_i, method='balanced')
        propensity_model.est_treatment_effect(matching_only=True)
        ate.append(propensity_model.causal_model.estimates['matching']['ate'].round(2))
        rmse.append(0)
        r2.append(0)
        propensity_model.propensity_model.reset()
        propensity_model.causal_model.reset()

    results = {'ate': ate, 'rmse': rmse, 'r2': r2}
    df_results = pd.DataFrame(data=results)

    return df_results

