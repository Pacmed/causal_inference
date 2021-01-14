from causalinference import CausalModel
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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

    def est_propensity(self, X, t, method = 'default'):

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
                                     max_iter=5000).fit(X, t)

            self.causal_model.raw_data._dict['pscore'] = clf.predict_proba(X)[:, 1]

            print("roc_auc:",roc_auc_score(t, clf.predict_proba(X)[:, 1]))

            coef = clf.coef_.round(2).T.tolist()
            print(coef)

        if method == 'polynomial':
            # Select variables with highest ndiff that have big covariates in pscore model
            return

        print(self.causal_model.propensity)

    def show_propensity(self):

        sns.distplot(self.causal_model.raw_data['pscore'][self.treatment],
                     hist=True,
                     bins=10,
                     kde=True,
                     label='Prone',
                     norm_hist=True)

        sns.distplot(self.causal_model.raw_data['pscore'][~self.treatment],
                     hist=True,
                     bins=10,
                     kde=True,
                     label='Supine',
                     norm_hist=True)

        # Plot formatting
        plt.legend(prop={'size': 12})
        plt.title('Estimated propensity score of being turned to prone position.')
        plt.xlabel('Propensity score')
        plt.ylabel('Density')
        plt.show()

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

    def est_treatment_effect(self):
        self.causal_model.est_via_ols()
        self.causal_model.est_via_weighting()
        self.causal_model.est_via_blocking()
        self.causal_model.est_via_matching(bias_adj=True, weights='maha')
        print(self.causal_model.estimates)


    def print_models(self, raw_effect=None, true_effect=None):

        y = []
        yerr = []
        x_label = []

        for method, result in dict(self.causal_model.estimates).items():
            y.append(result["ate"])
            yerr.append(result["ate_se"])
            x_label.append(method)

        if raw_effect:
            y.append(raw_effect)
            yerr.append(0)
            x_label.append("raw_effect")

        x = np.arange(len(y))

        plt.errorbar(x=x, y=y, yerr=yerr, linestyle="none", capsize=5, marker="o")
        plt.xticks(x, x_label)
        plt.title("Estimated Average Treatment Effect of proning", fontsize=18)
        plt.ylabel('Estimated ATE with 95%CI')
        if true_effect:
            plt.hlines(y=true_effect, xmin=-0.5, xmax=4.5, linestyles="dashed")
        # plt.xlim(-0.5,3.5);

# To do: add manual pscore estimation and variable selection. Add easy variable name checking