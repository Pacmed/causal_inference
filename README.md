## Estimating Treatment Effects from EHR Data
The repository contains code used by [Pacmed Labs](https://pacmed.ai/nl/labs) to estimate treatment effects from electronic health record data. It allows for easy replication of the experiment reported in:
* "A pragmatic approach to estimating average treatment
effects from EHR data: the effect of prone positioning on
mechanically ventilated COVID-19 patients" (Izdebski et al., 2021)

## Getting Started 
To install requirements:

```setup
conda env create -f environment.yml
```

>📋  To set up the environment create a conda environment using the `.yml` file.

Use the [BART](https://github.com/Pacmed/causal_inference/blob/master/causal_inference/model/bart.R) model within an R environment:

```setup
conda env create -f environment_R.yml
```

>📋  To set up the environment for BART model create a separate conda environment.

For using the CfR and TARNET models, use the [official implementation](https://github.com/clinicalml/cfrnet) with the configuration specified for the [CfR](https://github.com/Pacmed/causal_inference/blob/master/causal_inference/model/guerin_cfr_2_8.txt) and [TARNET](https://github.com/Pacmed/causal_inference/blob/master/causal_inference/model/guerin_tarnet_2_8.txt) models.
>📋  To set up the environment for CfR and TARNET models create a separate conda environment using `environment_cfrnet.yml`.

## Experiments 
For replicating the experiments for Outcome Regression, IPW and Blocking models run:

```experiment
from causal_inference.model.ols import OLS
from causal_inference.model.weighting import IPW
from causal_inference.model.blocking import Blocking
from causal_inference.experiments.run import Experiment

N_OF_ITERATIONS = 100

batch_of_models = [OLS(), IPW(), Blocking()]

for model in batch_of_models:

    experiment = Experiment(causal_model=model,
                            n_of_iterations=N_OF_ITERATIONS)
    experiment.run(y_train=y_train, t_train=t_train,
                   X_train=X_train, y_test=y_test,
                   t_test=t_test, X_test=X_test)

```

 >📋  To execute all the steps used in the experiments execute the code available [notebooks](https://github.com/Pacmed/causal_inference/tree/master/notebooks).

For replicating the BART experiment run:
```BART
Rscript BART.R
```

For replicating the CfR and TARNET experiments install the [official implementation](https://github.com/clinicalml/cfrnet) and run:
```CFRNET
mkdir results
mkdir results/<config_file_name>

python cfr_param_search.py ../causal_inference/model/<config_file> 20

python evaluate.py ../causal_inference/model/<config_file> 1
```

Note that:
>❗ To run the experiments you need to request access to [data](https://icudata.nl/index-en.html). For a detailed description see below. 

> ✨ To compare any scikit-learn models, simply add them to `batch_of_models':
```SCIKIT
batch_of_models = [OLS(), IPW(), Blocking(), RandomForestRegressor()]
```

  ## Meta
The project is part of the [Pacmed Labs](https://pacmed.ai/nl/labs) research [agenda](https://pacmed.ai/nl/media/press/pacmed-krijg-sidn-subsidie-onderzoek-causaliteit) and was funded by [SIDNFonds](https://www.sidnfonds.nl/projecten/using-machine-learning-on-observational-data-to-support-treatment-decisions).

## Contributing

1. Fork it (https://github.com/Pacmed/causal_inference)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

📇 Further Details
========

## Motivation
Estimating treatment effects from observational data remains a central problem in the field of causal inference. In recent years, significant advances were made using modern machine learning and deep learning approaches.

However, it is yet to be established how to utilize those advances
on real-world medical data in order to provide relevant clinical insights. In particular it is not clear, to what extend new models will outperform traditional methods based on propensity score models and allow for individual treatment effect estimation. 

The methods in this repository were employed to estimate the average treatment of prone positioning on mechanically ventilated COVID-19 patients. Prone positioning is a commonly used technique for the treatment of severely hypoxemic mechanically ventilated patients with acute respiratory distress syndrome and it’s effectiveness was not yet clinically confirmed, when performed on COVID-19 patients. This provides a direct clinical use case on which various models can be compared. 


## Data 
For the purpose of the observational study we used data collected in the Dutch Data Warehouse (DDW). The DDW is the result of a intensive care unit data sharing collaboration in the Nether-
lands that was initiated during the COVID-19 pandemic. The DDW includes data on demographics, comorbidities, monitoring and life support devices, laboratory results, clinical observations, medi-
cations, fluid balance, and outcomes. Request to data can be requested through https://icudata.nl/index-en.html. 

## Used Models
* Outcome Regressions
* IPW
* Blocking
* BART
* TARNET
* CFR
