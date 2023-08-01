#  Partially-Interpretable Neural Networks for modelling Extreme Values
Methodology for fitting marginal extreme value (and associated) models using partially-interpretable neural networks. Networks are trained using the [R interface to Keras](https://cloud.r-project.org/web/packages/keras/index.html) with custom loss functions taken to be penalised versions of the negative log-likelihood for associated models. For full details on the partially-interpretable deep-learning framework for extreme value analysis, see  Richards, J. and Huser, R., Regression modelling of spatiotemporal extreme U.S. wildfires via partially-interpretable neural networks</i> (2022).

Note that the partially-interpretable aspect does not need to be incorporated into the models; standard conditional density estimation neural networks can be implemented through function arguments. Missing values in the response variable `Y.train` are handled by setting said values to `-1e10`. For data where `-1e10` is within the range of reasonable values of `Y.train`, the models cannot be readily-applied; in these cases, the data must be scaled  or translated.

## Implemented models
* Generalised Pareto distribution (GPD) - see Coles, S.G. (2001) [doi:10.1007/978-1-4471-3675-0](https://doi.org/10.1007/978-1-4471-3675-0)
* Re-parameterised GPD (with offset scale) - see Richards, J., et al., (2022) [arXiv:2212.01796](https://arxiv.org/abs/2212.01796)
* Blended Generalised Extreme Value (bGEV) distribution - see Castro-Camilo, D., et al. (2022) [doi:10.1002/env.2742](https://doi.org/10.1002/env.2742)
* bGEV point process (bGEV-PP) - see Richards, J. and Huser, R. (2022) [arXiv:2208.07581](https://arxiv.org/abs/2208.07581)
* extended GPD (eGPD; with offset scale) - see Cisneros, D., et al., (2023) (In draft).
* Non-parametric quantile estimation - see Koenker, R. (2005) [doi:10.1257/jep.15.4.143](https://doi.org/10.1257/jep.15.4.143)
* Bernoulli/logistic
* Log-normal

Note that the bGEV and bGEV-PP models are less computationally efficient than the other models due to the complexity of the likelihood; bear that in mind when running the example code!

## Installation 

```r
py_version <- "3.8.10"
path_to_python <- reticulate::install_python(version=py_version)

#Create a virtual envionment 'myenv' with Python 3.8.10. Install tensorflow  within this environment.
reticulate::virtualenv_create(envname = 'myenv',
                              python=path_to_python,
                              version=py_version)

path<- paste0(reticulate::virtualenv_root(),"/myenv/bin/python")
Sys.setenv(RETICULATE_PYTHON = path) #Set Python interpreter to that installed in myenv

tf_version="2.11.0" 
reticulate::use_virtualenv("myenv", required = T)
tensorflow::install_tensorflow(method="virtualenv", envname="myenv",
                               version=tf_version) #Install version of tensorflow in virtual environment
keras::install_keras(method = c("virtualenv"), envname = "myenv",version=tf_version) #Install keras

keras::is_keras_available() #Check if keras is available


#Install spektral - this is for the graph NNs
reticulate::virtualenv_install("myenv",
                               packages = "spektral", version="1.3.0")


```

## Coming in future updates 
* Weight regularisation and dropout
* New statistical models - Gamma, mean/median regression
* Different architecture per parameter
* Non-stationary xi in GPD and bGEV models
* GPU installation

