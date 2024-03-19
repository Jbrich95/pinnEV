#  Partially-Interpretable Neural Networks for modelling Extreme Values
Methodology for fitting marginal extreme value (and associated) models using partially-interpretable neural networks (PINNs). Networks are trained using the [R interface to Keras](https://cloud.r-project.org/web/packages/keras/index.html) with custom loss functions taken to be penalised versions of the negative log-likelihood for associated models. For full details on the partially-interpretable deep-learning framework for extreme value analysis, see  Richards, J. and Huser, R., Regression modelling of spatiotemporal extreme U.S. wildfires via partially-interpretable neural networks</i> (2024+).

Models are defined for response $Y$ and covariates $\bf X$, where $Y$ can be an array of 1 to 3 dimensions and $\bf X$ has one extra dimension (corresponding to values of different predictors). All models are of the form $Y | \mathbf{X} \sim F(\boldsymbol{\theta}\{\mathbf{x})\}$, where $F$ is some statistical distribution model and the parameter set $\boldsymbol{\theta}(\mathbf{x})$ is modelled using a partially-interpretable neural network. The parameter set varies in length (denoted by $p$), depending on the choice of $F$, and each component $\theta_i(\mathbf{x}), i=1,\dots,p,$ is modelled using a PINN; this has the form
$$\theta_i(\mathbf{x})$$ 
where $\eta_0^{(i)}$ is a constant real intercept, $h_i$ is some link function, $m^{(i)}_\mathcal{I}$ is a semi-parametric function, and $m^{(i)_\mathcal{N}$ is a neural network. Note that these three functions, and their inputs, differ across $i=1,\dots,p$. We further split up $m_\mathcal{I}$ into a linear function and a thin-plate spline:
$$
m_\mathcal{I}^(i)(\mathbf{x}_\mathcal{I}^{(i)})=m_\mathcal{A}^(i)(\mathbf{x}_\mathcal{A}^{(i)})+m_\mathcal{L}^(i)(\mathbf{x}_\mathcal{L}^{(i)}),
$$
where $\mathbf{x}^{(i)}_\mathcal{I}$ and $\mathbf{x}^{(i)}_\mathcal{I}$ are complementary subsets of $\mathbf{x}^{(i)}_\mathcal{I}$. We model $m_\mathcal{A}$ using thin-plate splines and $m_\mathcal{L}$ as linear.



## Implemented statistical distributions

For the statistical distribution $F$, we have implemented:

* Generalised Pareto distribution (GPD) - see Coles, S.G. (2001) [doi:10.1007/978-1-4471-3675-0](https://doi.org/10.1007/978-1-4471-3675-0);
* Re-parameterised GPD (with offset scale) - see Richards, J., et al., (2023) [doi:10.1175/AIES-D-22-0095.1](https://doi.org/10.1175/AIES-D-22-0095.1);
* Blended Generalised Extreme Value (bGEV) distribution - see Castro-Camilo, D., et al. (2022) [doi:10.1002/env.2742](https://doi.org/10.1002/env.2742);
* bGEV point process (bGEV-PP) - see Richards, J. and Huser, R. (2024+) [arxiv:2208.07581](https://arxiv.org/abs/2208.07581);
* Extended GPD (eGPD; with offset scale) - see Cisneros, D., et al., (2024) [doi:10.1016/j.spasta.2024.100811](https://doi.org/10.1016/j.spasta.2024.100811);
* Bernoulli/logistic;
* Log-normal;
* Non-parametric quantile estimation - see Koenker, R. (2005) [doi:10.1257/jep.15.4.143](https://doi.org/10.1257/jep.15.4.143). Note that in this case $F$ is arbritary, and $\theta(\mathbf{x})$ is taken to be the conditional $\tau$ quantile for $\tau\in(0,1)$.

## Implemented neural networks

For $m_\mathcal{N}$, we have:

* A densely-connected neural network or multi-layered perceptron;
* A convolutional neural network. This requires that $Y$ is observed on a regular spatial grid.
* Graph convolutional neural network. Requires that $Y$ is spatial data and accompanies an adjacency matrix describing the graph structure.

Note that $\mathbf{x}_\mathcal{A}, \mathbf{x}_\mathcal{L},$ and $\mathcal{x}_\mathcal{N}$ can be taken as empty; hence, the partially-interpretable aspect of the PINN does not need to be incorporated into the models. Standard conditional density estimation neural networks can be implemented through function arguments. Missing values in the response variable `Y` are handled by setting said values to `-1e10`. For data where `-1e10` is within the range of reasonable values of `Y`, the models cannot be readily-applied; in these cases, the data must be scaled or translated.

## Installation 

We install CPU tensorflow and Keras in a virtual environment. See [this installation guide](https://tensorflow.rstudio.com/install/) for further details on installation of tensorflow in R. Currently the package has been developed for models to train on the CPU only. Relatively shallow neural network models (< 5 layers and 10,000s of parameters) are trainable on a standard laptop.

```r
py_version <- "3.9.18"
path_to_python <- reticulate::install_python(version=py_version)

#Create a virtual envionment 'pinnEV_env' with Python 3.9.18. Install tensorflow  within this environment.
reticulate::virtualenv_create(envname = 'pinnEV_env',
                              python=path_to_python,
                              version=py_version)

path<- paste0(reticulate::virtualenv_root(),"/pinnEV_env/bin/python")
Sys.setenv(RETICULATE_PYTHON = path) #Set Python interpreter to that installed in pinnEV_env

tf_version="2.13.1" 
reticulate::use_virtualenv("pinnEV_env", required = T)
tensorflow::install_tensorflow(method="virtualenv", envname="pinnEV_env",
                               version=tf_version) #Install version of tensorflow in virtual environment
keras::install_keras(method = c("virtualenv"), envname = "pinnEV_env",version=tf_version) #Install keras

keras::is_keras_available() #Check if keras is available


#Install spektral 1.3.0 - this is for the graph convolutional neural networks
reticulate::virtualenv_install("pinnEV_env",
                               packages = "spektral", version="1.3.0")


```

## Coming in future updates 
* Weight regularisation and dropout
* New statistical models - Gamma, mean/median regression
* Different architecture per parameter
* Non-stationary xi in GPD and bGEV models
* GPU installation

