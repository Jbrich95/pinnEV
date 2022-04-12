#  Extreme Value modelling using Partially-Interpretable Neural Networks
Methodology for fitting marginal extreme value (and associated) models using partially-interpretable neural networks. Networks are trained using the [R interface to Keras](https://cloud.r-project.org/web/packages/keras/index.html) with custom loss functions taken to be penalised versions of the negative log-likelihood for associated models. For full details on the partially-interpertable deep-learning framework for extreme value modelling, see  Richards, J., Huser, R. <i>High-dimensional extreme quantile regression using partially-interpretable neural networks: With application to U.S. wildfires</i> (2022).

## Implemented models
* Generalised Pareto distribution (GPD) - see Coles, S.G. (2001) [doi:10.1007/978-1-4471-3675-0](https://doi.org/10.1007%2F978-1-4471-3675-0)
* Blended Generalised Extreme Value (bGEV) distribution - see Castro-Camilo, D., et al. (2021) [10.48550/ARXIV.2106.13110](https://doi.org/10.48550/arXiv.2106.13110)
* bGEV point process - see Richards, J. and Huser, R. (2022)
* Non-parametric quantile estimation - see Koenker, R. (2005) [doi:10.1257/jep.15.4.143](https://doi.org/10.1257/jep.15.4.143)
* Bernoulli/logistic
## Installation

```r
library(devtools)
install_github("https://github.com/Jbrich95/EVpinn")

```

