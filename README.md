#  Extreme Value modelling using Partially-Interpretable Neural Networks
Methodology for fitting marginal extreme value (and associated) models using partially-interpretable neural networks. Networks are trained using the [R interface to Keras](https://cloud.r-project.org/web/packages/keras/index.html) with custom loss functions taken to be penalised versions of the negative log-likelihood for associated models. For full details on the partially-interpertable deep-learning framework for extreme value modelling, see  Richards, J., Huser, R. <i>High-dimensional extreme quantile regression using partially-interpretable neural networks: With application to U.S. wildfires</i> (2022).

## Implemented models
*Generalised Pareto distribution - see Coles, S.G. (2001) [<doi:10.1007/978-1-4471-3675-0>](doi:10.1007/978-1-4471-3675-0)
## Installation

```r
library(devtools)
install_github("https://github.com/Jbrich95/EVpinn")

```

