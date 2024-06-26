% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/USWild.R
\docType{data}
\name{USWild}
\alias{USWild}
\title{US Wildfires data}
\format{
A list with 7 elements:
\describe{
\item{BA}{An array with dimension (161, 129, 61), corresponding to the burnt area response data.}
\item{X}{A list with 4 elements:
\describe{
\item{X.t2m}{An array with dimension (161, 129, 61, 1) of 2m air temperature values, as described below.}
\item{X.SPI}{An array with dimension (161, 129, 61, 1) of standardised precipitation index (SPI) values, as described below.}
\item{X.N}{An array with dimension (161, 129, 61, 40) of the "non-interpretable" neural network predictors, as described below.}
\item{cov.names}{A vector of length 40 giving shorthand names for the covariates that correspond to the last dimension of \code{X.N}. See below.}

}
}
\item{times}{A vector of length 161 giving the monthly observation indices. Format is "year-month". Corresponds to first dimension of \code{BA}.}
\item{lon}{A vector of length 129 giving the longitude ordinate for the second dimension of \code{BA}.}
\item{lat}{A vector of length 61 giving the latitude ordinate for the third dimension of \code{BA}.}
\item{state.ids}{A (129,61) matrix providing the state ID number for each spatial grid-cell. A value of 0 corresponds to grid-cells outside of the CONUS, e.g., over sea.}
\item{state.flags}{A vector of length 45 providing the name associated with each state ID.}
}
}
\usage{
data(USWild)
}
\description{
Data used by Richards and Huser (2024+) for modelling extreme wildfires in the contiguous U.S.
with partially-interpretable neural networks
}
\details{
The response data \code{BA} are observations of monthly aggregated burnt area (acres) of 3503 spatial grid-cells
located across the contiguous United States, with the states of Alaska and Hawaii excluded.
The observation period covers 1993 to 2015, using only months between March and September, inclusive,
leaving 161 observed spatial fields. Grid-cells are arranged on a regular 129 by 61 latitude/longitude grid with spatial resolution
0.5deg by 0.5deg. Observations are provided by the Fire Program Analysis fire-occurrence database (Short, 2017)
which collates U.S. wildfire records from the reporting systems of federal, state and local organisations.

Both the response data and the subsequently described predictors have been re-gridded to a regular spatio-temporal grid with missing values set to \code{-1e10}.
For \code{BA} and entries to \code{X}, the first three dimensions correspond to time \eqn{\times} latitude \eqn{\times} longitude with
their respective ordinate values given in \code{times}, \code{lat}, and \code{lon}.

We consider three types of predictor variables, given in \code{X}: these are the two "interpretable" predictors, 2m air temperature (K; \code{X.t2m}) and the standardised precipitation index (SPI; unitless; \code{X.SPI}), that Richards and Huser (2024) model using linear and additive functions, and the 40
non-interpretable predictors (\code{X.N}) that feed the neural network component of the models.

For \code{X.N}, the names of each variable are contained with \code{X$cov.names}.
These include 11 meteorological variables, which are given as monthly means.
These variables are provided by the ERA5-reanalysis on single levels, available through the COPERNICUS Climate Data Service  (\href{https://doi.org/10.24381/cds.f17050d7}{doi:10.24381/cds.f17050d7}),
which is given on a 0.1deg \eqn{\times} 0.1deg grid; the values have then been aggregated to a 0.5deg \eqn{\times} 0.5deg resolution.
The variables are: eastern (u10; m/s) and northern (v10; m/s) components of wind velocity at 10m above ground level,
potential evaporation (pev; m), evaporation (e; m of water equivalent), surface pressure (sp; Pa),
surface net solar (ssr; J/m^2) and thermal radiation (str; J/m^2), snowfall (sf; m of water equivalent),
snow-melt (smlt; m of water equivalent), snow evaporation (es; m of water equivalent), and
surface run-off (sro; m). The reanalysis also provides three variables relating to sub-gridscale orography: angle (anor; radians),
anisotropy (isor; unitless), slope (slor; unitless), as well as the standard deviation of orography (sdor; unitless).

We also provide land cover variables in \code{X.N} that describe the proportion of a grid-cell which
is covered by one of 25 different types, e.g., urban, grassland, water (see Opitz (2022) for details on a subset of these data).
Land cover predictors are derived using a gridded land cover map, of spatial resolution 300m and temporal resolution one year,
produced by COPERNICUS and available through their Climate Data Service. For each 0.5deg \eqn{\times} 0.5deg grid-cell,
the proportion of a cell consisting of a specific land cover type is derived from the high-resolution product.
The land cover variables names are provided explicitly in \code{cov.names}.

Also provided are \code{state.flags} and \code{state.IDs}, which provide the state in which each of the (129 by 61) spatial grid-cells are located. A value of zero corresponds to a spatial grid-cell located outside of the CONUS. Note that some of the smaller states have been merged.
}
\examples{
data(USWild)
}
\references{
Richards, J. and Huser, R. (2024+), \emph{Regression modelling of spatiotemporal extreme U.S. wildfires via partially-interpretable neural networks}. (\href{https://arxiv.org/abs/2208.07581}{arXiv:2208.07581}).

Short, K. C. (2017), \emph{Spatial wildfire occurrence data for the United States, 1992-2015 \link{FPA_FOD_20170508}.
4th Ed. Fort Collins, CO: Forest Service Research Data Archive.}

Opitz, T.. (2022), \emph{Editorial: EVA 2021 Data Competition on spatio-temporal prediction of wildfire activity in the United States.} Extremes, to appear.
}
\keyword{datasets}
