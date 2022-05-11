#' US Wildfires data
#'
#'Data used by Richards and Huser (2022) for modelling extreme wildfires in the contiguous U.S.
#'with partially-interpretable neural networks
#'
#'The response data \code{BA} are observations of monthly aggregated burnt area (acres) of 3503 spatial grid-cells 
#'located across the contiguous United States, with the states of Alaska and Hawaii excluded.
#'The observation period covers 1993 to 2015, using only months between March and September, inclusive, 
#'leaving 161 observed spatial fields. Grid-cells are arranged on a regular latitude/longitude grid with spatial resolution
#'0.5deg by 0.5deg. Observations are provided by the Fire Program Analysis fire-occurrence database (Short, 2017)
#'which collates U.S. wildfire records from the reporting systems of federal, state and local organisations.
#'
#'Both the response data and the subsequently described predictors have been re-gridded to a regular spatio-temporal grid with missing values set to \code{-1e5}.
#'For \code{BA} and entries to \code{X}, the first three dimensions correspond to time \eqn{\times} latitude \eqn{\times} longitude with 
#'their respective ordinate values given in \code{times}, \code{lat} and \code{lon}.
#'
#'We consider three types of predictor variables, given in \code{X}: these are meteorological (\code{X.met}), land cover proportions (\code{X.lc}) and 
#'orograhical (\code{X.oro}).
#'
#'Ten meteorological variables are considered and given as monthly means in \code{X.met}. 
#'These variables are provided by the ERA5-reanalysis on land surface, available through the COPERNICUS Climate Data Service, 
#'which is given on a 0.1deg \eqn{\times} 0.1deg grid; the values are then aggregated to a 0.5deg \eqn{\times} 0.5deg resolution.
#'The variables are ordered as followed: both eastern and northern components of wind velocity at a 10m altitude (m/s), 
#'both dew-point temperature and temperature at a 2m altitude (Kelvin), potential evaporation (m), evaporation (m), precipitation (m),
#'surface pressure (Pa) and surface net solar, and thermal, radiation (J/m^2). 
#'This particular ERA5-reanalysis samples over land only, and so the meteorological conditions over the 
#'oceans are not available.
#'
#'The land cover variables that are given in \code{X.lc} describe the proportion of a grid-cell which
#'is covered by one of 18 different types, e.g., urban, grassland, water (see Opitz (2022) for full details).
#'Land cover predictors are derived using a gridded land cover map, of spatial resolution 300m and temporal resolution one year, 
#'produced by COPERNICUS and available through their Climate Data Service. For each 0.5deg \eqn{\times} 0.5deg grid-cell, 
#'the proportion of a cell consisting of a specific land cover type is derived from the high-resolution product.
#'The variables are ordered lc1 - lc18, as described by Opitz (2022).
#'
#'The two orographical predictors given in \code{X.oro} are the mean and standard deviation of the altitude (m) 
#'for each grid-cell; estimates are derived using a densely-sampled gridded 
#'output from the U.S. Geographical Survey Elevation Point Query Service. 
#' @docType data
#'
#' @usage data(USwild)
#'
#' @format A list with 5 elements:
#' \describe{
#' \item{BA}{An array with dimension (276, 119, 51), corresponding to the burnt area response data.}
#' \item{X}{A list with 3 elements:
#' \describe{
#' \item{X.met}{An array with dimension (276, 119, 51, 10) corresponding to the meterological predictors described below.}
#'  \item{X.lc}{An array with dimension (276, 119, 51, 18) corresponding to the land cover predictors described below.}
#' \item{X.oro}{An array with dimension (276, 119, 51, 2) corresponding to the oropgraphical predictors described below.}
#'
#' }
#' }
#' \item{times}{A vector of length 276 giving the observation indices. Format is "year-month". Corresponds to first dimension of \code{BA}.}
#'  \item{lon}{A vector of length 119 giving the longitude ordinate for the second dimension of \code{BA}.}
#'  \item{lat}{A vector of length 51 giving the latitude ordinate for the third dimension of \code{BA}.}

#' }
#'
#' @keywords datasets
#'
#'@references Richards, J. and Huser, R. (2022), \emph{High-dimensional extreme quantile regression using 
#'partially-interpretable neural networks: With application to U.S. wildfires.}
#'
#'Short, K. C. (2017), \emph{Spatial wildfire occurrence data for the United States, 1992-2015 [FPA\_FOD\_20170508].  
#'4th Ed. Fort Collins, CO: Forest Service Research Data Archive.}
#'
#'Opitz, T.. (2022), \emph{Editorial: EVA 2021 Data Competition on spatio-temporal prediction of wildfire activity in the United States.} Extremes, to appear.
#'
#' @examples
#' data(USwild)
"USwild"