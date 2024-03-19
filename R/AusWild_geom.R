#'Geometry file for Australian Wildfire data
#'
#'Data used by Cisneros et al. (2024) for modelling extreme wildfires in Australia
#'with eGPD graph convolutional neural networks
#'
#'See \code{help("AusWild")} for relevant response and covariates.
#'
#' @docType data
#'
#' @usage data(AusWild_geom)
#'
#' @format 
#' \describe{
#'  \item{AusWild_geom}{A sf data frame containing the geometry of the (7901) polygons that make up the spatial domain.}
#' }
#' @keywords datasets
#'
#' @references 
#'
#' Cisneros, D., Richards, J., Dahal, A., Lombardo, L., and Huser, R. (2024), \emph{Deep learning-based graphical regression for jointly moderate and extreme Australian wildfires.} Spatial Statistics, 53:100811. (\href{https://doi.org/10.1016/j.spasta.2024.100811}{doi}).
#'
#' @examples
#' 
#' data("AusWild_geom")
#' 
"AusWild_geom"
