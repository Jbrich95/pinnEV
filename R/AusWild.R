#'Australian Wildfire data
#'
#'Data used by Cisneros et al. (2023+) for modelling extreme wildfires in Australia
#'with eGPD graph convolutional neural networks
#'
#'The response data \code{BA} are observations of monthly aggregated burnt area (km^2) over 7901 artificially constructed spatial polygons 
#'that discretise Australia and Tasmania. These polygons were constructed using Statistical Area level-1 and level-2 (SA1/SA2) census regions (ABS, 2011) to ensure that the polygons have comparable population density.
#'The observation period covers June 1999 to December 2018, inclusive, leaving 235 observed spatial fields. Observations are derived from historical reported bushfire boundaries (Lizundia-Loiola et al., 2020). 
#'Alongside monthly values of BA, we provide the area of each polygon; this is given in \code{a.s}, which is equivalent to \eqn{a(s)} in Cisneros et al. (2023), 
#'and can be specified as an offset term in [pinnEV::eGPD.NN]. The boundaries of the polygons are provided separately, see \code{help("AusWild_geom")}.
#'
#'Values of \code{BA} are missing in the Northern territories and have been set to \code{-1e10}; this leaves 7590 polygons with observed \code{BA >= 0}. Stored in \code{X} are the thirteen model predictors used by Cisneros et al. (2023).
#'For \code{BA} and entries to \code{X}, the first two dimensions correspond to time \eqn{\times} location with 
#'their respective ordinate values given in \code{times} and \code{coords}; \code{coords} is a 7901 by 2 matrix correspond to the longitude and latitude coordinates of the centroid of each polygon.
#'
#'We have three types of predictor variables in \code{X}: meteorological (\code{X.met}), NDVI (\code{X.NDVI}) and topographical (\code{X.topo}).
#'
#'Ten meteorological variables are considered and given as monthly maxima and means in \code{X.met}. 
#'These were provided by the ERA5-reanalysis monthly land averages (Muñoz-Sabater, 2019), available through the COPERNICUS Climate Data Service. 
#'The variables are ordered as followed: the first five variables are maximum evaporation (m of water equiv.), precipitation (m), temperature at a 2m altitude (K),
#'and both eastern (U) and northern (V) components of wind velocity at a 10m altitude (m/s); these are followed by the corresponding mean values.
#'
#'Average NDVI (unitless) is provided in \code{X.NDVI} and taken from the Tier-1 orthorectified Landsat-7 scenes converted to the top of atmosphere reflectance (Chander et al., 2009). 
#'
#'Two topographical predictors are given in \code{X.topo}: the average slope (◦) and aspect (◦) of each polygon, which were derived from the 
#'Shuttle Radar Topography Mission digital elevation model (Farr et al., 2000). Note that whilst these variables are static, 
#'we have stacked them into an array to have the same dimension as \code{X.met} and \code{X.NDVI}.
#'
#'See Cisneros et al. (2023+) for details on the construction of this dataset. Note that the example code describes the fitting of the eGPD-GCNN model used by these authors. This model exploits
#'the graph convolutional neural network (GCNN) of Kipf and Welling (2016), with trainable skip connection (see GCSConv layer of \href{https://graphneural.network/layers/convolution/}{https://graphneural.network/layers/convolution/}).
#' @docType data
#'
#' @usage data(AusWild)
#'
#' @format A list with 6 elements:
#' \describe{
#' \item{BA}{An array with dimension (235, 7901), corresponding to the monthly burnt area response data.}

#'  \item{a.s}{A vector of length (7901), giving the area of each spatial polygon.}

#' \item{X}{A list with 4 elements:
#' \describe{
#' \item{X.met}{An array with dimension (235, 7901, 10) corresponding to the meterological predictors described below.}
#'  \item{X.NDVI}{An array with dimension (235, 7901, 1) corresponding to NDVI as described below.}
#' \item{X.topo}{An array with dimension (235, 7901, 2) corresponding to the topographical predictors described below.}
#'
#' }
#' }
#' \item{times}{A vector of length (235) giving the observation indices. Format is "year-month". Corresponds to first dimension of \code{BA}.}
#'  \item{coords}{A matrix of dimension (7901, 2) giving the longitude/latitude coordinate for the second dimension of \code{BA}.}
#' }
#'
#' @keywords datasets
#'
#'@references 
#' Farr, T. G. and Kobrick, M. (2000). \emph{Shuttle radar topography mission produces a wealth of data.} Eos, Transactions American Geophysical Union, 81(48):583–585. (\href{https://doi.org/10.1029/EO081i048p00583}{doi}).
#'
#' Chander, G., Markham, B. L., and Helder, D. L. (2009). \emph{Summary of current radiometric calibration coefficients for landsat MSS, TM, ETM+, and EO-1 ALI sensors.} Remote sensing of environment, 113(5):893–903. (\href{https://doi.org/10.1016/j.rse.2009.01.007}{doi}).
#'
#' Australian Bureau of Statistics (ABS, 2011). \emph{Australian statistical geography standard (ASGS): Volume 5–remoteness structure.} (\href{https://www.abs.gov.au/ausstats/abs@.nsf/mf/1270.0.55.005}{Link}).
#' 
#' Kipf, T. N. and Welling, M. (2016). \emph{Semi-supervised classification with graph convolutional networks}. (\href{https://arxiv.org/abs/1412.6980}{arXiv:1412.6980}).
#' 
#' Muñoz-Sabater, J. (2019).\emph{ERA5-land monthly averaged data from 1981 to present, Copernicus climate change service (C3S) climate data store (CDS).} Earth Syst. Sci. Data, 55:5679–5695. (\href{https://doi.org/10.24381/cds.e2161bac}{doi}).
#'
#' Cisneros, D., Richards, J., Dahal, A., Lombardo, L., and Huser, R. (2023+), \emph{Deep learning-based graphical regression for jointly moderate and extreme Australian wildfires.}. (\href{}{In draft}).
# '
#' @examples
#' data("AusWild")
#' 
#' 
#' #Create adjacency matrix
#' require(fields)
#' h <- rdist.earth(AusWild$coords,miles=F) #Distance matrix
#' 
#' range.par <- 650
#' A <- exp(-(h/range.par)^2)  # or alternatively, exp(-h/range.par)
#' 
#' cut.off.dist <- 700
#' 
#' A[h>cut.off.dist] <- 0 #Induce sparsity by setting values with h < cut.off.dist to zero
#' 
#' diag(A) <- 0 #Remove self-loops
#' 
#' #Make response
#' Y<-sqrt(AusWild$BA) # Square-root average BA per fire
#' Y[is.na(Y)] <- -1e10 # Any NA values set to -1e10. These are removed from evaluation of the loss function
#' 
#' #Make covariates
#' X <- array(dim=c(dim(Y),15))
#' 
#' X[,,1:10] <- AusWild$X$X.met
#' X[,,11] <- AusWild$X$X.NDVI
#' X[,,12:13] <- AusWild$X$X.topo
#' #We also add the coordinates
#' for(i in 1:dim(X)[1]){
#'   X[i,,14]=AusWild$coords[,1]
#'   X[i,,15]=AusWild$coords[,2]
#' }
#' #Normalise inputs
#' for(i in 1:dim(X)[3]){
#'   m=mean( X[,,i][Y > 0],na.rm=T)
#'   s=sd( X[,,i][Y > 0 ],na.rm=T)
#'   X[,,i]=( X[,,i]-m)/s
#' }
#' 
#' #We replicate the polygon areas to have the same dimension as Y. Note that we use the square root area as the offset in sigma.
#' offset <- matrix(rep(sqrt(AusWild$a.s),nrow(Y)),nrow=nrow(Y),ncol=ncol(Y),byrow=T)
#' 
#' #Subset into validation and training data
#' valid.inds=sample(1:length(Y),length(Y)/5)
#' Y.train<-Y.valid<-Y
#' Y.train[valid.inds]=-1e10
#' Y.valid[-valid.inds]=-1e10
#' 
#' 
#' # Set inital parameters
#' init.xi<- 0.3; init.kappa <- 0.85; init.scale <- 50/mean(AusWild$a.s) #We scale the latter by the area
#' 
#' #Define architecture
#' widths <- c(8,8,8)
#' 
#' # Define predictors for sigma/scale parameter. Note that we do not use the PINN framework of Richard and Huser (2022+), and
#' # so we set interpetable components to NULL values.
#' X.s=list("X.nn.s"=X,
#'          "X.lin.s"=NULL, "X.add.basis.s"=NULL)
#' 
#' 
#' #Fit the eGPD model. 
#' NN.fit<-eGPD.NN.train(Y.train, Y.valid,X.s,X.k=NULL, #X.k=NULL corresponds to a stationary kappa parameters
#'                       type="GCNN", A=A, offset=offset,
#'                       n.ep=3500, batch.size=235, 
#'                       init.scale=init.scale, init.kappa=init.kappa,init.xi=init.xi,
#'                       widths=widths, seed=1)
#' 
#' 
#' preds<-eGPD.NN.predict(X.s=X.s,X.k=NULL,NN.fit$model, offset)
#' 
#' print("Plot scale parameter estimates")
#' hist(preds$pred.sigma, xlab=expression(sigma),main="")
#' 
#' print(paste0("kappa = ", round(preds$pred.kappa[1,1],3)))
#' print(paste0("xi = ", round(preds$pred.xi[1,1],3)))
#' 
#' # To plot a map using the geometry file. See help(AusWild_geom):
#' #----------------------------------------------------------------------
#' # require(ggplot2)
#' # require(viridis)
#' # data("AusWild_geom")
#' # plot.df <- AusWild_geom
#' # plot.df$plot.spread <- preds$pred.sigma[1,] #Plot first month of sigma estimates
#'  
#' # ggplot(data = plot.df) + xlab("")+ylab("")+
#' #   geom_sf(mapping = aes_string(fill="plot.spread"), color = "black",size = 0.1 ) +
#' #   scale_fill_viridis(name="",option = "F",direction=-1,alpha=.7)
#' 
"AusWild"
