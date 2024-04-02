#' blended-GEV point process PINN model
#'
#' Build and train a partially-interpretable neural network for fitting a bGEV point-process model
#'
#' @param types  named list of strings defining the types of neural network to be built for each parameter. Has three entries: \code{q.NN}, \code{s.NN}, and \code{xi.NN}. Each entry takes one of two values: if \code{types$q.NN=="MLP"}, 
#' the neural network in the model for \eqn{q_\alpha} will have all densely-connected layers; if \code{types$q.NN=="CNN"}, this neural network will instead have all convolutional layers (with 3 by 3 filters). If only a single string provided, e.g., \code{types=="MLP"}, then the neural network type is shared across all parameters. Defaults to an MLP. 
#' If \code{any(types=="CNN")}, then \code{Y.train} and \code{Y.valid} must have three dimensions with the latter two corresponding to an \eqn{M} by \eqn{N} regular grid of spatial locations.
#' @param Y.train,Y.valid a 2 or 3 dimensional array of training or validation real-valued response data.
#' Missing values can be handled by setting corresponding entries of \code{Y.train} or \code{Y.valid} to \code{-1e10}.
#' The first dimension should be the observation indices, e.g., time.
#'
#' If \code{Y.valid==NULL}, no validation loss will be computed and the returned model will be that which minimises the training loss over \code{n.ep} epochs.
#'
#' @param u an array with the same dimension as \code{Y.train}. Gives the threshold above which the bGEV-PP model is fitted, see below. Note that \code{u} is applied to both \code{Y.train} and \code{Y.valid}.
#' @param X  a list of arrays corresponding to the complementary subsets of the \eqn{d\geq 1} predictors which are used in the PINN model for each parameter. Must contain at least one of the following named entries:\describe{
#' \item{\code{X_L.q}}{A 3 or 4 dimensional array of "linear" predictor values used in modelling \eqn{q_\alpha}. Must have more dimension than \code{Y.train}.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the last dimension corresponds to the chosen \eqn{l_1\geq 0} 'linear' predictor values.}
#' \item{\code{X_A.basis.q}}{A 4 or 5 dimensional array of basis function evaluations for the "additive" predictor values used in modelling \eqn{q_\alpha}.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the penultimate dimensions corresponds to the chosen \eqn{a_1\geq 0} 'linear' predictor values and the last dimension is equal to the number of knots used for estimating the splines. See example code.}
#' \item{\code{X_N.q}}{A 3 or 4 dimensional array of "non-additive" predictor values used in the neural network component of the PINN for \eqn{q_\alpha}.  If \code{NULL}, a model without the NN component is built and trained; if this is the case, then \code{types$q.NN} has no effect on the model.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the last dimension corresponds to the chosen \eqn{d-l_1-a_1\geq 0} 'non-additive' predictor values.}
#' \item{\code{X_L.s}, \code{X_A.basis.s}, \code{X_N.s}, \code{X_L.xi}, \code{X_A.basis.xi}, \code{X_N.xi}}{As above, but these predictors feature in the PINN models for \eqn{s_\beta} and \eqn{\xi}.}
#' }
#' Note that entries to \code{X} denote the predictors for both \code{Y.train} and \code{Y.valid}. If any of the arrays in \code{X} are missing or set to \code{NULL}, the corresponding component of the PINN model is removed. For example, if \code{is.null(X$X_L.xi)}, no linear model is used within the PINN for \eqn{\xi}.
#' @param n.ep number of epochs used for training. Defaults to 1000.
#' @param alpha,beta,p_a,p_b,c1,c2 hyper-parameters associated with the bGEV distribution. Defaults to those used by Castro-Camilo, D., et al. (2021). Require \code{alpha >= p_b} and \code{beta/2 >= p_b}.
#' @param batch.size mini-batch size used for training with stochastic gradient descent. If larger than \code{dim(Y.train)[1]}, i.e., the number of observations, then regular gradient descent used.
#' @param init.loc,init.spread,init.xi sets the initial estimate of \eqn{q_\alpha,s_\beta}, and \eqn{\xi\in(0,1)} estimates across all dimensions of \code{Y.train}. Overridden by \code{init.wb_path} if \code{!is.null(init.wb_path)}, but otherwise the initial parameters must be supplied.
#' @param init.wb_path filepath to a \code{keras} model which is then used as initial weights and biases for training the new model. The original model must have
#' the exact same architecture and trained with the same input data as the new model. If \code{NULL}, then initial weights and biases are random (with seed \code{seed}) but the
#' final layer has zero initial weights to ensure that the initial location, spread and shape estimates are \code{init.loc, init.spread}, and \code{init.xi}, respectively,  across all dimensions.
#' @param widths named list of vectors giving the widths/filters for the hidden dense/convolution layers for each parameter, see example. Entries take the same names as argument \code{types}. The number of hidden layers in the neural network of the corresponding PINN model is equal to the length of the provided vector. For example, setting \code{types$q.NN="MLP"} and \code{widths$q.NN=c(6,6,6)} will construct a MLP for \eqn{q_\alpha} that has three hidden layers, each with width six. If \code{widths} provides a single vector in place of a list, this architecture will be shared across all parameters. Defaults to (6,6).
#' @param seed seed for random initial weights and biases.
#' @param loc.link string defining the link function used for the location parameter, see \eqn{h_1} below. If \code{link=="exp"}, then \eqn{h_1=\exp(x)}; if \code{link=="identity"}, then \eqn{h_1(x)=x}.
#' @param model fitted \code{keras} model. Output from \code{bGEVPP.NN.train}.
#' @param n_b number of observations per block, e.g., if observations correspond to months and the interest is annual maxima, then \code{n_b=12}.

#' @name bGEVPP.NN

#' @details{
#' Consider a real-valued random variable \eqn{Y} and let \eqn{\mathbf{X}} denote a \eqn{d}-dimensional predictor set with observations \eqn{\mathbf{x}}.
#' For \eqn{i=1,2,3}, we define integers \eqn{l_i\geq 0,a_i \geq 0}, and \eqn{0\leq l_i+a_i \leq d}, and let \eqn{\mathbf{X}^{(i)}_L, \mathbf{X}^{(i)}_A}, and \eqn{\mathbf{X}^{(i)}_N} be distinct sub-vectors
#' of \eqn{\mathbf{X}}, with observations of each component denoted by \eqn{\mathbf{x}^{(i)}_L, \mathbf{x}^{(i)}_A}, and \eqn{\mathbf{x}^{(i)}_N}, respectively; the lengths of the sub-vectors are \eqn{l_i,a_i}, and \eqn{d_i-l_i-a}, respectively.
#' For a fixed threshold \eqn{u(\mathbf{x})}, dependent on predictors, we model \eqn{Y|\mathbf{X}=\mathbf{x}\sim\mbox{bGEV-PP}(q_\alpha(\mathbf{x}),s_\beta(\mathbf{x}),\xi(\mathbf{x});u(\mathbf{x}))} for \eqn{\xi\in(0,1)} with
#' \deqn{q_\alpha (\mathbf{x})=h_1\{\eta^{(1)}_0+m^{(1)}_L(\mathbf{x}^{(1)}_L)+m^{(1)}_A(x^{(1)}_A)+m^{(1)}_N(\mathbf{x}^{(1)}_N)\},}
#' \deqn{s_\beta (\mathbf{x})=\exp\{\eta^{(2)}_0+m^{(2)}_L(\mathbf{x}^{(2)}_L)+m^{(2)}_A(x^{(2)}_A)+m^{(2)}_N(\mathbf{x}^{(2)}_N)\},} and
#'  \deqn{\xi(\mathbf{x})=logistic\{\eta^{(3)}_0+m^{(3)}_L(\mathbf{x}^{(3)}_L)+m^{(3)}_A(x^{(3)}_A)+m^{(3)}_N(\mathbf{x}^{(3)}_N)\},}
#' where \eqn{h_1} is some link-function and \eqn{\eta^{(i)}_0} are constant intercepts. The unknown functions \eqn{m^{(i)}_L} and
#' \eqn{m^{(i)}_A} are estimated using linear functions and splines, respectively, and are
#' both returned as outputs by \code{bGEVPP.N.predict}; each \eqn{m^{(i)}_N} are estimated using neural networks.
#'
#'Note that for sufficiently large \eqn{u} that \eqn{Y\sim\mbox{bGEV-PP}(q_\alpha,s_\beta,\xi;u)} implies that \eqn{\max_{i=1,\dots,n_b}\{Y_i\}\sim \mbox{bGEV}(q_\alpha,s_\beta,\xi)},
#'i.e., the \eqn{n_b}-block maxima of independent realisations of \eqn{Y} follow a bGEV distribution (see \code{help(pbGEV)}). The size of the block can be specified by the parameter \code{n_b}.
#'
#' The model is fitted by minimising the negative log-likelihood associated with the bGEV-PP model; training is performed over \code{n.ep} training epochs.
#' Although the model is trained by minimising the loss evaluated for \code{Y.train}, the final returned model may minimise some other loss.
#' The current state of the model is saved after each epoch, using \code{keras::callback_model_checkpoint}, if the value of some criterion subcedes that of the model from the previous checkpoint; this criterion is the loss evaluated for validation set \code{Y.valid} if \code{!is.null(Y.valid)} and for \code{Y.train}, otherwise.
#'
#'}
#' @return \code{bGEVPP.NN.train} returns the fitted Keras \code{model}.  \code{bGEVPP.NN.predict} is a wrapper for \code{keras::predict} that returns the predicted parameter estimates, and, if applicable, their corresponding linear regression coefficients and spline bases weights.
#'
#'@references
#' Castro-Camilo, D., Huser, R., and Rue, H. (2021), \emph{Practical strategies for generalized extreme value-based regression models for extremes}, Environmetrics, e274.
#' (\href{https://doi.org/10.1002/env.2742}{doi})
#'
#' Richards, J. and Huser, R. (2024+), \emph{Regression modelling of spatiotemporal extreme U.S. wildfires via partially-interpretable neural networks}. (\href{https://arxiv.org/abs/2208.07581}{arXiv:2208.07581}).
#'
#' @examples
#'
#' # Build and train a simple MLP for toy data
#' 
#' set.seed(1)
#' 
#' # Create  predictors
#' preds<-rnorm(prod(c(200,10,10,8)))
#' 
#' 
#' # Re-shape to a 4d array. First dimension corresponds to observations,
#' # last to the different components of the predictor set.
#' # Other dimensions correspond to indices of predictors, e.g., a grid of locations. Can be just a 1D grid.
#' dim(preds)=c(200,10,10,8)
#' # We have 200 observations of eight predictors on a 10 by 10 grid.
#' 
#' 
#' # Split predictors into linear, additive, and nn. Different for the location and scale parameters.
#' X_N.q=preds[,,,1:4] #Four nn predictors for q_\alpha
#' X_L.q=preds[,,,5:6] #Two additive predictors for q_\alpha
#' X_A.q=preds[,,,7:8] #Two additive predictors for q_\alpha
#' 
#' X_N.s=preds[,,,1:2] #Two nn predictors for s_\beta
#' X_L.s=preds[,,,3] #One linear predictor for s_\beta
#' dim(X_L.s)=c(dim(X_L.s),1) #Change dimension so consistent
#' X_A.s=preds[,,,4] #One additive predictor for s_\beta
#' dim(X_A.s)=c(dim(X_A.s),1) #Change dimension so consistent
#' 
#' # Create toy response data
#' 
#' # Contribution to location parameter
#' # Linear contribution
#' m_L_1 = 0.3*X_L.q[,,,1]+0.6*X_L.q[,,,2]
#' 
#' # Additive contribution
#' m_A_1 = 0.1*X_A.q[,,,1]^3+0.2*X_A.q[,,,1]-
#'   0.1*X_A.q[,,,2]^3+0.5*X_A.q[,,,2]^2
#' 
#' # Non-additive contribution - to be estimated by NN
#' m_N_1 = 0.5*exp(-3+X_N.q[,,,4]+X_N.q[,,,1])+
#'   sin(X_N.q[,,,1]-X_N.q[,,,2])*(X_N.q[,,,4]+X_N.q[,,,2])-
#'   cos(X_N.q[,,,4]-X_N.q[,,,1])*(X_N.q[,,,3]+X_N.q[,,,1])
#' 
#' q_alpha=1+m_L_1+m_A_1+m_N_1 #Identity link
#' 
#' # Contribution to scale parameter
#' # Linear contribution
#' m_L_2 = 0.5*X_L.s[,,,1]
#' 
#' # Additive contribution
#' m_A_2 = 0.1*X_A.s[,,,1]^2+0.2*X_A.s[,,,1]
#' 
#' # Non-additive contribution - to be estimated by NN
#' m_N_2 = 0.2*exp(-4+X_N.s[,,,2]+X_N.s[,,,1])+
#'   sin(X_N.s[,,,1]-X_N.s[,,,2])*(X_N.s[,,,1]+X_N.s[,,,2])
#' 
#' s_beta=0.2*exp(m_L_2+m_A_2+m_N_2) #Exponential link
#' 
#' # We will keep xi fixed across predictors
#' xi=0.1 # Set xi
#' 
#' theta=array(dim=c(dim(s_beta),3))
#' theta[,,,1]=q_alpha; theta[,,,2] = s_beta; theta[,,,3]=xi
#' 
#' # We simulate data from the extreme value point process model with u take as the 80% quantile
#' 
#' # Gives the 80% quantile of Y
#' u<-apply(theta,1:3,function(x) qPP(prob=0.8,loc=x[1],scale=x[2],xi=x[3],re.par = T))
#' 
#' # Simulate from re-parametrised point process model using same u as given above
#' Y=apply(theta,1:3,function(x) rPP(1,u.prob=0.8,loc=x[1],scale=x[2],xi=x[3],re.par=T))
#' 
#' # Note that the point process model is only valid for Y > u. If Y < u, then rPP gives NA.
#' # We can set NA values to some c < u as these do not contribute to model fitting.
#' Y[is.na(Y)]=u[is.na(Y)]-1
#' 
#' 
#' 
#' # Create training and validation, respectively.
#' # We mask 20% of the Y values and use this for validation
#' # Masked values must be set to -1e10 and are treated as missing whilst training
#' 
#' mask_inds=sample(1:length(Y),size=length(Y)*0.8)
#' 
#' Y.train<-Y.valid<-Y # Create training and validation data, respectively.
#' Y.train[-mask_inds]=-1e10
#' Y.valid[mask_inds]=-1e10
#' 
#' 
#' 
#' # To build a model with an additive component, we require an array of evaluations of
#' # the basis functions for each pre-specified knot and entry to X_A.q and X_A.s
#' 
#' rad=function(x,c){ # Define a basis function. Here we use radial basis
#'   out=abs(x-c)^2*log(abs(x-c))
#'   out[(x-c)==0]=0
#'   return(out)
#' }
#' 
#' n.knot.q = 5; n.knot.s = 4 # set number of knots.
#' # Must be the same for each additive predictor,
#' # but can differ between the parameters q_\alpha and s_\beta
#' 
#' # Get knots for q_\alpha predictors
#' knots.q=matrix(nrow=dim(X_A.q)[4],ncol=n.knot.q)
#' 
#' # We set knots to be equally-spaced marginal quantiles
#' for( i in 1:dim(X_A.q)[4]){
#'   knots.q[i,]=quantile(X_A.q[,,,i],probs=seq(0,1,length=n.knot.q))
#' }
#' 
#' # Evaluate radial basis functions for q_\alpha predictors
#' X_A.basis.q<-array(dim=c(dim(X_A.q),n.knot.q))
#' for( i in 1:dim(X_A.q)[4]) {
#'   for(k in 1:n.knot.q) {
#'     X_A.basis.q[,,,i,k]= rad(x=X_A.q[,,,i],c=knots.q[i,k])
#'     # Evaluate rad at all entries to X_A.q and for all knots
#'   }}
#' 
#' 
#' 
#' # Get knots for s_\beta predictor
#' knots.s=matrix(nrow=dim(X_A.s)[4],ncol=n.knot.s)
#' for( i in 1:dim(X_A.s)[4]){
#'   knots.s[i,]=quantile(X_A.s[,,,i],probs=seq(0,1,length=n.knot.s))
#' }
#' 
#' # Evaluate radial basis functions for s_\beta predictor
#' X_A.basis.s<-array(dim=c(dim(X_A.s),n.knot.s))
#' for( i in 1:dim(X_A.s)[4]) {
#'   for(k in 1:n.knot.s) {
#'     X_A.basis.s[,,,i,k]= rad(x=X_A.s[,,,i],c=knots.s[i,k])
#'     #Evaluate rad at all entries to X_A.q and for all knots
#'   }}
#' 
#' 
#' 
#' # We define PINN (lin+GAM+NN) models for both the location and scale parameter
#' # Combine into a list of predictors
#' X = list(
#'   "X_N.q"=X_N.q, "X_L.q"=X_L.q,
#'   "X_A.basis.q"=X_A.basis.q, #Predictors for q_\alpha
#'   "X_N.s"=X_N.s, "X_L.s"=X_L.s,
#'   "X_A.basis.s"=X_A.basis.s,
#'   "X_A.basis.xi"=X_A.basis.s) #Predictors for s_\beta
#' # Note that we have not defined covariates for xi, 
#' # so this will be treated as constant with respect to the predictors
#' 
#' 
#' # We here treat u as fixed and known. In an application, u can be estimated using quant.N.train.
#' 
#' # Choose NN archiecture for different parameters
#' NN.types = list("q.NN"="MLP","s.NN"="CNN") # Layer types
#' NN.widths = list("q.NN"=c(6,6),"s.NN"=c(4,4)) # Layers/widths
#' 
#' # Fit the bGEV-PP model using u. Note that training is not run to completion.
#' NN.fit<-bGEVPP.NN.train(Y.train, Y.valid,X, u=u, types=NN.types,
#'                        link.loc="identity",
#'                        n.ep=20, batch.size=50,
#'                        init.loc=2, init.spread=2, init.xi=0.1,
#'                        widths=NN.widths, seed=1, n_b=12)
#' out<-bGEVPP.NN.predict(X,u=u,NN.fit$model)
#' 
#' print("q_alpha linear coefficients: "); print(round(out$lin.coeff_q,2))
#' print("s_beta linear coefficients: "); print(round(out$lin.coeff_s,2))
#' 
#' # Note that this is a simple example that can be run in a personal computer.
#' # Whilst the q_alpha functions are well estimated, more data/larger n.ep are required for more accurate
#' # estimation of s_beta functions and xi
#' 
#' # To save model, run
#' NN.fit$model %>% save_model_tf("model_bGEVPP")
#' # To load model, run
#' model  <- load_model_tf("model_bGEVPP",
#'                         custom_objects=list(
#'                           "bgev_PP_loss_alpha__beta__p_a__p_b__c1__c2__n_b_"=
#'                             bgev_PP_loss(n_b=12))
#' )
#' 
#' # Note that bGEV_PP_loss() can take custom alpha,beta, p_a, p_b, c1 and c2 arguments if defaults not used.
#' 
#' 
#' # Plot splines for the additive predictors
#' 
#' # Location predictors
#' n.A.preds_q=dim(X_A.q)[length(dim(X_A.q))]
#' par(mfrow=c(1,n.A.preds_q))
#' for(i in 1:n.A.preds_q){
#'   plt.x=seq(from=min(knots.q[i,]),to=max(knots.q[i,]),length=1000)  #Create sequence for x-axis
#' 
#'   tmp=matrix(nrow=length(plt.x),ncol=n.knot.q)
#'   for(j in 1:n.knot.q){
#'     tmp[,j]=rad(plt.x,knots.q[i,j]) #Evaluate radial basis function of plt.x and all knots
#'   }
#'   plt.y=tmp%*%out$gam.weights_q[i,]
#'   plot(plt.x,plt.y,type="l",main=paste0("q_alpha spline: predictor ",i),xlab="x",ylab="f(x)")
#'   points(knots.q[i,],rep(mean(plt.y),n.knot.q),col="red",pch=2)
#'   # Adds red triangles that denote knot locations
#' 
#' }
#' 
#' # Spread predictors
#' n.A.preds_s=dim(X_A.s)[length(dim(X_A.s))]
#' par(mfrow=c(1,n.A.preds_s))
#' for(i in 1:n.A.preds_s){
#'   plt.x=seq(from=min(knots.s[i,]),to=max(knots.s[i,]),length=1000)  # Create sequence for x-axis
#' 
#'   tmp=matrix(nrow=length(plt.x),ncol=n.knot.s)
#'   for(j in 1:n.knot.s){
#'     tmp[,j]=rad(plt.x,knots.s[i,j]) # Evaluate radial basis function of plt.x and all knots
#'   }
#'   plt.y=tmp%*%out$gam.weights_s[i,]
#'   plot(plt.x,plt.y,type="l",main=paste0("s_beta spline: predictor ",i), xlab="x", ylab="f(x)")
#'   points(knots.s[i,],rep(mean(plt.y),n.knot.s),col="red",pch=2)
#'   # Adds red triangles that denote knot locations
#' 
#' }
#' 
#' @import reticulate keras tensorflow
#'
#' @rdname bGEVPP.NN
#' @export

bGEVPP.NN.train=function(Y.train, Y.valid = NULL, X, u = NULL, types="MLP",link.loc="identity",
                        n.ep=100, batch.size=100, init.loc=NULL, init.spread=NULL,init.xi=NULL,
                        widths=c(6,6), seed=NULL, init.wb_path=NULL,
                        alpha=0.5,beta=0.5,p_a=0.05,p_b=0.2,c1=5,c2=5,n_b=1)
{

  valid.X.names=c("X_N.q","X_L.q","X_A.basis.q",
            "X_N.s","X_L.s","X_A.basis.s",
            "X_N.xi","X_L.xi","X_A.basis.xi")
  
  for(i in names(X)) if(!(i %in% valid.X.names)) stop("Predictors named incorrectly. Double-check documentation.")
  
  if(length(types)!=length(widths)) stop("length(types)!=length(widths)")
  valid.type.names=c("q.NN","s.NN","xi.NN")
  if(length(types)>1) for(i in names(types)) if(!(i %in% valid.type.names)) stop("Widths and types named incorrectly. Double-check documentation.")
  if(length(widths)>1) for(i in names(widths)) if(!(i %in% valid.type.names)) stop("Widths and types named incorrectly. Double-check documentation.")
  if(class(types)=="character" & class(widths)=="vector" ) print("One architecture provided. Will be shared across parameters.")

  if(is.null(Y.train)) stop("No training response data provided")
  if(is.null(u)) stop("No threshold u provided")

  if(is.null(init.loc) & is.null(init.wb_path)  ) stop("Inital location estimate not provided")
  if(is.null(init.spread) & is.null(init.wb_path)   ) stop("Inital spread estimate not provided")
  if(is.null(init.xi)  & is.null(init.wb_path) ) stop("Inital shape estimate not provided")

  

  print(paste0("Creating bGEV-PP model with ",n_b,"-block maxima following bGEV"))
  
  X_N.q=X$X_N.q
  X_L.q=X$X_L.q
  X_A.basis.q=X$X_A.basis.q


  if(!is.null(X_N.q) & !is.null(X_A.basis.q) & !is.null(X_L.q) ) {  train.data= list(X_L.q,X_A.basis.q,X_N.q); print("Defining lin+GAM+NN model for q_\alpha" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_q=X_L.q,add_input_q=X_A.basis.q,  nn_input_q=X_N.q),Y.valid)}
  if(is.null(X_N.q) & !is.null(X_A.basis.q) & !is.null(X_L.q) ) {   train.data= list(X_L.q,X_A.basis.q); print("Defining lin+GAM model for q_\alpha" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_q=X_L.q,add_input_q=X_A.basis.q),Y.valid)}
  if(!is.null(X_N.q) & is.null(X_A.basis.q) & !is.null(X_L.q) ) { train.data= list(X_L.q,X_N.q); print("Defining lin+NN model for q_\alpha" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_q=X_L.q, nn_input_q=X_N.q),Y.valid)}
  if(!is.null(X_N.q) & !is.null(X_A.basis.q) & is.null(X_L.q) ) {train.data= list(X_A.basis.q,X_N.q); print("Defining GAM+NN model for q_\alpha" );  if(!is.null(Y.valid)) validation.data=list(list(add_input_q=X_A.basis.q,  nn_input_q=X_N.q),Y.valid)}
  if(is.null(X_N.q) & is.null(X_A.basis.q) & !is.null(X_L.q) )   {train.data= list(X_L.q); print("Defining fully-linear model for q_\alpha" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_q=X_L.q),Y.valid)}
  if(is.null(X_N.q) & !is.null(X_A.basis.q) & is.null(X_L.q) )   {train.data= list(X_A.basis.q); print("Defining fully-additive model for q_\alpha" );  if(!is.null(Y.valid)) validation.data=list(list(add_input_q=X_A.basis.q),Y.valid)}
  if(!is.null(X_N.q) & is.null(X_A.basis.q) & is.null(X_L.q) )   {train.data= list(X_N.q); print("Defining fully-NN model for q_\alpha" );  if(!is.null(Y.valid)) validation.data=list(list( nn_input_q=X_N.q),Y.valid)}
  if(is.null(X_N.q) & is.null(X_A.basis.q) & is.null(X_L.q) )   {train.data= list(); print("Defining stationary model for q_\alpha" );  if(!is.null(Y.valid)) validation.data=list(list( ),Y.valid)}
  
  type.q=types$q.NN
  widths.q=widths$q.NN
  if(class(types)=="character") type.q=types
  if(class(widths)=="vector") widths.q=widths
  
  if(!is.null(X_N.q) & is.null(type.q)) stop("No architecture provided for q NN")
  if(!is.null(X_N.q) & !is.null(type.q)){
    if(type.q=="CNN"  )  print(paste0("Building ",length(widths.q),"-layer convolutional neural network for q_\alpha" ))
    if(type.q=="MLP"  ) print(paste0("Building ",length(widths.q),"-layer denqely-connected neural network for q_\alpha" ))
  }
  
  X_N.s=X$X_N.s
  X_L.s=X$X_L.s
  X_A.basis.s=X$X_A.basis.s

  if(!is.null(X_N.s) & !is.null(X_A.basis.s) & !is.null(X_L.s) ) {  train.data= c(train.data,list(X_L.s,X_A.basis.s,X_N.s)); print("Defining lin+GAM+NN model for s_\beta" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X_L.s,add_input_s=X_A.basis.s,  nn_input_s=X_N.s)),Y.valid)}
  if(is.null(X_N.s) & !is.null(X_A.basis.s) & !is.null(X_L.s) ) {   train.data= c(train.data,list(X_L.s,X_A.basis.s)); print("Defining lin+GAM model for s_\beta" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X_L.s,add_input_s=X_A.basis.s)),Y.valid)}
  if(!is.null(X_N.s) & is.null(X_A.basis.s) & !is.null(X_L.s) ) { train.data= c(train.data,list(X_L.s,X_N.s)); print("Defining lin+NN model for s_\beta" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X_L.s, nn_input_s=X_N.s)),Y.valid)}
  if(!is.null(X_N.s) & !is.null(X_A.basis.s) & is.null(X_L.s) ) {train.data= c(train.data,list(X_A.basis.s,X_N.s)); print("Defining GAM+NN model for s_\beta" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(add_input_s=X_A.basis.s,  nn_input_s=X_N.s)),Y.valid)}
  if(is.null(X_N.s) & is.null(X_A.basis.s) & !is.null(X_L.s) )   {train.data= c(train.data,list(X_L.s)); print("Defining fully-linear model for s_\beta" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X_L.s)),Y.valid)}
  if(is.null(X_N.s) & !is.null(X_A.basis.s) & is.null(X_L.s) )   {train.data= c(train.data,list(X_A.basis.s)); print("Defining fully-additive model for s_\beta" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(add_input_s=X_A.basis.s)),Y.valid)}
  if(!is.null(X_N.s) & is.null(X_A.basis.s) & is.null(X_L.s) )   {train.data= c(train.data,list(X_N.s)); print("Defining fully-NN model for s_\beta" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(nn_input_s=X_N.s)),Y.valid)}
  if(is.null(X_N.s) & is.null(X_A.basis.s) & is.null(X_L.s) )   { print("Defining stationary model for s_\beta" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]]),Y.valid)}

  type.s=types$s.NN
  widths.s=widths$s.NN
  if(class(types)=="character") type.s=types
  if(class(widths)=="vector") widths.s=widths
  
  if(!is.null(X_N.s) & is.null(type.s)) stop("No architecture provided for s NN")
  if(!is.null(X_N.s) & !is.null(type.s)){
    if(type.s=="CNN"  )  print(paste0("Building ",length(widths.s),"-layer convolutional neural network for s_\beta" ))
    if(type.s=="MLP"  ) print(paste0("Building ",length(widths.s),"-layer densely-connected neural network for s_\beta" ))
  }
  
  X_N.xi=X$X_N.xi
  X_L.xi=X$X_L.xi
  X_A.basis.xi=X$X_A.basis.xi
  
  if(!is.null(X_N.xi) & !is.null(X_A.basis.xi) & !is.null(X_L.xi) ) {  train.data= c(train.data,list(X_L.xi,X_A.basis.xi,X_N.xi,u)); print("Defining lin+GAM+NN model for xi" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_xi=X_L.xi,add_input_xi=X_A.basis.xi,  nn_input_xi=X_N.xi,u_input=u)),Y.valid)}
  if(is.null(X_N.xi) & !is.null(X_A.basis.xi) & !is.null(X_L.xi) ) {   train.data= c(train.data,list(X_L.xi,X_A.basis.xi,u)); print("Defining lin+GAM model for xi" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_xi=X_L.xi,add_input_xi=X_A.basis.xi,u_input=u)),Y.valid)}
  if(!is.null(X_N.xi) & is.null(X_A.basis.xi) & !is.null(X_L.xi) ) { train.data= c(train.data,list(X_L.xi,X_N.xi,u)); print("Defining lin+NN model for xi" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_xi=X_L.xi, nn_input_xi=X_N.xi,u_input=u)),Y.valid)}
  if(!is.null(X_N.xi) & !is.null(X_A.basis.xi) & is.null(X_L.xi) ) {train.data= c(train.data,list(X_A.basis.xi,X_N.xi,u)); print("Defining GAM+NN model for xi" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(add_input_xi=X_A.basis.xi,  nn_input_xi=X_N.xi,u_input=u)),Y.valid)}
  if(is.null(X_N.xi) & is.null(X_A.basis.xi) & !is.null(X_L.xi) )   {train.data= c(train.data,list(X_L.xi,u)); print("Defining fully-linear model for xi" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_xi=X_L.xi,u_input=u)),Y.valid)}
  if(is.null(X_N.xi) & !is.null(X_A.basis.xi) & is.null(X_L.xi) )   {train.data= c(train.data,list(X_A.basis.xi,u)); print("Defining fully-additive model for xi" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(add_input_xi=X_A.basis.xi,u_input=u)),Y.valid)}
  if(!is.null(X_N.xi) & is.null(X_A.basis.xi) & is.null(X_L.xi) )   {train.data= c(train.data,list(X_N.xi,u)); print("Defining fully-NN model for xi" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(nn_input_xi=X_N.xi,u_input=u)),Y.valid)}
  if(is.null(X_N.xi) & is.null(X_A.basis.xi) & is.null(X_L.xi) )   {train.data=  c(train.data,list(u)); print("Defining stationary model for xi" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(u_input=u)),Y.valid)}
  
  type.xi=types$xi.NN
  widths.xi=widths$xi.NN
  if(class(types)=="character") type.xi=types
  if(class(widths)=="vector") widths.xi=widths
  
  if(!is.null(X_N.xi) & is.null(type.xi)) stop("No architecture provided for xi NN")
  if(!is.null(X_N.xi) & !is.null(type.xi)){
    if(type.xi=="CNN"  )  print(paste0("Building ",length(widths.xi),"-layer convolutional neural network for xi" ))
    if(type.xi=="MLP"  ) print(paste0("Building ",length(widths.xi),"-layer densely-connected neural network for xi" ))
  }

  reticulate::use_virtualenv("pinnEV_env", required = T)

  if(!is.null(seed)) tf$random$set_seed(seed)

  if(length(dim(u))!=length(dim(Y.train))+1) dim(u)=c(dim(u),1)
  model<-bGEVPP.NN.build(X_N.q,X_L.q,X_A.basis.q,
                         X_N.s,X_L.s,X_A.basis.s,
                         X_N.xi, X_L.xi, X_A.basis.xi,
                         u,
                        type.q, type.s, type.xi, init.loc,init.spread,init.xi, widths.q, widths.s, widths.xi,link.loc,alpha,beta,p_a,p_b)
  if(!is.null(init.wb_path)) model <- load_model_weights_tf(model,filepath=init.wb_path)

  model %>% compile(
    optimizer="adam",
    loss = bgev_PP_loss(alpha,beta,p_a,p_b,c1,c2,n_b),
    run_eagerly=T
    
  )

  if(!is.null(Y.valid)) checkpoint <- callback_model_checkpoint(paste0("model_bGEVPP_checkpoint"), monitor = "val_loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch") else checkpoint <- callback_model_checkpoint(paste0("model_bGEVPP_checkpoint"), monitor = "loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch")

  .GlobalEnv$model <- model
  if(!is.null(Y.valid)){
    history <- model %>% fit(
      train.data, Y.train,
      epochs = n.ep, batch_size = batch.size,
      callback=list(checkpoint),
      validation_data=validation.data

    )
  }else{

    history <- model %>% fit(
      train.data, Y.train,
      epochs = n.ep, batch_size = batch.size,
      callback=list(checkpoint)
    )
  }

  print("Loading checkpoint weights")
  model <- load_model_weights_tf(model,filepath=paste0("model_bGEVPP_checkpoint"))
  print("Final training loss")
  loss.train<-model %>% evaluate(train.data,Y.train, batch_size=batch.size)
  if(!is.null(Y.valid)){
    print("Final validation loss")
    loss.valid<-model %>% evaluate(train.data,Y.valid, batch_size=batch.size)
    return(list("model"=model,"Training loss"=loss.train, "Validation loss"=loss.valid))
  }else{
    return(list("model"=model,"Training loss"=loss.train))
  }
  

  return(model)
}
#' @rdname bGEVPP.NN
#' @export
#'
bGEVPP.NN.predict=function(X,u, model)
{
    library(tensorflow)
  valid.X.names=c("X_N.q","X_L.q","X_A.basis.q",
                  "X_N.s","X_L.s","X_A.basis.s",
                  "X_N.xi","X_L.xi","X_A.basis.xi")
  
  for(i in names(X)) if(!(i %in% valid.X.names)) stop("Predictors named incorrectly. Double-check documentation.")
  

  X_N.q=X$X_N.q
  X_L.q=X$X_L.q
  X_A.basis.q=X$X_A.basis.q


  if(!is.null(X_N.q) & !is.null(X_A.basis.q) & !is.null(X_L.q) )   train.data= list(X_L.q,X_A.basis.q,X_N.q)
  if(is.null(X_N.q) & !is.null(X_A.basis.q) & !is.null(X_L.q) )   train.data= list(X_L.q,X_A.basis.q)
  if(!is.null(X_N.q) & is.null(X_A.basis.q) & !is.null(X_L.q) )  train.data= list(X_L.q,X_N.q)
  if(!is.null(X_N.q) & !is.null(X_A.basis.q) & is.null(X_L.q) ) train.data= list(X_A.basis.q,X_N.q)
  if(is.null(X_N.q) & is.null(X_A.basis.q) & !is.null(X_L.q) )   train.data= list(X_L.q)
  if(is.null(X_N.q) & !is.null(X_A.basis.q) & is.null(X_L.q) )   train.data= list(X_A.basis.q)
  if(!is.null(X_N.q) & is.null(X_A.basis.q) & is.null(X_L.q) )   train.data= list(X_N.q)
  if(is.null(X_N.q) & is.null(X_A.basis.q) & is.null(X_L.q) )   train.data= list()
  
  X_N.s=X$X_N.s
  X_L.s=X$X_L.s
  X_A.basis.s=X$X_A.basis.s

  if(!is.null(X_N.s) & !is.null(X_A.basis.s) & !is.null(X_L.s) )   train.data= c(train.data,list(X_L.s,X_A.basis.s,X_N.s))
  if(is.null(X_N.s) & !is.null(X_A.basis.s) & !is.null(X_L.s) )   train.data= c(train.data,list(X_L.s,X_A.basis.s))
  if(!is.null(X_N.s) & is.null(X_A.basis.s) & !is.null(X_L.s) )  train.data= c(train.data,list(X_L.s,X_N.s))
  if(!is.null(X_N.s) & !is.null(X_A.basis.s) & is.null(X_L.s) ) train.data= c(train.data,list(X_A.basis.s,X_N.s))
  if(is.null(X_N.s) & is.null(X_A.basis.s) & !is.null(X_L.s) )   train.data= c(train.data,list(X_L.s))
  if(is.null(X_N.s) & !is.null(X_A.basis.s) & is.null(X_L.s) )   train.data= c(train.data,list(X_A.basis.s))
  if(!is.null(X_N.s) & is.null(X_A.basis.s) & is.null(X_L.s) ) train.data= c(train.data,list(X_N.s))

  X_N.xi=X$X_N.xi
  X_L.xi=X$X_L.xi
  X_A.basis.xi=X$X_A.basis.xi
  
  if(!is.null(X_N.xi) & !is.null(X_A.basis.xi) & !is.null(X_L.xi) )   train.data= c(train.data,list(X_L.xi,X_A.basis.xi,X_N.xi,u))
  if(is.null(X_N.xi) & !is.null(X_A.basis.xi) & !is.null(X_L.xi) )   train.data= c(train.data,list(X_L.xi,X_A.basis.xi,u))
  if(!is.null(X_N.xi) & is.null(X_A.basis.xi) & !is.null(X_L.xi) )  train.data= c(train.data,list(X_L.xi,X_N.xi,u))
  if(!is.null(X_N.xi) & !is.null(X_A.basis.xi) & is.null(X_L.xi) ) train.data= c(train.data,list(X_A.basis.xi,X_N.xi,u))
  if(is.null(X_N.xi) & is.null(X_A.basis.xi) & !is.null(X_L.xi) )   train.data= c(train.data,list(X_L.xi,u))
  if(is.null(X_N.xi) & !is.null(X_A.basis.xi) & is.null(X_L.xi) )   train.data= c(train.data,list(X_A.basis.xi,u))
  if(!is.null(X_N.xi) & is.null(X_A.basis.xi) & is.null(X_L.xi) ) train.data= c(train.data,list(X_N.xi,u))
  if(is.null(X_N.xi) & is.null(X_A.basis.xi) & is.null(X_L.xi) ) train.data=c(train.data,list(u))
    
    predictions<-model %>% predict( train.data)
    predictions <- k_constant(predictions)
    pred.loc=k_get_value(predictions[all_dims(),2])
    pred.spread=k_get_value(predictions[all_dims(),3])
    pred.xi=k_get_value(predictions[all_dims(),4])

  if(!is.null(X_A.basis.q))  gam.weights_q<-matrix(t(model$get_layer("add_q")$get_weights()[[1]]),nrow=dim(X_A.basis.q)[length(dim(X_A.basis.q))-1],ncol=dim(X_A.basis.q)[length(dim(X_A.basis.q))],byrow=T)
    if(!is.null(X_A.basis.s))  gam.weights_s<-matrix(t(model$get_layer("add_s")$get_weights()[[1]]),nrow=dim(X_A.basis.s)[length(dim(X_A.basis.s))-1],ncol=dim(X_A.basis.s)[length(dim(X_A.basis.s))],byrow=T)
    if(!is.null(X_A.basis.xi))  gam.weights_xi<-matrix(t(model$get_layer("add_xi")$get_weights()[[1]]),nrow=dim(X_A.basis.xi)[length(dim(X_A.basis.xi))-1],ncol=dim(X_A.basis.xi)[length(dim(X_A.basis.xi))],byrow=T)
    
    out=list("pred.loc"=pred.loc,"pred.spread"=pred.spread,"pred.xi"=pred.xi)
  if(!is.null(X_L.q) ) out=c(out,list("lin.coeff_q"=c(model$get_layer("lin_q")$get_weights()[[1]])))
  if(!is.null(X_L.s) ) out=c(out,list("lin.coeff_s"=c(model$get_layer("lin_s")$get_weights()[[1]])))
    if(!is.null(X_L.xi) ) out=c(out,list("lin.coeff_xi"=c(model$get_layer("lin_xi")$get_weights()[[1]])))
    if(!is.null(X_A.basis.q) ) out=c(out,list("gam.weights_q"=gam.weights_q))
    if(!is.null(X_A.basis.s) ) out=c(out,list("gam.weights_s"=gam.weights_s))
    if(!is.null(X_A.basis.xi) ) out=c(out,list("gam.weights_xi"=gam.weights_xi))
    
  return(out)

}
#'
#'
bGEVPP.NN.build=function(X_N.q,X_L.q,X_A.basis.q,
                         X_N.s,X_L.s,X_A.basis.s,
                        X_N.xi, X_L.xi, X_A.basis.xi,
                         u,
                         type.q,type.s,type.xi,
                         init.loc,init.spread,init.xi, 
                         widths.q, widths.s, widths.xi,
                         link.loc,alpha,beta,p_a,p_b,c1,c2)
{
  #Additive inputs
  if(!is.null(X_A.basis.q))  input_add_q<- layer_input(shape = dim(X_A.basis.q)[-1], name = 'add_input_q')
  if(!is.null(X_A.basis.s))  input_add_s<- layer_input(shape = dim(X_A.basis.s)[-1], name = 'add_input_s')
  if(!is.null(X_A.basis.xi))  input_add_xi<- layer_input(shape = dim(X_A.basis.xi)[-1], name = 'add_input_xi')
  
  #NN input

  if(!is.null(X_N.q))   input_nn_q <- layer_input(shape = dim(X_N.q)[-1], name = 'nn_input_q')
  if(!is.null(X_N.s))   input_nn_s <- layer_input(shape = dim(X_N.s)[-1], name = 'nn_input_s')
  if(!is.null(X_N.xi))   input_nn_xi <- layer_input(shape = dim(X_N.xi)[-1], name = 'nn_input_xi')
  
  #Linear input

  if(!is.null(X_L.q)) input_lin_q <- layer_input(shape = dim(X_L.q)[-1], name = 'lin_input_q')
  if(!is.null(X_L.s)) input_lin_s <- layer_input(shape = dim(X_L.s)[-1], name = 'lin_input_s')
  if(!is.null(X_L.xi)) input_lin_xi <- layer_input(shape = dim(X_L.xi)[-1], name = 'lin_input_xi')
  
  #Threshold input
  input_u <- layer_input(shape = dim(u)[-1], name = 'u_input')




  if(link.loc=="exp") init.loc=log(init.loc) else if(link.loc =="identity") init.loc=init.loc else stop("Invalid link function for location parameter")
  init.spread = log(init.spread)
  init.xi = qlogis(init.xi)
  #NN towers

  #Location
  if(!is.null(X_N.q)){

    nunits=c(widths.q,1)
    n.layers=length(nunits)-1

    nnBranchq <- input_nn_q
    if(type.q=="MLP"){
      for(i in 1:n.layers){
        nnBranchq <- nnBranchq  %>% layer_dense(units=nunits[i],activation = 'relu',
                                                input_shape =dim(X_N.q)[-1], name = paste0('nn_q_dense',i) )
      }
    }else if(type.q=="CNN"){
      for(i in 1:n.layers){
        nnBranchq <- nnBranchq  %>% layer_conv_2d(filters=nunits[i],activation = 'relu',kernel_size=c(3,3), padding='same',
                                                  input_shape =dim(X_N.q)[-1], name = paste0('nn_q_cnn',i) )
      }

    }

    nnBranchq <-   nnBranchq  %>%   layer_dense(units = nunits[n.layers+1], activation = "linear", name = 'nn_q_dense_final',
                                                weights=list(matrix(0,nrow=nunits[n.layers],ncol=1), array(init.loc)))

  }
  
  
#Spread
    if(!is.null(X_N.s)){

    nunits=c(widths.s,1)
    n.layers=length(nunits)-1

    nnBranchs <- input_nn_s
    if(type.s=="MLP"){
      for(i in 1:n.layers){
        nnBranchs <- nnBranchs  %>% layer_dense(units=nunits[i],activation = 'relu',
                                                input_shape =dim(X_N.s)[-1], name = paste0('nn_s_dense',i) )
      }
    }else if(type.s=="CNN"){
      for(i in 1:n.layers){
        nnBranchs <- nnBranchs  %>% layer_conv_2d(filters=nunits[i],activation = 'relu',kernel_size=c(3,3), padding='same',
                                                  input_shape =dim(X_N.s)[-1], name = paste0('nn_s_cnn',i) )
      }

    }

    nnBranchs <-   nnBranchs  %>%   layer_dense(units = nunits[n.layers+1], activation = "linear", name = 'nn_s_dense_final',
                                                weights=list(matrix(0,nrow=nunits[n.layers],ncol=1), array(init.spread)))

    }
  
  #xi
  if(!is.null(X_N.xi)){
    
    nunits=c(widths.xi,1)
    n.layers=length(nunits)-1
    
    nnBranchxi <- input_nn_xi
    if(type.xi=="MLP"){
      for(i in 1:n.layers){
        nnBranchxi <- nnBranchxi  %>% layer_dense(units=nunits[i],activation = 'relu',
                                                input_shape =dim(X_N.xi)[-1], name = paste0('nn_xi_dense',i) )
      }
    }else if(type.xi=="CNN"){
      for(i in 1:n.layers){
        nnBranchxi <- nnBranchxi  %>% layer_conv_2d(filters=nunits[i],activation = 'relu',kernel_size=c(3,3), padding='same',
                                                  input_shape =dim(X_N.xi)[-1], name = paste0('nn_xi_cnn',i) )
      }
      
    }
    
    nnBranchxi <-   nnBranchxi  %>%   layer_dense(units = nunits[n.layers+1], activation = "linear", name = 'nn_xi_dense_final',
                                                weights=list(matrix(0,nrow=nunits[n.layers],ncol=1), array(init.xi)))
    
  }
  
  #Additive towers
  #Location
  n.dim.add_q=length(dim(X_A.basis.q))
  if(!is.null(X_A.basis.q) & !is.null(X_A.basis.q) ) {

    addBranchq <- input_add_q %>%
      layer_reshape(target_shape=c(dim(X_A.basis.q)[2:(n.dim.add_q-2)],prod(dim(X_A.basis.q)[(n.dim.add_q-1):n.dim.add_q]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_q',
                  weights=list(matrix(0,nrow=prod(dim(X_A.basis.q)[(n.dim.add_q-1):n.dim.add_q]),ncol=1)),use_bias = F)
  }
  if(!is.null(X_A.basis.q) & is.null(X_A.basis.q) ) {

    addBranchq <- input_add_q %>%
      layer_reshape(target_shape=c(dim(X_A.basis.q)[2:(n.dim.add_q-2)],prod(dim(X_A.basis.q)[(n.dim.add_q-1):n.dim.add_q]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_q',
                  weights=list(matrix(0,nrow=prod(dim(X_A.basis.q)[(n.dim.add_q-1):n.dim.add_q]),ncol=1),array(init.loc)),use_bias = T)
  }
  #Spread
  n.dim.add_s=length(dim(X_A.basis.s))
  if(!is.null(X_A.basis.s) & !is.null(X_A.basis.s) ) {

    addBranchs <- input_add_s %>%
      layer_reshape(target_shape=c(dim(X_A.basis.s)[2:(n.dim.add_s-2)],prod(dim(X_A.basis.s)[(n.dim.add_s-1):n.dim.add_s]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_s',
                  weights=list(matrix(0,nrow=prod(dim(X_A.basis.s)[(n.dim.add_s-1):n.dim.add_s]),ncol=1)),use_bias = F)
  }
  if(!is.null(X_A.basis.s) & is.null(X_A.basis.s) ) {

    addBranchs <- input_add_s %>%
      layer_reshape(target_shape=c(dim(X_A.basis.s)[2:(n.dim.add_s-2)],prod(dim(X_A.basis.s)[(n.dim.add_s-1):n.dim.add_s]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_s',
                  weights=list(matrix(0,nrow=prod(dim(X_A.basis.s)[(n.dim.add_s-1):n.dim.add_s]),ncol=1),array(init.spread)),use_bias = T)
  }
  #xi
  n.dim.add_xi=length(dim(X_A.basis.xi))
  if(!is.null(X_A.basis.xi) & !is.null(X_A.basis.xi) ) {
    
    addBranchxi <- input_add_xi %>%
      layer_reshape(target_shape=c(dim(X_A.basis.xi)[2:(n.dim.add_xi-2)],prod(dim(X_A.basis.xi)[(n.dim.add_xi-1):n.dim.add_xi]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_xi',
                  weights=list(matrix(0,nrow=prod(dim(X_A.basis.xi)[(n.dim.add_xi-1):n.dim.add_xi]),ncol=1)),use_bias = F)
  }
  if(!is.null(X_A.basis.xi) & is.null(X_A.basis.xi) ) {
    
    addBranchxi <- input_add_xi %>%
      layer_reshape(target_shape=c(dim(X_A.basis.xi)[2:(n.dim.add_xi-2)],prod(dim(X_A.basis.xi)[(n.dim.add_xi-1):n.dim.add_xi]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_xi',
                  weights=list(matrix(0,nrow=prod(dim(X_A.basis.xi)[(n.dim.add_xi-1):n.dim.add_xi]),ncol=1),array(init.xi)),use_bias = T)
  }
  
  
  #Linear towers

  #Location
  if(!is.null(X_L.q) ) {
    n.dim.lin_q=length(dim(X_L.q))

    if(is.null(X_N.q) & is.null(X_A.basis.q )){
      linBranchq <- input_lin_q%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X_L.q)[-1], name = 'lin_q',
                    weights=list(matrix(0,nrow=dim(X_L.q)[n.dim.lin_q],ncol=1),array(init.loc)),use_bias=T)
    }else{
      linBranchq <- input_lin_q%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X_L.q)[-1], name = 'lin_q',
                    weights=list(matrix(0,nrow=dim(X_L.q)[n.dim.lin_q],ncol=1)),use_bias=F)
    }
  }
  #Spread
  if(!is.null(X_L.s) ) {
    n.dim.lin_s=length(dim(X_L.s))

    if(is.null(X_N.s) & is.null(X_A.basis.s )){
      linBranchs <- input_lin_s%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X_L.s)[-1], name = 'lin_s',
                    weights=list(matrix(0,nrow=dim(X_L.s)[n.dim.lin_s],ncol=1),array(init.spread)),use_bias=T)
    }else{
      linBranchs <- input_lin_s%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X_L.s)[-1], name = 'lin_s',
                    weights=list(matrix(0,nrow=dim(X_L.s)[n.dim.lin_s],ncol=1)),use_bias=F)
    }
  }
  #xi
  if(!is.null(X_L.xi) ) {
    n.dim.lin_xi=length(dim(X_L.xi))
    
    if(is.null(X_N.xi) & is.null(X_A.basis.xi)){
      linBranchxi <- input_lin_xi%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X_L.xi)[-1], name = 'lin_xi',
                    weights=list(matrix(0,nrow=dim(X_L.xi)[n.dim.lin_xi],ncol=1),array(init.xi)),use_bias=T)
    }else{
      linBranchxi <- input_lin_xi%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X_L.xi)[-1], name = 'lin_xi',
                    weights=list(matrix(0,nrow=dim(X_L.xi)[n.dim.lin_xi],ncol=1)),use_bias=F)
    }
  }
  
  
  
  #Stationary towers
  
  #Location
  if(is.null(X_N.q) & is.null(X_A.basis.q) & is.null(X_L.q)) {
  
      statBranchq <- input_u %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(u)[-1], trainable=F,
                                                weights=list(matrix(0,nrow=dim(u)[length(dim(u))],ncol=1),array(1,dim=c(1))), name = 'q_stationary_dense1') %>%
        layer_dense(units = 1 ,activation = 'linear',use_bias = F,weights=list(matrix(array(init.loc),nrow=1,ncol=1)), name = 'q_stationary_dense2')
   
    
  }
  
  #Spread
  if(is.null(X_N.s) & is.null(X_A.basis.s) & is.null(X_L.s)) {
    
    
      statBranchs <- input_u %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(u)[-1], trainable=F,
                                                weights=list(matrix(0,nrow=dim(u)[length(dim(u))],ncol=1),array(1,dim=c(1))), name = 's_stationary_dense1') %>%
        layer_dense(units = 1 ,activation = 'linear',use_bias = F,weights=list(matrix(array(init.spread),nrow=1,ncol=1)), name = 's_stationary_dense2')
   
    
  }
  #Spread
  if(is.null(X_N.xi) & is.null(X_A.basis.xi) & is.null(X_L.xi)) {
    
    
    statBranchxi <- input_u %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(u)[-1], trainable=F,
                                           weights=list(matrix(0,nrow=dim(u)[length(dim(u))],ncol=1),array(1,dim=c(1))), name = 'xi_stationary_dense1') %>%
      layer_dense(units = 1 ,activation = 'linear',use_bias = F,weights=list(matrix(array(init.xi),nrow=1,ncol=1)), name = 'xi_stationary_dense2')
    
    
  }
  #Combine towers
  
  
  #Location
  if(!is.null(X_N.q) & !is.null(X_A.basis.q) & !is.null(X_L.q) )  qBranchjoined <- layer_add(inputs=c(addBranchq,  linBranchq,nnBranchq))  #Add all towers
  if(is.null(X_N.q) & !is.null(X_A.basis.q) & !is.null(X_L.q) )  qBranchjoined <- layer_add(inputs=c(addBranchq,  linBranchq))  #Add GAM+lin towers
  if(!is.null(X_N.q) & is.null(X_A.basis.q) & !is.null(X_L.q) )  qBranchjoined <- layer_add(inputs=c(  linBranchq,nnBranchq))  #Add nn+lin towers
  if(!is.null(X_N.q) & !is.null(X_A.basis.q) & is.null(X_L.q) )  qBranchjoined <- layer_add(inputs=c(addBranchq,  nnBranchq))  #Add nn+GAM towers
  if(is.null(X_N.q) & is.null(X_A.basis.q) & !is.null(X_L.q) )  qBranchjoined <- linBranchq  #Just lin tower
  if(is.null(X_N.q) & !is.null(X_A.basis.q) & is.null(X_L.q) )  qBranchjoined <- addBranchq  #Just GAM tower
  if(!is.null(X_N.q) & is.null(X_A.basis.q) & is.null(X_L.q) )  qBranchjoined <- nnBranchq  #Just nn tower
  if(is.null(X_N.q) & is.null(X_A.basis.q) & is.null(X_L.q) )  qBranchjoined <- statBranchq  #Just stationary tower
  
  #Spread
  if(!is.null(X_N.s) & !is.null(X_A.basis.s) & !is.null(X_L.s) )  sBranchjoined <- layer_add(inputs=c(addBranchs,  linBranchs,nnBranchs)) #Add all towers
  if(is.null(X_N.s) & !is.null(X_A.basis.s) & !is.null(X_L.s) )  sBranchjoined <- layer_add(inputs=c(addBranchs,  linBranchs))  #Add GAM+lin towers
  if(!is.null(X_N.s) & is.null(X_A.basis.s) & !is.null(X_L.s) )  sBranchjoined <- layer_add(inputs=c(  linBranchs,nnBranchs))  #Add nn+lin towers
  if(!is.null(X_N.s) & !is.null(X_A.basis.s) & is.null(X_L.s) )  sBranchjoined <- layer_add(inputs=c(addBranchs,  nnBranchs))  #Add nn+GAM towers
  if(is.null(X_N.s) & is.null(X_A.basis.s) & !is.null(X_L.s) )  sBranchjoined <- linBranchs  #Just lin tower
  if(is.null(X_N.s) & !is.null(X_A.basis.s) & is.null(X_L.s) )  sBranchjoined <- addBranchs  #Just GAM tower
  if(!is.null(X_N.s) & is.null(X_A.basis.s) & is.null(X_L.s) )  sBranchjoined <- nnBranchs  #Just nn tower
  if(is.null(X_N.s) & is.null(X_A.basis.s) & is.null(X_L.s) )  sBranchjoined <- statBranchs  #Just stationary tower
  
  #xi
  if(!is.null(X_N.xi) & !is.null(X_A.basis.xi) & !is.null(X_L.xi) )  xiBranchjoined <- layer_add(inputs=c(addBranchxi,  linBranchxi,nnBranchxi)) #Add all towers
  if(is.null(X_N.xi) & !is.null(X_A.basis.xi) & !is.null(X_L.xi) )  xiBranchjoined <- layer_add(inputs=c(addBranchxi,  linBranchxi))  #Add GAM+lin towers
  if(!is.null(X_N.xi) & is.null(X_A.basis.xi) & !is.null(X_L.xi) )  xiBranchjoined <- layer_add(inputs=c(  linBranchxi,nnBranchxi))  #Add nn+lin towers
  if(!is.null(X_N.xi) & !is.null(X_A.basis.xi) & is.null(X_L.xi) )  xiBranchjoined <- layer_add(inputs=c(addBranchxi,  nnBranchxi))  #Add nn+GAM towers
  if(is.null(X_N.xi) & is.null(X_A.basis.xi) & !is.null(X_L.xi) )  xiBranchjoined <- linBranchxi #Just lin tower
  if(is.null(X_N.xi) & !is.null(X_A.basis.xi) & is.null(X_L.xi) )  xiBranchjoined <- addBranchxi  #Just GAM tower
  if(!is.null(X_N.xi) & is.null(X_A.basis.xi) & is.null(X_L.xi) )  xiBranchjoined <- nnBranchxi  #Just nn tower
  if(is.null(X_N.xi) & is.null(X_A.basis.xi) & is.null(X_L.xi) )  xiBranchjoined <- statBranchxi  #Just stationary tower
  
  #Apply link functions
  if(link.loc=="exp") qBranchjoined <- qBranchjoined %>% layer_activation( activation = 'exponential', name = "q_activation") else if(link.loc=="identity") qBranchjoined <- qBranchjoined %>% layer_activation( activation = 'linear', name = "q_activation")
  sBranchjoined <- sBranchjoined %>% layer_activation( activation = 'exponential', name = "s_activation")
  xiBranchjoined <- xiBranchjoined %>% layer_activation( activation = 'sigmoid', name = "xi_activation")
  
  input=c()
  if(!is.null(X_L.q) ) input=c(input,input_lin_q)
  if(!is.null(X_A.basis.q) ) input=c(input,input_add_q)
  if(!is.null(X_N.q) ) input=c(input,input_nn_q)
  if(!is.null(X_L.s) ) input=c(input,input_lin_s)
  if(!is.null(X_A.basis.s) ) input=c(input,input_add_s)
  if(!is.null(X_N.s) ) input=c(input,input_nn_s)
  if(!is.null(X_L.xi) ) input=c(input,input_lin_xi)
  if(!is.null(X_A.basis.xi) ) input=c(input,input_add_xi)
  if(!is.null(X_N.xi) ) input=c(input,input_nn_xi)
  input=c(input,input_u)


  output <- layer_concatenate(c(input_u,qBranchjoined,sBranchjoined, xiBranchjoined),name="Combine_parameter_tensors")

  model <- keras_model(  inputs = input,   outputs = output,name=paste0("bGEV-PP"))
  print(model)

  return(model)

}


l=function(a,xi){
  K <- backend()

  # K$exp(-xi*K$log(-K$log(a)))

  (-K$log(a))^(-xi)
}
l0 = function(a){
  K <- backend()

  K$log(-K$log(a))
}

Finverse = function(x,q_a,s_b,xi,alpha,beta){
  K <- backend()

  ( (-K$log(x))^(-xi)-l(alpha,xi))*s_b/(l(1-beta/2,xi)-l(beta/2,xi))+q_a
}

logH=function(y,q_a,s_b,xi,alpha,beta,a,b,p_a,p_b,c1,c2,obsInds){
  K <- backend()

  #Upper tail
  z1=(y-q_a)/(s_b/(l(1-beta/2,xi)-l(beta/2,xi)))+l(alpha,xi)
  z1=K$relu(z1)

  zeroz1_inds=1-K$sign(z1)
  t1=(z1+(1-obsInds)+zeroz1_inds)^(-1/xi)

  #Weight

  temp=(y-a)/(b-a)*obsInds #Need to set values <0 and >1 to 0 and 1, otherwise function breaks
  temp=K$relu(temp)
  temp=1-temp
  temp=K$relu(temp)
  temp=1-temp
  p =tf$math$betainc(c1,c2,temp)

  #Lower tail
  q_a_tilde=a-(b-a)*(l0(alpha)-l0(p_a))/(l0(p_a)-l0(p_b))
  s_b_tilde=(b-a)*(l0(beta/2)-l0(1-beta/2))/(l0(p_a)-l0(p_b))
  s_b_tilde =s_b_tilde + (1-obsInds)

  z2=(y-q_a_tilde)/(s_b_tilde/(l0(beta/2)-l0(1-beta/2)))-l0(alpha)
  z2=z2*obsInds

  t2=K$exp(-z2)




  return((p*(-t1)*(1-zeroz1_inds)+(1-p)*(-t2))*obsInds)
}

lambda=function(y,q_a,s_b,xi,alpha,beta,a,b,p_a,p_b,c1,c2,obsInds,exceedInds){
  K <- backend()
  #Upper tail
  z1=((y-q_a)/(s_b/(l(1-beta/2,xi)-l(beta/2,xi))))+l(alpha,xi)
  z1=K$relu(z1)
  z1=z1*exceedInds
  zeroz1_inds=1-K$sign(z1)
  t1=(z1+(1-obsInds)+zeroz1_inds)^(-1/xi)

  #Weight

  temp=(y-a)/(b-a)*obsInds #Need to set values <0 and >1 to 0 and 1, otherwise function breaks
  temp=K$relu(temp)
  temp=1-temp
  temp=K$relu(temp)
  temp=1-temp
  p =tf$math$betainc(c1,c2,temp)
  
  pprime = temp^(c1-1)*(1-temp)^(c2-1)/beta(c1,c2)
  pprime=pprime/(b-a)*exceedInds


  #Lower tail
  q_a_tilde=a-(b-a)*(l0(alpha)-l0(p_a))/(l0(p_a)-l0(p_b))
  s_b_tilde=(b-a)*(l0(beta/2)-l0(1-beta/2))/(l0(p_a)-l0(p_b))
  s_b_tilde=s_b_tilde+(1-obsInds)
  z2=(y-q_a_tilde)/(s_b_tilde/(l0(beta/2)-l0(1-beta/2)))-l0(alpha)
  z2=z2*exceedInds
  t2=K$exp(-z2)


  z1prime=(l(1-beta/2,xi)-l(beta/2,xi))/s_b

  z2prime=(l0(beta/2)-l0(1-beta/2))/s_b_tilde

  out=(
    (pprime*(-t1)
     +p*(1/xi)*t1/(z1+zeroz1_inds)*z1prime)*(1-zeroz1_inds)
    - pprime*(-t2)
    +(1-p)*t2*z2prime
  )
  return(
    out*exceedInds
  )
}

bgev_PP_loss <-function(alpha=0.5,beta=0.5,p_a=0.05,p_b=0.2,c1=5,c2=5,n_b=1){
  
 
loss<- function( y_true, y_pred) {

  library(tensorflow)
  K <- backend()

  u=y_pred[all_dims(),1]
  q_a=y_pred[all_dims(),2]
  s_b=y_pred[all_dims(),3]
  xi=y_pred[all_dims(),4]

  y <- y_true


  # Find inds of non-missing obs.  Remove missing obs, i.e., -1e10. This is achieved by adding an
  # arbitrarily large (<1e10) value to y and then taking the sign ReLu
  obsInds=K$sign(K$relu(y+9e9))

  #Find exceedance inds
  exceed=y-u
  exceedInds=K$sign(K$relu(exceed))


  a=Finverse(p_a,q_a,s_b,xi,alpha,beta)
  b=Finverse(p_b,q_a,s_b,xi,alpha,beta)
  b =b + (1-obsInds)
  s_b=s_b+(1-obsInds)

  #Use exceedance only only
  lam=lambda(y,q_a,s_b,xi,alpha,beta,a,b,p_a,p_b,c1,c2,obsInds,exceedInds)
  loglam=K$log(lam+(1-exceedInds))*exceedInds



  #Use all values of y i.e., non-exceedances + exceedances.

  LAM=-logH(u,q_a,s_b,xi,alpha,beta,a,b,p_a,p_b,c1,c2,obsInds) #1/12 as 12 obs per year
  return(-(
    K$sum(loglam)
    -(1/n_b)*K$sum(LAM)
  ))
}
 
  

return(loss)

}

