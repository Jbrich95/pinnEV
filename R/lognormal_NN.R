#'log-normal PINN
#'
#' Build and train a partially-interpretable neural network for fitting a log-normal model
#'

#' @param type  string defining the type of network to be built. If \code{type=="MLP"}, the network will have all densely connected layers; if \code{type=="CNN"},
#'  the network will have all convolutional layers (with 3 by 3 filters). Defaults to an MLP. (Currently the same network is used for all parameters, may change in future versions)
#' @param Y.train,Y.valid a 2 or 3 dimensional array of training or validation real response values.
#' Missing values can be handled by setting corresponding entries to \code{Y.train} or \code{Y.valid} to \code{-1e10}.
#' The first dimension should be the observation indices, e.g., time.
#'
#' If \code{type=="CNN"}, then \code{Y.train} and \code{Y.valid} must have three dimensions with the latter two corresponding to an \eqn{M} by \eqn{N} regular grid of spatial locations.
#' If \code{Y.valid==NULL}, no validation loss will be computed and the returned model will be that which minimises the training loss over \code{n.ep} epochs.
#'
#' @param X.mu  list of arrays corresponding to complementary subsets of the \eqn{d\geq 1} predictors which are used for modelling the location parameter \eqn{\mu}. Must contain at least one of the following three named entries:\describe{
#' \item{\code{X.lin.mu}}{A 3 or 4 dimensional array of "linear" predictor values. One more dimension than \code{Y.train}. If \code{NULL}, a model without the linear component is built and trained.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the last dimension corresponds to the chosen \eqn{l_1\geq 0} 'linear' predictor values.}
#' \item{\code{X.add.basis.mu}}{A 4 or 5 dimensional array of basis function evaluations for the "additive" predictor values.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the penultimate dimensions corresponds to the chosen \eqn{a_1\geq 0} 'linear' predictor values and the last dimension is equal to the number of knots used for estimating the splines. See example.
#' If \code{NULL}, a model without the additive component is built and trained.}
#' \item{\code{X.nn.mu}}{A 3 or 4 dimensional array of "non-additive" predictor values.  If \code{NULL}, a model without the NN component is built and trained; if this is the case, then \code{type} has no effect.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the last dimension corresponds to the chosen \eqn{d-l_1-a_1\geq 0} 'non-additive' predictor values.}
#' }
#' Note that \code{X.mu} and \code{X.sig} are the predictors for both \code{Y.train} and \code{Y.valid}. If \code{is.null(X.mu)}, then \eqn{\mu} will be treated as fixed over the predictors.
#' @param X.sig similarly to \code{X.mu}, but for modelling the shape parameter \eqn{\sigma>0}. Note that we require at least one of \code{!is.null(X.mu)} or \code{!is.null(X.sig)}, otherwise the formulated model will be fully stationary and will not be fitted.
#' @param n.ep number of epochs used for training. Defaults to 1000.
#' @param batch.size batch size for stochastic gradient descent. If larger than \code{dim(Y.train)[1]}, i.e., the number of observations, then regular gradient descent used.
#' @param init.loc,init.sig sets the initial \eqn{\mu} and \eqn{\sigma} estimates across all dimensions of \code{Y.train}. Overridden by \code{init.wb_path} if \code{!is.null(init.wb_path)}. Defaults to empirical estimates of mean and standard deviation, respectively, of \code{log(Y.train)}.
#' @param init.wb_path filepath to a \code{keras} model which is then used as initial weights and biases for training the new model. The original model must have
#' the exact same architecture and trained with the same input data as the new model. If \code{NULL}, then initial weights and biases are random (with seed \code{seed}) but the
#' final layer has zero initial weights to ensure that the initial location and shape estimates are \code{init.loc} and \code{init.sig}, respectively,  across all dimensions.
#' @param widths vector of widths/filters for hidden dense/convolution layers. Number of layers is equal to \code{length(widths)}. Defaults to (6,3).
#' @param seed seed for random initial weights and biases.
#' @param loc.link string defining the link function used for the location parameter, see \eqn{h_1} below. If \code{link=="exp"}, then \eqn{h_1=\exp(x)}; if \code{link=="identity"}, then \eqn{h_1(x)=x}.
#' @param model fitted \code{keras} model. Output from \code{bGEVPP.NN.train}.

#'@name lognormal.NN

#' @details{
#' Consider a real-valued random variable \eqn{Y} and let \eqn{\mathbf{X}} denote a \eqn{d}-dimensional predictor set with observations \eqn{\mathbf{x}}.
#' For \eqn{i=1,2}, we define integers \eqn{l_i\geq 0,a_i \geq 0} and \eqn{0\leq l_i+a_i \leq d}, and let \eqn{\mathbf{X}^{(i)}_L, \mathbf{X}^{(i)}_A} and \eqn{\mathbf{X}^{(i)}_N} be distinct sub-vectors
#' of \eqn{\mathbf{X}}, with observations of each component denoted \eqn{\mathbf{x}^{(i)}_L, \mathbf{x}^{(i)}_A} and \eqn{\mathbf{x}^{(i)}_N}, respectively; the lengths of the sub-vectors are \eqn{l_i,a_i} and \eqn{d_i-l_i-a}, respectively.
#' We model \eqn{Y|\mathbf{X}=\mathbf{x}\sim\mbox{Lognormal}(\mu(\mathbf{x}),\sigma(\mathbf{x}))} with \eqn{\sigma>0} and 
#' \deqn{\mu (\mathbf{x})=h_1\{\eta^{(1)}_0+m^{(1)}_L(\mathbf{x}^{(1)}_L)+m^{(1)}_A(x^{(1)}_A)+m^{(1)}_N(\mathbf{x}^{(1)}_N)\}} and
#' \deqn{\sigma (\mathbf{x})=\exp\{\eta^{(2)}_0+m^{(2)}_L(\mathbf{x}^{(2)}_L)+m^{(2)}_A(x^{(2)}_A)+m^{(2)}_N(\mathbf{x}^{(2)}_N)\}}
#' where \eqn{h_1} is some link-function and \eqn{\eta^{(1)}_0,\eta^{(2)}_0} are constant intercepts. The unknown functions \eqn{m^{(1)}_L,m^{(2)}_L} and
#' \eqn{m^{(1)}_A,m^{(2)}_A} are estimated using linear functions and splines, respectively, and are
#' both returned as outputs by \code{lognormal.NN.predict}; \eqn{m^{(1)}_N,m^{(2)}_N} are estimated using neural networks
#' (currently the same architecture is used for both parameters). 
#'
#' For details of the log-normal parameterisation, see \code{help(Lognormal)}. 
#'
#' The model is fitted by minimising the negative log-likelihood associated with the lognormal distribution; training is performed over \code{n.ep} training epochs.
#' Although the model is trained by minimising the loss evaluated for \code{Y.train}, the final returned model may minimise some other loss.
#' The current state of the model is saved after each epoch, using \code{keras::callback_model_checkpoint}, if the value of some criterion subcedes that of the model from the previous checkpoint; this criterion is the loss evaluated for validation set \code{Y.valid} if \code{!is.null(Y.valid)} and for \code{Y.train}, otherwise.
#'
#'}
#' @return \code{lognormal.NN.train} returns the fitted \code{model}.  \code{lognormal.NN.predict} is a wrapper for \code{keras::predict} that returns the predicted parameter estimates, and, if applicable, their corresponding linear regression coefficients and spline bases weights.
#'
#'@references{
#'
#' Richards, J. and Huser, R. (2024+), \emph{Regression modelling of spatiotemporal extreme U.S. wildfires via partially-interpretable neural networks}. (\href{https://arxiv.org/abs/2208.07581}{arXiv:2208.07581}).
#'}
#' @examples
#'
#' # Build and train a simple MLP for toy data
#'
#'set.seed(1)
#'
#' # Create  predictors
#' preds<-rnorm(prod(c(200,10,10,8)))
#'
#' #Re-shape to a 4d array. First dimension corresponds to observations,
#' #last to the different components of the predictor set.
#' #Other dimensions correspond to indices of predictors, e.g., a grid of locations.
#' dim(preds)=c(200,10,10,8) 
#' #We have 200 observations of eight predictors on a 10 by 10 grid.

#'
#' #Split predictors into linear, additive and nn. Different for the location and shape parameters.
#'X.nn.mu=preds[,,,1:4] #Four nn predictors for mu
#'X.lin.mu=preds[,,,5:6] #Two additive predictors for mu
#'X.add.mu=preds[,,,7:8] #Two additive predictors for mu
#'
#'X.nn.sig=preds[,,,1:2] #Two nn predictors for sigma
#'X.lin.sig=preds[,,,3] #One linear predictor for sigma
#'dim(X.lin.sig)=c(dim(X.lin.sig),1) #Change dimension so consistent
#'X.add.sig=preds[,,,4] #One additive predictor for sigma
#'dim(X.add.sig)=c(dim(X.add.sig),1) #Change dimension so consistent
#'
#'
#' # Create toy response data
#' 
#' #Contribution to location parameter
#' #Linear contribution
#' m_L_1 = 0.3*X.lin.mu[,,,1]+0.6*X.lin.mu[,,,2]
#' 
#' # Additive contribution
#' m_A_1 = 0.05*X.add.mu[,,,1]^3+0.2*X.add.mu[,,,1]-
#'  0.05*X.add.mu[,,,2]^3+0.5*X.add.mu[,,,2]^2
#' 
#' #Non-additive contribution - to be estimated by NN
#' m_N_1 = 0.25*exp(-3+X.nn.mu[,,,4]+X.nn.mu[,,,1])+
#'   sin(X.nn.mu[,,,1]-X.nn.mu[,,,2])*(X.nn.mu[,,,4]+X.nn.mu[,,,2])-
#'   cos(X.nn.mu[,,,4]-X.nn.mu[,,,1])*(X.nn.mu[,,,3]+X.nn.mu[,,,1])
#' 
#' mu=m_L_1+m_A_1+m_N_1 #Identity link
#' 
#' #Contribution to shape parameter
#' #Linear contribution
#' m_L_2 = 0.5*X.lin.sig[,,,1]
#' 
#' # Additive contribution
#' m_A_2 = 0.1*X.add.sig[,,,1]^2+0.2*X.add.sig[,,,1]
#' 
#' #Non-additive contribution - to be estimated by NN
#' m_N_2 = 0.1*exp(-4+X.nn.sig[,,,2]+X.nn.sig[,,,1])+
#'   0.1* sin(X.nn.sig[,,,1]-X.nn.sig[,,,2])*(X.nn.sig[,,,1]+X.nn.sig[,,,2])
#' 
#' sig=0.1*exp(m_L_2+m_A_2+m_N_2) #Exponential link
#' 
#' 
#' theta=array(dim=c(dim(sig),2))
#' theta[,,,1]=mu; theta[,,,2] = sig
#' #We simulate data from a lognormal distribution
#' 
#' Y=apply(theta,1:3,function(x) rlnorm(1,meanlog =x[1],sdlog=x[2]))
#'
#' #Create training and validation, respectively.
#' #We mask 20% of the Y values and use this for validation
#' #Masked values must be set to -1e10 and are treated as missing whilst training
#'
#' mask_inds=sample(1:length(Y),size=length(Y)*0.8)
#'
#' Y.train<-Y.valid<-Y #Create training and validation, respectively.
#' Y.train[-mask_inds]=-1e10
#' Y.valid[mask_inds]=-1e10
#'
#'
#'
#' #To build a model with an additive component, we require an array of evaluations of
#' #the basis functions for each pre-specified knot and entry to X.add.mu and X.add.sig
#'
#'rad=function(x,c){ #Define a basis function. Here we use the radial bases
#'   out=abs(x-c)^2*log(abs(x-c))
#'   out[(x-c)==0]=0
#'  return(out)
#' }
#'
#'n.knot.mu = 5; n.knot.sig = 4 # set number of knots.
#'#Must be the same for each additive predictor,
#'#but can differ between the parameters mu and sigma
#'
#' #Get knots for mu predictors
#' knots.mu=matrix(nrow=dim(X.add.mu)[4],ncol=n.knot.mu)
#'
#' #We set knots to be equally-spaced marginal quantiles
#' for( i in 1:dim(X.add.mu)[4]){
#' knots.mu[i,]=quantile(X.add.mu[,,,i],probs=seq(0,1,length=n.knot.mu))
#' }
#' #Evaluate radial basis functions for mu predictors
#' X.add.basis.mu<-array(dim=c(dim(X.add.mu),n.knot.mu))
#' for( i in 1:dim(X.add.mu)[4]) {
#'   for(k in 1:n.knot.mu) {
#'     X.add.basis.mu[,,,i,k]= rad(x=X.add.mu[,,,i],c=knots.mu[i,k])
#'     #Evaluate rad at all entries to X.add.mu and for all knots
#'   }}
#'   
#'
#' #Get knots for sigma predictor
#' 
#' knots.sig=matrix(nrow=dim(X.add.sig)[4],ncol=n.knot.sig)
#' 
#' for( i in 1:dim(X.add.sig)[4]){
#'  knots.sig[i,]=quantile(X.add.sig[,,,i],probs=seq(0,1,length=n.knot.sig))
#' }
#'
#' #Evaluate radial basis functions for sigma predictor
#' 
#' X.add.basis.sig<-array(dim=c(dim(X.add.sig),n.knot.sig))
#' 
#' for( i in 1:dim(X.add.sig)[4]) {
#'   for(k in 1:n.knot.sig) {
#'     X.add.basis.sig[,,,i,k]= rad(x=X.add.sig[,,,i],c=knots.sig[i,k])
#'     #Evaluate rad at all entries to X.add.mu and for all knots
#'   }}
#'
#'
#' #lin+GAM+NN models defined for both location and scale parameters
#' X.mu=list("X.nn.mu"=X.nn.mu, "X.lin.mu"=X.lin.mu,
#'                "X.add.basis.mu"=X.add.basis.mu) #Predictors for mu
#' X.sig=list("X.nn.sig"=X.nn.sig, "X.lin.sig"=X.lin.sig,
#'                "X.add.basis.sig"=X.add.basis.sig) #Predictors for sigma
#'
#'
#' #Fit the log-normal model. Note that training is not run to completion.
#' NN.fit<-lognormal.NN.train(Y.train, Y.valid,X.mu,X.sig, type="MLP",link.loc="identity",
#'                           n.ep=50, batch.size=50,
#'                           widths=c(6,3),seed=1)
#' out<-lognormal.NN.predict(X.mu=X.mu,X.sig=X.sig,NN.fit$model)
#'
#' print("mu linear coefficients: "); print(round(out$lin.coeff_loc,3))
#' print("sig linear coefficients: "); print(round(out$lin.coeff_sig,3))
#' 
#' # Note that this is a simple example that can be run in a personal computer. 
#' 
#' ## To save model, run
#' # NN.fit$model %>% save_model_tf("model_lognormal")
#' ## To load model, run
#' #model  <- load_model_tf("model_lognormal",
#' #                        custom_objects=list(
#' #                          "lognormal_loss__"=
#' #                            lognormal_loss())
#' #)
#'
#'
#'
#' # Plot splines for the additive predictors
#'
#' #Location predictors
#' n.add.preds_loc=dim(X.add.mu)[length(dim(X.add.mu))]
#' par(mfrow=c(1,n.add.preds_loc))
#' for(i in 1:n.add.preds_loc){
#'   plt.x=seq(from=min(knots.mu[i,]),to=max(knots.mu[i,]),length=1000)  #Create sequence for x-axis
#'
#'   tmp=matrix(nrow=length(plt.x),ncol=n.knot.mu)
#'   for(j in 1:n.knot.mu){
#'     tmp[,j]=rad(plt.x,knots.mu[i,j]) #Evaluate radial basis function of plt.x and all knots
#'   }
#'   plt.y=tmp%*%out$gam.weights_loc[i,]
#'   plot(plt.x,plt.y,type="l",main=paste0("mu spline: predictor ",i),xlab="x",ylab="f(x)")
#'   points(knots.mu[i,],rep(mean(plt.y),n.knot.mu),col="red",pch=2)
#'   #Adds red triangles that denote knot locations
#'
#' }
#'
#' #Shape predictors
#' n.add.preds_sig=dim(X.add.sig)[length(dim(X.add.sig))]
#' par(mfrow=c(1,n.add.preds_sig))
#' for(i in 1:n.add.preds_sig){
#'   plt.x=seq(from=min(knots.sig[i,]),to=max(knots.sig[i,]),length=1000)  #Create sequence for x-axis
#'
#'   tmp=matrix(nrow=length(plt.x),ncol=n.knot.sig)
#'   for(j in 1:n.knot.sig){
#'     tmp[,j]=rad(plt.x,knots.sig[i,j]) #Evaluate radial basis function of plt.x and all knots
#'   }
#'   plt.y=tmp%*%out$gam.weights_sig[i,]
#'   plot(plt.x,plt.y,type="l",main=paste0("sigma spline: predictor ",i),xlab="x",ylab="f(x)")
#'   points(knots.sig[i,],rep(mean(plt.y),n.knot.sig),col="red",pch=2)
#'   #Adds red triangles that denote knot locations
#'
#' }
#'
#' @import reticulate keras tensorflow 
#' @rdname lognormal.NN
#' @export

lognormal.NN.train=function(Y.train, Y.valid = NULL,X.mu,X.sig, type="MLP",link.loc="identity",
                       n.ep=100, batch.size=100,init.loc=NULL, init.sig=NULL,
                       widths=c(6,3),seed=NULL,init.wb_path=NULL)
{
  
  
  if(min(Y.train[Y.train > -1e4])<=0 | min(Y.valid[Y.valid > -1e4])<=0 ) stop("Reponse data must be strictly positive!")
  
  
  if(is.null(X.mu) &  is.null(X.sig)  ) stop("No predictors provided for mu or sigma: Stationary models are not permitted ")
  if(is.null(Y.train)) stop("No training response data provided")
  
  if(is.null(init.loc) & is.null(init.wb_path)  ) init.loc=mean(log(Y.train[Y.train>0]))
  if(is.null(init.sig) & is.null(init.wb_path)   ) init.sig=sd(log(Y.train[Y.train>0]))
  
  
  print(paste0("Creating log-normal model"))
  X.nn.mu=X.mu$X.nn.mu
  X.lin.mu=X.mu$X.lin.mu
  X.add.basis.mu=X.mu$X.add.basis.mu
  
  
  if(!is.null(X.nn.mu) & !is.null(X.add.basis.mu) & !is.null(X.lin.mu) ) {  train.data= list(X.lin.mu,X.add.basis.mu,X.nn.mu); print("Defining lin+GAM+NN model for mu" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_loc=X.lin.mu,add_input_loc=X.add.basis.mu,  nn_input_loc=X.nn.mu),Y.valid)}
  if(is.null(X.nn.mu) & !is.null(X.add.basis.mu) & !is.null(X.lin.mu) ) {   train.data= list(X.lin.mu,X.add.basis.mu); print("Defining lin+GAM model for mu" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_loc=X.lin.mu,add_input_loc=X.add.basis.mu),Y.valid)}
  if(!is.null(X.nn.mu) & is.null(X.add.basis.mu) & !is.null(X.lin.mu) ) { train.data= list(X.lin.mu,X.nn.mu); print("Defining lin+NN model for mu" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_loc=X.lin.mu, nn_input_loc=X.nn.mu),Y.valid)}
  if(!is.null(X.nn.mu) & !is.null(X.add.basis.mu) & is.null(X.lin.mu) ) {train.data= list(X.add.basis.mu,X.nn.mu); print("Defining GAM+NN model for mu" );  if(!is.null(Y.valid)) validation.data=list(list(add_input_loc=X.add.basis.mu,  nn_input_loc=X.nn.mu),Y.valid)}
  if(is.null(X.nn.mu) & is.null(X.add.basis.mu) & !is.null(X.lin.mu) )   {train.data= list(X.lin.mu); print("Defining fully-linear model for mu" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_loc=X.lin.mu),Y.valid)}
  if(is.null(X.nn.mu) & !is.null(X.add.basis.mu) & is.null(X.lin.mu) )   {train.data= list(X.add.basis.mu); print("Defining fully-additive model for mu" );  if(!is.null(Y.valid)) validation.data=list(list(add_input_loc=X.add.basis.mu),Y.valid)}
  if(!is.null(X.nn.mu) & is.null(X.add.basis.mu) & is.null(X.lin.mu) )   {train.data= list(X.nn.mu); print("Defining fully-NN model for mu" );  if(!is.null(Y.valid)) validation.data=list(list( nn_input_loc=X.nn.mu),Y.valid)}
  if(is.null(X.nn.mu) & is.null(X.add.basis.mu) & is.null(X.lin.mu) )   {train.data= list(); print("Defining stationary model for mu" );  if(!is.null(Y.valid)) validation.data=list(list( ),Y.valid)}

  
  X.nn.sig=X.sig$X.nn.sig
  X.lin.sig=X.sig$X.lin.sig
  X.add.basis.sig=X.sig$X.add.basis.sig
  
  if(!is.null(X.nn.sig) & !is.null(X.add.basis.sig) & !is.null(X.lin.sig) ) {  train.data= c(train.data,list(X.lin.sig,X.add.basis.sig,X.nn.sig)); print("Defining lin+GAM+NN model for sigma" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X.lin.sig,add_input_s=X.add.basis.sig,  nn_input_s=X.nn.sig)),Y.valid)}
  if(is.null(X.nn.sig) & !is.null(X.add.basis.sig) & !is.null(X.lin.sig) ) {   train.data= c(train.data,list(X.lin.sig,X.add.basis.sig)); print("Defining lin+GAM model for sigma" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X.lin.sig,add_input_s=X.add.basis.sig)),Y.valid)}
  if(!is.null(X.nn.sig) & is.null(X.add.basis.sig) & !is.null(X.lin.sig) ) { train.data= c(train.data,list(X.lin.sig,X.nn.sig)); print("Defining lin+NN model for sigma" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X.lin.sig, nn_input_s=X.nn.sig)),Y.valid)}
  if(!is.null(X.nn.sig) & !is.null(X.add.basis.sig) & is.null(X.lin.sig) ) {train.data= c(train.data,list(X.add.basis.sig,X.nn.sig)); print("Defining GAM+NN model for sigma" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(add_input_s=X.add.basis.sig,  nn_input_s=X.nn.sig)),Y.valid)}
  if(is.null(X.nn.sig) & is.null(X.add.basis.sig) & !is.null(X.lin.sig) )   {train.data= c(train.data,list(X.lin.sig)); print("Defining fully-linear model for sigma" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X.lin.sig)),Y.valid)}
  if(is.null(X.nn.sig) & !is.null(X.add.basis.sig) & is.null(X.lin.sig) )   {train.data= c(train.data,list(X.add.basis.sig)); print("Defining fully-additive model for sigma" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(add_input_s=X.add.basis.sig)),Y.valid)}
  if(!is.null(X.nn.sig) & is.null(X.add.basis.sig) & is.null(X.lin.sig) )   {train.data= c(train.data,list(X.nn.sig)); print("Defining fully-NN model for sigma" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(nn_input_s=X.nn.sig)),Y.valid)}
  if(is.null(X.nn.sig) & is.null(X.add.basis.sig) & is.null(X.lin.sig) )   {train.data= train.data; print("Defining stationary model for sigma" );  if(!is.null(Y.valid)) validation.data=validation.data}
  

  if(type=="CNN" & (!is.null(X.nn.mu) | !is.null(X.nn.sig)))print(paste0("Building ",length(widths),"-layer convolutional neural network with ", filter.dim[1]," by ", filter.dim[2]," filter" ))
  if(type=="MLP"  & (!is.null(X.nn.mu) | !is.null(X.nn.sig)) ) print(paste0("Building ",length(widths),"-layer densely-connected neural network" ))
  
  reticulate::use_virtualenv("pinnEV_env", required = T)
  
  if(!is.null(seed)) tf$random$set_seed(seed)
  
  model<-lognormal.NN.build(X.nn.mu,X.lin.mu,X.add.basis.mu,
                       X.nn.sig,X.lin.sig,X.add.basis.sig,
                       type, init.loc,init.sig, widths,link.loc)
  
  if(!is.null(init.wb_path)) model <- load_model_weights_tf(model,filepath=init.wb_path)
  
  model %>% compile(
    optimizer="adam",
    loss = lognormal_loss(),
    run_eagerly=T
  )
  
  
  if(!is.null(Y.valid)) checkpoint <- callback_model_checkpoint(paste0("model_lognormal_checkpoint"), monitor = "val_loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch") else checkpoint <- callback_model_checkpoint(paste0("model_lognormal_checkpoint"), monitor = "loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch")
  .GlobalEnv$model <- model
  if(!is.null(Y.valid)){
    history <- model %>% keras::fit(
      x=train.data, y=Y.train,
      epochs = n.ep, batch_size = batch.size,
      callback=list(checkpoint),
      validation_data=validation.data
      
    )
  }else{
    
    history <- model %>% keras::fit(
      x=train.data, y=Y.train,
      epochs = n.ep, batch_size = batch.size,
      callback=list(checkpoint)
    )
  }
  
  print("Loading checkpoint weights")
  model <- load_model_weights_tf(model,filepath=paste0("model_lognormal_checkpoint"))
  print("Final training loss")
  loss.train<-model %>% evaluate(train.data,Y.train, batch_size=batch.size)
  if(!is.null(Y.valid)){
    print("Final validation loss")
    loss.valid<-model %>% evaluate(train.data,Y.valid, batch_size=batch.size)
    return(list("model"=model,"Training loss"=loss.train, "Validation loss"=loss.valid))
  }else{
    return(list("model"=model,"Training loss"=loss.train))
  }
  
  
}
#' @rdname lognormal.NN
#' @export
#'
lognormal.NN.predict=function(X.mu,X.sig, model)
{
  library(tensorflow)
  if(is.null(X.mu) &  is.null(X.sig)  ) stop("No predictors provided for mu or sigma: Stationary models are not permitted ")
  
  
  
  X.nn.mu=X.mu$X.nn.mu
  X.lin.mu=X.mu$X.lin.mu
  X.add.basis.mu=X.mu$X.add.basis.mu
  
  
  if(!is.null(X.nn.mu) & !is.null(X.add.basis.mu) & !is.null(X.lin.mu) )   train.data= list(X.lin.mu,X.add.basis.mu,X.nn.mu)
  if(is.null(X.nn.mu) & !is.null(X.add.basis.mu) & !is.null(X.lin.mu) )   train.data= list(X.lin.mu,X.add.basis.mu)
  if(!is.null(X.nn.mu) & is.null(X.add.basis.mu) & !is.null(X.lin.mu) )  train.data= list(X.lin.mu,X.nn.mu)
  if(!is.null(X.nn.mu) & !is.null(X.add.basis.mu) & is.null(X.lin.mu) ) train.data= list(X.add.basis.mu,X.nn.mu)
  if(is.null(X.nn.mu) & is.null(X.add.basis.mu) & !is.null(X.lin.mu) )   train.data= list(X.lin.mu)
  if(is.null(X.nn.mu) & !is.null(X.add.basis.mu) & is.null(X.lin.mu) )   train.data= list(X.add.basis.mu)
  if(!is.null(X.nn.mu) & is.null(X.add.basis.mu) & is.null(X.lin.mu) )   train.data= list(X.nn.mu)
  if(is.null(X.nn.mu) & is.null(X.add.basis.mu) & is.null(X.lin.mu) )   train.data= list()
  
  X.nn.sig=X.sig$X.nn.sig
  X.lin.sig=X.sig$X.lin.sig
  X.add.basis.sig=X.sig$X.add.basis.sig
  
  if(!is.null(X.nn.sig) & !is.null(X.add.basis.sig) & !is.null(X.lin.sig) )   train.data= c(train.data,list(X.lin.sig,X.add.basis.sig,X.nn.sig))
  if(is.null(X.nn.sig) & !is.null(X.add.basis.sig) & !is.null(X.lin.sig) )   train.data= c(train.data,list(X.lin.sig,X.add.basis.sig))
  if(!is.null(X.nn.sig) & is.null(X.add.basis.sig) & !is.null(X.lin.sig) )  train.data= c(train.data,list(X.lin.sig,X.nn.sig))
  if(!is.null(X.nn.sig) & !is.null(X.add.basis.sig) & is.null(X.lin.sig) ) train.data= c(train.data,list(X.add.basis.sig,X.nn.sig))
  if(is.null(X.nn.sig) & is.null(X.add.basis.sig) & !is.null(X.lin.sig) )   train.data= c(train.data,list(X.lin.sig))
  if(is.null(X.nn.sig) & !is.null(X.add.basis.sig) & is.null(X.lin.sig) )   train.data= c(train.data,list(X.add.basis.sig))
  if(!is.null(X.nn.sig) & is.null(X.add.basis.sig) & is.null(X.lin.sig) ) train.data= c(train.data,list(X.nn.sig))
  if(is.null(X.nn.sig) & is.null(X.add.basis.sig) & is.null(X.lin.sig) ) train.data= train.data
  
  
  predictions<-model %>% predict( train.data)
  predictions <- k_constant(predictions)
  pred.loc=k_get_value(predictions[all_dims(),1])
  pred.sig=k_get_value(predictions[all_dims(),2])

  if(!is.null(X.add.basis.mu))  gam.weights_loc<-matrix(t(model$get_layer("add_loc")$get_weights()[[1]]),nrow=dim(X.add.basis.mu)[length(dim(X.add.basis.mu))-1],ncol=dim(X.add.basis.mu)[length(dim(X.add.basis.mu))],byrow=T)
  if(!is.null(X.add.basis.sig))  gam.weights_sig<-matrix(t(model$get_layer("add_s")$get_weights()[[1]]),nrow=dim(X.add.basis.sig)[length(dim(X.add.basis.sig))-1],ncol=dim(X.add.basis.sig)[length(dim(X.add.basis.sig))],byrow=T)
  
  out=list("pred.loc"=pred.loc,"pred.sig"=pred.sig)
  if(!is.null(X.lin.mu) ) out=c(out,list("lin.coeff_loc"=c(model$get_layer("lin_loc")$get_weights()[[1]])))
  if(!is.null(X.lin.sig) ) out=c(out,list("lin.coeff_sig"=c(model$get_layer("lin_s")$get_weights()[[1]])))
  if(!is.null(X.add.basis.mu) ) out=c(out,list("gam.weights_loc"=gam.weights_loc))
  if(!is.null(X.add.basis.sig) ) out=c(out,list("gam.weights_sig"=gam.weights_sig))
  
  return(out)
  
}
#'
#'
lognormal.NN.build=function(X.nn.mu,X.lin.mu,X.add.basis.mu,
                       X.nn.sig,X.lin.sig,X.add.basis.sig,
                       type, init.loc,init.sig, widths,link.loc)
{
  #Additive inputs
  if(!is.null(X.add.basis.mu))  input_add_loc<- layer_input(shape = dim(X.add.basis.mu)[-1], name = 'add_input_loc')
  if(!is.null(X.add.basis.sig))  input_add_s<- layer_input(shape = dim(X.add.basis.sig)[-1], name = 'add_input_s')
  
  #NN input
  
  if(!is.null(X.nn.mu))   input_nn_loc <- layer_input(shape = dim(X.nn.mu)[-1], name = 'nn_input_loc')
  if(!is.null(X.nn.sig))   input_nn_s <- layer_input(shape = dim(X.nn.sig)[-1], name = 'nn_input_s')
  
  #Linear input
  
  if(!is.null(X.lin.mu)) input_lin_loc <- layer_input(shape = dim(X.lin.mu)[-1], name = 'lin_input_loc')
  if(!is.null(X.lin.sig)) input_lin_s <- layer_input(shape = dim(X.lin.sig)[-1], name = 'lin_input_s')
  

  
  if(link.loc=="exp") init.loc=log(init.loc) else if(link.loc =="identity") init.loc=init.loc else stop("Invalid link function for location parameter")
  init.sig=log(init.sig)
  
  #NN towers
  
  #Location
  if(!is.null(X.nn.mu)){
    
    nunits=c(widths,1)
    n.layers=length(nunits)-1
    
    nnBranchloc <- input_nn_loc
    if(type=="MLP"){
      for(i in 1:n.layers){
        nnBranchloc <- nnBranchloc  %>% layer_dense(units=nunits[i],activation = 'relu',
                                                input_shape =dim(X.nn.mu)[-1], name = paste0('nn_loc_dense',i) )
      }
    }else if(type=="CNN"){
      for(i in 1:n.layers){
        nnBranchloc <- nnBranchloc  %>% layer_conv_2d(filters=nunits[i],activation = 'relu',kernel_size=c(3,3), padding='same',
                                                  input_shape =dim(X.nn.mu)[-1], name = paste0('nn_loc_cnn',i) )
      }
      
    }
    
    nnBranchloc <-   nnBranchloc  %>%   layer_dense(units = nunits[n.layers+1], activation = "linear", name = 'nn_loc_dense_final',
                                                weights=list(matrix(0,nrow=nunits[n.layers],ncol=1), array(init.loc)))
    
  }
  #Shape
  if(!is.null(X.nn.sig)){
    
    nunits=c(widths,1)
    n.layers=length(nunits)-1
    
    nnBranchs <- input_nn_s
    if(type=="MLP"){
      for(i in 1:n.layers){
        nnBranchs <- nnBranchs  %>% layer_dense(units=nunits[i],activation = 'relu',
                                                input_shape =dim(X.nn.sig)[-1], name = paste0('nn_s_dense',i) )
      }
    }else if(type=="CNN"){
      for(i in 1:n.layers){
        nnBranchs <- nnBranchs  %>% layer_conv_2d(filters=nunits[i],activation = 'relu',kernel_size=c(3,3), padding='same',
                                                  input_shape =dim(X.nn.sig)[-1], name = paste0('nn_s_cnn',i) )
      }
      
    }
    
    nnBranchs <-   nnBranchs  %>%   layer_dense(units = nunits[n.layers+1], activation = "linear", name = 'nn_s_dense_final',
                                                weights=list(matrix(0,nrow=nunits[n.layers],ncol=1), array(init.sig)))
    
  }
  #Additive towers
  #Location
  n.dim.add_loc=length(dim(X.add.basis.mu))
  if(!is.null(X.add.basis.mu) & !is.null(X.add.basis.mu) ) {
    
    addBranchloc <- input_add_loc %>%
      layer_reshape(target_shape=c(dim(X.add.basis.mu)[2:(n.dim.add_loc-2)],prod(dim(X.add.basis.mu)[(n.dim.add_loc-1):n.dim.add_loc]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_loc',
                  weights=list(matrix(0,nrow=prod(dim(X.add.basis.mu)[(n.dim.add_loc-1):n.dim.add_loc]),ncol=1)),use_bias = F)
  }
  if(!is.null(X.add.basis.mu) & is.null(X.add.basis.mu) ) {
    
    addBranchloc <- input_add_loc %>%
      layer_reshape(target_shape=c(dim(X.add.basis.mu)[2:(n.dim.add_loc-2)],prod(dim(X.add.basis.mu)[(n.dim.add_loc-1):n.dim.add_loc]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_loc',
                  weights=list(matrix(0,nrow=prod(dim(X.add.basis.mu)[(n.dim.add_loc-1):n.dim.add_loc]),ncol=1),array(init.loc)),use_bias = T)
  }
  #Shape
  n.dim.add_s=length(dim(X.add.basis.sig))
  if(!is.null(X.add.basis.sig) & !is.null(X.add.basis.sig) ) {
    
    addBranchs <- input_add_s %>%
      layer_reshape(target_shape=c(dim(X.add.basis.sig)[2:(n.dim.add_s-2)],prod(dim(X.add.basis.sig)[(n.dim.add_s-1):n.dim.add_s]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_s',
                  weights=list(matrix(0,nrow=prod(dim(X.add.basis.sig)[(n.dim.add_s-1):n.dim.add_s]),ncol=1)),use_bias = F)
  }
  if(!is.null(X.add.basis.sig) & is.null(X.add.basis.sig) ) {
    
    addBranchs <- input_add_s %>%
      layer_reshape(target_shape=c(dim(X.add.basis.sig)[2:(n.dim.add_s-2)],prod(dim(X.add.basis.sig)[(n.dim.add_s-1):n.dim.add_s]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_s',
                  weights=list(matrix(0,nrow=prod(dim(X.add.basis.sig)[(n.dim.add_s-1):n.dim.add_s]),ncol=1),array(init.sig)),use_bias = T)
  }
  #Linear towers
  
  #Location
  if(!is.null(X.lin.mu) ) {
    n.dim.lin_loc=length(dim(X.lin.mu))
    
    if(is.null(X.nn.mu) & is.null(X.add.basis.mu )){
      linBranchloc <- input_lin_loc%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X.lin.mu)[-1], name = 'lin_loc',
                    weights=list(matrix(0,nrow=dim(X.lin.mu)[n.dim.lin_loc],ncol=1),array(init.loc)),use_bias=T)
    }else{
      linBranchloc <- input_lin_loc%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X.lin.mu)[-1], name = 'lin_loc',
                    weights=list(matrix(0,nrow=dim(X.lin.mu)[n.dim.lin_loc],ncol=1)),use_bias=F)
    }
  }
  #Shape
  if(!is.null(X.lin.sig) ) {
    n.dim.lin_s=length(dim(X.lin.sig))
    
    if(is.null(X.nn.sig) & is.null(X.add.basis.sig )){
      linBranchs <- input_lin_s%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X.lin.sig)[-1], name = 'lin_s',
                    weights=list(matrix(0,nrow=dim(X.lin.sig)[n.dim.lin_s],ncol=1),array(init.sig)),use_bias=T)
    }else{
      linBranchs <- input_lin_s%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X.lin.sig)[-1], name = 'lin_s',
                    weights=list(matrix(0,nrow=dim(X.lin.sig)[n.dim.lin_s],ncol=1)),use_bias=F)
    }
  }
  
  #Stationary towers
  
  #Location
  if(is.null(X.nn.mu) & is.null(X.add.basis.mu) & is.null(X.lin.mu)) {
    
    if(!is.null(X.nn.sig)){
      statBranchloc <- input_nn_s %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X.nn.sig)[-1], trainable=F,
                                                weights=list(matrix(0,nrow=dim(X.nn.sig)[length(dim(X.nn.sig))],ncol=1),array(1,dim=c(1))), name = 'q_stationary_dense1') %>%
        layer_dense(units = 1 ,activation = 'linear',use_bias = F,weights=list(matrix(array(init.loc),nrow=1,ncol=1)), name = 'q_stationary_dense2')
    }else  if(!is.null(X.lin.sig)){
      statBranchloc <- input_lin_s %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X.lin.sig)[-1], trainable=F,
                                                 weights=list(matrix(0,nrow=dim(X.lin.sig)[length(dim(X.lin.sig))],ncol=1),array(1,dim=c(1))), name = 'q_stationary_dense1') %>%
        layer_dense(units = 1 ,activation = 'linear',use_bias = F,weights=list(matrix(array(init.loc),nrow=1,ncol=1)), name = 'q_stationary_dense2')
    }else  if(!is.null(X.add.basis.sig)){
      statBranchloc <- input_add_s %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X.add.basis.sig)[-1], trainable=F,
                                                 weights=list(matrix(0,nrow=dim(X.add.basis.sig)[length(dim(X.add.basis.sig))],ncol=1),array(1,dim=c(1))), name = 'q_stationary_dense1') %>%
        layer_dense(units = 1 ,activation = 'linear',use_bias = F,weights=list(matrix(array(init.loc),nrow=1,ncol=1)), name = 'q_stationary_dense2')
    }
    
  }
  
  #Shape
  if(is.null(X.nn.sig) & is.null(X.add.basis.sig) & is.null(X.lin.sig)) {
    
    if(!is.null(X.nn.mu)){
      statBranchs <- input_nn_loc %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X.nn.mu)[-1], trainable=F,
                                                weights=list(matrix(0,nrow=dim(X.nn.mu)[length(dim(X.nn.mu))],ncol=1),array(1,dim=c(1))), name = 's_stationary_dense1') %>%
        layer_dense(units = 1 ,activation = 'linear',use_bias = F,weights=list(matrix(array(init.sig),nrow=1,ncol=1)), name = 's_stationary_dense2')
    }else  if(!is.null(X.lin.mu)){
      statBranchs <- input_lin_loc %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X.nn.mu)[-1], trainable=F,
                                                 weights=list(matrix(0,nrow=dim(X.nn.mu)[length(dim(X.nn.mu))],ncol=1),array(1,dim=c(1))), name = 's_stationary_dense1') %>%
        layer_dense(units = 1 ,activation = 'linear',use_bias = F,weights=list(matrix(array(init.sig),nrow=1,ncol=1)), name = 's_stationary_dense2')
    }else  if(!is.null(X.add.basis.mu)){
      statBranchs <- input_add_loc %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X.nn.mu)[-1], trainable=F,
                                                 weights=list(matrix(0,nrow=dim(X.nn.mu)[length(dim(X.nn.mu))],ncol=1),array(1,dim=c(1))), name = 's_stationary_dense1') %>%
        layer_dense(units = 1 ,activation = 'linear',use_bias = F,weights=list(matrix(array(init.sig),nrow=1,ncol=1)), name = 's_stationary_dense2')
    }
    
  }
  
  
  #Combine towers
  
  #Location
  if(!is.null(X.nn.mu) & !is.null(X.add.basis.mu) & !is.null(X.lin.mu) )  locBranchjoined <- layer_add(inputs=c(addBranchloc,  linBranchloc,nnBranchloc))  #Add all towers
  if(is.null(X.nn.mu) & !is.null(X.add.basis.mu) & !is.null(X.lin.mu) )  locBranchjoined <- layer_add(inputs=c(addBranchloc,  linBranchloc)) #Add GAM+lin towers
  if(!is.null(X.nn.mu) & is.null(X.add.basis.mu) & !is.null(X.lin.mu) )  locBranchjoined <- layer_add(inputs=c(  linBranchloc,nnBranchloc))  #Add nn+lin towers
  if(!is.null(X.nn.mu) & !is.null(X.add.basis.mu) & is.null(X.lin.mu) )  locBranchjoined <- layer_add(inputs=c(addBranchloc,  nnBranchloc))  #Add nn+GAM towers
  if(is.null(X.nn.mu) & is.null(X.add.basis.mu) & !is.null(X.lin.mu) )  locBranchjoined <- linBranchloc  #Just lin tower
  if(is.null(X.nn.mu) & !is.null(X.add.basis.mu) & is.null(X.lin.mu) )  locBranchjoined <- addBranchloc  #Just GAM tower
  if(!is.null(X.nn.mu) & is.null(X.add.basis.mu) & is.null(X.lin.mu) )  locBranchjoined <- nnBranchloc  #Just nn tower
  if(is.null(X.nn.mu) & is.null(X.add.basis.mu) & is.null(X.lin.mu) )  locBranchjoined <- statBranchloc  #Just stationary tower
  
  #Shape
  if(!is.null(X.nn.sig) & !is.null(X.add.basis.sig) & !is.null(X.lin.sig) )  sBranchjoined <- layer_add(inputs=c(addBranchs,  linBranchs,nnBranchs))  #Add all towers
  if(is.null(X.nn.sig) & !is.null(X.add.basis.sig) & !is.null(X.lin.sig) )  sBranchjoined <- layer_add(inputs=c(addBranchs,  linBranchs))  #Add GAM+lin towers
  if(!is.null(X.nn.sig) & is.null(X.add.basis.sig) & !is.null(X.lin.sig) )  sBranchjoined <- layer_add(inputs=c(  linBranchs,nnBranchs)) #Add nn+lin towers
  if(!is.null(X.nn.sig) & !is.null(X.add.basis.sig) & is.null(X.lin.sig) )  sBranchjoined <- layer_add(inputs=c(addBranchs,  nnBranchs))  #Add nn+GAM towers
  if(is.null(X.nn.sig) & is.null(X.add.basis.sig) & !is.null(X.lin.sig) )  sBranchjoined <- linBranchs  #Just lin tower
  if(is.null(X.nn.sig) & !is.null(X.add.basis.sig) & is.null(X.lin.sig) )  sBranchjoined <- addBranchs  #Just GAM tower
  if(!is.null(X.nn.sig) & is.null(X.add.basis.sig) & is.null(X.lin.sig) )  sBranchjoined <- nnBranchs  #Just nn tower
  if(is.null(X.nn.sig) & is.null(X.add.basis.sig) & is.null(X.lin.sig) )  sBranchjoined <- statBranchs  #Just stationary tower
  
  #Apply link functions
  if(link.loc=="exp") locBranchjoined <- locBranchjoined %>% layer_activation( activation = 'exponential', name = "loc_activation") else if(link.loc=="identity") locBranchjoined <- locBranchjoined %>% layer_activation( activation = 'linear', name = "loc_activation")
  sBranchjoined <- sBranchjoined %>% layer_activation( activation = 'exponential', name = "s_activation")
  
  input=c()
  if(!is.null(X.lin.mu) ) input=c(input,input_lin_loc)
  if(!is.null(X.add.basis.mu) ) input=c(input,input_add_loc)
  if(!is.null(X.nn.mu) ) input=c(input,input_nn_loc)
  if(!is.null(X.lin.sig) ) input=c(input,input_lin_s)
  if(!is.null(X.add.basis.sig) ) input=c(input,input_add_s)
  if(!is.null(X.nn.sig) ) input=c(input,input_nn_s)
  input=c(input)
  
  
  output <- layer_concatenate(c(locBranchjoined,sBranchjoined))
  
  model <- keras_model(  inputs = input,   outputs = output,name=paste0("log-normal"))
  print(model)
  
  return(model)
  
}


lognormal_loss <-function(){
  

 
    loss<- function( y_true, y_pred) {
      
      library(tensorflow)
      K <- backend()
      
      
      mu=y_pred[all_dims(),1]
      sig=y_pred[all_dims(),2]
      
      y <- y_true
  
      # Find inds of non-missing obs.  Remove missing obs, i.e., -1e10. This is achieved by adding an
      # arbitrarily large (<1e10) value to y and then taking the sign ReLu
      obsInds=K$sign(K$relu(y+9e9))
   
      y=K$relu(y)+(1-obsInds)*1 #Sets all missing response values to 1
      sig=sig*obsInds+(1-obsInds)*1 #Sets all sigmas for missing response values to 1
      mu=mu*obsInds+(1-obsInds)*1 #Sets all mus for missing response values to 1
      
      
      #Evaluate log-likelihood
      ll1=-K$log(y)-K$log(sig)-0.5*K$log(2*pi)
      ll1=ll1*obsInds # set all ll1 contribution to zero for missing responses
      
   
      ll2= -(K$log(y)-mu)^2/(2*sig^2)
      ll2=ll2*obsInds # set all ll2 contribution to zero for missing responses
      
      return( -K$sum(ll1)   -K$sum(ll2) 
      )
    }
  
  return(loss)
}
