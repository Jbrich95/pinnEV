#'eGPD PINN
#'
#' Build and train a partially-interpretable neural network for fitting an eGPD model
#'

#' @param type  string defining the type of network to be built. If \code{type=="MLP"}, the network will have all densely connected layers; if \code{type=="CNN"},
#'  the network will have all convolutional layers. If \code{type=="GCNN"}, then a graph convolutional neural network (with skip connections) is used and require \code{!is.null(A)}. Defaults to an MLP (currently the same network is used for all parameters, may change in future versions).
#' @param Y.train,Y.valid a 2 or 3 dimensional array of training or validation real response values.
#' Missing values can be handled by setting corresponding entries to \code{Y.train} or \code{Y.valid} to \code{-1e10}.
#' The first dimension should be the observation indices, e.g., time.
#'
#' If \code{type=="CNN"}, then \code{Y.train} and \code{Y.valid} must have three dimensions with the latter two corresponding to an \eqn{M} by \eqn{N} regular grid of spatial locations. If \code{type=="GCNN"}, then \code{Y.train} and \code{Y.valid} must have two dimensions with the latter corresponding to \eqn{M} spatial locations.
#' If \code{Y.valid==NULL}, no validation loss will be computed and the returned model will be that which minimises the training loss over \code{n.ep} epochs.
#'
#' @param X.s  list of arrays corresponding to complementary subsets of the \eqn{d\geq 1} predictors which are used for modelling the scale parameter \eqn{sigma}. Must contain at least one of the following three named entries:\describe{
#' \item{\code{X.lin.s}}{A 3 or 4 dimensional array of "linear" predictor values. One more dimension than \code{Y.train}. If \code{NULL}, a model without the linear component is built and trained.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the last dimension corresponds to the chosen \eqn{l_1\geq 0} 'linear' predictor values.}
#' \item{\code{X.add.basis.s}}{A 4 or 5 dimensional array of basis function evaluations for the "additive" predictor values.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the penultimate dimensions corresponds to the chosen \eqn{a_1\geq 0} 'linear' predictor values and the last dimension is equal to the number of knots used for estimating the splines. See example.
#' If \code{NULL}, a model without the additive component is built and trained.}
#' \item{\code{X.nn.s}}{A 3 or 4 dimensional array of "non-additive" predictor values.  If \code{NULL}, a model without the NN component is built and trained; if this is the case, then \code{type} has no effect.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the last dimension corresponds to the chosen \eqn{d-l_1-a_1\geq 0} 'non-additive' predictor values.}
#' }
#' Note that \code{X.s} and \code{X.k} are the predictors for both \code{Y.train} and \code{Y.valid}. If \code{is.null(X.s)}, then \eqn{\sigma} will be treated as fixed over the predictors.
#' @param X.k similarly to \code{X.s}, but for modelling the shape parameter \eqn{\kappa>0}. Note that we require at least one of \code{!is.null(X.s)} or \code{!is.null(X.k)}, otherwise the formulated model will be fully stationary and will not be fitted.
#' @param offset an array of strictly positive scalars the same dimension as \code{Y.train}, containing the offset values used in modelling the scale parameter. If \code{offset=NULL}, then no offset is used in the scale parameter (equivalently, \code{offset} is populated with ones). Defaults to \code{NULL}.
#' @param n.ep number of epochs used for training. Defaults to 1000.
#' @param batch.size batch size for stochastic gradient descent. If larger than \code{dim(Y.train)[1]}, i.e., the number of observations, then regular gradient descent used.
#' @param init.scale,init.kappa,init.xi sets the initial \eqn{\sigma,\kappa} and \eqn{\xi} estimates across all dimensions of \code{Y.train}. Overridden by \code{init.wb_path} if \code{!is.null(init.wb_path)}, but otherwise the initial parameters must be supplied.
#' @param init.wb_path filepath to a \code{keras} model which is then used as initial weights and biases for training the new model. The original model must have
#' the exact same architecture and trained with the same input data as the new model. If \code{NULL}, then initial weights and biases are random (with seed \code{seed}) but the
#' final layer has zero initial weights to ensure that the initial scale, kappa and shape estimates are \code{init.scale, init.kappa} and \code{init.xi}, respectively,  across all dimensions.
#' @param widths vector of widths/filters for hidden layers. Number of layers is equal to \code{length(widths)}. Defaults to (6,3).
#' @param filter.dim if \code{type=="CNN"}, this 2-vector gives the dimensions of the convolution filter kernel; must have odd integer inputs. Note that filter.dim=c(1,1) is equivalent to \code{type=="MLP"}. The same filter is applied for each hidden layer across all parameters with NN predictors.
#' @param seed seed for random initial weights and biases.
#' @param model fitted \code{keras} model. Output from \code{bGEVPP.NN.train}.
#' @param S_lambda list of smoothing penalty matrices for the splines modelling the effects of \code{X.add.basis.s} and \code{X.add.basis.k} on their respective parameters; each element only used if \code{!is.null(X.add.basis.s)} and \code{!is.null(X.add.basis.k)}, respectively. If \code{is.null(S_lambda[[1]])}, then no smoothing penalty used for \eqn{\sigma}; similarly for the second element and \eqn{\kappa}. 
#' @param A \eqn{M \times M} adjacency matrix used if and only if \code{type=="GCNN"}. Must be supplied, defaults to \code{NULL}.
#'
#'@name eGPD.NN

#' @details{
#' Consider a real-valued random variable \eqn{Y} and let \eqn{\mathbf{X}} denote a \eqn{d}-dimensional predictor set with observations \eqn{\mathbf{x}}.
#' For \eqn{i=1,2}, we define integers \eqn{l_i\geq 0,a_i \geq 0} and \eqn{0\leq l_i+a_i \leq d}, and let \eqn{\mathbf{X}^{(i)}_L, \mathbf{X}^{(i)}_A} and \eqn{\mathbf{X}^{(i)}_N} be distinct sub-vectors
#' of \eqn{\mathbf{X}}, with observations of each component denoted \eqn{\mathbf{x}^{(i)}_L, \mathbf{x}^{(i)}_A} and \eqn{\mathbf{x}^{(i)}_N}, respectively; the lengths of the sub-vectors are \eqn{l_i,a_i} and \eqn{d_i-l_i-a}, respectively.
#' We model \eqn{Y|\mathbf{X}=\mathbf{x}\sim\mbox{eGPD}(\sigma(\mathbf{x}),\kappa(\mathbf{x}),\xi)} for \eqn{\xi>0} with
#' \deqn{\sigma (\mathbf{x})=C(\mathbf{x})\exp\{\eta^{(1)}_0+m^{(1)}_L(\mathbf{x}^{(1)}_L)+m^{(1)}_A(x^{(1)}_A)+m^{(1)}_N(\mathbf{x}^{(1)}_N)\}} and
#' \deqn{\kappa (\mathbf{x})=\exp\{\eta^{(2)}_0+m^{(2)}_L(\mathbf{x}^{(2)}_L)+m^{(2)}_A(x^{(2)}_A)+m^{(2)}_N(\mathbf{x}^{(2)}_N)\}}
#' where \eqn{\eta^{(1)}_0,\eta^{(2)}_0} are constant intercepts and \eqn{C(\mathbf{x})>0} is a fixed offset term. The unknown functions \eqn{m^{(1)}_L,m^{(2)}_L} and
#' \eqn{m^{(1)}_A,m^{(2)}_A} are estimated using linear functions and splines, respectively, and are
#' both returned as outputs by \code{eGPD.NN.predict}; \eqn{m^{(1)}_N,m^{(2)}_N} are estimated using neural networks
#' (currently the same architecture is used for both parameters). The offset term is, by default, \eqn{C(\mathbf{x})=1} for all \eqn{\mathbf{x}}; if \code{!is.null(offset)}, then \code{offset} determines \eqn{C(\mathbf{x})} (see Cisneros et al., 2023). Note that \eqn{\xi>0} is fixed across all predictors; this may change in future versions. 
#' Note that \eqn{\xi>0} is fixed across all predictors; this may change in future versions.
#'
#' For details of the eGPD distribution, see \code{help(peGPD)}. 
#'
#' The model is fitted by minimising the negative log-likelihood associated with the bGEV model plus some smoothing penalty for the additive functions (determined by \code{S_lambda}; see Richards and Huser, 2022); training is performed over \code{n.ep} training epochs.
#' Although the model is trained by minimising the loss evaluated for \code{Y.train}, the final returned model may minimise some other loss.
#' The current state of the model is saved after each epoch, using \code{keras::callback_model_checkpoint}, if the value of some criterion subcedes that of the model from the previous checkpoint; this criterion is the loss evaluated for validation set \code{Y.valid} if \code{!is.null(Y.valid)} and for \code{Y.train}, otherwise.
#'
#'A non-interpretable version of this model was exploited by Cisneros et al. (2023). Equivalence with their model is achieved by setting \code{X.k=NULL}, \code{X.lin.s=NULL}, \code{X.add.basis.s=NULL} and \code{type="GCNN"}. See \code{help(AusWild)}.
#'
#'}
#' @return \code{eGPD.NN.train} returns the fitted \code{model}.  \code{eGPD.NN.predict} is a wrapper for \code{keras::predict} that returns the predicted parameter estimates, and, if applicable, their corresponding linear regression coefficients and spline bases weights.
#'
#'@references{
#' Papastathopoulos, I. and Tawn, J. A. (2013), \emph{xtended generalised Pareto models for tail estimation}, Journal of Statistical Planning and Inference, 43(1):131–1439.
#' (\href{https://doi.org/10.1016/j.jspi.2012.07.001}{doi})
#'
#' Naveau, P., Huser, R., Ribereau, P., and Hannart, A. (2016), \emph{Modeling jointly low, moderate, and heavy rainfall intensities without a threshold selection}, Water Resources Research, 2(4):2753–2769.
#' (\href{https://doi.org/10.1002/2015WR018552}{doi})
#'
#' Richards, J. and Huser, R. (2022), \emph{Regression modelling of spatiotemporal extreme U.S. wildfires via partially-interpretable neural networks}. (\href{https://arxiv.org/abs/2208.07581}{arXiv:2208.07581}).
#' 
#' Cisneros, D., Richards, J., Dahal, A., Lombardo, L., and Huser, R. (2023), \emph{Deep learning-based graphical regression for jointly moderate and extreme Australian wildfires.}. (\href{}{In draft}).
#'}
#' @examples
# Build and train a simple MLP for toy data
#' 
#' set.seed(1)
#' 
#' # Create  predictors
#' preds<-rnorm(prod(c(2500,10,8)))
#' 
#' 
#' #Re-shape to a 3d array. First dimension corresponds to observations,
#' #last to the different components of the predictor set.
#' #Other dimensions correspond to indices of predictors, e.g., a grid of locations. Can be a 1D or 2D grid.
#' dim(preds)=c(2500,10,8)
#' #We have 2000 observations of eight predictors at 10 sites.
#' 
#' 
#' #Split predictors into linear, additive and nn. Different for kappa and scale parameters.
#' X.nn.k=preds[,,1:4] #Four nn predictors for kappa
#' X.lin.k=preds[,,5:6] #Two additive predictors for kappa
#' X.add.k=preds[,,7:8] #Two additive predictors for kappa
#' 
#' X.nn.s=preds[,,1:2] #Two nn predictors for sigma
#' X.lin.s=preds[,,3] #One linear predictor for sigma
#' dim(X.lin.s)=c(dim(X.lin.s),1) #Change dimension so consistent
#' X.add.s=preds[,,4] #One additive predictor for sigma
#' dim(X.add.s)=c(dim(X.add.s),1) #Change dimension so consistent
#' 
#' # Create toy response data
#' 
#' #Contribution to scale parameter
#' #Linear contribution
#' m_L_1 = 0.2*X.lin.s[,,1]
#' 
#' # Additive contribution
#' m_A_1 = 0.1*X.add.s[,,1]^2+0.2*X.add.s[,,1]
#' 
#' plot(X.add.s[,,1],m_A_1)
#' 
#' #Non-additive contribution - to be estimated by NN
#' m_N_1 = 0.2*exp(-4+X.nn.s[,,2]+X.nn.s[,,1])+
#'   0.1*sin(X.nn.s[,,1]-X.nn.s[,,2])*(X.nn.s[,,1]+X.nn.s[,,2])
#' 
#' sigma=0.4*exp(0.5+m_L_1+m_A_1+m_N_1+1) #Exponential link
#' 
#' 
#' #Contribution to kappa parameter
#' #Linear contribution
#' m_L_2 = 0.1*X.lin.k[,,1]-0.02*X.lin.k[,,2]
#' 
#' # Additive contribution
#' m_A_2 = 0.1*X.add.k[,,1]^2+0.1*X.add.k[,,1]-
#'   0.025*X.add.k[,,2]^3+0.025*X.add.k[,,2]^2
#' 
#' #Non-additive contribution - to be estimated by NN
#' m_N_2 = 0.5*exp(-3+X.nn.k[,,4]+X.nn.k[,,1])+
#'   sin(X.nn.k[,,1]-X.nn.k[,,2])*(X.nn.k[,,4]+X.nn.k[,,2])-
#'   cos(X.nn.k[,,4]-X.nn.k[,,1])*(X.nn.k[,,3]+X.nn.k[,,1])
#' 
#' kappa=exp(m_L_2+m_A_2+0.05 *m_N_2)  #Exponential link
#' 
#' 
#' xi=0.1 # Set xi
#' 
#' theta=array(dim=c(dim(sigma),3))
#' theta[,,1]=sigma; theta[,,2] = kappa; theta[,,3]=xi
#' #We simulate data from an eGPD model
#' 
#' #Simulate from eGPD model using same u as given above
#' Y=apply(theta,1:2,function(x) reGPD(1,sigma=x[1],kappa=x[2],xi=x[3]))
#' 
#' 
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
#' #the basis functions for each pre-specified knot and entry to X.add.k and X.add.s
#' 
#' rad=function(x,c){ #Define a basis function. Here we use the radial bases
#'   out=abs(x-c)^2*log(abs(x-c))
#'   out[(x-c)==0]=0
#'   return(out)
#' }
#' 
#' n.knot.s = 4; n.knot.k = 5# set number of knots.
#' #Must be the same for each additive predictor,
#' #but can differ between the parameters sigma and kappa
#' 
#' 
#' #Get knots for sigma predictor
#' knots.s=matrix(nrow=dim(X.add.s)[3],ncol=n.knot.s)
#' for( i in 1:dim(X.add.s)[3]){
#'   knots.s[i,]=quantile(X.add.s[,,i],probs=seq(0,1,length=n.knot.s))
#' }
#' 
#' #Evaluate radial basis functions for s_\beta predictor
#' X.add.basis.s<-array(dim=c(dim(X.add.s),n.knot.s))
#' for( i in 1:dim(X.add.s)[3]) {
#'   for(k in 1:n.knot.s) {
#'     X.add.basis.s[,,i,k]= rad(x=X.add.s[,,i],c=knots.s[i,k])
#'     #Evaluate rad at all entries to X.add.k and for all knots
#'   }}
#' 
#' 
#' 
#' 
#' #Create smoothing penalty matrix for the sigma additive function
#' 
#' # Set smoothness parameter
#' lambda = c(0.2)
#' 
#' S_lambda.s=matrix(0,nrow=n.knot.s*dim(X.add.s)[3],ncol=n.knot.s*dim(X.add.s)[3])
#' for(i in 1:dim(X.add.s)[3]){
#'   for(j in 1:n.knot.s){
#'     for(k in 1:n.knot.s){
#'       S_lambda.s[(j+(i-1)*n.knot.s),(k+(i-1)*n.knot.s)]=lambda[i]*rad(knots.s[i,j],knots.s[i,k])
#'     }
#'   }
#' }
#' #Get knots for kappa predictors
#' knots.k=matrix(nrow=dim(X.add.k)[3],ncol=n.knot.k)
#' 
#' #We set knots to be equally-spaced marginal quantiles
#' for( i in 1:dim(X.add.k)[3]){
#'   knots.k[i,]=quantile(X.add.k[,,i],probs=seq(0,1,length=n.knot.k))
#' }
#' 
#' 
#' #Evaluate radial basis functions for kappa predictors
#' X.add.basis.k<-array(dim=c(dim(X.add.k),n.knot.k))
#' for( i in 1:dim(X.add.k)[3]) {
#'   for(k in 1:n.knot.k) {
#'     X.add.basis.k[,,i,k]= rad(x=X.add.k[,,i],c=knots.k[i,k])
#'     #Evaluate rad at all entries to X.add.k and for all knots
#'   }}
#' 
#' #'#Create smoothing penalty matrix for the two kappa additive functions
#' 
#' # Set smoothness parameters for two functions
#' lambda = c(0.1,0.2)
#' 
#' S_lambda.k=matrix(0,nrow=n.knot.k*dim(X.add.k)[3],ncol=n.knot.k*dim(X.add.k)[3])
#' for(i in 1:dim(X.add.k)[3]){
#'   for(j in 1:n.knot.k){
#'     for(k in 1:n.knot.k){
#'       S_lambda.k[(j+(i-1)*n.knot.k),(k+(i-1)*n.knot.k)]=lambda[i]*rad(knots.k[i,j],knots.k[i,k])
#'     }
#'   }
#' }
#' 
#' 
#' #Join in one list
#' S_lambda =list("S_lambda.k"=S_lambda.k, "S_lambda.s"=S_lambda.s)
#' 
#' #lin+GAM+NN models defined for both scale and kappa parameters
#' X.s=list("X.nn.s"=X.nn.s, "X.lin.s"=X.lin.s,
#'          "X.add.basis.s"=X.add.basis.s) #Predictors for sigma
#' X.k=list("X.nn.k"=X.nn.k, "X.lin.k"=X.lin.k,
#'          "X.add.basis.k"=X.add.basis.k) #Predictors for kappa
#' 
#' 
#' #Fit the eGPD model. Note that training is not run to completion.
#' NN.fit<-eGPD.NN.train(Y.train, Y.valid,X.s,X.k, type="MLP",
#'                       n.ep=50, batch.size=50,init.scale=1, init.kappa=1,init.xi=0.1,
#'                       widths=c(6,3),seed=1,S_lambda=S_lambda)
#' out<-eGPD.NN.predict(X.s=X.s,X.k=X.k,NN.fit$model)
#' 
#' print("sigma linear coefficients: "); print(round(out$lin.coeff_s,2))
#' print("kappa linear coefficients: "); print(round(out$lin.coeff_k,2))
#' 
#' # Note that this is a simple example that can be run in a personal computer.
#' 
#' 
#' # #To save model, run
#' # NN.fit$model %>% save_model_tf("model_eGPD")
#' # #To load model, run
#' #  model  <- load_model_tf("model_eGPD",
#' #   custom_objects=list(
#' #     "eGPD_loss_S_lambda___S_lambda_"=
#' #       eGPD_loss(S_lambda=S_lambda))
#' #         )
#' 
#' 
#' # Plot splines for the additive predictors
#' 
#' 
#' 
#' #Sigma predictors
#' n.add.preds_s=dim(X.add.s)[length(dim(X.add.s))]
#' par(mfrow=c(1,n.add.preds_s))
#' for(i in 1:n.add.preds_s){
#'   plt.x=seq(from=min(knots.s[i,]),to=max(knots.s[i,]),length=1000)  #Create sequence for x-axis
#'   
#'   tmp=matrix(nrow=length(plt.x),ncol=n.knot.s)
#'   for(j in 1:n.knot.s){
#'     tmp[,j]=rad(plt.x,knots.s[i,j]) #Evaluate radial basis function of plt.x and all knots
#'   }
#'   plt.y=tmp%*%out$gam.weights_s[i,]
#'   plot(plt.x,plt.y,type="l",main=paste0("sigma spline: predictor ",i),xlab="x",ylab="f(x)")
#'   points(knots.s[i,],rep(mean(plt.y),n.knot.s),col="red",pch=2)
#'   #Adds red triangles that denote knot locations
#'   
#' }
#' 
#' #Kappa predictors
#' n.add.preds_k=dim(X.add.k)[length(dim(X.add.k))]
#' par(mfrow=c(1,n.add.preds_k))
#' for(i in 1:n.add.preds_k){
#'   plt.x=seq(from=min(knots.k[i,]),to=max(knots.k[i,]),length=1000)  #Create sequence for x-axis
#'   
#'   tmp=matrix(nrow=length(plt.x),ncol=n.knot.k)
#'   for(j in 1:n.knot.k){
#'     tmp[,j]=rad(plt.x,knots.k[i,j]) #Evaluate radial basis function of plt.x and all knots
#'   }
#'   plt.y=tmp%*%out$gam.weights_k[i,]
#'   plot(plt.x,plt.y,type="l",main=paste0("kappa spline: predictor ",i),xlab="x",ylab="f(x)")
#'   points(knots.k[i,],rep(mean(plt.y),n.knot.k),col="red",pch=2)
#'   #Adds red triangles that denote knot locations
#'   
#' }


#' @import reticulate keras tensorflow 
#' @rdname eGPD.NN
#' @export

eGPD.NN.train=function(Y.train, Y.valid = NULL, X.s, X.k, type="MLP", offset=NULL, A=NULL,
                       n.ep=100, batch.size=100, init.scale=NULL, init.kappa=NULL, init.xi=NULL,
                       widths=c(6,3), filter.dim=c(3,3), seed=NULL, init.wb_path=NULL, S_lambda=NULL)
{
  

  
  
  if(is.null(X.k) &  is.null(X.s)  ) stop("No predictors provided for sigma or kappa: Stationary models are not permitted ")
  if(is.null(Y.train)) stop("No training response data provided")
  
  if(is.null(X.s)) offset=NULL
  if(!is.null(offset) & any(offset <= 0)) stop("Negative or zero offset values provided")
  
  if(is.null(init.kappa) & is.null(init.wb_path)  ) stop("Inital kappa estimate not provided")
  if(is.null(init.scale) & is.null(init.wb_path)   ) stop("Inital scale estimate not provided")
  if(is.null(init.xi)  & is.null(init.wb_path) ) stop("Inital xi estimate not provided")
  if(any(c(init.scale,init.kappa,init.xi) < 0) ) stop("Negative parameters are not feasible")
  
  if(is.null(A) & type=="GCNN")stop("Adjacency matrix must be supplied if GCNN required")
  if(!is.null(A)) if(type=="GCNN" & (length(dim(Y.train))!=2 | dim(A)[1]!=dim(A)[2] | dim(A)[1]!=dim(Y.train)[2]))stop("Dimensions of the adjacency matrix are incorrect")
  
  
  print(paste0("Creating eGPD model"))
  if(!is.null(offset)) print(paste0("Using offset scale parameter"))
  
  X.nn.s=X.s$X.nn.s
  X.lin.s=X.s$X.lin.s
  X.add.basis.s=X.s$X.add.basis.s
  
  if(!is.null(offset)){
    if(!is.null(X.nn.s) & !is.null(X.add.basis.s) & !is.null(X.lin.s) ) {  train.data= list(X.lin.s,X.add.basis.s,X.nn.s,offset); print("Defining lin+GAM+NN model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_s=X.lin.s,add_input_s=X.add.basis.s,  nn_input_s=X.nn.s,offset_input=offset),Y.valid)}
    if(is.null(X.nn.s) & !is.null(X.add.basis.s) & !is.null(X.lin.s) ) {   train.data= list(X.lin.s,X.add.basis.s,offset); print("Defining lin+GAM model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_s=X.lin.s,add_input_s=X.add.basis.s,offset_input=offset),Y.valid)}
    if(!is.null(X.nn.s) & is.null(X.add.basis.s) & !is.null(X.lin.s) ) { train.data= list(X.lin.s,X.nn.s,offset); print("Defining lin+NN model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_s=X.lin.s, nn_input_s=X.nn.s,offset_input=offset),Y.valid)}
    if(!is.null(X.nn.s) & !is.null(X.add.basis.s) & is.null(X.lin.s) ) {train.data= list(X.add.basis.s,X.nn.s,offset); print("Defining GAM+NN model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list(add_input_s=X.add.basis.s,  nn_input_s=X.nn.s,offset_input=offset),Y.valid)}
    if(is.null(X.nn.s) & is.null(X.add.basis.s) & !is.null(X.lin.s) )   {train.data= list(X.lin.s,offset); print("Defining fully-linear model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_s=X.lin.s,offset_input=offset),Y.valid)}
    if(is.null(X.nn.s) & !is.null(X.add.basis.s) & is.null(X.lin.s) )   {train.data= list(X.add.basis.s,offset); print("Defining fully-additive model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list(add_input_s=X.add.basis.s,offset_input=offset),Y.valid)}
    if(!is.null(X.nn.s) & is.null(X.add.basis.s) & is.null(X.lin.s) )   {train.data= list(X.nn.s,offset); print("Defining fully-NN model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list( nn_input_s=X.nn.s,offset_input=offset),Y.valid)} 
  
    }else if(is.null(offset)){
  if(!is.null(X.nn.s) & !is.null(X.add.basis.s) & !is.null(X.lin.s) ) {  train.data= list(X.lin.s,X.add.basis.s,X.nn.s); print("Defining lin+GAM+NN model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_s=X.lin.s,add_input_s=X.add.basis.s,  nn_input_s=X.nn.s),Y.valid)}
  if(is.null(X.nn.s) & !is.null(X.add.basis.s) & !is.null(X.lin.s) ) {   train.data= list(X.lin.s,X.add.basis.s); print("Defining lin+GAM model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_s=X.lin.s,add_input_s=X.add.basis.s),Y.valid)}
  if(!is.null(X.nn.s) & is.null(X.add.basis.s) & !is.null(X.lin.s) ) { train.data= list(X.lin.s,X.nn.s); print("Defining lin+NN model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_s=X.lin.s, nn_input_s=X.nn.s),Y.valid)}
  if(!is.null(X.nn.s) & !is.null(X.add.basis.s) & is.null(X.lin.s) ) {train.data= list(X.add.basis.s,X.nn.s); print("Defining GAM+NN model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list(add_input_s=X.add.basis.s,  nn_input_s=X.nn.s),Y.valid)}
  if(is.null(X.nn.s) & is.null(X.add.basis.s) & !is.null(X.lin.s) )   {train.data= list(X.lin.s); print("Defining fully-linear model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_s=X.lin.s),Y.valid)}
  if(is.null(X.nn.s) & !is.null(X.add.basis.s) & is.null(X.lin.s) )   {train.data= list(X.add.basis.s); print("Defining fully-additive model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list(add_input_s=X.add.basis.s),Y.valid)}
  if(!is.null(X.nn.s) & is.null(X.add.basis.s) & is.null(X.lin.s) )   {train.data= list(X.nn.s); print("Defining fully-NN model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list( nn_input_s=X.nn.s),Y.valid)}
  if(is.null(X.nn.s) & is.null(X.add.basis.s) & is.null(X.lin.s) )   {train.data= list(); print("Defining stationary model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list( ),Y.valid)}
  }
  S_lambda.s=S_lambda$S_lambda.s
  if(!is.null(X.add.basis.s) & is.null(S_lambda.s)){print("No smoothing penalty used for sigma")}
  if(is.null(X.add.basis.s)){S_lambda.s=NULL}
  
  X.nn.k=X.k$X.nn.k
  X.lin.k=X.k$X.lin.k
  X.add.basis.k=X.k$X.add.basis.k
  
  if(!is.null(X.nn.k) & !is.null(X.add.basis.k) & !is.null(X.lin.k) ) {  train.data= c(train.data,list(X.lin.k,X.add.basis.k,X.nn.k)); print("Defining lin+GAM+NN model for kappa" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_k=X.lin.k,add_input_k=X.add.basis.k,  nn_input_k=X.nn.k)),Y.valid)}
  if(is.null(X.nn.k) & !is.null(X.add.basis.k) & !is.null(X.lin.k) ) {   train.data= c(train.data,list(X.lin.k,X.add.basis.k)); print("Defining lin+GAM model for kappa" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_k=X.lin.k,add_input_k=X.add.basis.k)),Y.valid)}
  if(!is.null(X.nn.k) & is.null(X.add.basis.k) & !is.null(X.lin.k) ) { train.data= c(train.data,list(X.lin.k,X.nn.k)); print("Defining lin+NN model for kappa" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_k=X.lin.k, nn_input_k=X.nn.k)),Y.valid)}
  if(!is.null(X.nn.k) & !is.null(X.add.basis.k) & is.null(X.lin.k) ) {train.data= c(train.data,list(X.add.basis.k,X.nn.k)); print("Defining GAM+NN model for kappa" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(add_input_k=X.add.basis.k,  nn_input_k=X.nn.k)),Y.valid)}
  if(is.null(X.nn.k) & is.null(X.add.basis.k) & !is.null(X.lin.k) )   {train.data= c(train.data,list(X.lin.k)); print("Defining fully-linear model for kappa" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_k=X.lin.k)),Y.valid)}
  if(is.null(X.nn.k) & !is.null(X.add.basis.k) & is.null(X.lin.k) )   {train.data= c(train.data,list(X.add.basis.k)); print("Defining fully-additive model for kappa" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(add_input_k=X.add.basis.k)),Y.valid)}
  if(!is.null(X.nn.k) & is.null(X.add.basis.k) & is.null(X.lin.k) )   {train.data= c(train.data,list(X.nn.k)); print("Defining fully-NN model for kappa" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(nn_input_k=X.nn.k)),Y.valid)}
  if(is.null(X.nn.k) & is.null(X.add.basis.k) & is.null(X.lin.k) )   {train.data= train.data; print("Defining stationary model for kappa" );  if(!is.null(Y.valid)) validation.data=validation.data}
  
  S_lambda.k=S_lambda$S_lambda.k
  if(!is.null(X.add.basis.k) & is.null(S_lambda.k)){print("No smoothing penalty used for kappa")}
  if(is.null(X.add.basis.k)){S_lambda.k=NULL}
  
  S_lambda =list("S_lambda.k"=S_lambda.k, "S_lambda.s"=S_lambda.s)
  
  if(type=="CNN" & (!is.null(X.nn.k) | !is.null(X.nn.s)))print(paste0("Building ",length(widths),"-layer convolutional neural network with ", filter.dim[1]," by ", filter.dim[2]," filter" ))
  if(type=="MLP"  & (!is.null(X.nn.k) | !is.null(X.nn.s)) ) print(paste0("Building ",length(widths),"-layer densely-connected neural network" ))
  if(type=="GCNN"  & (!is.null(X.nn.k) | !is.null(X.nn.s)) ) print(paste0("Building ",length(widths),"-layer graph convolutional neural network" ))
  
  reticulate::use_virtualenv("pinnEV_env", required = T)
  if(is.null(seed)) seed=1
  tf$random$set_seed(seed)
  
  if(!is.null(offset) & length(dim(offset))!=length(dim(Y.train))+1) dim(offset)=c(dim(offset),1)
  
  model<-eGPD.NN.build(X.nn.s,X.lin.s,X.add.basis.s,
                       X.nn.k,X.lin.k,X.add.basis.k,
                       offset,type,init.scale,init.kappa,init.xi, widths, filter.dim, A, seed)
  
  if(!is.null(init.wb_path)) model <- load_model_weights_tf(model,filepath=init.wb_path)
  
  model %>% compile(
    optimizer="adam",
    loss = eGPD_loss(S_lambda=S_lambda),
    run_eagerly=T
  )
  
  
  if(!is.null(Y.valid)) checkpoint <- callback_model_checkpoint(paste0("model_eGPD_checkpoint"), monitor = "val_loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch") else checkpoint <- callback_model_checkpoint(paste0("model_eGPD_checkpoint"), monitor = "loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch")
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
  model <- load_model_weights_tf(model,filepath=paste0("model_eGPD_checkpoint"))
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
#' @rdname eGPD.NN
#' @export
#'
eGPD.NN.predict=function(X.s,X.k, model, offset=NULL)
{
  if(is.null(X.k) &  is.null(X.s)  ) stop("No predictors provided for sigma or kappa: Stationary models are not permitted ")
  if(is.null(X.s)) offset=NULL

  if(!is.null(offset) & any(offset <= 0)) stop("Negative or zero offset values provided")
  
  X.nn.k=X.k$X.nn.k
  X.lin.k=X.k$X.lin.k
  X.add.basis.k=X.k$X.add.basis.k
  
  X.nn.s=X.s$X.nn.s
  X.lin.s=X.s$X.lin.s
  X.add.basis.s=X.s$X.add.basis.s
  if(!is.null(offset)){
    if(!is.null(X.nn.s) & !is.null(X.add.basis.s) & !is.null(X.lin.s) )   train.data= list(X.lin.s,X.add.basis.s,X.nn.s,offset)
    if(is.null(X.nn.s) & !is.null(X.add.basis.s) & !is.null(X.lin.s) )   train.data= list(X.lin.s,X.add.basis.s,offset)
    if(!is.null(X.nn.s) & is.null(X.add.basis.s) & !is.null(X.lin.s) )  train.data= list(X.lin.s,X.nn.s,offset)
    if(!is.null(X.nn.s) & !is.null(X.add.basis.s) & is.null(X.lin.s) ) train.data= list(X.add.basis.s,X.nn.s,offset)
    if(is.null(X.nn.s) & is.null(X.add.basis.s) & !is.null(X.lin.s) )   train.data= list(X.lin.s,offset)
    if(is.null(X.nn.s) & !is.null(X.add.basis.s) & is.null(X.lin.s) )   train.data= list(X.add.basis,u,offset)
    if(!is.null(X.nn.s) & is.null(X.add.basis.s) & is.null(X.lin.s) )   train.data= list(X.nn.s,offset)
  }else{
  if(!is.null(X.nn.s) & !is.null(X.add.basis.s) & !is.null(X.lin.s) )   train.data= list(X.lin.s,X.add.basis.s,X.nn.s)
  if(is.null(X.nn.s) & !is.null(X.add.basis.s) & !is.null(X.lin.s) )   train.data= list(X.lin.s,X.add.basis.s)
  if(!is.null(X.nn.s) & is.null(X.add.basis.s) & !is.null(X.lin.s) )  train.data= list(X.lin.s,X.nn.s)
  if(!is.null(X.nn.s) & !is.null(X.add.basis.s) & is.null(X.lin.s) ) train.data= list(X.add.basis.s,X.nn.s)
  if(is.null(X.nn.s) & is.null(X.add.basis.s) & !is.null(X.lin.s) )   train.data= list(X.lin.s)
  if(is.null(X.nn.s) & !is.null(X.add.basis.s) & is.null(X.lin.s) )   train.data= list(X.add.basis.s)
  if(!is.null(X.nn.s) & is.null(X.add.basis.s) & is.null(X.lin.s) )   train.data= list(X.nn.s)
  if(is.null(X.nn.s) & is.null(X.add.basis.s) & is.null(X.lin.s) )   train.data= list()
  }
  
  if(!is.null(X.nn.k) & !is.null(X.add.basis.k) & !is.null(X.lin.k) )   train.data= c(train.data,list(X.lin.k,X.add.basis.k,X.nn.k))
  if(is.null(X.nn.k) & !is.null(X.add.basis.k) & !is.null(X.lin.k) )   train.data= c(train.data,list(X.lin.k,X.add.basis.k))
  if(!is.null(X.nn.k) & is.null(X.add.basis.k) & !is.null(X.lin.k) )  train.data= c(train.data,list(X.lin.k,X.nn.k))
  if(!is.null(X.nn.k) & !is.null(X.add.basis.k) & is.null(X.lin.k) ) train.data= c(train.data,list(X.add.basis.k,X.nn.k))
  if(is.null(X.nn.k) & is.null(X.add.basis.k) & !is.null(X.lin.k) )   train.data= c(train.data,list(X.lin.k))
  if(is.null(X.nn.k) & !is.null(X.add.basis.k) & is.null(X.lin.k) )   train.data= c(train.data,list(X.add.basis.k))
  if(!is.null(X.nn.k) & is.null(X.add.basis.k) & is.null(X.lin.k) ) train.data= c(train.data,list(X.nn.k))
  if(is.null(X.nn.k) & is.null(X.add.basis.k) & is.null(X.lin.k) ) train.data= train.data
  
  

  
  
  predictions<-model %>% predict( train.data)
  predictions <- k_constant(predictions)
  pred.sigma=k_get_value(predictions[all_dims(),1])
  pred.kappa=k_get_value(predictions[all_dims(),2])
  pred.xi=k_get_value(predictions[all_dims(),3])
  
  if(!is.null(X.add.basis.k))  gam.weights_k<-matrix(t(model$get_layer("add_k")$get_weights()[[1]]),nrow=dim(X.add.basis.k)[length(dim(X.add.basis.k))-1],ncol=dim(X.add.basis.k)[length(dim(X.add.basis.k))],byrow=T)
  if(!is.null(X.add.basis.s))  gam.weights_s<-matrix(t(model$get_layer("add_s")$get_weights()[[1]]),nrow=dim(X.add.basis.s)[length(dim(X.add.basis.s))-1],ncol=dim(X.add.basis.s)[length(dim(X.add.basis.s))],byrow=T)
  
  out=list("pred.sigma"=pred.sigma,"pred.kappa"=pred.kappa,"pred.xi"=pred.xi)
  if(!is.null(X.lin.k) ) out=c(out,list("lin.coeff_k"=c(model$get_layer("lin_k")$get_weights()[[1]])))
  if(!is.null(X.lin.s) ) out=c(out,list("lin.coeff_s"=c(model$get_layer("lin_s")$get_weights()[[1]])))
  if(!is.null(X.add.basis.k) ) out=c(out,list("gam.weights_k"=gam.weights_k))
  if(!is.null(X.add.basis.s) ) out=c(out,list("gam.weights_s"=gam.weights_s))
  
  return(out)
  
}
#'
#'
eGPD.NN.build=function(X.nn.s,X.lin.s,X.add.basis.s,
                       X.nn.k,X.lin.k,X.add.basis.k,
                       offset, type, init.scale,init.kappa,init.xi, widths,filter.dim, A, seed)
{
  
  if(type=="GCNN"){
  spk <<- reticulate::import("spektral", delay_load = list(
    priority = 10,
    environment = "pinnEV_env"
  ))
  
  
  layer_graph_conv <- function(object,
                               channels,
                               activation = NULL,
                               use_bias = TRUE,
                               kernel_initializer = 'glorot_uniform',
                               bias_initializer = 'zeros',
                               kernel_regularizer = NULL,
                               bias_regularizer = NULL,
                               activity_regularizer = NULL,
                               kernel_constraint = NULL,
                               bias_constraint = NULL,
                               name=NULL,
                               ...)
  {
    args <- list(channels = as.integer(channels),
                 activation = activation,
                 use_bias = use_bias,
                 kernel_initializer = kernel_initializer,
                 bias_initializer = bias_initializer,
                 kernel_regularizer = kernel_regularizer,
                 bias_regularizer = bias_regularizer,
                 activity_regularizer = activity_regularizer,
                 kernel_constraint = kernel_constraint,
                 bias_constraint = bias_constraint,
                 name=name
    )
    keras::create_layer(spk$layers$GCSConv, object, args)
  }
  
  print("Normalising adjacency matrix")
    ML<-spk$utils$convolution$normalized_adjacency(A)
  }
  
  
  if(!is.null(offset) & any(offset <= 0)) stop("Negative or zero offset values provided")
  #Additive inputs
  if(!is.null(X.add.basis.k))  input_add_k<- layer_input(shape = dim(X.add.basis.k)[-1], name = 'add_input_k')
  if(!is.null(X.add.basis.s))  input_add_s<- layer_input(shape = dim(X.add.basis.s)[-1], name = 'add_input_s')
  
  #NN input
  
  if(!is.null(X.nn.k))   input_nn_k <- layer_input(shape = dim(X.nn.k)[-1], name = 'nn_input_k')
  if(!is.null(X.nn.s))   input_nn_s <- layer_input(shape = dim(X.nn.s)[-1], name = 'nn_input_s')
  
  #Linear input
  
  if(!is.null(X.lin.k)) input_lin_k <- layer_input(shape = dim(X.lin.k)[-1], name = 'lin_input_k')
  if(!is.null(X.lin.s)) input_lin_s <- layer_input(shape = dim(X.lin.s)[-1], name = 'lin_input_s')
  
  #offset input
  if(!is.null(offset))  input_offset <- layer_input(shape = dim(offset)[-1], name = 'offset_input')
  
  
  #Create xi branch
  
  if(!is.null(X.nn.k)){
    xiBranch <- input_nn_k %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X.nn.k)[-1], trainable=F,
                                           weights=list(matrix(0,nrow=dim(X.nn.k)[length(dim(X.nn.k))],ncol=1),array(1,dim=c(1))), name = 'xi_stationary_dense') %>%
      layer_dense(units = 1 ,activation = 'sigmoid',use_bias = F,weights=list(matrix(qlogis(init.xi),nrow=1,ncol=1)), name = 'xi_activation')
  }else  if(!is.null(X.nn.s)){
    xiBranch <- input_nn_s %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X.nn.s)[-1], trainable=F,
                                           weights=list(matrix(0,nrow=dim(X.nn.s)[length(dim(X.nn.s))],ncol=1),array(1,dim=c(1))), name = 'xi_stationary_dense') %>%
      layer_dense(units = 1 ,activation = 'sigmoid',use_bias = F,weights=list(matrix(qlogis(init.xi),nrow=1,ncol=1)), name = 'xi_activation')
  }else  if(!is.null(X.lin.k)){
    xiBranch <- input_lin_k %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X.lin.k)[-1], trainable=F,
                                            weights=list(matrix(0,nrow=dim(X.lin.k)[length(dim(X.lin.k))],ncol=1),array(1,dim=c(1))), name = 'xi_stationary_dense') %>%
      layer_dense(units = 1 ,activation = 'sigmoid',use_bias = F,weights=list(matrix(qlogis(init.xi),nrow=1,ncol=1)), name = 'xi_activation')
  }else  if(!is.null(X.lin.s)){
    xiBranch <- input_lin_s %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X.lin.s)[-1], trainable=F,
                                            weights=list(matrix(0,nrow=dim(X.lin.s)[length(dim(X.lin.s))],ncol=1),array(1,dim=c(1))), name = 'xi_stationary_dense') %>%
      layer_dense(units = 1 ,activation = 'sigmoid',use_bias = F,weights=list(matrix(qlogis(init.xi),nrow=1,ncol=1)), name = 'xi_activation')
  }else if(!is.null(X.add.basis.k)){
    xiBranch <- input_add_k %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X.add.basis.k)[-1], trainable=F,
                                            weights=list(matrix(0,nrow=dim(X.add.basis.k)[length(dim(X.add.basis.k))],ncol=1),array(1,dim=c(1))), name = 'xi_stationary_dense') %>%
      layer_dense(units = 1 ,activation = 'sigmoid',use_bias = F,weights=list(matrix(qlogis(init.xi),nrow=1,ncol=1)), name = 'xi_activation')
  }else  if(!is.null(X.add.basis.s)){
    xiBranch <- input_add_s %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X.add.basis.s)[-1], trainable=F,
                                            weights=list(matrix(0,nrow=dim(X.add.basis.s)[length(dim(X.add.basis.s))],ncol=1),array(1,dim=c(1))), name = 'xi_stationary_dense') %>%
      layer_dense(units = 1 ,activation = 'sigmoid',use_bias = F,weights=list(matrix(qlogis(init.xi),nrow=1,ncol=1)), name = 'xi_activation')
  }
  
  
  
  init.kappa=log(init.kappa)
  init.scale=log(init.scale)
  
  #NN towers
  
  #Kappa
  if(!is.null(X.nn.k)){
    
    nunits=c(widths,1)
    n.layers=length(nunits)-1
    
    nnBranchk <- input_nn_k
    if(type=="MLP"){
      for(i in 1:n.layers){
        nnBranchk <- nnBranchk  %>% layer_dense(units=nunits[i],activation = 'relu',
                                                input_shape =dim(X.nn.k)[-1], name = paste0('nn_k_dense',i) )
      }
    }else if(type=="CNN"){
      for(i in 1:n.layers){
        nnBranchk <- nnBranchk  %>% layer_conv_2d(filters=nunits[i],activation = 'relu',kernel_size=c(filter.dim[1],filter.dim[2]), padding='same',
                                                  input_shape =dim(X.nn.k)[-1], name = paste0('nn_k_cnn',i) )
      }
      
    }else if(type=="GCNN"){
      for(i in 1:n.layers){
        nnBranchk <- list(nnBranchk,ML)  %>% layer_graph_conv(channels=nunits[i],activation = 'relu',
                                                  input_shape =dim(X.nn.k)[-1], name = paste0('nn_k_gcnn',i) ,
                                                  kernel_initializer=initializer_glorot_uniform(seed=seed))
      }
    
    }
    nnBranchk <-   nnBranchk  %>%   layer_dense(units = nunits[n.layers+1], activation = "linear", name = 'nn_k_dense_final',
                                                weights=list(matrix(0,nrow=nunits[n.layers],ncol=1), array(init.kappa)))
    
  }
  #Scale
  if(!is.null(X.nn.s)){
    
    nunits=c(widths,1)
    n.layers=length(nunits)-1
    
    nnBranchs <- input_nn_s
    if(type=="MLP"){
      for(i in 1:n.layers){
        nnBranchs <- nnBranchs  %>% layer_dense(units=nunits[i],activation = 'relu',
                                                input_shape =dim(X.nn.s)[-1], name = paste0('nn_s_dense',i) )
      }
    }else if(type=="CNN"){
      for(i in 1:n.layers){
        nnBranchs <- nnBranchs  %>% layer_conv_2d(filters=nunits[i],activation = 'relu',kernel_size=c(filter.dim[1],filter.dim[2]), padding='same',
                                                  input_shape =dim(X.nn.s)[-1], name = paste0('nn_s_cnn',i) )
      }
      
    }else if(type=="GCNN"){

      for(i in 1:n.layers){
        nnBranchs <- list(nnBranchs,ML)  %>% layer_graph_conv(channels=nunits[i],activation = 'relu',
                                                  input_shape =dim(X.nn.s)[-1], name = paste0('nn_s_gcnn',i) ,
                                                  kernel_initializer=initializer_glorot_uniform(seed=seed))
      }
      
    }
    
    nnBranchs <-   nnBranchs  %>%   layer_dense(units = nunits[n.layers+1], activation = "linear", name = 'nn_s_dense_final',
                                                weights=list(matrix(0,nrow=nunits[n.layers],ncol=1), array(init.scale)))
    
  }
  #Additive towers
  #Kappa
  n.dim.add_k=length(dim(X.add.basis.k))
  if(!is.null(X.add.basis.k) & !is.null(X.add.basis.k) ) {
    
    addBranchk <- input_add_k %>%
      layer_reshape(target_shape=c(dim(X.add.basis.k)[2:(n.dim.add_k-2)],prod(dim(X.add.basis.k)[(n.dim.add_k-1):n.dim.add_k]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_k',
                  weights=list(matrix(0,nrow=prod(dim(X.add.basis.k)[(n.dim.add_k-1):n.dim.add_k]),ncol=1)),use_bias = F)
  }
  if(!is.null(X.add.basis.k) & is.null(X.add.basis.k) ) {
    
    addBranchk <- input_add_k %>%
      layer_reshape(target_shape=c(dim(X.add.basis.k)[2:(n.dim.add_k-2)],prod(dim(X.add.basis.k)[(n.dim.add_k-1):n.dim.add_k]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_k',
                  weights=list(matrix(0,nrow=prod(dim(X.add.basis.k)[(n.dim.add_k-1):n.dim.add_k]),ncol=1),array(init.kappa)),use_bias = T)
  }
  #Scale
  n.dim.add_s=length(dim(X.add.basis.s))
  if(!is.null(X.add.basis.s) & !is.null(X.add.basis.s) ) {
    
    addBranchs <- input_add_s %>%
      layer_reshape(target_shape=c(dim(X.add.basis.s)[2:(n.dim.add_s-2)],prod(dim(X.add.basis.s)[(n.dim.add_s-1):n.dim.add_s]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_s',
                  weights=list(matrix(0,nrow=prod(dim(X.add.basis.s)[(n.dim.add_s-1):n.dim.add_s]),ncol=1)),use_bias = F)
  }
  if(!is.null(X.add.basis.s) & is.null(X.add.basis.s) ) {
    
    addBranchs <- input_add_s %>%
      layer_reshape(target_shape=c(dim(X.add.basis.s)[2:(n.dim.add_s-2)],prod(dim(X.add.basis.s)[(n.dim.add_s-1):n.dim.add_s]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_s',
                  weights=list(matrix(0,nrow=prod(dim(X.add.basis.s)[(n.dim.add_s-1):n.dim.add_s]),ncol=1),array(init.scale)),use_bias = T)
  }
  #Linear towers
  
  #Kappa
  if(!is.null(X.lin.k) ) {
    n.dim.lin_k=length(dim(X.lin.k))
    
    if(is.null(X.nn.k) & is.null(X.add.basis.k )){
      linBranchk <- input_lin_k%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X.lin.k)[-1], name = 'lin_k',
                    weights=list(matrix(0,nrow=dim(X.lin.k)[n.dim.lin_k],ncol=1),array(init.kappa)),use_bias=T)
    }else{
      linBranchk <- input_lin_k%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X.lin.k)[-1], name = 'lin_k',
                    weights=list(matrix(0,nrow=dim(X.lin.k)[n.dim.lin_k],ncol=1)),use_bias=F)
    }
  }
  #Scale
  if(!is.null(X.lin.s) ) {
    n.dim.lin_s=length(dim(X.lin.s))
    
    if(is.null(X.nn.s) & is.null(X.add.basis.s )){
      linBranchs <- input_lin_s%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X.lin.s)[-1], name = 'lin_s',
                    weights=list(matrix(0,nrow=dim(X.lin.s)[n.dim.lin_s],ncol=1),array(init.scale)),use_bias=T)
    }else{
      linBranchs <- input_lin_s%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X.lin.s)[-1], name = 'lin_s',
                    weights=list(matrix(0,nrow=dim(X.lin.s)[n.dim.lin_s],ncol=1)),use_bias=F)
    }
  }
  
  #Stationary towers
  
  #Kappa
  if(is.null(X.nn.k) & is.null(X.add.basis.k) & is.null(X.lin.k)) {
    
    if(!is.null(X.nn.s)){
      statBranchk <- input_nn_s %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X.nn.s)[-1], trainable=F,
                                                weights=list(matrix(0,nrow=dim(X.nn.s)[length(dim(X.nn.s))],ncol=1),array(1,dim=c(1))), name = 'kappa_stationary_dense1') %>%
        layer_dense(units = 1 ,activation = 'linear',use_bias = F,weights=list(matrix(array(init.kappa),nrow=1,ncol=1)), name = 'kappa_stationary_dense2')
    }else  if(!is.null(X.lin.s)){
      statBranchk <- input_lin_s %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X.lin.s)[-1], trainable=F,
                                                 weights=list(matrix(0,nrow=dim(X.lin.s)[length(dim(X.lin.s))],ncol=1),array(1,dim=c(1))), name = 'kappa_stationary_dense1') %>%
        layer_dense(units = 1 ,activation = 'linear',use_bias = F,weights=list(matrix(array(init.kappa),nrow=1,ncol=1)), name = 'kappa_stationary_dense2')
    }else  if(!is.null(X.add.basis.s)){
      statBranchk <- input_add_s %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X.add.basis.s)[-1], trainable=F,
                                                 weights=list(matrix(0,nrow=dim(X.add.basis.s)[length(dim(X.add.basis.s))],ncol=1),array(1,dim=c(1))), name = 'kappa_stationary_dense1') %>%
        layer_dense(units = 1 ,activation = 'linear',use_bias = F,weights=list(matrix(array(init.kappa),nrow=1,ncol=1)), name = 'kappa_stationary_dense2')
    }
    
  }
  
  #Sigma
  if(is.null(X.nn.s) & is.null(X.add.basis.s) & is.null(X.lin.s)) {
    
    if(!is.null(X.nn.k)){
      statBranchs <- input_nn_k %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X.nn.k)[-1], trainable=F,
                                                weights=list(matrix(0,nrow=dim(X.nn.k)[length(dim(X.nn.k))],ncol=1),array(1,dim=c(1))), name = 'sigma_stationary_dense1') %>%
        layer_dense(units = 1 ,activation = 'linear',use_bias = F,weights=list(matrix(array(init.scale),nrow=1,ncol=1)), name = 'sigma_stationary_dense2')
    }else  if(!is.null(X.lin.k)){
      statBranchs <- input_lin_k %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X.nn.k)[-1], trainable=F,
                                                 weights=list(matrix(0,nrow=dim(X.nn.k)[length(dim(X.nn.k))],ncol=1),array(1,dim=c(1))), name = 'sigma_stationary_dense1') %>%
        layer_dense(units = 1 ,activation = 'linear',use_bias = F,weights=list(matrix(array(init.scale),nrow=1,ncol=1)), name = 'sigma_stationary_dense2')
    }else  if(!is.null(X.add.basis.k)){
      statBranchs <- input_add_k %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X.nn.k)[-1], trainable=F,
                                                 weights=list(matrix(0,nrow=dim(X.nn.k)[length(dim(X.nn.k))],ncol=1),array(1,dim=c(1))), name = 'sigma_stationary_dense1') %>%
        layer_dense(units = 1 ,activation = 'linear',use_bias = F,weights=list(matrix(array(init.scale),nrow=1,ncol=1)), name = 'sigma_stationary_dense2')
    }
    
  }
  
  
  #Combine towers
  
  #Kappa
  if(!is.null(X.nn.k) & !is.null(X.add.basis.k) & !is.null(X.lin.k) )  kBranchjoined <- layer_add(inputs=c(addBranchk,  linBranchk,nnBranchk))  #Add all towers
  if(is.null(X.nn.k) & !is.null(X.add.basis.k) & !is.null(X.lin.k) )  kBranchjoined <- layer_add(inputs=c(addBranchk,  linBranchk))  #Add GAM+lin towers
  if(!is.null(X.nn.k) & is.null(X.add.basis.k) & !is.null(X.lin.k) )  kBranchjoined <- layer_add(inputs=c(  linBranchk,nnBranchk))  #Add nn+lin towers
  if(!is.null(X.nn.k) & !is.null(X.add.basis.k) & is.null(X.lin.k) )  kBranchjoined <- layer_add(inputs=c(addBranchk,  nnBranchk))  #Add nn+GAM towers
  if(is.null(X.nn.k) & is.null(X.add.basis.k) & !is.null(X.lin.k) )  kBranchjoined <- linBranchk  #Just lin tower
  if(is.null(X.nn.k) & !is.null(X.add.basis.k) & is.null(X.lin.k) )  kBranchjoined <- addBranchk  #Just GAM tower
  if(!is.null(X.nn.k) & is.null(X.add.basis.k) & is.null(X.lin.k) )  kBranchjoined <- nnBranchk  #Just nn tower
  if(is.null(X.nn.k) & is.null(X.add.basis.k) & is.null(X.lin.k) )  kBranchjoined <- statBranchk  #Just stationary tower
  
  #Scale
  if(!is.null(X.nn.s) & !is.null(X.add.basis.s) & !is.null(X.lin.s) )  sBranchjoined <- layer_add(inputs=c(addBranchs,  linBranchs,nnBranchs))  #Add all towers
  if(is.null(X.nn.s) & !is.null(X.add.basis.s) & !is.null(X.lin.s) )  sBranchjoined <- layer_add(inputs=c(addBranchs,  linBranchs))  #Add GAM+lin towers
  if(!is.null(X.nn.s) & is.null(X.add.basis.s) & !is.null(X.lin.s) )  sBranchjoined <- layer_add(inputs=c(  linBranchs,nnBranchs))  #Add nn+lin towers
  if(!is.null(X.nn.s) & !is.null(X.add.basis.s) & is.null(X.lin.s) )  sBranchjoined <- layer_add(inputs=c(addBranchs,  nnBranchs))  #Add nn+GAM towers
  if(is.null(X.nn.s) & is.null(X.add.basis.s) & !is.null(X.lin.s) )  sBranchjoined <- linBranchs  #Just lin tower
  if(is.null(X.nn.s) & !is.null(X.add.basis.s) & is.null(X.lin.s) )  sBranchjoined <- addBranchs  #Just GAM tower
  if(!is.null(X.nn.s) & is.null(X.add.basis.s) & is.null(X.lin.s) )  sBranchjoined <- nnBranchs  #Just nn tower
  if(is.null(X.nn.s) & is.null(X.add.basis.s) & is.null(X.lin.s) )  sBranchjoined <- statBranchs  #Just stationary tower
  
  #Apply link functions
  kBranchjoined <- kBranchjoined %>% layer_activation( activation = 'exponential', name = "kappa_activation") 
  sBranchjoined <- sBranchjoined %>% layer_activation( activation = 'exponential', name = "sigma_activation")
  
  #Accommodate offset if available
  if(!is.null(offset))    sBranchjoined <- layer_multiply(inputs=c(input_offset,sBranchjoined))
  input=c()

  if(!is.null(X.lin.s) ) input=c(input,input_lin_s)
  if(!is.null(X.add.basis.s) ) input=c(input,input_add_s)
  if(!is.null(X.nn.s) ) input=c(input,input_nn_s)
  if(!is.null(offset)) input=c(input,input_offset)
  if(!is.null(X.lin.k) ) input=c(input,input_lin_k)
  if(!is.null(X.add.basis.k) ) input=c(input,input_add_k)
  if(!is.null(X.nn.k) ) input=c(input,input_nn_k)
  input=c(input)

  output <- layer_concatenate(c(sBranchjoined, kBranchjoined, xiBranch),name="Combine_parameter_tensors")
  
  model <- keras_model(  inputs = input,   outputs = output,name=paste0("eGPD"))
  print(model)
  
  return(model)
  
}


eGPD_loss <-function(S_lambda=NULL){
  
  S_lambda.k=S_lambda$S_lambda.k;   S_lambda.s=S_lambda$S_lambda.s
  
  if(is.null(S_lambda.k) & is.null(S_lambda.s)){
    loss<- function( y_true, y_pred) {
      
      K <- backend()
      
      sig=y_pred[all_dims(),1]
      kappa=y_pred[all_dims(),2]
      xi=y_pred[all_dims(),3]
      y=K$relu(y_true)
      
      
      sig=sig- sig*(1-K$sign(y))+(1-K$sign(y)) #If no exceedance, set sig to 1
      kappa=kappa- kappa*(1-K$sign(y))+(1-K$sign(y)) #If no exceedance, set kappa to 1
      xi=xi- xi*(1-K$sign(y))+(1-K$sign(y)) #If no exceedance, set kappa to 1
      
      
      #Evaluate log-likelihood
      ll1=-(1/xi+1)*tf$math$log1p(xi*y/sig)
      
      #Uses non-zero response values only
      ll2= K$log(sig) *K$sign(ll1)
      
      ll3=-K$log(kappa) *K$sign(ll1)
      
      y=y- y*(1-K$sign(y))+(1-K$sign(y)) #If zero, set y to 1
      
      ll4=(kappa-1)*K$log(1-(1+xi*y/sig)^(-1/xi))
      
      
      return(-(K$sum(ll1+ll2+ll3+ll4)))
    }
  }else if(!is.null(S_lambda.k) & !is.null(S_lambda.s)){
    loss<- function( y_true, y_pred) {
      
      library(tensorflow)
      K <- backend()

      kappa=y_pred[all_dims(),2]
      sig=y_pred[all_dims(),1]
      xi=y_pred[all_dims(),3]
      y=K$relu(y_true)
      
      
      sig=sig- sig*(1-K$sign(y))+(1-K$sign(y)) #If no exceedance, set sig to 1
      kappa=kappa- kappa*(1-K$sign(y))+(1-K$sign(y)) #If no exceedance, set kappa to 1
      xi=xi- xi*(1-K$sign(y))+(1-K$sign(y)) #If no exceedance, set kappa to 1
      
      
      t.gam.weights.k=K$constant(t(model$get_layer("add_k")$get_weights()[[1]]))
      gam.weights.k=K$constant(model$get_layer("add_k")$get_weights()[[1]])
      S_lambda.k.tensor=K$constant(S_lambda.k)
      
      t.gam.weights.s=K$constant(t(model$get_layer("add_s")$get_weights()[[1]]))
      gam.weights.s=K$constant(model$get_layer("add_s")$get_weights()[[1]])
      S_lambda.s.tensor=K$constant(S_lambda.s)
      
      penalty = 0.5*K$dot(t.gam.weights.k,K$dot(S_lambda.k.tensor,gam.weights.k))+0.5*K$dot(t.gam.weights.s,K$dot(S_lambda.s.tensor,gam.weights.s))
      
      
      #Evaluate log-likelihood
      ll1=-(1/xi+1)*tf$math$log1p(xi*y/sig)
      
      #Uses non-zero response values only
      ll2= K$log(sig) *K$sign(ll1)
      
      ll3=-K$log(kappa) *K$sign(ll1)
      
      y=y- y*(1-K$sign(y))+(1-K$sign(y)) #If zero, set y to 1
      
      ll4=(kappa-1)*K$log(1-(1+xi*y/sig)^(-1/xi))
      
      
      return(penalty-
               (K$sum(ll1+ll2+ll3+ll4))
             )
 
    }
  }else if(is.null(S_lambda.k) & !is.null(S_lambda.s)){
    loss<- function( y_true, y_pred) {
      
      library(tensorflow)
      K <- backend()
      
      sig=y_pred[all_dims(),1]
      kappa=y_pred[all_dims(),2]
      xi=y_pred[all_dims(),3]
      y=K$relu(y_true)
      
      
      sig=sig- sig*(1-K$sign(y))+(1-K$sign(y)) #If no exceedance, set sig to 1
      kappa=kappa- kappa*(1-K$sign(y))+(1-K$sign(y)) #If no exceedance, set kappa to 1
      xi=xi- xi*(1-K$sign(y))+(1-K$sign(y)) #If no exceedance, set kappa to 1
      
      
      t.gam.weights.s=K$constant(t(model$get_layer("add_s")$get_weights()[[1]]))
      gam.weights.s=K$constant(model$get_layer("add_s")$get_weights()[[1]])
      S_lambda.s.tensor=K$constant(S_lambda.s)
      
      penalty = 0.5*K$dot(t.gam.weights.s,K$dot(S_lambda.s.tensor,gam.weights.s))
  
      
      #Evaluate log-likelihood
      ll1=-(1/xi+1)*tf$math$log1p(xi*y/sig)
      
      #Uses non-zero response values only
      ll2= K$log(sig) *K$sign(ll1)
      
      ll3=-K$log(kappa) *K$sign(ll1)
      
      y=y- y*(1-K$sign(y))+(1-K$sign(y)) #If zero, set y to 1
      
      ll4=(kappa-1)*K$log(1-(1+xi*y/sig)^(-1/xi))
      
      
      return(penalty-
               (K$sum(ll1+ll2+ll3+ll4))
      )
      
    }
  }else if(!is.null(S_lambda.k) & is.null(S_lambda.s)){
    loss<- function( y_true, y_pred) {
      
      library(tensorflow)
      K <- backend()
      
      sig=y_pred[all_dims(),1]
      kappa=y_pred[all_dims(),2]
      xi=y_pred[all_dims(),3]
      y=K$relu(y_true)
      
      
      sig=sig- sig*(1-K$sign(y))+(1-K$sign(y)) #If no exceedance, set sig to 1
      kappa=kappa- kappa*(1-K$sign(y))+(1-K$sign(y)) #If no exceedance, set kappa to 1
      xi=xi- xi*(1-K$sign(y))+(1-K$sign(y)) #If no exceedance, set kappa to 1
      
  
  
      t.gam.weights.k=K$constant(t(model$get_layer("add_k")$get_weights()[[1]]))
      gam.weights.k=K$constant(model$get_layer("add_k")$get_weights()[[1]])
      S_lambda.k.tensor=K$constant(S_lambda.k)

      penalty = 0.5*K$dot(t.gam.weights.k,K$dot(S_lambda.k.tensor,gam.weights.k))
      
      
      #Evaluate log-likelihood
      ll1=-(1/xi+1)*tf$math$log1p(xi*y/sig)
      
      #Uses non-zero response values only
      ll2= K$log(sig) *K$sign(ll1)
      
      ll3=-K$log(kappa) *K$sign(ll1)
      
      y=y- y*(1-K$sign(y))+(1-K$sign(y)) #If zero, set y to 1
      
      ll4=(kappa-1)*K$log(1-(1+xi*y/sig)^(-1/xi))
      
      
      return(penalty-
               (K$sum(ll1+ll2+ll3+ll4))
      )
    }
  }
  return(loss)
}
