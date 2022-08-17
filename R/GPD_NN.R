#'GPD PINN
#'
#' Build and train a partially-interpretable neural network for fitting a GPD model to threshold exceedances 
#'

#' @param type  string defining the type of network to be built. If \code{type=="MLP"}, the network will have all densely connected layers; if \code{type=="CNN"},
#'  the network will have all convolutional layers. Defaults to an MLP. (Currently the same network is used for all parameters, may change in future versions)
#' @param Y.train,Y.valid a 2 or 3 dimensional array of training or validation real response values.
#' Missing values can be handled by setting corresponding entries to \code{Y.train} or \code{Y.valid} to \code{-1e5}.
#' The first dimension should be the observation indices, e.g., time.
#'
#' If \code{type=="CNN"}, then \code{Y.train} and \code{Y.valid} must have three dimensions with the latter two corresponding to an \eqn{M} by \eqn{N} regular grid of spatial locations.
#' If \code{Y.valid==NULL}, no validation loss will be computed and the returned model will be that which minimises the training loss over \code{n.ep} epochs.
#'
#'@param u.train an array with the same dimension as \code{Y.train}. Gives the threshold used to create exceedances of \code{Y.train}, see below. Note that \code{u.train} is applied to both \code{Y.train} and \code{Y.valid}.
#' @param X.train  list of arrays corresponding to complementary subsets of the \eqn{d\geq 1} predictors which are used for modelling the scale parameter \eqn{\sigma}. Must contain at least one of the following three named entries:\describe{
#' \item{\code{X.train.lin}}{A 3 or 4 dimensional array of "linear" predictor values. One more dimension then \code{Y.train}. If \code{NULL}, a model without the linear component is built and trained.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the last dimension corresponds to the chosen \eqn{l\geq 0} 'linear' predictor values.}
#' \item{\code{X.train.add.basis}}{A 4 or 5 dimensional array of basis function evaluations for the "additive" predictor values.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the penultimate dimensions corresponds to the chosen \eqn{a\geq 0} 'linear' predictor values and the last dimension is equal to the number of knots used for estimating the splines. See example.
#' If \code{NULL}, a model without the additive component is built and trained.}
#' \item{\code{X.train.nn}}{A 3 or 4 dimensional array of "non-additive" predictor values.  If \code{NULL}, a model without the NN component is built and trained; if this is the case, then \code{type} has no effect.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the last dimension corresponds to the chosen \eqn{d-l-a\geq 0} 'non-additive' predictor values.}
#' }
#' Note that \code{X.train} is the predictors for both \code{Y.train} and \code{Y.valid}.
#' @param n.ep number of epochs used for training. Defaults to 1000.
#' @param batch.size batch size for stochastic gradient descent. If larger than \code{dim(Y.train)[1]}, i.e., the number of observations, then regular gradient descent used.
#' @param init.scale,init.xi sets the initial \eqn{sigma} and \eqn{\xi\in(0,1)} estimates across all dimensions of \code{Y.train}. Overridden by \code{init.wb_path} if \code{!is.null(init.wb_path)}, but otherwise the initial parameters must be supplied.
#' @param init.wb_path filepath to a \code{keras} model which is then used as initial weights and biases for training the new model. The original model must have
#' the exact same architecture and trained with the same input data as the new model. If \code{NULL}, then initial weights and biases are random (with seed \code{seed}) but the
#' final layer has zero initial weights to ensure that the initial scale and shape estimates are \code{ init.scale} and \code{init.xi}, respectively,  across all dimensions.
#' @param widths vector of widths/filters for hidden dense/convolution layers. Number of layers is equal to \code{length(widths)}. Defaults to (6,3).
#' @param filter.dim if \code{type=="CNN"}, this 2-vector gives the dimensions of the convolution filter kernel; must have odd integer inputs. Note that filter.dim=c(1,1) is equivalent to \code{type=="MLP"}. The same filter is applied for each hidden layer across all parameters with NN predictors.
#' @param seed seed for random initial weights and biases.
#' @param model fitted \code{keras} model. Output from \code{GPD.NN.train}.
#' @param S_lamda smoothing penalty matrix for the splines modelling the effect of \code{X.train.add.basis} on \eqn{\log(\sigma)}; only used if \code{!is.null(X.train.add.basis)}. If \code{is.null(S_lambda)}, then no smoothing penalty used.

#'@name GPD.NN

#' @details{
#' Consider a real-valued random variable \eqn{Y} and let \eqn{\mathbf{X}} denote a \eqn{d}-dimensional predictor set with observations \eqn{\mathbf{x}}.
#' For integers \eqn{l\geq 0,a \geq 0} and \eqn{0\leq l+a \leq d}, let \eqn{\mathbf{X}_L, \mathbf{X}_A} and \eqn{\mathbf{X}_N} be distinct sub-vectors of \eqn{\mathbf{X}},
#'  with observations of each component denoted \eqn{\mathbf{x}_L, \mathbf{x}_A} and \eqn{\mathbf{x}_N}, respectively; the lengths of the sub-vectors are \eqn{l,a} and \eqn{d-l-a}, respectively.
#' For a fixed threshold \eqn{u(\mathbf{x})}, dependent on predictors, we model \eqn{(Y-u(\mathbf{x}))|\mathbf{X}=\mathbf{x}\sim\mbox{GPD}(\sigma(\mathbf{X})),\xi;u(\mathbf{x}))} for \eqn{\xi\in(0,1)} with
#' \deqn{\sigma (\mathbf{x})=\exp[\eta_0+m_L\{\mathbf{x}_L\}+m_A\{x_A\}+m_N\{\mathbf{x}_N\}]}
#' where \eqn{\eta_0} is a constant intercept. The unknown functions \eqn{m_L} and
#' \eqn{m_A} are estimated using linear functions and splines, respectively, and are
#' both returned as outputs by \code{GPD.NN.predict}; \eqn{m_N} is estimated using a neural network
#' (currently the same architecture is used for both parameters). Note that \eqn{\xi>0} is fixed across all predictors; this may change in future versions.
#'
#' The model is fitted by minimising the negative log-likelihood associated with the GPD model over \code{n.ep} training epochs.
#' Although the model is trained by minimising the loss evaluated for \code{Y.train}, the final returned model may minimise some other loss.
#' The current state of the model is saved after each epoch, using \code{keras::callback_model_checkpoint}, if the value of some criterion subcedes that of the model from the previous checkpoint; this criterion is the loss evaluated for validation set \code{Y.valid} if \code{!is.null(Y.valid)} and for \code{Y.train}, otherwise.
#'
#'}
#' @return \code{bGEVPP.NN.train} returns the fitted \code{model}.  \code{bGEVPP.NN.predict} is a wrapper for \code{keras::predict} that returns the predicted parameter estimates, and, if applicable, their corresponding linear regression coefficients and spline bases weights.
#'
#'@references{
#' Coles, S. G. (2001), \emph{An Introduction to Statistical Modeling of Extreme Values}. Volume 208, Springer. (\href{https://doi.org/10.1007/F978-1-4471-3675-0}{doi})
#' 
#' Richards, J. and Huser, R. (2022), \emph{A unifying partially-interpretable framework for neural network-based extreme quantile regression}. (\href{https://arxiv.org/abs/2208.07581}{arXiv:2208.07581}).
#'}
#' @examples
#'
#'#Apply model to toy data
#'
#' # Create  predictors
#' preds<-rnorm(128000)
#' 
#' #Re-shape to a 4d array. First dimension corresponds to observations,
#' #last to the different components of the predictor set
#' dim(preds)=c(200,8,8,10) #We have ten predictors
#' 
#' #Split predictors into linear, additive and nn. 
#' 
#' X.train.nn=preds[,,,1:5] #Five nn predictors 
#' X.train.lin=preds[,,,6:8] #Three linear predictors 
#' X.train.add=preds[,,,9:10] #Two additive predictors 
#' 
#' 
#' # Create response data
#' 
#' #Contribution to scale parameter
#' #Linear contribution
#' m_L = 0.5*X.train.lin[,,,1]+0.3*X.train.lin[,,,2]-0.4*X.train.lin[,,,3]
#' 
#' # Additive contribution
#' m_A = 0.2*X.train.add[,,,1]^2+0.05*X.train.add[,,,1]-0.1*X.train.add[,,,2]^2+
#'   0.01*X.train.add[,,,2]^3
#' 
#' #Non-additive contribution - to be estimated by NN
#' m_N =0.5*(exp(-4+X.train.nn[,,,2]+X.train.nn[,,,3])+
#'             sin(X.train.nn[,,,1]-X.train.nn[,,,2])*(X.train.nn[,,,1]+X.train.nn[,,,2])-
#'             cos(X.train.nn[,,,3]-X.train.nn[,,,4])*(X.train.nn[,,,2]+X.train.nn[,,,5]))
#' 
#'sigma=2*exp(-2+m_L+m_A+m_N) #Exponential link
#' xi=0.1 # Set xi
#' 
#' #We simulate data as exceedances above some random positive threshold u. 
#' u<-apply(sigma,1:3,function(x) rgpd(n=1,loc=0,scale=1,shape=0.1) ) #Random threshold
#' 
#' theta=array(dim=c(dim(sigma),3))
#' theta[,,,1]=sigma; theta[,,,2]  =xi; theta[,,,3]  =u
#' 
#' #If u were the true 80% quantile, say,  of the response, then only 20% of the data should exceed u. 
#' #We achieve this by simulating a Bernoulli variable to determine if Y exceeds u
#' 
#' Y=apply(theta,1:3,function(x){ 
#'   if(rbinom(1,1,0.8)==1) rgpd(n=1,loc=x[3],scale=x[1],shape=x[2]) else  runif(1,0,x[3])
#' }
#' ) #Simulate GPD exceedances above u as given above
#'
#' 
#' #Create training and validation, respectively.
#' #We mask 20% of the Y values and use this for validation
#' #Masked values must be set to -1e5 and are treated as missing whilst training
#' 
#' mask_inds=sample(1:length(Y),size=length(Y)*0.8)
#' 
#' Y.train<-Y.valid<-Y #Create training and validation, respectively.
#' Y.train[-mask_inds]=-1e5
#' Y.valid[mask_inds]=-1e5
#' 
#' 
#' 
#' #To build a model with an additive component, we require an array of evaluations of
#' #the basis functions for each pre-specified knot and entry to X.train.add
#' 
#' rad=function(x,c){ #Define a basis function. Here we use the radial bases
#'   out=abs(x-c)^2*log(abs(x-c))
#'   out[(x-c)==0]=0
#'   return(out)
#' }
#' 
#' n.knot = 5 # set number of knots. Must be the same for each additive predictor
#' 
#' knots=matrix(nrow=dim(X.train.add)[4],ncol=n.knot)
#' 
#' #We set knots to be equally-spaced marginal quantiles
#' for( i in 1:dim(X.train.add)[4]){
#'  knots[i,]=quantile(X.train.add[,,,i],probs=seq(0,1,length=n.knot))
#' }
#' 
#' X.train.add.basis<-array(dim=c(dim(X.train.add),n.knot))
#' for( i in 1:dim(X.train.add)[4]) {
#'   for(k in 1:n.knot) {
#'     X.train.add.basis[,,,i,k]= rad(x=X.train.add[,,,i],c=knots[i,k])
#'     #Evaluate rad at all entries to X.train.add and for all knots
#'   }}
#' 
#' #Create smoothing penalty matrix for the two additive functions
#' 
#'# Set smoothness parameters for three functions
#'  lambda = c(0.5,1) 
#'
#' S_lambda=matrix(0,nrow=n.knot*dim(X.train.add)[4],ncol=n.knot*dim(X.train.add)[4])
#'for(i in 1:dim(X.train.add)[4]){
#'  for(j in 1:n.knot){
#'   for(k in 1:n.knot){
#'      S_lambda[(j+(i-1)*n.knot),(k+(i-1)*n.knot)]=lambda[i]*rad(knots[i,j],knots[i,k])
#'   }
#' }
#'}
#' 
#' #lin+GAM+NN models defined for scale parameter
#' X.train=list("X.train.nn"=X.train.nn, "X.train.lin"=X.train.lin,
#'              "X.train.add.basis"=X.train.add.basis) 
#' 
#' #We treat u as fixed and known. In an application, u can be estimated using quant.NN.train.
#' 
#' u.train <- u
#' 
#' #Fit the GPD model for exceedances above u.train
#' model<-GPD.NN.train(Y.train, Y.valid,X.train, u.train, type="MLP",
#'                     n.ep=500, batch.size=50,init.scale=1, init.xi=0.1,
#'                     widths=c(6,3),seed=1,S_lambda=S_lambda)
#' out<-GPD.NN.predict(X.train=X.train,u.train=u.train,model)
#' 
#' print("sigma linear coefficients: "); print(round(out$lin.coeff_sigma,2))
#' 
#' #To save model, run
#' #model %>% save_model_tf("model_GPD")
#' To load model, run
#' #model  <- load_model_tf("model_GPD",
#' #custom_objects=list("GPD_loss_S_lambda___S_lambda_"=GPD_loss(S_lambda=S_lambda)))
#' 
#' 
#' # Plot splines for the additive predictors
#' n.add.preds=dim(X.train.add)[length(dim(X.train.add))]
#' par(mfrow=c(1,n.add.preds))
#' for(i in 1:n.add.preds){
#'   plt.x=seq(from=min(knots[i,]),to=max(knots[i,]),length=1000)  #Create sequence for x-axis
#'   
#'   tmp=matrix(nrow=length(plt.x),ncol=n.knot)
#'   for(j in 1:n.knot){
#'     tmp[,j]=rad(plt.x,knots[i,j]) #Evaluate radial basis function of plt.x and all knots
#'   }
#'   plt.y=tmp%*%out$gam.weights_sigma[i,]
#'   plot(plt.x,plt.y,type="l",main=paste0("sigma spline: predictor ",i),xlab="x",ylab="f(x)")
#'   points(knots[i,],rep(mean(plt.y),n.knot),col="red",pch=2)
#'   #Adds red triangles that denote knot locations
#'   
#' }
#' @import reticulate keras tensorflow
#' @importFrom evd rgpd
#'
#' @rdname GPD.NN
#' @export


  GPD.NN.train=function(Y.train, Y.valid = NULL,X.train,u.train = NULL,   type="MLP",link="identity",tau=NULL,
                          n.ep=100, batch.size=100,init.scale=NULL,init.xi=NULL, widths=c(6,3), filter.dim=c(3,3),seed=NULL,init.wb_path=NULL,S_lambda=NULL)
  {
    
  
  
  
  if(is.null(X.train)  ) stop("No predictors provided for sigma")
  if(is.null(Y.train)) stop("No training response data provided")
  if(is.null(u.train)) stop("No threshold u.train provided")
  
  if(is.null(init.scale) & is.null(init.wb_path)   ) stop("Inital scale estimate not provided")
  if(is.null(init.xi)  & is.null(init.wb_path) ) stop("Inital shape estimate not provided")
  
  
  print(paste0("Creating GPD model"))
  X.train.nn=X.train$X.train.nn
  X.train.lin=X.train$X.train.lin
  X.train.add.basis=X.train$X.train.add.basis
  
  if(!is.null(X.train.nn) & !is.null(X.train.add.basis) & !is.null(X.train.lin) ) {  train.data= list(X.train.lin,X.train.add.basis,X.train.nn,u.train); print("Defining lin+GAM+NN model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_sigma=X.train.lin,add_input_sigma=X.train.add.basis,  nn_input_sigma=X.train.nn,u_input=u.train),Y.valid)}
  if(is.null(X.train.nn) & !is.null(X.train.add.basis) & !is.null(X.train.lin) ) {   train.data= list(X.train.lin,X.train.add.basis,u.train); print("Defining lin+GAM model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_sigma=X.train.lin,add_input_sigma=X.train.add.basis,u_input=u.train),Y.valid)}
  if(!is.null(X.train.nn) & is.null(X.train.add.basis) & !is.null(X.train.lin) ) { train.data= list(X.train.lin,X.train.nn,u.train); print("Defining lin+NN model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_sigma=X.train.lin, nn_input_sigma=X.train.nn,u_input=u.train),Y.valid)}
  if(!is.null(X.train.nn) & !is.null(X.train.add.basis) & is.null(X.train.lin) ) {train.data= list(X.train.add.basis,X.train.nn,u.train); print("Defining GAM+NN model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list(add_input_sigma=X.train.add.basis,  nn_input_sigma=X.train.nn,u_input=u.train),Y.valid)}
  if(is.null(X.train.nn) & is.null(X.train.add.basis) & !is.null(X.train.lin) )   {train.data= list(X.train.lin,u.train); print("Defining fully-linear model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_sigma=X.train.lin,u_input=u.train),Y.valid)}
  if(is.null(X.train.nn) & !is.null(X.train.add.basis) & is.null(X.train.lin) )   {train.data= list(X.train.add.basis,u.train); print("Defining fully-additive model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list(add_input_sigma=X.train.add.basis,u_input=u.train),Y.valid)}
  if(!is.null(X.train.nn) & is.null(X.train.add.basis) & is.null(X.train.lin) )   {train.data= list(X.train.nn,u.train); print("Defining fully-NN model for sigma" );  if(!is.null(Y.valid)) validation.data=list(list( nn_input_sigma=X.train.nn,u_input=u.train),Y.valid)}
  
  if(is.null(S_lambda) & !is.null(X.train.add.basis)){print("No smoothing penalty used")}
  if(is.null(X.train.add.basis)){S_lambda=NULL}
  
  if(type=="CNN" & !is.null(X.train.nn)) print(paste0("Building ",length(widths),"-layer convolutional neural network with ", filter.dim[1]," by ", filter.dim[2]," filter" ))
  if(type=="MLP"  & !is.null(X.train.nn) ) print(paste0("Building ",length(widths),"-layer densely-connected neural network" ))
  
  reticulate::use_virtualenv("myenv", required = T)
  
  if(!is.null(seed)) tf$random$set_seed(seed)
  
  if(length(dim(u.train))!=length(dim(Y.train))+1) dim(u.train)=c(dim(u.train),1)
  model<-GPD.NN.build(X.train.nn,X.train.lin,X.train.add.basis,
                         u.train,type,init.scale,init.xi, widths,filter.dim)
  if(!is.null(init.wb_path)) model <- load_model_weights_tf(model,filepath=init.wb_path)
  
  model %>% compile(
    optimizer="adam",
    loss = GPD_loss(S_lambda=S_lambda),
    run_eagerly=T
  )
  
  if(!is.null(Y.valid)) checkpoint <- callback_model_checkpoint(paste0("model_GPD_checkpoint"), monitor = "val_loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch") else checkpoint <- callback_model_checkpoint(paste0("model_GPD_checkpoint"), monitor = "loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch")
  
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
  model <- load_model_weights_tf(model,filepath=paste0("model_GPD_checkpoint"))
  
  
  return(model)
}
#' @rdname GPD.NN
#' @export
#'
  GPD.NN.predict=function(X.train,u.train, model)
{
  library(tensorflow)
  if(is.null(X.train)  ) stop("No predictors provided for sigma")

  
  
  X.train.nn=X.train$X.train.nn
  X.train.lin=X.train$X.train.lin
  X.train.add.basis=X.train$X.train.add.basis
  
  
  if(!is.null(X.train.nn) & !is.null(X.train.add.basis) & !is.null(X.train.lin) )   train.data= list(X.train.lin,X.train.add.basis,X.train.nn,u.train)
  if(is.null(X.train.nn) & !is.null(X.train.add.basis) & !is.null(X.train.lin) )   train.data= list(X.train.lin,X.train.add.basis,u.train)
  if(!is.null(X.train.nn) & is.null(X.train.add.basis) & !is.null(X.train.lin) )  train.data= list(X.train.lin,X.train.nn,u.train)
  if(!is.null(X.train.nn) & !is.null(X.train.add.basis) & is.null(X.train.lin) ) train.data= list(X.train.add.basis,X.train.nn,u.train)
  if(is.null(X.train.nn) & is.null(X.train.add.basis) & !is.null(X.train.lin) )   train.data= list(X.train.lin,u.train)
  if(is.null(X.train.nn) & !is.null(X.train.add.basis) & is.null(X.train.lin) )   train.data= list(X.train.add.basis,u.train)
  if(!is.null(X.train.nn) & is.null(X.train.add.basis) & is.null(X.train.lin) )   train.data= list(X.train.nn,u.train)
  

  
  predictions<-model %>% predict( train.data)
  predictions <- k_constant(predictions)
  pred.sigma=k_get_value(predictions[all_dims(),2])
  pred.xi=k_get_value(predictions[all_dims(),3])
  
  if(!is.null(X.train.add.basis))  gam.weights_sigma<-matrix(t(model$get_layer("add_sigma")$get_weights()[[1]]),nrow=dim(X.train.add.basis)[length(dim(X.train.add.basis))-1],ncol=dim(X.train.add.basis)[length(dim(X.train.add.basis))],byrow=T)

  out=list("pred.sigma"=pred.sigma,"pred.xi"=pred.xi)
  if(!is.null(X.train.lin) ) out=c(out,list("lin.coeff_sigma"=c(model$get_layer("lin_sigma")$get_weights()[[1]])))
  if(!is.null(X.train.add.basis) ) out=c(out,list("gam.weights_sigma"=gam.weights_sigma))
  
  return(out)
  
}
#'
#'
GPD.NN.build=function(X.train.nn,X.train.lin,X.train.add.basis,
                         u.train,
                         type,init.scale,init.xi, widths,filter.dim)
{
  #Additive input
  if(!is.null(X.train.add.basis))  input_add_sigma<- layer_input(shape = dim(X.train.add.basis)[-1], name = 'add_input_sigma')
  
  #NN input
  
  if(!is.null(X.train.nn))   input_nn_sigma <- layer_input(shape = dim(X.train.nn)[-1], name = 'nn_input_sigma')
  
  #Linear input
  if(!is.null(X.train.lin)) input_lin_sigma <- layer_input(shape = dim(X.train.lin)[-1], name = 'lin_input_sigma')
  
  #Threshold input
  input_u <- layer_input(shape = dim(u.train)[-1], name = 'u_input')
  
  #Create xi branch
  
  xiBranch <- input_u %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(u.train)[-1], trainable=F,
                                      weights=list(matrix(0,nrow=dim(u.train)[length(dim(u.train))],ncol=1),array(1,dim=c(1))), name = 'xi_dense') %>%
    layer_dense(units = 1 ,activation = 'sigmoid',use_bias = F,weights=list(matrix(qlogis(init.xi),nrow=1,ncol=1)), name = 'xi_activation')
  
  
  
  
  init.scale=log(init.scale)
  #NN towers
 
  #Sigma
  if(!is.null(X.train.nn)){
    
    nunits=c(widths,1)
    n.layers=length(nunits)-1
    
    nnBranchsigma <- input_nn_sigma
    if(type=="MLP"){
      for(i in 1:n.layers){
        nnBranchsigma <- nnBranchsigma  %>% layer_dense(units=nunits[i],activation = 'relu',
                                                input_shape =dim(X.train.nn)[-1], name = paste0('nn_sigma_dense',i) )
      }
    }else if(type=="CNN"){
      for(i in 1:n.layers){
        nnBranchsigma <- nnBranchsigma  %>% layer_conv_2d(filters=nunits[i],activation = 'relu',kernel_size=c(filter.dim[1],filter.dim[2]), padding='same',
                                                  input_shape =dim(X.train.nn)[-1], name = paste0('nn_sigma_cnn',i) )
      }
      
    }
    
    nnBranchsigma <-   nnBranchsigma  %>%   layer_dense(units = nunits[n.layers+1], activation = "linear", name = 'nn_sigma_dense_final',
                                                weights=list(matrix(0,nrow=nunits[n.layers],ncol=1), array(init.scale)))
    
  }
  #Additive towers
  
  #Scale
  n.dim.add_sigma=length(dim(X.train.add.basis))
  if(!is.null(X.train.add.basis) & !is.null(X.train.add.basis) ) {
    
    addBranchsigma <- input_add_sigma %>%
      layer_reshape(target_shape=c(dim(X.train.add.basis)[2:(n.dim.add_sigma-2)],prod(dim(X.train.add.basis)[(n.dim.add_sigma-1):n.dim.add_sigma]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_sigma',
                  weights=list(matrix(0,nrow=prod(dim(X.train.add.basis)[(n.dim.add_sigma-1):n.dim.add_sigma]),ncol=1)),use_bias = F)
  }
  if(!is.null(X.train.add.basis) & is.null(X.train.add.basis) ) {
    
    addBranchsigma <- input_add_sigma %>%
      layer_reshape(target_shape=c(dim(X.train.add.basis)[2:(n.dim.add_sigma-2)],prod(dim(X.train.add.basis)[(n.dim.add_sigma-1):n.dim.add_sigma]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_sigma',
                  weights=list(matrix(0,nrow=prod(dim(X.train.add.basis)[(n.dim.add_sigma-1):n.dim.add_sigma]),ncol=1),array(init.scale)),use_bias = T)
  }
  #Linear towers
  
  
  #Scale
  if(!is.null(X.train.lin) ) {
    n.dim.lin_sigma =length(dim(X.train.lin))
    
    if(is.null(X.train.nn) & is.null(X.train.add.basis )){
      linBranchsigma <- input_lin_sigma%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X.train.lin)[-1], name = 'lin_sigma',
                    weights=list(matrix(0,nrow=dim(X.train.lin)[n.dim.lin_sigma],ncol=1),array(init.scale)),use_bias=T)
    }else{
      linBranchsigma <- input_lin_sigma%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X.train.lin)[-1], name = 'lin_sigma',
                    weights=list(matrix(0,nrow=dim(X.train.lin)[n.dim.lin_sigma],ncol=1)),use_bias=F)
    }
  }
  
 
  #Scale
  if(!is.null(X.train.nn) & !is.null(X.train.add.basis) & !is.null(X.train.lin) )  sigmaBranchjoined <- layer_add(inputs=c(addBranchsigma,  linBranchsigma,nnBranchsigma),name="Combine_sigma_components")  #Add all towers
  if(is.null(X.train.nn) & !is.null(X.train.add.basis) & !is.null(X.train.lin) )  sigmaBranchjoined <- layer_add(inputs=c(addBranchsigma,  linBranchsigma),name="Combine_sigma_components")  #Add GAM+lin towers
  if(!is.null(X.train.nn) & is.null(X.train.add.basis) & !is.null(X.train.lin) )  sigmaBranchjoined <- layer_add(inputs=c(  linBranchsigma,nnBranchsigma),name="Combine_sigma_components")  #Add nn+lin towers
  if(!is.null(X.train.nn) & !is.null(X.train.add.basis) & is.null(X.train.lin) )  sigmaBranchjoined <- layer_add(inputs=c(addBranchsigma,  nnBranchsigma),name="Combine_sigma_components")  #Add nn+GAM towers
  if(is.null(X.train.nn) & is.null(X.train.add.basis) & !is.null(X.train.lin) )  sigmaBranchjoined <- linBranchsigma  #Just lin tower
  if(is.null(X.train.nn) & !is.null(X.train.add.basis) & is.null(X.train.lin) )  sigmaBranchjoined <- addBranchsigma  #Just GAM tower
  if(!is.null(X.train.nn) & is.null(X.train.add.basis) & is.null(X.train.lin) )  sigmaBranchjoined <- nnBranchsigma  #Just nn tower
  
  #Apply link functions
  sigmaBranchjoined <- sigmaBranchjoined %>% layer_activation( activation = 'exponential', name = "sigma_activation")
  
  input=c()

  if(!is.null(X.train.lin) ) input=c(input,input_lin_sigma)
  if(!is.null(X.train.add.basis) ) input=c(input,input_add_sigma)
  if(!is.null(X.train.nn) ) input=c(input,input_nn_sigma)
  input=c(input,input_u)
  
  
  output <- layer_concatenate(c(input_u,sigmaBranchjoined, xiBranch),name="Combine_parameter_tensors")
  
  model <- keras_model(  inputs = input,   outputs = output,name=paste0("GPD"))
  print(model)
  
  return(model)
  
}


GPD_loss <- function(S_lambda=NULL){
  
  if(is.null(S_lambda)){
    loss<-function( y_true, y_pred) {
      
      K <- backend()
      
      u=y_pred[all_dims(),1]
      sig=y_pred[all_dims(),2]
      xi=y_pred[all_dims(),3]
      y=K$relu(y_true-u)
      
      
      #Evaluate log-likelihood
      ll1=-(1/xi+1)*K$log(1+xi*y/sig)
      
      #Uses non-zero response values only
      ll2= K$log(sig) *K$sign(ll1)
      
      return(-(K$sum(ll1+ll2)))
    }
  }else{
    loss<-function( y_true, y_pred) {
      
    K <- backend()
    
    t.gam.weights=K$constant(t(model$get_layer("add_sigma")$get_weights()[[1]]))
    gam.weights=K$constant(model$get_layer("add_sigma")$get_weights()[[1]])
    S_lambda.tensor=K$constant(S_lambda)
    
    penalty = 0.5*K$dot(t.gam.weights,K$dot(S_lambda.tensor,gam.weights))
    
    u=y_pred[all_dims(),1]
    sig=y_pred[all_dims(),2]
    xi=y_pred[all_dims(),3]
    y=K$relu(y_true-u)
    
    
    #Evaluate log-likelihood
    ll1=-(1/xi+1)*K$log(1+xi*y/sig)
    
    #Uses non-zero response values only
    ll2= K$log(sig) *K$sign(ll1)
    
    return(penalty-(K$sum(ll1+ll2)))
    
    
    }
    }
  return(loss)
}
