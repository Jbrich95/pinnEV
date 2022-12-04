#' Non-parametric single quantile regression PINN
#'
#' Build and train a partially-interpretable neural network for non-parametric single quantile regression
#'
#'@name quant.NN

#' @param type string defining the type of network to be built. If \code{type=="MLP"}, the network will have all densely connected layers; if \code{type=="CNN"}, the network will have all convolutional layers. Defaults to an MLP.
#' @param Y.train,Y.valid a 2 or 3 dimensional array of training or validation real response values.
#' Missing values can be handled by setting corresponding entries to \code{Y.train} or \code{Y.valid} to \code{-1e10}.
#' The first dimension should be the observation indices, e.g., time.
#'
#' If \code{type=="CNN"}, then \code{Y.train} and \code{Y.valid} must have three dimensions with the latter two corresponding to an \eqn{M} by \eqn{N} regular grid of spatial locations.
#' If \code{Y.valid==NULL}, no validation loss will be computed and the returned model will be that which minimises the training loss over \code{n.ep} epochs.
#'
#' @param X list of arrays corresponding to complementary subsets of the \eqn{d\geq 1} predictors which are used for modelling. Must contain at least one of the following three named entries:\describe{
#' \item{\code{X.lin}}{A 3 or 4 dimensional array of "linear" predictor values. One more dimension than \code{Y.train}. If \code{NULL}, a model without the linear component is built and trained.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the last dimension corresponds to the chosen \eqn{l\geq 0} 'linear' predictor values.}
#' \item{\code{X.add.basis}}{A 4 or 5 dimensional array of basis function evaluations for the "additive" predictor values.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the penultimate dimensions corresponds to the chosen \eqn{a\geq 0} 'linear' predictor values and the last dimension is equal to the number of knots used for estimating the splines. See example.
#' If \code{NULL}, a model without the additive component is built and trained.}
#' \item{\code{X.nn}}{A 3 or 4 dimensional array of "non-additive" predictor values.  If \code{NULL}, a model without the NN component is built and trained; if this is the case, then \code{type} has no effect.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the last dimension corresponds to the chosen \eqn{d-l-a\geq 0} 'non-additive' predictor values.}
#' }
#' Note that \code{X} is the predictors for both \code{Y.train} and \code{Y.valid}.
#' @param n.ep number of epochs used for training. Defaults to 1000.
#' @param tau  quantile level. Must satisfy \code{0 < tau < 1}.
#' @param offset an array of strictly positive scalars the same dimension as \code{Y.train}, containing the offset values used for modelling the quantile. If \code{offset=NULL}, then no offset is used (equivalently, \code{offset} is populated with ones). Defaults to \code{NULL}.
#' @param batch.size batch size for stochastic gradient descent. If larger than \code{dim(Y.train)[1]}, i.e., the number of observations, then regular gradient descent used.
#' @param init.q sets the initial \code{tau}-quantile estimate across all dimensions of \code{Y.train}. Defaults to empirical estimate. Overriden by \code{init.wb_path} if \code{!is.null(init.wb_path)}. Note that if \code{!is.null(offset)}, then the initial quantile array will be \code{init.q*offset}.
#' @param init.wb_path filepath to a \code{keras} model which is then used as initial weights and biases for training the new model. The original model must have
#' the exact same architecture and trained with the same input data as the new model. If \code{NULL}, then initial weights and biases are random (with seed \code{seed}) but the
#' final layer has zero initial weights to ensure that the initial quantile estimate is \code{init.q} across all dimensions.
#' @param widths vector of widths/filters for hidden dense/convolution layers. Number of layers is equal to \code{length(widths)}. Defaults to (6,3).
#' @param filter.dim if \code{type=="CNN"}, this 2-vector gives the dimensions of the convolution filter kernel; must have odd integer inputs. Note that filter.dim=c(1,1) is equivalent to \code{type=="MLP"}. The same filter is applied for each hidden layer.
#' @param seed seed for random initial weights and biases.
#' @param link string defining the link function used, see \eqn{h} below. If \code{link=="exp"}, then \eqn{h=\exp(x)}; if \code{link=="identity"}, then \eqn{h(x)=x}.
#' @param model fitted \code{keras} model. Output from \code{quant.NN.train}.
#' @param S_lamda smoothing penalty matrix for the splines modelling the effect of \code{X.add.basis} on the inverse-\code{link} of the \code{tau}-quantile; only used if \code{!is.null(X.add.basis)}. If \code{is.null(S_lambda)}, then no smoothing penalty used.

#' @details{
#' Consider a real-valued random variable \eqn{Y} and let \eqn{\mathbf{X}} denote a \eqn{d}-dimensional predictor set with observations \eqn{\mathbf{x}}.
#' For integers \eqn{l\geq 0,a \geq 0} and \eqn{0\leq l+a \leq d}, let \eqn{\mathbf{X}_L, \mathbf{X}_A} and \eqn{\mathbf{X}_N} be distinct sub-vectors of \eqn{\mathbf{X}}, with observations of each component denoted \eqn{\mathbf{x}_L, \mathbf{x}_A} and \eqn{\mathbf{x}_N}, respectively; the lengths of the sub-vectors are \eqn{l,a} and \eqn{d-l-a}, respectively.
#'  We model \eqn{\Pr \{ Y \leq y_\tau (\mathbf{x}) |\mathbf{X}=\mathbf{x}\}=\tau} with
#' \deqn{y_\tau (\mathbf{x})=C(\mathbf{x})h\{\eta_0+m_L(\mathbf{x}_L)+m_A(\mathbf{x}_A)+m_N(\mathbf{x}_N)\}} where \eqn{h} is some link-function, \eqn{\eta_0} is a
#' constant intercept and \eqn{C(\mathbf{x})} is a fixed offset term (see Richards et al., 2022). The unknown functions \eqn{m_L} and \eqn{m_A} are estimated using a linear function and spline, respectively, and are
#' both returned as outputs by \code{quant.NN.predict}; \eqn{m_N} is estimated using a neural network. The offset term is, by default, \eqn{C(\mathbf{x})=1} for all \eqn{\mathbf{x}}; if \code{!is.null(offset)}, then \code{offset} determines \eqn{C(\mathbf{x})}.
#'
#' The model is fitted by minimising the penalised tilted loss over \code{n.ep} training epochs; the loss is given by
#' \deqn{l(y_\tau; y)=\max\{\tau(y-y_\tau),(\tau-1)(y-y_\tau)\}} plus some smoothing penalty for the additive functions (determined by \code{S_lambda}; see Richards and Huser, 2022) and is averaged over all entries to \code{Y.train} (or \code{Y.valid}).
#' Although the model is trained by minimising the loss evaluated for \code{Y.train}, the final returned model may minimise some other loss.
#' The current state of the model is saved after each epoch, using \code{keras::callback_model_checkpoint}, if the value of some criterion subcedes that of the model from the previous checkpoint; this criterion is the loss evaluated for validation set \code{Y.valid} if \code{!is.null(Y.valid)} and for \code{Y.train}, otherwise.
#'
#'}
#' @return \code{quant.NN.train} returns the fitted \code{model}.  \code{quant.NN.predict} is a wrapper for \code{keras::predict} that returns the predicted \code{tau}-quantile estimates, and, if applicable, the linear regression coefficients and spline bases weights.
#'
#' @references{
#'Richards, J. and Huser, R. (2022), \emph{A unifying partially-interpretable framework for neural network-based extreme quantile regression}. (\href{https://arxiv.org/abs/2208.07581}{arXiv:2208.07581}).
#'
#'Richards, J., Huser, R., Bevacqua, E., Zscheischler, J,  (2022), \emph{Insights into the drivers and spatio-temporal trends of extreme Mediterranean wildfires with statistical deep-learning.}
#'}
#' @examples
#'
#'
#'# Build and train a simple MLP for toy data
#'
#'set.seed(1)
#'
#'# Create  predictors
#'preds<-rnorm(prod(c(500,12,12,10)))
#'
#'#Re-shape to a 4d array. First dimension corresponds to observations,
#'#last to the different components of the predictor set.
#'#Other dimensions correspond to indices of predictors, e.g., a grid of locations. Can be just a 1D grid.
#'dim(preds)=c(500,12,12,10) 
#'#' #We have 500 observations of ten predictors on a 12 by 12 grid.
#'
#Split predictors into linear, additive and nn. 
#'
#'X.nn=preds[,,,1:5] #Five nn predictors
#'X.lin=preds[,,,6:8] #Three linear predictors
#'X.add=preds[,,,9:10] #Two additive predictors
#'
#' # Create toy response data
#'
#' #Linear contribution
#' m_L = 0.3*X.lin[,,,1]+0.6*X.lin[,,,2]-0.2*X.lin[,,,3]
#'
#' # Additive contribution
#' m_A = 0.2*X.add[,,,1]^2+0.05*X.add[,,,1]-0.1*X.add[,,,2]^2+
#' 0.1*X.add[,,,2]^3
#'
#' #Non-additive contribution - to be estimated by NN
#' m_N = exp(-3+X.nn[,,,2]+X.nn[,,,3])+
#' sin(X.nn[,,,1]-X.nn[,,,2])*(X.nn[,,,4]+X.nn[,,,5])
#'
#'theta=1+m_L+m_A+m_N #Identity link
#' #We simulate normal data and estimate the median, i.e., the 50% quantile or mean,
#' #as the form for this is known
#' Y=apply(theta,1:3,function(x) rnorm(1,mean=x,sd=2))
#'
#'
#' #Create training and validation, respectively.
#' #We mask 20% of the Y values and use this for validation.
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
#' #the basis functions for each pre-specified knot and entry to X.add
#'
#' rad=function(x,c){ #Define a basis function. Here we use the radial bases
#'   out=abs(x-c)^2*log(abs(x-c))
#'   out[(x-c)==0]=0
#'   return(out)
#' }
#'
#' n.knot = 5 # set number of knots. Must be the same for each additive predictor
#' knots=matrix(nrow=dim(X.add)[4],ncol=n.knot)
#'
#' #We set knots to be equally-spaced marginal quantiles
#' for( i in 1:dim(X.add)[4]){
#'  knots[i,]=quantile(X.add[,,,i],probs=seq(0,1,length=n.knot))
#'  }
#'
#' X.add.basis<-array(dim=c(dim(X.add),n.knot))
#' for( i in 1:dim(X.add)[4]) {
#' for(k in 1:n.knot) {
#' X.add.basis[,,,i,k]= rad(x=X.add[,,,i],c=knots[i,k])
#' #Evaluate rad at all entries to X.add and for all knots
#' }}
#' 
#' 
#' #Penalty matrix for additive functions
#' 
#'# Set smoothness parameters for first and second additive functions
#'  lambda = c(0.2,0.1) 
#'  
#'S_lambda=matrix(0,nrow=n.knot*dim(X.add)[4],ncol=n.knot*dim(X.add)[4])
#'for(i in 1:dim(X.add)[4]){
#'  for(j in 1:n.knot){
 #'   for(k in 1:n.knot){
#'      S_lambda[(j+(i-1)*n.knot),(k+(i-1)*n.knot)]=lambda[i]*rad(knots[i,j],knots[i,k])
 #'   }
 #' }
#'}
#'
#'#Build lin+GAM+NN model.
#' X=list("X.nn"=X.nn, "X.lin"=X.lin,
#' "X.add.basis"=X.add.basis)
#'
#'#Build and train a two-layered "lin+GAM+NN" MLP. Note that training is not run to completion.
#' NN.fit<-quant.NN.train(Y.train, Y.valid,X,  type="MLP",link="identity",tau=0.5,n.ep=600,
#'                       batch.size=100, widths=c(6,3),S_lambda=S_lambda)
#'
#' out<-quant.NN.predict(X,model=NN.fit$model)
#'hist(out$pred.q) #Plot histogram of predicted quantiles

#' print(out$lin.coeff)
#'
#'n.add.preds=dim(X.add)[length(dim(X.add))]
#'par(mfrow=c(1,n.add.preds))
#'for(i in 1:n.add.preds){
#'  plt.x=seq(from=min(knots[i,]),to=max(knots[i,]),length=1000)  #Create sequence for x-axis
#'
#'  tmp=matrix(nrow=length(plt.x),ncol=n.knot)
#'  for(j in 1:n.knot){
#'    tmp[,j]=rad(plt.x,knots[i,j]) #Evaluate radial basis function of plt.x and all knots
#'  }
#'  plt.y=tmp%*%out$gam.weights[i,]
#'  plot(plt.x,plt.y,type="l",main=paste0("Quantile spline: predictor ",i),xlab="x",ylab="f(x)")
#'  points(knots[i,],rep(mean(plt.y),n.knot),col="red",pch=2)
#'  #Adds red triangles that denote knot locations
#'}

#'
#' tau <- 0.5
#'#To save model, run
#'# NN.fit$model %>% save_model_tf(paste0("model_",tau,"-quantile"))
#'#To load model, run
#'#model  <- load_model_tf(paste0("model_",tau,"-quantile"),
#'#custom_objects=list("tilted_loss_tau___tau__S_lambda_"=tilted.loss(tau,S_lambda)))
#'
#' @import reticulate tensorflow keras
#'
#' @rdname quant.NN
#' @export

quant.NN.train=function(Y.train, Y.valid = NULL,X, type="MLP",link="identity",tau=NULL,offset=NULL,
                       n.ep=100, batch.size=100,init.q=NULL, widths=c(6,3), filter.dim=c(3,3),seed=NULL,init.wb_path=NULL,
                       S_lambda=NULL)
{

  
  if(is.null(X)  ) stop("No predictors provided")
  if(is.null(Y.train)) stop("No training response data provided")
  if(is.null(tau)  ) stop("tau not provided")
  


  print(paste0("Creating ",tau,"-quantile model"))
  if(!is.null(offset)) print(paste0("Using offset"))
  X.nn=X$X.nn
  X.lin=X$X.lin
  X.add.basis=X$X.add.basis

  if(is.null(offset)){
  if(!is.null(X.nn) & !is.null(X.add.basis) & !is.null(X.lin) ) {  train.data= list(X.lin,X.add.basis,X.nn); print("Defining lin+GAM+NN model for tau-quantile" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_q=X.lin,add_input_q=X.add.basis,  nn_input_q=X.nn),Y.valid)}
  if(is.null(X.nn) & !is.null(X.add.basis) & !is.null(X.lin) ) {   train.data= list(X.lin,X.add.basis); print("Defining lin+GAM model for tau-quantile" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_q=X.lin,add_input_q=X.add.basis),Y.valid)}
  if(!is.null(X.nn) & is.null(X.add.basis) & !is.null(X.lin) ) { train.data= list(X.lin,X.nn); print("Defining lin+NN model for tau-quantile" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_q=X.lin, nn_input_q=X.nn),Y.valid)}
  if(!is.null(X.nn) & !is.null(X.add.basis) & is.null(X.lin) ) {train.data= list(X.add.basis,X.nn); print("Defining GAM+NN model for tau-quantile" );  if(!is.null(Y.valid)) validation.data=list(list(add_input_q=X.add.basis,  nn_input_q=X.nn),Y.valid)}
  if(is.null(X.nn) & is.null(X.add.basis) & !is.null(X.lin) )   {train.data= list(X.lin); print("Defining fully-linear model for tau-quantile" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_q=X.lin),Y.valid)}
  if(is.null(X.nn) & !is.null(X.add.basis) & is.null(X.lin) )   {train.data= list(X.add.basis); print("Defining fully-additive model for tau-quantile" );  if(!is.null(Y.valid)) validation.data=list(list(add_input_q=X.add.basis),Y.valid)}
  if(!is.null(X.nn) & is.null(X.add.basis) & is.null(X.lin) )   {train.data= list(X.nn); print("Defining fully-NN model for tau-quantile" ) ; if(!is.null(Y.valid)) validation.data=list(list( nn_input_q=X.nn),Y.valid)}
  }else if(!is.null(offset)){
    if(!is.null(X.nn) & !is.null(X.add.basis) & !is.null(X.lin) ) {  train.data= list(X.lin,X.add.basis,X.nn,offset); print("Defining lin+GAM+NN model for tau-quantile" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_q=X.lin,add_input_q=X.add.basis,  nn_input_q=X.nn,offset_input=offset),Y.valid)}
    if(is.null(X.nn) & !is.null(X.add.basis) & !is.null(X.lin) ) {   train.data= list(X.lin,X.add.basis,offset); print("Defining lin+GAM model for tau-quantile" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_q=X.lin,add_input_q=X.add.basis,offset_input=offset),Y.valid)}
    if(!is.null(X.nn) & is.null(X.add.basis) & !is.null(X.lin) ) { train.data= list(X.lin,X.nn,offset); print("Defining lin+NN model for tau-quantile" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_q=X.lin, nn_input_q=X.nn,offset_input=offset),Y.valid)}
    if(!is.null(X.nn) & !is.null(X.add.basis) & is.null(X.lin) ) {train.data= list(X.add.basis,X.nn,offset); print("Defining GAM+NN model for tau-quantile" );  if(!is.null(Y.valid)) validation.data=list(list(add_input_q=X.add.basis,  nn_input_q=X.nn,offset_input=offset),Y.valid)}
    if(is.null(X.nn) & is.null(X.add.basis) & !is.null(X.lin) )   {train.data= list(X.lin,offset); print("Defining fully-linear model for tau-quantile" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_q=X.lin,offset_input=offset),Y.valid)}
    if(is.null(X.nn) & !is.null(X.add.basis) & is.null(X.lin) )   {train.data= list(X.add.basis,offset); print("Defining fully-additive model for tau-quantile" );  if(!is.null(Y.valid)) validation.data=list(list(add_input_q=X.add.basis,offset_input=offset),Y.valid)}
    if(!is.null(X.nn) & is.null(X.add.basis) & is.null(X.lin) )   {train.data= list(X.nn,offset); print("Defining fully-NN model for tau-quantile" ) ; if(!is.null(Y.valid)) validation.data=list(list( nn_input_q=X.nn,offset_input=offset),Y.valid)}

  }
  if(is.null(S_lambda) & !is.null(X.add.basis)){print("No smoothing penalty used")}

  if(is.null(X.add.basis)){S_lambda=NULL}

  if(type=="CNN" & !is.null(X.nn)) print(paste0("Building ",length(widths),"-layer convolutional neural network with ", filter.dim[1]," by ", filter.dim[2]," filter" ))
  if(type=="MLP"  & !is.null(X.nn) ) print(paste0("Building ",length(widths),"-layer densely-connected neural network" ))

  reticulate::use_virtualenv("myenv", required = T)

  if(!is.null(seed)) tf$random$set_seed(seed)
  if(!is.null(offset) & length(dim(offset))!=length(dim(Y.train))+1) dim(offset)=c(dim(offset),1)
  if(is.null(init.q)) init.q=quantile(Y.train[Y.train!=-1e10],prob=tau)
  model<-quant.NN.build(X.nn,X.lin,X.add.basis, offset,
                        type, init.q, widths,filter.dim,link,tau)
  
  if(!is.null(init.wb_path)) model <- load_model_weights_tf(model,filepath=init.wb_path)
  
  model %>% compile(
    optimizer="adam",
    loss = tilted.loss(tau=tau,S_lambda=S_lambda),
    run_eagerly=T
  )

  if(!is.null(Y.valid)) checkpoint <- callback_model_checkpoint(paste0("model_",tau,"-quantile_checkpoint"), monitor = "val_loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch") else checkpoint <- callback_model_checkpoint(paste0("model_",tau,"-quantile_checkpoint"), monitor = "loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch")

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
  model <- load_model_weights_tf(model,filepath=paste0("model_",tau,"-quantile_checkpoint"))
  print("Final training loss")
  loss.train<-model %>% evaluate(train.data,Y.train, batch_size=50)
  if(!is.null(Y.valid)){
    print("Final validation loss")
    loss.valid<-model %>% evaluate(train.data,Y.valid, batch_size=50)
    return(list("model"=model,"Training loss"=loss.train, "Validation loss"=loss.valid))
  }else{
    return(list("model"=model,"Training loss"=loss.train))
  }
  

  return(model)
}
#' @rdname quant.NN
#' @export
#'
quant.NN.predict=function(X, model,offset=NULL)
{



  if(is.null(X)  ) stop("No predictors provided")
  

  X.nn=X$X.nn
  X.lin=X$X.lin
  X.add.basis=X$X.add.basis
  
  if(is.null(offset)){
  if(!is.null(X.nn) & !is.null(X.add.basis) & !is.null(X.lin) ) pred.q<-model %>% predict(  list(X.lin,X.add.basis,X.nn))
  if(is.null(X.nn) & !is.null(X.add.basis) & !is.null(X.lin) )    pred.q<-model %>% predict( list(X.lin,X.add.basis))
  if(!is.null(X.nn) & is.null(X.add.basis) & !is.null(X.lin) ) pred.q<-model %>% predict( list(X.lin,X.nn))
  if(!is.null(X.nn) & !is.null(X.add.basis) & is.null(X.lin) ) pred.q<-model %>% predict( list(X.add.basis,X.nn))
  if(is.null(X.nn) & is.null(X.add.basis) & !is.null(X.lin) )  pred.q<-model %>% predict(list(X.lin))
  if(is.null(X.nn) & !is.null(X.add.basis) & is.null(X.lin) )   pred.q<-model %>% predict( list(X.add.basis))
  if(!is.null(X.nn) & is.null(X.add.basis) & is.null(X.lin) )   pred.q<-model %>% predict( list(X.nn))
  }else if(!is.null(offset)){
    if(!is.null(X.nn) & !is.null(X.add.basis) & !is.null(X.lin) ) pred.q<-model %>% predict(  list(X.lin,X.add.basis,X.nn,offset))
    if(is.null(X.nn) & !is.null(X.add.basis) & !is.null(X.lin) )    pred.q<-model %>% predict( list(X.lin,X.add.basis,offset))
    if(!is.null(X.nn) & is.null(X.add.basis) & !is.null(X.lin) ) pred.q<-model %>% predict( list(X.lin,X.nn,offset))
    if(!is.null(X.nn) & !is.null(X.add.basis) & is.null(X.lin) ) pred.q<-model %>% predict( list(X.add.basis,X.nn,offset))
    if(is.null(X.nn) & is.null(X.add.basis) & !is.null(X.lin) )  pred.q<-model %>% predict(list(X.lin,offset))
    if(is.null(X.nn) & !is.null(X.add.basis) & is.null(X.lin) )   pred.q<-model %>% predict( list(X.add.basis,offset))
    if(!is.null(X.nn) & is.null(X.add.basis) & is.null(X.lin) )   pred.q<-model %>% predict( list(X.nn,offset))
    
  }
  if(!is.null(X.add.basis))  gam.weights<-matrix(t(model$get_layer("add_q")$get_weights()[[1]]),nrow=dim(X.add.basis)[length(dim(X.add.basis))-1],ncol=dim(X.add.basis)[length(dim(X.add.basis))],byrow=T)

  if(!is.null(X.add.basis) & !is.null(X.lin)) return(list("pred.q"=pred.q, "lin.coeff"=c(model$get_layer("lin_q")$get_weights()[[1]]),"gam.weights"=gam.weights))
  if(is.null(X.add.basis) & !is.null(X.lin)) return(list("pred.q"=pred.q, "lin.coeff"=c(model$get_layer("lin_q")$get_weights()[[1]])))
  if(!is.null(X.add.basis) & is.null(X.lin)) return(list("pred.q"=pred.q,"gam.weights"=gam.weights))
  if(is.null(X.add.basis) & is.null(X.lin)) return(list("pred.q"=pred.q))


}
#'
#'
quant.NN.build=function(X.nn,X.lin,X.add.basis, offset, type, init.q, widths,filter.dim,link,tau=tau)
{


  #offset input
  if(!is.null(offset))  input_offset <- layer_input(shape = dim(offset)[-1], name = 'offset_input')
  

  #Additive input
  if(!is.null(X.add.basis))  input_add<- layer_input(shape = dim(X.add.basis)[-1], name = 'add_input_q')

  #NN input

  if(!is.null(X.nn))   input_nn <- layer_input(shape = dim(X.nn)[-1], name = 'nn_input_q')

  #Linear input

  if(!is.null(X.lin)) input_lin <- layer_input(shape = dim(X.lin)[-1], name = 'lin_input_q')

  if(link=="exp") init.q=log(init.q) else if(link =="identity") init.q=init.q else stop("Invalid link function")
  #NN tower
  if(!is.null(X.nn)){

    nunits=c(widths,1)
    n.layers=length(nunits)-1

    nnBranchq <- input_nn
    if(type=="MLP"){
      for(i in 1:n.layers){
        nnBranchq <- nnBranchq  %>% layer_dense(units=nunits[i],activation = 'relu',
                                                input_shape =dim(X.nn)[-1], name = paste0('nn_q_dense',i) )
      }
    }else if(type=="CNN"){
      for(i in 1:n.layers){
        nnBranchq <- nnBranchq  %>% layer_conv_2d(filters=nunits[i],activation = 'relu',kernel_size=c(filter.dim[1],filter.dim[2]), padding='same',
                                                  input_shape =dim(X.nn)[-1], name = paste0('nn_q_cnn',i) )
      }

    }

    nnBranchq <-   nnBranchq  %>%   layer_dense(units = nunits[n.layers+1], activation = "linear", name = 'nn_q_dense_final',
                                                weights=list(matrix(0,nrow=nunits[n.layers],ncol=1), array(init.q)))

  }

  #Additive tower
  n.dim.add=length(dim(X.add.basis))
  if(!is.null(X.add.basis) & !is.null(X.add.basis) ) {

    addBranchq <- input_add %>%
      layer_reshape(target_shape=c(dim(X.add.basis)[2:(n.dim.add-2)],prod(dim(X.add.basis)[(n.dim.add-1):n.dim.add]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_q',
                  weights=list(matrix(0,nrow=prod(dim(X.add.basis)[(n.dim.add-1):n.dim.add]),ncol=1)),use_bias = F)
  }
  if(!is.null(X.add.basis) & is.null(X.add.basis) ) {

    addBranchq <- input_add %>%
      layer_reshape(target_shape=c(dim(X.add.basis)[2:(n.dim.add-2)],prod(dim(X.add.basis)[(n.dim.add-1):n.dim.add]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_q',
                  weights=list(matrix(0,nrow=prod(dim(X.add.basis)[(n.dim.add-1):n.dim.add]),ncol=1),array(init.q)),use_bias = T)
  }

  #Linear tower


  if(!is.null(X.lin) ) {
    n.dim.lin=length(dim(X.lin))

    if(is.null(X.nn) & is.null(X.add.basis )){
      linBranchq <- input_lin%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X.lin)[-1], name = 'lin_q',
                    weights=list(matrix(0,nrow=dim(X.lin)[n.dim.lin],ncol=1),array(init.q)),use_bias=T)
    }else{
      linBranchq <- input_lin%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X.lin)[-1], name = 'lin_q',
                    weights=list(matrix(0,nrow=dim(X.lin)[n.dim.lin],ncol=1)),use_bias=F)
    }
  }



  if(!is.null(X.nn) & !is.null(X.add.basis) & !is.null(X.lin) )  qBranchjoined <- layer_add(inputs=c(addBranchq,  linBranchq,nnBranchq),name="Combine_q_components")  #Add all towers
  if(is.null(X.nn) & !is.null(X.add.basis) & !is.null(X.lin) )  qBranchjoined <- layer_add(inputs=c(addBranchq,  linBranchq),name="Combine_q_components")  #Add GAM+lin towers
  if(!is.null(X.nn) & is.null(X.add.basis) & !is.null(X.lin) )  qBranchjoined <- layer_add(inputs=c(  linBranchq,nnBranchq),name="Combine_q_components")  #Add NN+lin towers
  if(!is.null(X.nn) & !is.null(X.add.basis) & is.null(X.lin) )  qBranchjoined <- layer_add(inputs=c(addBranchq,  nnBranchq),name="Combine_q_components")  #Add NN+GAM towers
  if(is.null(X.nn) & is.null(X.add.basis) & !is.null(X.lin) )  qBranchjoined <- linBranchq  #Just lin tower
  if(is.null(X.nn) & !is.null(X.add.basis) & is.null(X.lin) )  qBranchjoined <- addBranchq  #Just GAM tower
  if(!is.null(X.nn) & is.null(X.add.basis) & is.null(X.lin) )  qBranchjoined <- nnBranchq  #Just NN tower


  #Apply link functions
  #Accommodate offset if available

  
  if(link=="exp") qBranchjoined <- qBranchjoined %>% layer_activation( activation = 'exponential', name = "q_activation") else if(link=="linear") qBranchjoined <- qBranchjoined %>% layer_activation( activation = 'identity', name = "q_activation")

  if(!is.null(offset))    qBranchjoined <- layer_multiply(inputs=c(input_offset,qBranchjoined))
  
  input=c()
  
  if(!is.null(X.lin) ) input=c(input,input_lin)
  if(!is.null(X.add.basis) ) input=c(input,input_add)
  if(!is.null(X.nn) ) input=c(input,input_nn)
  if(!is.null(offset)) input=c(input,input_offset)
  
model <- keras_model(  inputs = input,   outputs = c(qBranchjoined),name=paste0("quantile")) 
  

  print(model)

  return(model)

}



tilted.loss <- function( tau,S_lambda=NULL) {

  if(is.null(S_lambda)){
  loss <- function( y_true, y_pred) {
  K <- backend()

  # Find inds of non-missing obs.  Remove missing obs, i.e., -1e10. This is achieved by adding an
  # arbitrarily large (<1e10) value to y_true and then taking the sign ReLu
  obsInds=K$sign(K$relu(y_true+9e9))

  error = y_true - y_pred
  return(K$sum(K$maximum(tau*error, (tau-1)*error)*obsInds)/K$sum(obsInds))
  }
  }else{
    loss <- function( y_true, y_pred) {
      K <- backend()
      
      t.gam.weights=K$constant(t(model$get_layer("add_q")$get_weights()[[1]]))
      gam.weights=K$constant(model$get_layer("add_q")$get_weights()[[1]])
      S_lambda.tensor=K$constant(S_lambda)
      
      penalty = 0.5*K$dot(t.gam.weights,K$dot(S_lambda.tensor,gam.weights))
      # Find inds of non-missing obs.  Remove missing obs, i.e., -1e10. This is achieved by adding an
      # arbitrarily large (<1e10) value to y_true and then taking the sign ReLu
      obsInds=K$sign(K$relu(y_true+9e9))
      
      error = y_true - y_pred
      return(K$sum(K$maximum(tau*error, (tau-1)*error)*obsInds)/K$sum(obsInds)+penalty)
    }
    
  }
  return(loss)
}


