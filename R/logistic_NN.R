#' Logistic regression PINN

#'
#' Build and train a partially-interpretable neural network for a logistic regression model
#'
#'@name logistic.NN

#' @param type  string defining the type of network to be built. If \code{type=="MLP"}, the network will have all densely connected layers; if \code{type=="CNN"}, the network will have all convolutional layers (with 3 by 3 filters). If \code{type=="GCNN"}, then a graph convolutional neural network (with skip connections) is used and require \code{!is.null(A)}. Defaults to an MLP (currently the same network is used for all parameters, may change in future versions). Defaults to an MLP.
#' @param Y.train,Y.valid a 2 or 3 dimensional array of training or validation response values, with entries of 0/1 for failure/success.
#' Missing values can be handled by setting corresponding entries to \code{Y.train} or \code{Y.valid} to \code{-1e10}.
#' The first dimension should be the observation indices, e.g., time.
#'
#' If \code{type=="CNN"}, then \code{Y.train} and \code{Y.valid} must have three dimensions with the latter two corresponding to an \eqn{M} by \eqn{N} regular grid of spatial locations. If \code{type=="GCNN"}, then \code{Y.train} and \code{Y.valid} must have two dimensions with the latter corresponding to \eqn{M} spatial locations.
#' If \code{Y.valid==NULL}, no validation loss will be computed and the returned model will be that which minimises the training loss over \code{n.ep} epochs.
#'
#' @param X  list of arrays corresponding to complementary subsets of the \eqn{d\geq 1} predictors which are used for modelling. Must contain at least one of the following three named entries:\describe{
#' \item{\code{X.lin}}{A 3 or 4 dimensional array of "linear" predictor values. One more dimension than \code{Y.train}. If \code{NULL}, a model without the linear component is built and trained.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the last dimension corresponds to the chosen \eqn{l\geq 0} 'linear' predictor values.}
#' \item{\code{X.add.basis}}{A 4 or 5 dimensional array of basis function evaluations for the "additive" predictor values.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the penultimate dimensions corresponds to the chosen \eqn{a\geq 0} 'linear' predictor values and the last dimension is equal to the number of knots used for estimating the splines. See example.
#' If \code{NULL}, a model without the additive component is built and trained.}
#' \item{\code{X.nn}}{A 3 or 4 dimensional array of "non-additive" predictor values.  If \code{NULL}, a model without the NN component is built and trained; if this is the case, then \code{type} has no efect.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the last dimension corresponds to the chosen \eqn{d-l-a\geq 0} 'non-additive' predictor values.}
#' }
#' Note that \code{X} is the predictors for both \code{Y.train} and \code{Y.valid}.
#' @param n.ep number of epochs used for training. Defaults to 1000.
#' @param batch.size batch size for stochastic gradient descent. If larger than \code{dim(Y.train)[1]}, i.e., the number of observations, then regular gradient descent used.
#' @param init.p sets the initial probability estimate across all dimensions of \code{Y.train}. Defaults to empirical estimate. Overriden by \code{init.wb_path} if \code{!is.null(init.wb_path)}.
#' @param init.wb_path filepath to a \code{keras} model which is then used as initial weights and biases for training the new model. The original model must have
#' the exact same architecture and trained with the same input data as the new model. If \code{NULL}, then initial weights and biases are random (with seed \code{seed}) but the
#' final layer has zero initial weights to ensure that the initial probability estimate is \code{init.p} across all dimensions.
#' @param widths vector of widths/filters for hidden dense/convolution layers. Number of layers is equal to \code{length(widths)}. Defaults to (6,3).
#' @param seed seed for random initial weights and biases.
#' @param model fitted \code{keras} model. Output from \code{logistic.NN.train}.
#' @param A \eqn{M \times M} adjacency matrix used if and only if \code{type=="GCNN"}. Must be supplied in this case.

#' @details{
#' Consider a Bernoulli random variable, say \eqn{Z\sim\mbox{Bernoulli}(p)}, with probability mass function \eqn{\Pr(Z=1)=p=1-\Pr(Z=0)=1-p}. Let \eqn{Y\in\{0,1\}} be a univariate Boolean response and let \eqn{\mathbf{X}} denote a \eqn{d}-dimensional predictor set with observations \eqn{\mathbf{x}}.
#' For integers \eqn{l\geq 0,a \geq 0} and \eqn{0\leq l+a \leq d}, let \eqn{\mathbf{X}_L, \mathbf{X}_A} and \eqn{\mathbf{X}_N} be distinct sub-vectors of \eqn{\mathbf{X}}, with observations of each component denoted \eqn{\mathbf{x}_L, \mathbf{x}_A} and \eqn{\mathbf{x}_N}, respectively; the lengths of the sub-vectors are \eqn{l,a} and \eqn{d-l-a}, respectively.
#'  We model \eqn{Y|\mathbf{X}=\mathbf{x}\sim\mbox{Bernoulli}(p\{\mathbf{x})\}} with
#' \deqn{p(\mathbf{x})=h\{\eta_0+m_L(\mathbf{x}_L)+m_A(\mathbf{x}_A)+m_N(\mathbf{x}_N)\}} where \eqn{h} is the logistic link-function and
#' \eqn{\eta_0} is a constant intercept. The unknown functions \eqn{m_L} and \eqn{m_A} are estimated using a linear function and spline, respectively,
#' and are both returned as outputs by \code{logistic.NN.predict}; \eqn{m_N} is estimated using a neural network.
#'
#' The model is fitted by minimising the binary cross-entropy loss over \code{n.ep} training epochs.
#' Although the model is trained by minimising the loss evaluated for \code{Y.train}, the final returned model may minimise some other loss.
#' The current state of the model is saved after each epoch, using \code{keras::callback_model_checkpoint}, if the value of some criterion subcedes that of the model from the previous checkpoint; this criterion is the loss evaluated for validation set \code{Y.valid} if \code{!is.null(Y.valid)} and for \code{Y.train}, otherwise.
#'
#'A non-interpretable version of this model was exploited by Cisneros et al. (2023). Equivalence with their model is achieved by setting \code{X.lin=NULL}, \code{X.add.basis=NULL} and \code{type="GCNN"}, and using the adjacency matrix \code{A} given in the example of \code{help(AusWild)}.
#'
#'}
#' @return \code{logistic.NN.train} returns the fitted \code{model}.  \code{logistic.NN.predict} is a wrapper for \code{keras::predict} that returns the predicted probability estimates, and, if applicable, the linear regression coefficients and spline bases weights.
#'
#'@references{
#' Richards, J. and Huser, R. (2022), \emph{Regression modelling of spatiotemporal extreme U.S. wildfires via partially-interpretable neural networks}. (\href{https://arxiv.org/abs/2208.07581}{arXiv:2208.07581}).
#'}
#'
#' @examples
#'
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
#'#Split predictors into linear, additive and nn. 
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
#'   m_A = 0.1*X.add[,,,1]^2+0.2*X.add[,,,1]-0.1*X.add[,,,2]^2+
#' 0.1*X.add[,,,2]^3-0.5*X.add[,,,2]
#'
#' #Non-additive contribution - to be estimated by NN
#' m_N = exp(-3+X.nn[,,,2]+X.nn[,,,3])+
#' sin(X.nn[,,,1]-X.nn[,,,2])*(X.nn[,,,4]+X.nn[,,,5])
#'
#' p=0.5+0.5*tanh((m_L+m_A+m_N)/2) #Logistic link
#' Y=apply(p,1:3,function(x) rbinom(1,1,x))
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
#' for( i in 1:dim(X.add)[4]) {
#' knots[i,]=quantile(X.add[,,,i],probs=seq(0,1,length=n.knot))}
#'
#' X.add.basis<-array(dim=c(dim(X.add),n.knot))
#' for( i in 1:dim(X.add)[4]) {
#' for(k in 1:n.knot) {
#' X.add.basis[,,,i,k]= rad(x=X.add[,,,i],c=knots[i,k])
#' #Evaluate rad at all entries to X.add and for all knots
#' }}
#' 
#'
#' X=list("X.nn"=X.nn, "X.lin"=X.lin,
#' "X.add.basis"=X.add.basis)
#'
#' #Build and train a two-layered "lin+GAM+NN" logistic MLP. 
#' #Note that training is not run to completion.
#' NN.fit<-logistic.NN.train(Y.train, Y.valid,X,  type="MLP",n.ep=600,
#'                       batch.size=100,init.p=0.4, widths=c(6,3))
#'
#' out<-logistic.NN.predict(X,NN.fit$model)
#' hist(out$pred.p) #Plot histogram of predicted probability
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
#'#To save model, run NN.fit$model %>% save_model_tf("model_Bernoulli")
#'#To load model, run model  <- load_model_tf("model_Bernoulli",
#'#custom_objects=list("bce_loss__"=bce.loss()))
#'
#' @import reticulate tensorflow keras
#'
#' @rdname logistic.NN
#' @export

logistic.NN.train=function(Y.train, Y.valid = NULL,X, type="MLP", A=NULL,
                       seed=NULL, init.wb_path=NULL)
{




  if(is.null(X)  ) stop("No predictors provided")
  if(is.null(Y.train)) stop("No training response data provided")

  if(is.null(A) & type=="GCNN")stop("Adjacency matrix must be supplied if GCNN required")
  if(!is.null(A)) if(type=="GCNN" & (length(dim(Y.train))!=2 | dim(A)[1]!=dim(A)[2] | dim(A)[1]!=dim(Y.train)[2]))stop("Dimensions of the adjacency matrix are incorrect")
  
  X.nn=X$X.nn
  X.lin=X$X.lin
  X.add.basis=X$X.add.basis

  if(!is.null(X.nn) & !is.null(X.add.basis) & !is.null(X.lin) ) {  train.data= list(X.lin,X.add.basis,X.nn); print("Defining lin+GAM+NN model for p" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_p=X.lin,add_input_p=X.add.basis,  nn_input_p=X.nn),Y.valid)}
  if(is.null(X.nn) & !is.null(X.add.basis) & !is.null(X.lin) ) {   train.data= list(X.lin,X.add.basis); print("Defining lin+GAM model for p" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_p=X.lin,add_input_p=X.add.basis),Y.valid)}
  if(!is.null(X.nn) & is.null(X.add.basis) & !is.null(X.lin) ) { train.data= list(X.lin,X.nn); print("Defining lin+NN model for p" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_p=X.lin, nn_input_p=X.nn),Y.valid)}
  if(!is.null(X.nn) & !is.null(X.add.basis) & is.null(X.lin) ) {train.data= list(X.add.basis,X.nn); print("Defining GAM+NN model for p" );  if(!is.null(Y.valid)) validation.data=list(list(add_input_p=X.add.basis,  nn_input_p=X.nn),Y.valid)}
  if(is.null(X.nn) & is.null(X.add.basis) & !is.null(X.lin) )   {train.data= list(X.lin); print("Defining fully-linear model for p" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_p=X.lin),Y.valid)}
  if(is.null(X.nn) & !is.null(X.add.basis) & is.null(X.lin) )   {train.data= list(X.add.basis); print("Defining fully-additive model for p" );  if(!is.null(Y.valid)) validation.data=list(list(add_input_p=X.add.basis),Y.valid)}
  if(!is.null(X.nn) & is.null(X.add.basis) & is.null(X.lin) )   {train.data= list(X.nn); print("Defining fully-NN model for p" );  if(!is.null(Y.valid)) validation.data=list(list( nn_input_p=X.nn),Y.valid)}


  if(type=="CNN" & !is.null(X.nn)) print(paste0("Building ",length(widths),"-layer convolutional neural network with ",3," by ", 3," filter" ))
  if(type=="MLP"  & !is.null(X.nn) ) print(paste0("Building ",length(widths),"-layer densely-connected neural network" ))
  if(type=="GCNN"  & (!is.null(X.nn) ) ) print(paste0("Building ",length(widths),"-layer graph convolutional neural network" ))
  
  print(paste0("Training for ", n.ep," epochs with a batch size of ", batch.size))
  reticulate::use_virtualenv("pinnEV_env", required = T)

  if(is.null(seed)) seed=1
  tf$random$set_seed(seed)
  if(is.null(init.p)) init.p=mean(Y.train[Y.train>=0]==1)


  model<-logistic.NN.build(X.nn,X.lin,X.add.basis, type, init.p, widths, A, seed)
  if(!is.null(init.wb_path)) model <- load_model_weights_tf(model,filepath=init.wb_path)
  model %>% compile(
    optimizer="adam",
    loss = bce.loss(),
    run_eagerly=T
  )

  if(!is.null(Y.valid)) checkpoint <- callback_model_checkpoint(paste0("model_bernoulli_checkpoint"), monitor = "val_loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch") else checkpoint <- callback_model_checkpoint(paste0("model_bernoulli_checkpoint"), monitor = "loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch")

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
  model <- load_model_weights_tf(model,filepath=paste0("model_bernoulli_checkpoint"))
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
#' @rdname logistic.NN
#' @export
#'
logistic.NN.predict=function(X, model)
{



  if(is.null(X)  ) stop("No predictors provided")

  X.nn=X$X.nn
  X.lin=X$X.lin
  X.add.basis=X$X.add.basis

  if(!is.null(X.nn) & !is.null(X.add.basis) & !is.null(X.lin) ) pred.p<-model %>% predict(  list(X.lin,X.add.basis,X.nn))
  if(is.null(X.nn) & !is.null(X.add.basis) & !is.null(X.lin) )    pred.p<-model %>% predict( list(X.lin,X.add.basis))
  if(!is.null(X.nn) & is.null(X.add.basis) & !is.null(X.lin) ) pred.p<-model %>% predict( list(X.lin,X.nn))
  if(!is.null(X.nn) & !is.null(X.add.basis) & is.null(X.lin) ) pred.p<-model %>% predict( list(X.add.basis,X.nn))
  if(is.null(X.nn) & is.null(X.add.basis) & !is.null(X.lin) )  pred.p<-model %>% predict(list(X.lin))
  if(is.null(X.nn) & !is.null(X.add.basis) & is.null(X.lin) )   pred.p<-model %>% predict( list(X.add.basis))
  if(!is.null(X.nn) & is.null(X.add.basis) & is.null(X.lin) )   pred.p<-model %>% predict( list(X.nn))

  if(!is.null(X.add.basis))  gam.weights<-matrix(t(model$get_layer("add_p")$get_weights()[[1]]),nrow=dim(X.add.basis)[length(dim(X.add.basis))-1],ncol=dim(X.add.basis)[length(dim(X.add.basis))],byrow=T)

  if(!is.null(X.add.basis) & !is.null(X.lin)) return(list("pred.p"=pred.p, "lin.coeff"=c(model$get_layer("lin_p")$get_weights()[[1]]),"gam.weights"=gam.weights))
  if(is.null(X.add.basis) & !is.null(X.lin)) return(list("pred.p"=pred.p, "lin.coeff"=c(model$get_layer("lin_p")$get_weights()[[1]])))
  if(!is.null(X.add.basis) & is.null(X.lin)) return(list("pred.p"=pred.p,"gam.weights"=gam.weights))
  if(is.null(X.add.basis) & is.null(X.lin)) return(list("pred.p"=pred.p))


}
#'
#'
logistic.NN.build=function(X.nn,X.lin,X.add.basis, type, init.p, widths,  A, seed)
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
  
  #Additive input
  if(!is.null(X.add.basis))  input_add<- layer_input(shape = dim(X.add.basis)[-1], name = 'add_input_p')

  #NN input

  if(!is.null(X.nn))   input_nn <- layer_input(shape = dim(X.nn)[-1], name = 'nn_input_p')

  #Linear input

  if(!is.null(X.lin)) input_lin <- layer_input(shape = dim(X.lin)[-1], name = 'lin_input_p')


  #NN tower
  if(!is.null(X.nn)){

    nunits=c(widths,1)
    n.layers=length(nunits)-1

    nnBranchp <- input_nn
    if(type=="MLP"){
      for(i in 1:n.layers){
        nnBranchp <- nnBranchp  %>% layer_dense(units=nunits[i],activation = 'relu',
                                                input_shape =dim(X.nn)[-1], name = paste0('nn_p_dense',i) )
      }
    }else if(type=="CNN"){
      for(i in 1:n.layers){
        nnBranchp <- nnBranchp  %>% layer_conv_2d(filters=nunits[i],activation = 'relu',kernel_size=c(3,3), padding='same',
                                                  input_shape =dim(X.nn)[-1], name = paste0('nn_p_cnn',i) )
      }

    }else if(type=="GCNN"){
      for(i in 1:n.layers){
        nnBranchp <- list(nnBranchp,ML)  %>% layer_graph_conv(channels=nunits[i],activation = 'relu',
                                                              input_shape =dim(X.nn)[-1], name = paste0('nn_p_gcnn',i) ,
                                                              kernel_initializer=initializer_glorot_uniform(seed=seed))
      }
      
    }

    nnBranchp <-   nnBranchp  %>%   layer_dense(units = nunits[n.layers+1], activation = "linear", name = 'nn_p_dense_final',
                                                weights=list(matrix(0,nrow=nunits[n.layers],ncol=1), array(log(init.p/(1-init.p)))))

  }

  #Additive tower
  n.dim.add=length(dim(X.add.basis))
  if(!is.null(X.add.basis) & !is.null(X.add.basis) ) {

    addBranchp <- input_add %>%
      layer_reshape(target_shape=c(dim(X.add.basis)[2:(n.dim.add-2)],prod(dim(X.add.basis)[(n.dim.add-1):n.dim.add]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_p',
                  weights=list(matrix(0,nrow=prod(dim(X.add.basis)[(n.dim.add-1):n.dim.add]),ncol=1)),use_bias = F)
  }
  if(!is.null(X.add.basis) & is.null(X.add.basis) ) {

    addBranchp <- input_add %>%
      layer_reshape(target_shape=c(dim(X.add.basis)[2:(n.dim.add-2)],prod(dim(X.add.basis)[(n.dim.add-1):n.dim.add]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_p',
                  weights=list(matrix(0,nrow=prod(dim(X.add.basis)[(n.dim.add-1):n.dim.add]),ncol=1),array(log(init.p/(1-init.p)))),use_bias = T)
  }

  #Linear tower


  if(!is.null(X.lin) ) {
    n.dim.lin=length(dim(X.lin))

    if(is.null(X.nn) & is.null(X.add.basis )){
      linBranchp <- input_lin%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X.lin)[-1], name = 'lin_p',
                    weights=list(matrix(0,nrow=dim(X.lin)[n.dim.lin],ncol=1),array(log(init.p/(1-init.p)))),use_bias=T)
    }else{
      linBranchp <- input_lin%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X.lin)[-1], name = 'lin_p',
                    weights=list(matrix(0,nrow=dim(X.lin)[n.dim.lin],ncol=1)),use_bias=F)
    }
  }



  if(!is.null(X.nn) & !is.null(X.add.basis) & !is.null(X.lin) )  pBranchjoined <- layer_add(inputs=c(addBranchp,  linBranchp,nnBranchp))  #Add all towers
  if(is.null(X.nn) & !is.null(X.add.basis) & !is.null(X.lin) )  pBranchjoined <- layer_add(inputs=c(addBranchp,  linBranchp))  #Add GAM+lin towers
  if(!is.null(X.nn) & is.null(X.add.basis) & !is.null(X.lin) )  pBranchjoined <- layer_add(inputs=c(  linBranchp,nnBranchp))  #Add NN+lin towers
  if(!is.null(X.nn) & !is.null(X.add.basis) & is.null(X.lin) )  pBranchjoined <- layer_add(inputs=c(addBranchp,  nnBranchp))  #Add NN+GAM towers
  if(is.null(X.nn) & is.null(X.add.basis) & !is.null(X.lin) )  pBranchjoined <- linBranchp  #Just lin tower
  if(is.null(X.nn) & !is.null(X.add.basis) & is.null(X.lin) )  pBranchjoined <- addBranchp  #Just GAM tower
  if(!is.null(X.nn) & is.null(X.add.basis) & is.null(X.lin) )  pBranchjoined <- nnBranchp  #Just NN tower


  #Apply link functions
  pBranchjoined <- pBranchjoined %>%
    layer_activation( activation = 'sigmoid', name = "p_activation")


  if(!is.null(X.nn) & !is.null(X.add.basis) & !is.null(X.lin) ) model <- keras_model(  inputs = c(input_lin,input_add,input_nn),   outputs = c(pBranchjoined),name="Bernoulli" )
  if(is.null(X.nn) & !is.null(X.add.basis) & !is.null(X.lin) )  model <- keras_model(  inputs = c(input_lin,input_add),   outputs = c(pBranchjoined),name="Bernoulli" )
  if(!is.null(X.nn) & is.null(X.add.basis) & !is.null(X.lin) ) model <- keras_model(  inputs = c(input_lin,input_nn),   outputs = c(pBranchjoined) ,name="Bernoulli")
  if(!is.null(X.nn) & !is.null(X.add.basis) & is.null(X.lin) )  model <- keras_model(  inputs = c(input_add,input_nn),   outputs = c(pBranchjoined) ,name="Bernoulli")
  if(is.null(X.nn) & is.null(X.add.basis) & !is.null(X.lin) )  model <- keras_model(  inputs = c(input_lin),   outputs = c(pBranchjoined) ,name="Bernoulli")
  if(is.null(X.nn) & !is.null(X.add.basis) & is.null(X.lin) )  model <- keras_model(  inputs = c(input_add),   outputs = c(pBranchjoined) ,name="Bernoulli")
  if(!is.null(X.nn) & is.null(X.add.basis) & is.null(X.lin) )  model <- keras_model(  inputs = c(input_nn),   outputs = c(pBranchjoined),name="Bernoulli" )

  print(model)

  return(model)

}



bce.loss <- function(){
  

   loss <- function( y_true, y_pred) {
      
      K <- backend()
      p=y_pred
      
      obsInds=K$sign(K$relu(y_true+1e4))
      
      #This will change the predicted p to 0.5 where there are no observations. Will fix likelihood evaluation issues!
      p=p-3*(1-obsInds)
      p=K$relu(p)+0.5*(1-obsInds)
      
      pc=1-p
      
      zeroInds = 1-K$sign(K$abs(y_true))
      
      #This will change the predicted p to 0.5 where there are zero values in y_true. Stops issues multiplying infinity with 0 which can occur for log(p) if p very small
      p=p-3*(zeroInds)
      p=K$relu(p)+0.5*(zeroInds)
      
      
      
      #This will change the predicted 1-p to 0.5 where there are one values in y_true. Stops issues multiplying infinity with 0 which can occur for log(1-p) if p close to one
      pc=pc-3*(1-zeroInds)
      pc=K$relu(pc)+0.5*(1-zeroInds)
      
      out <- K$abs(y_true)*K$log(p)+K$abs(1-y_true)*K$log(pc)
      out <- -K$sum(out * obsInds)/K$sum(obsInds)
      
      return(out)
    }
   
  return(loss)
}
