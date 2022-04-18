#' Build and train a partially-interpretable neural network to fit a logistic regression model
#'

#' @param type A string defining the type of network to be built. If \code{type=="MLP"}, the network will have all densely connected layers; if \code{type=="CNN"}, the network will have all convolutional layers. Defaults to an MLP.
#' @param Y_train,Y_test A 2 or 3 dimensional array of training or test response values, with entries of 0/1 for failure/success.
#' Missing values can be handled by setting corresponding entries to \code{Y_train} or \code{Y_test} to \code{-1e5}.
#' The first dimension should be the observation indices, e.g., time.
#' If \code{type=="CNN"}, then \code{Y_train} and \code{Y_test} must have three dimensions with the latter two corresponding to an \eqn{M} by \eqn{N} regular grid of spatial locations.
#' If \code{Y_test==NULL}, no validation loss will be computed and the returned model will be that which minimises the training loss over \code{n.ep} epochs.
#'
#' @param X_train_nn A 3 or 4 dimensional array of "non-additive" predictor values.  If \code{NULL}, a model without the NN component is built and trained; if this is the case, then \code{type} has no efect.
#' The first 2/3 dimensions should be equal to that of \code{Y_train}; the last dimension corresponds to the chosen \eqn{d-l-a} 'non-additive' predictor values.
#' @param X_train_lin A 3 or 4 dimensional array of "linear" predictor values. Same number of dimensions as \code{X_train_nn}. If \code{NULL}, a model without the linear component is built and trained.
#' The first 2/3 dimensions should be equal to that of \code{Y_train}; the last dimension corresponds to the chosen \eqn{l} 'linear' predictor values.
#' @param X_train_add_basis A 4 or 5 dimensional array of basis function evaluations for the "additive" predictor values.
#' The first 2/3 dimensions should be equal to that of \code{Y_train}; the penultimate dimensions corresponds to the chosen \eqn{a} 'linear' predictor values and the last dimension is equal to the number of knots used for estimating the splines. See example.
#' If \code{NULL}, a model without the additive component is built and trained.
#' @param n.ep Number of epochs used for training. Defaults to 1000.
#' @param batch.size Batch size for stochastic gradient descent. If larger than \code{dim(Y_train)[1]}, i.e., the number of observations, then regular gradient descent used.
#' @param init.p Sets the initial probability estimate across all dimensions of \code{Y_train}.
#' @param widths Vector of widths/filters for hidden dense/convolution layers. Number of layers is equal to \code{length(widths)}. Defaults to 0.5.
#' @param filter.dim If \code{type=="CNN"}, this 2-vector gives the dimensions of the convolution filter kernel. The same filter is applied for each hidden layer.
#' @param seed Seed for random initial weights and biases.

#' @details
#' Model is fitted by minimising the binary cross-entropy loss over \code{n.ep} epochs, with a logistic link function used to ensure that the estimated probabilities are in \eqn{(0,1)}.
#' Although the model is trained by minimising the loss evaluated for \code{Y_train}, the final returned model may minimise some other loss.
#' The current state of the model is saved after each epoch, using \code{keras::callback_model_checkpoint}, if the value of some criterion subcedes that of the model from the previous checkpoint; this criterion is the loss evaluated for validation/test set \code{Y_test} if \code{!is.null(Y_test)} and for \code{Y_train}, otherwise.
#'


#' @return Returns the fitted model
#'
#'
#' @examples
#'
#' ##Build and train a simple MLP for toy data
#'X_train_nn<-rnorm(2000); X_train_add<-rnorm(2000); X_test_lin<-rnorm(2000) # Create 'nn', 'additive' and 'linear' predictors
#'
#'dim(X_train_nn)=c(10,10,2,10) #Re-shape to a 4-d array. First dimension corresponds to observations, last to the different components of the predictor set
#'dim(X_train_lin)=c(10,10,2,10)
#'dim(X_train_add)=c(10,10,2,10)
#'
#'#To build a model with an additive component, we require an array of evaluations of the basis functions for each pre-specified knot and entry to X_train_add
#'
#' rad=function(x,c){ #Define a basis function. Here we use the radial bases
#' out=abs(x-c)^2*log(abs(x-c))
#' out[(x-c)==0]=0
#' return(out)
#' }
#'
#'n.knot = 5 # set number of knots. Must be the same for each additive predictor
#'knots=matrix(nrow=dim(X_train_add)[4],ncol=n.knot)
#'
#' #We set knots to be equally-spaced marginal quantiles
#'for( i in 1:dim(X_train_add)[4]) knots[i,]=quantile(X_train_add[,,,i],probs=seq(0,1,length=n.knot))
#'
#'X_train_add_basis<-array(dim=c(dim(X_train_add),n.knot))
#'for( i in 1:dim(X_train_add)[4]) for(k in 1:n.knot)  X_train_add_basis[,,,i,k]= rad(x=X_train_add[,,,i],c=knots[i,k]) #Evaluate rad at all entries to X_train_add and for all knots
#'
#'Y_train<-Y_test<-rnorm(200) #Create response data; training and test, respectively
#'dim(Y_train)=c(10,10,2) #Re-shape to a 4-d array. First dimension corresponds to observations
#'dim(Y_test)=c(10,10,2)
#'
#' #Build and train a two-layered "lin+GAM+NN" MLP
#'model<-fit_bernoulli_nn(Y_train, Y_test ,X_train_nn,X_train_lin,X_train_add_basis, type="MLP",n.ep=100, batch.size=50,init.p=0.5, widths=c(6,3))
#'
#'


#'
#' @rdname fit_bern_NN
#' @export

fit_bernoulli_nn=function(Y_train, Y_test = NULL,X_train_nn,X_train_lin,X_train_add_basis, type="MLP",
                          n.ep=100, batch.size=100,init.p=0.5, widths=c(6,3), filter.dim=c(3,3),seed=NULL)
{
  if(is.null(X_train_nn) & is.null(X_train_add_basis) & is.null(X_train_lin) ) stop("No predictors provided")
  if(is.null(Y_train)) stop("No training response data provided")
  if(type=="CNN") print(paste0("Building ",length(widths),"-layer convolutional neural network with ", filter.dim[1]," by ", filter.dim[2]," filter" ))
  if(type=="MLP") print(paste0("Building ",length(widths),"-layer densely-connected neural network" ))


  reticulate::use_virtualenv("myenv", required = T)

  if(!is.null(seed)) tf$random$set_seed(seed)

  model<-build_bernoulli_nn(X_train_nn,X_train_lin,X_train_add_basis, type, init.p, widths,filter.dim)

  model %>% compile(
    optimizer="adam",
    loss = bce_loss,
    run_eagerly=T
  )

  if(!is.null(Y_test)) checkpoint <- callback_model_checkpoint(paste0("model_bernoulli_checkpoint"), monitor = "val_loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch") else checkpoint <- callback_model_checkpoint(paste0("model_bernoulli_checkpoint"), monitor = "loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch")

  if(!is.null(X_train_nn) & !is.null(X_train_add_basis) & !is.null(X_train_lin) ) {  test.data= list(X_train_lin,X_train_add_basis,X_train_nn);  if(!is.null(Y_test)) validation.data=list(list(lin_input_p=X_train_lin,add_input_p=X_train_add_basis,  nn_input_p=X_train_nn),Y_test)}
  if(is.null(X_train_nn) & !is.null(X_train_add_basis) & !is.null(X_train_lin) ) {   test.data= list(X_train_lin,X_train_add_basis);  if(!is.null(Y_test)) validation.data=list(list(lin_input_p=X_train_lin,add_input_p=X_train_add_basis),Y_test)}
  if(!is.null(X_train_nn) & is.null(X_train_add_basis) & !is.null(X_train_lin) ) { test.data= list(X_train_lin,X_train_nn);  if(!is.null(Y_test)) validation.data=list(list(lin_input_p=X_train_lin, nn_input_p=X_train_nn),Y_test)}
  if(!is.null(X_train_nn) & !is.null(X_train_add_basis) & is.null(X_train_lin) ) {test.data= list(X_train_add_basis,X_train_nn);  if(!is.null(Y_test)) validation.data=list(list(add_input_p=X_train_add_basis,  nn_input_p=X_train_nn),Y_test)}
  if(is.null(X_train_nn) & is.null(X_train_add_basis) & !is.null(X_train_lin) )   {test.data= list(X_train_lin);  if(!is.null(Y_test)) validation.data=list(list(lin_input_p=X_train_lin),Y_test)}
  if(is.null(X_train_nn) & !is.null(X_train_add_basis) & is.null(X_train_lin) )   {test.data= list(X_train_add_basis);  if(!is.null(Y_test)) validation.data=list(list(add_input_p=X_train_add_basis),Y_test)}
  if(!is.null(X_train_nn) & is.null(X_train_add_basis) & is.null(X_train_lin) )   {test.data= list(X_train_nn);  if(!is.null(Y_test)) validation.data=list(list( nn_input_p=X_train_nn),Y_test)}

  if(!is.null(Y_test)){
  history <- model %>% fit(
    test.data, Y_train,
    epochs = n.ep, batch_size = batch.size,
    callback=list(checkpoint),
    validation_data=validation.data

  )
  }else{

    history <- model %>% fit(
      test.data, Y_train,
      epochs = n.ep, batch_size = batch.size,
      callback=list(checkpoint)
    )
  }

  return(model)
}

build_bernoulli_nn=function(X_train_nn,X_train_lin,X_train_add_basis, type, init.p=0.5, widths=c(6,3),filter.dim=c(3,3))
{
  #Additive input
  if(!is.null(X_train_add_basis))  input_add_p<- layer_input(shape = dim(X_train_add_basis)[-1], name = 'add_input_p')

  #NN input

  if(!is.null(X_train_nn))   input_nn <- layer_input(shape = dim(X_train_nn)[-1], name = 'nn_input_p')

  #Linear input

  if(!is.null(X_train_lin)) input_lin <- layer_input(shape = dim(X_train_lin)[-1], name = 'lin_input_p')


  #NN tower
  if(!is.null(X_train_nn)){

    nunits=c(widths,1)
    n.layers=length(nunits)-1

    nnBranchp <- input_nn
    if(type=="MLP"){
    for(i in 1:n.layers){
      nnBranchp <- nnBranchp  %>% layer_dense(units=nunits[i],activation = 'relu',
                                              input_shape =dim(X_train_nn)[-1], name = paste0('nn_p_dense',i) )
    }
    }else if(type=="CNN"){
      for(i in 1:n.layers){
        nnBranchp <- nnBranchp  %>% layer_conv_2d(filters=nunits[i],activation = 'relu',kernel_size=c(filter.dim[1],filter.dim[2]), padding='same',
                                                input_shape =dim(X_train_nn)[-1], name = paste0('nn_p_cnn',i) )
      }

    }

    nnBranchp <-   nnBranchp  %>%   layer_dense(units = nunits[n.layers+1], activation = "linear", name = 'nn_p_dense_final',
                                                weights=list(matrix(0,nrow=nunits[n.layers],ncol=1), array(log(init.p/(1-init.p)))))

  }

  #Additive tower
  n.dim.add=length(dim(X_train_add_basis))
  if(!is.null(X_train_add_basis) & !is.null(X_train_add_basis) ) {

    addBranchp <- input_add_p %>%
      layer_reshape(target_shape=c(dim(X_train_add_basis)[2:(n.dim.add-2)],prod(dim(X_train_add_basis)[(n.dim.add-1):n.dim.add]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_p',
                  weights=list(matrix(0,nrow=prod(dim(X_train_add_basis)[(n.dim.add-1):n.dim.add]),ncol=1)),use_bias = F)
  }
  if(!is.null(X_train_add_basis) & is.null(X_train_add_basis) ) {

    addBranchp <- input_add_p %>%
      layer_reshape(target_shape=c(dim(X_train_add_basis)[2:(n.dim.add-2)],prod(dim(X_train_add_basis)[(n.dim.add-1):n.dim.add]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_p',
                  weights=list(matrix(0,nrow=prod(dim(X_train_add_basis)[(n.dim.add-1):n.dim.add]),ncol=1),array(log(init.p/(1-init.p)))),use_bias = T)
  }

  #Linear tower


  if(!is.null(X_train_lin) ) {
    n.dim.lin=length(dim(X_train_lin))

    if(is.null(X_train_nn) & is.null(X_train_add_basis )){
      linBranchp <- input_lin%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X_train_lin)[-1], name = 'lin_p',
                    weights=list(matrix(0,nrow=dim(X_train_lin)[n.dim.lin],ncol=1),array(log(init.p/(1-init.p)))),use_bias=T)
    }else{
      linBranchp <- input_lin%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X_train_lin)[-1], name = 'lin_p',
                    weights=list(matrix(0,nrow=dim(X_train_lin)[n.dim.lin],ncol=1)),use_bias=F)
    }
  }



  if(!is.null(X_train_nn) & !is.null(X_train_add_basis) & !is.null(X_train_lin) )  pBranchjoined <- layer_add(inputs=c(addBranchp,  linBranchp,nnBranchp))  #Add all towers
  if(is.null(X_train_nn) & !is.null(X_train_add_basis) & !is.null(X_train_lin) )  pBranchjoined <- layer_add(inputs=c(addBranchp,  linBranchp))  #Add GAM+lin towers
  if(!is.null(X_train_nn) & is.null(X_train_add_basis) & !is.null(X_train_lin) )  pBranchjoined <- layer_add(inputs=c(  linBranchp,nnBranchp))  #Add NN+lin towers
  if(!is.null(X_train_nn) & !is.null(X_train_add_basis) & is.null(X_train_lin) )  pBranchjoined <- layer_add(inputs=c(addBranchp,  nnBranchp))  #Add NN+GAM towers
  if(is.null(X_train_nn) & is.null(X_train_add_basis) & !is.null(X_train_lin) )  pBranchjoined <- linBranchp  #Just lin tower
  if(is.null(X_train_nn) & !is.null(X_train_add_basis) & is.null(X_train_lin) )  pBranchjoined <- addBranchp  #Just GAM tower
  if(!is.null(X_train_nn) & is.null(X_train_add_basis) & is.null(X_train_lin) )  pBranchjoined <- nnBranchp  #Just NN tower


  #Use exponential activation so sig > 0
  pBranchjoined <- pBranchjoined %>%
    layer_activation( activation = 'sigmoid')


  if(!is.null(X_train_nn) & !is.null(X_train_add_basis) & !is.null(X_train_lin) ) model <- keras_model(  inputs = c(input_lin,input_add_p,input_nn),   outputs = c(pBranchjoined),name="Bernoulli" )
  if(is.null(X_train_nn) & !is.null(X_train_add_basis) & !is.null(X_train_lin) )  model <- keras_model(  inputs = c(input_lin,input_add_p),   outputs = c(pBranchjoined),name="Bernoulli" )
  if(!is.null(X_train_nn) & is.null(X_train_add_basis) & !is.null(X_train_lin) ) model <- keras_model(  inputs = c(input_lin,input_nn),   outputs = c(pBranchjoined) ,name="Bernoulli")
  if(!is.null(X_train_nn) & !is.null(X_train_add_basis) & is.null(X_train_lin) )  model <- keras_model(  inputs = c(input_add_p,input_nn),   outputs = c(pBranchjoined) ,name="Bernoulli")
  if(is.null(X_train_nn) & is.null(X_train_add_basis) & !is.null(X_train_lin) )  model <- keras_model(  inputs = c(input_lin),   outputs = c(pBranchjoined) ,name="Bernoulli")
  if(is.null(X_train_nn) & !is.null(X_train_add_basis) & is.null(X_train_lin) )  model <- keras_model(  inputs = c(input_add_p),   outputs = c(pBranchjoined) ,name="Bernoulli")
  if(!is.null(X_train_nn) & is.null(X_train_add_basis) & is.null(X_train_lin) )  model <- keras_model(  inputs = c(input_nn),   outputs = c(pBranchjoined),name="Bernoulli" )

  print(model)

  return(model)

}



bce_loss <- function( y_true, y_pred) {

  K <- backend()
  p=y_pred

  obsInds=K$sign(K$relu(y_true+1e4))

  loss <- K$abs(y_true)*K$log(p)+K$abs(1-y_true)*K$log(1-p)
  loss <- -K$sum(loss * obsInds)/K$sum(obsInds)

  return(loss)
}

