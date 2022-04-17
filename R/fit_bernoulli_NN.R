#' Build and train a partially-interpretable neural network to fit a logistic regression model
#'

#' @param type A string defining the type of network to be built. If \code{type=="MLP"}, the network will have all densely connected layers; if \code{type=="CNN"}, the network will have all convolutional layers. Defaults to an MLP.
#' @param Y_train,Y_test A 2 or 3 dimensional array of training or test response values, with entries of 0/1 for failure/success.
#' Missing values can be handled by setting corresponding entries to \code{Y_train} or \code{Y_test} to \code{-1e5}.
#' The first dimension should be the observation indices, e.g., time.
#' If \code{type=="CNN"}, then \code{Y_train} and \code{Y_test} must have three dimensions with the latter two corresponding to an \eqn{M} by \eqn{N} regular grid of spatial locations.
#' If \code{Y_test==NULL}, no validation loss will be computed and the returned model will be that which minimises the training loss over \code{n.ep} epochs.
#'
#' @param X_train_nn A 3 or 4 dimensional array of "non-additive" predictor values.  If \code{NULL}, a model without the NN component is built and trained; if thise is the case, then \code{type} has no efect.
#' The first 2/3 dimensions should be equal to that of \code{Y_train}; the last dimension corresponds to the chosen \eqn{d-l-a} 'non-additive' predictor values.
#' @param X_train_lin A 3 or 4 dimensional array of "linear" predictor values. Same number of dimensions as \code{X_train_nn}. If \code{NULL}, a model without the linear component is built and trained.
#' The first 2/3 dimensions should be equal to that of \code{Y_train}; the last dimension corresponds to the chosen \eqn{l} 'linear' predictor values.
#' @param X_train_add_basis A 4 or 5 dimensional array of basis function evaluations for the "additive" predictor values.
#' The first 2/3 dimensions should be equal to that of \code{Y_train}; the penultimate dimensions is equal to the number of knots used for estimating the splines and last dimension corresponds to the chosen \eqn{a} 'linear' predictor values.
#' If \code{NULL}, a model without the additive component is built and trained.
#' @param n.ep Number of epochs used for training. Defaults to 1000.
#' @param init.p Initial probability estimate. Applied across all predictor values.
#' @param widths Vector of widths/filters for hidden dense/convolution layers. Number of layers is equal to \code{length(widths)}. Defaults to 0.5
#' @param filter.dim If \code{type=="CNN"}, this 2-vector gives the dimensions of the convolution filter kernel. The same filter is applied for each hidden layer.

#' @details
#' Model is fitted by minimisation of binary cross-entropy loss over \code{n.ep} epochs using a logsitic link function to ensure that estimated probability is in \eqn{(0,1)}.
#'


#' @return Returns the fitted model which minimises some loss over the specified number of epochs; if \code{!is.null(Y_test)}, minimises the validation loss and minmises the training loss, otherwise.
#'
#'
#' @examples

#'
#' @rdname fit_bern_NN
#' @export

fit_bernoulli_nn=function(Y_train, Y_test = NULL,X_train_nn=NULL,X_train_lin=NULL,X_train_add_basis=NULL, type="MLP",
                          n.ep=1000, init.p=0.5, widths, filter.dim)
{

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


  if(!is.null(X_train_nn) & !is.null(X_train_add_basis) & !is.null(X_train_lin) ) model <- keras_model(  inputs = c(input_lin,input_add_p,input_nn),   outputs = c(pBranchjoined) )
  if(is.null(X_train_nn) & !is.null(X_train_add_basis) & !is.null(X_train_lin) )  model <- keras_model(  inputs = c(input_lin,input_add_p),   outputs = c(pBranchjoined) )
  if(!is.null(X_train_nn) & is.null(X_train_add_basis) & !is.null(X_train_lin) ) model <- keras_model(  inputs = c(input_lin,input_nn),   outputs = c(pBranchjoined) )
  if(!is.null(X_train_nn) & !is.null(X_train_add_basis) & is.null(X_train_lin) )  model <- keras_model(  inputs = c(input_add_p,input_nn),   outputs = c(pBranchjoined) )
  if(is.null(X_train_nn) & is.null(X_train_add_basis) & !is.null(X_train_lin) )  model <- keras_model(  inputs = c(input_lin),   outputs = c(pBranchjoined) )
  if(is.null(X_train_nn) & !is.null(X_train_add_basis) & is.null(X_train_lin) )  model <- keras_model(  inputs = c(input_add_p),   outputs = c(pBranchjoined) )
  if(!is.null(X_train_nn) & is.null(X_train_add_basis) & is.null(X_train_lin) )  model <- keras_model(  inputs = c(input_nn),   outputs = c(pBranchjoined) )

  print(summary(model))

  return(model)

}
