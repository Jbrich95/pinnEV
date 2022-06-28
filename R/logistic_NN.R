#' Logistic regression PINN

#'
#' Build and train a partially-interpretable neural network for a logistic regression model
#'
#'@name logistic.NN

#' @param type  string defining the type of network to be built. If \code{type=="MLP"}, the network will have all densely connected layers; if \code{type=="CNN"}, the network will have all convolutional layers. Defaults to an MLP.
#' @param Y_train,Y_valid a 2 or 3 dimensional array of training or validation response values, with entries of 0/1 for failure/success.
#' Missing values can be handled by setting corresponding entries to \code{Y_train} or \code{Y_valid} to \code{-1e5}.
#' The first dimension should be the observation indices, e.g., time.
#'
#' If \code{type=="CNN"}, then \code{Y_train} and \code{Y_valid} must have three dimensions with the latter two corresponding to an \eqn{M} by \eqn{N} regular grid of spatial locations.
#' If \code{Y_valid==NULL}, no validation loss will be computed and the returned model will be that which minimises the training loss over \code{n.ep} epochs.
#'
#' @param X_train  list of arrays corresponding to complementary subsets of the \eqn{d\geq 1} predictors which are used for modelling. Must contain at least one of the following three named entries:\describe{
#' \item{\code{X_train_lin}}{A 3 or 4 dimensional array of "linear" predictor values. One more dimension than \code{Y_train}. If \code{NULL}, a model without the linear component is built and trained.
#' The first 2/3 dimensions should be equal to that of \code{Y_train}; the last dimension corresponds to the chosen \eqn{l\geq 0} 'linear' predictor values.}
#' \item{\code{X_train_add_basis}}{A 4 or 5 dimensional array of basis function evaluations for the "additive" predictor values.
#' The first 2/3 dimensions should be equal to that of \code{Y_train}; the penultimate dimensions corresponds to the chosen \eqn{a\geq 0} 'linear' predictor values and the last dimension is equal to the number of knots used for estimating the splines. See example.
#' If \code{NULL}, a model without the additive component is built and trained.}
#' \item{\code{X_train_nn}}{A 3 or 4 dimensional array of "non-additive" predictor values.  If \code{NULL}, a model without the NN component is built and trained; if this is the case, then \code{type} has no efect.
#' The first 2/3 dimensions should be equal to that of \code{Y_train}; the last dimension corresponds to the chosen \eqn{d-l-a\geq 0} 'non-additive' predictor values.}
#' }
#' Note that \code{X_train} is the predictors for both \code{Y_train} and \code{Y_valid}.
#' @param n.ep number of epochs used for training. Defaults to 1000.
#' @param batch.size batch size for stochastic gradient descent. If larger than \code{dim(Y_train)[1]}, i.e., the number of observations, then regular gradient descent used.
#' @param init.p sets the initial probability estimate across all dimensions of \code{Y_train}. Defaults to empirical estimate. Overriden by \code{init.wb_path} if \code{!is.null(init.wb_path)}.
#' @param init.wb_path filepath to a \code{keras} model which is then used as initial weights and biases for training the new model. The original model must have
#' the exact same architecture and trained with the same input data as the new model. If \code{NULL}, then initial weights and biases are random (with seed \code{seed}) but the
#' final layer has zero initial weights to ensure that the initial probability estimate is \code{init.p} across all dimensions.
#' @param widths vector of widths/filters for hidden dense/convolution layers. Number of layers is equal to \code{length(widths)}. Defaults to (6,3).
#' @param filter.dim if \code{type=="CNN"}, this 2-vector gives the dimensions of the convolution filter kernel; must have odd integer inputs. Note that filter.dim=c(1,1) is equivalent to \code{type=="MLP"}. The same filter is applied for each hidden layer.
#' @param seed seed for random initial weights and biases.
#' @param model fitted \code{keras} model. Output from \code{logistic.NN.train}.

#' @details{
#' Consider a Bernoulli random variable, say \eqn{Z\sim\mbox{Bernoulli}(p)}, with probability mass function \eqn{\Pr(Z=1)=p=1-\Pr(Z=0)=1-p}. Let \eqn{Y\in\{0,1\}} be a univariate Boolean response and let \eqn{\mathbf{X}} denote a \eqn{d}-dimensional predictor set with observations \eqn{\mathbf{x}}.
#' For integers \eqn{l\geq 0,a \geq 0} and \eqn{0\leq l+a \leq d}, let \eqn{\mathbf{X}_L, \mathbf{X}_A} and \eqn{\mathbf{X}_N} be distinct sub-vectors of \eqn{\mathbf{X}}, with observations of each component denoted \eqn{\mathbf{x}_L, \mathbf{x}_A} and \eqn{\mathbf{x}_N}, respectively; the lengths of the sub-vectors are \eqn{l,a} and \eqn{d-l-a}, respectively.
#'  We model \eqn{Y|\mathbf{X}=\mathbf{x}\sim\mbox{Bernoulli}(p(\mathbf{x}))} with
#' \deqn{p(\mathbf{x})=h[\eta_0+m_L\{\mathbf{x}_L\}+m_A\{x_A\}+m_N\{\mathbf{x}_N\}],} where \eqn{h} is the logistic link-function and
#' \eqn{\eta_0} is a constant intercept. The unknown functions \eqn{m_L} and \eqn{m_A} are estimated using a linear function and spline, respectively,
#' and are both returned as outputs by \code{logistic.NN.predict}; \eqn{m_N} is estimated using a neural network.
#'
#' The model is fitted by minimising the binary cross-entropy loss over \code{n.ep} training epochs.
#' Although the model is trained by minimising the loss evaluated for \code{Y_train}, the final returned model may minimise some other loss.
#' The current state of the model is saved after each epoch, using \code{keras::callback_model_checkpoint}, if the value of some criterion subcedes that of the model from the previous checkpoint; this criterion is the loss evaluated for validation set \code{Y_valid} if \code{!is.null(Y_valid)} and for \code{Y_train}, otherwise.
#'
#'}
#' @return \code{logistic.NN.train} returns the fitted \code{model}.  \code{logistic.NN.predict} is a wrapper for \code{keras::predict} that returns the predicted probability estimates, and, if applicable, the linear regression coefficients and spline bases weights.
#'
#'
#' @examples
#'
#'
#'
#' # Build and train a simple MLP for toy data
#'
#' # Create 'nn', 'additive' and 'linear' predictors
#' X_train_nn<-rnorm(5000); X_train_add<-rnorm(2000); X_train_lin<-rnorm(3000)
#'
#' #Re-shape to a 4d array. First dimension corresponds to observations,
#' #last to the different components of the predictor set
#' dim(X_train_nn)=c(10,10,10,5) #Five nn predictors
#' dim(X_train_lin)=c(10,10,10,3) #Three linear predictors
#' dim(X_train_add)=c(10,10,10,2) #Two additive predictors
#'
#' # Create toy response data
#'
#' #Linear contribution
#' m_L = 0.3*X_train_lin[,,,1]+0.6*X_train_lin[,,,2]-0.2*X_train_lin[,,,3]
#'
#' # Additive contribution
#' m_A = 0.1*X_train_add[,,,1]^2+0.2*X_train_add[,,,1]-0.1*X_train_add[,,,2]^3+
#' 0.5*X_train_add[,,,2]^2
#'
#' #Non-additive contribution - to be estimated by NN
#' m_N = exp(-3+X_train_nn[,,,2]+X_train_nn[,,,3])+
#' sin(X_train_nn[,,,1]-X_train_nn[,,,2])*(X_train_nn[,,,4]+X_train_nn[,,,5])
#'
#' p=0.5+0.5*tanh((m_L+m_A+m_N)/2) #Logistic link
#' Y=apply(p,1:3,function(x) rbinom(1,1,x))
#'
#' #Create training and validation, respectively.
#' #We mask 20% of the Y values and use this for validation
#' #Masked values must be set to -1e5 and are treated as missing whilst training
#'
#' mask_inds=sample(1:length(Y),size=length(Y)*0.8)
#'
#' Y_train<-Y_valid<-Y #Create training and validation, respectively.
#' Y_train[-mask_inds]=-1e5
#' Y_valid[mask_inds]=-1e5
#'
#'
#'
#' #To build a model with an additive component, we require an array of evaluations of
#' #the basis functions for each pre-specified knot and entry to X_train_add
#'
#' rad=function(x,c){ #Define a basis function. Here we use the radial bases
#'   out=abs(x-c)^2*log(abs(x-c))
#'   out[(x-c)==0]=0
#'   return(out)
#' }
#'
#' n.knot = 5 # set number of knots. Must be the same for each additive predictor
#' knots=matrix(nrow=dim(X_train_add)[4],ncol=n.knot)
#'
#' #We set knots to be equally-spaced marginal quantiles
#' for( i in 1:dim(X_train_add)[4]) {
#' knots[i,]=quantile(X_train_add[,,,i],probs=seq(0,1,length=n.knot))}
#'
#' X_train_add_basis<-array(dim=c(dim(X_train_add),n.knot))
#' for( i in 1:dim(X_train_add)[4]) {
#' for(k in 1:n.knot) {
#' X_train_add_basis[,,,i,k]= rad(x=X_train_add[,,,i],c=knots[i,k])
#' #Evaluate rad at all entries to X_train_add and for all knots
#' }}
#'
#' X_train=list("X_train_nn"=X_train_nn, "X_train_lin"=X_train_lin,
#' "X_train_add_basis"=X_train_add_basis)
#'
#' #Build and train a two-layered "lin+GAM+NN" MLP
#' model<-logistic.NN.train(Y_train, Y_valid,X_train,  type="MLP",n.ep=2000,
#'                       batch.size=50,init.p=0.4, widths=c(6,3))
#'
#' out<-predict_bernoulli_nn(X_train,model)
#' hist(out$predictions) #Plot histogram of predicted probability
#' print(out$lin.coeff)
#'
#'n.add.preds=dim(X_train_add)[length(dim(X_train_add))]
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
#'#To save model, run model %>% save_model_tf("model_Bernoulli")
#'#To load model, run model  <- load_model_tf("model_Bernoulli",
#'#custom_objects=list("bce_loss"=bce_loss))
#'
#' @rdname logistic.NN
#' @export

logistic.NN.train=function(Y_train, Y_valid = NULL,X_train, type="MLP",
                          n.ep=100, batch.size=100,init.p=NULL, widths=c(6,3), filter.dim=c(3,3),
                       seed=NULL, init.wb_path=NULL)
{




  if(is.null(X_train)  ) stop("No predictors provided")
  if(is.null(Y_train)) stop("No training response data provided")

  X_train_nn=X_train$X_train_nn
  X_train_lin=X_train$X_train_lin
  X_train_add_basis=X_train$X_train_add_basis

  if(!is.null(X_train_nn) & !is.null(X_train_add_basis) & !is.null(X_train_lin) ) {  train.data= list(X_train_lin,X_train_add_basis,X_train_nn); print("Defining lin+GAM+NN model for p" );  if(!is.null(Y_valid)) validation.data=list(list(lin_input_p=X_train_lin,add_input_p=X_train_add_basis,  nn_input_p=X_train_nn),Y_valid)}
  if(is.null(X_train_nn) & !is.null(X_train_add_basis) & !is.null(X_train_lin) ) {   train.data= list(X_train_lin,X_train_add_basis); print("Defining lin+GAM model for p" );  if(!is.null(Y_valid)) validation.data=list(list(lin_input_p=X_train_lin,add_input_p=X_train_add_basis),Y_valid)}
  if(!is.null(X_train_nn) & is.null(X_train_add_basis) & !is.null(X_train_lin) ) { train.data= list(X_train_lin,X_train_nn); print("Defining lin+NN model for p" );  if(!is.null(Y_valid)) validation.data=list(list(lin_input_p=X_train_lin, nn_input_p=X_train_nn),Y_valid)}
  if(!is.null(X_train_nn) & !is.null(X_train_add_basis) & is.null(X_train_lin) ) {train.data= list(X_train_add_basis,X_train_nn); print("Defining GAM+NN model for p" );  if(!is.null(Y_valid)) validation.data=list(list(add_input_p=X_train_add_basis,  nn_input_p=X_train_nn),Y_valid)}
  if(is.null(X_train_nn) & is.null(X_train_add_basis) & !is.null(X_train_lin) )   {train.data= list(X_train_lin); print("Defining fully-linear model for p" );  if(!is.null(Y_valid)) validation.data=list(list(lin_input_p=X_train_lin),Y_valid)}
  if(is.null(X_train_nn) & !is.null(X_train_add_basis) & is.null(X_train_lin) )   {train.data= list(X_train_add_basis); print("Defining fully-additive model for p" );  if(!is.null(Y_valid)) validation.data=list(list(add_input_p=X_train_add_basis),Y_valid)}
  if(!is.null(X_train_nn) & is.null(X_train_add_basis) & is.null(X_train_lin) )   {train.data= list(X_train_nn); print("Defining fully-NN model for p" );  if(!is.null(Y_valid)) validation.data=list(list( nn_input_p=X_train_nn),Y_valid)}

  if(type=="CNN" & !is.null(X_train_nn)) print(paste0("Building ",length(widths),"-layer convolutional neural network with ", filter.dim[1]," by ", filter.dim[2]," filter" ))
  if(type=="MLP"  & !is.null(X_train_nn) ) print(paste0("Building ",length(widths),"-layer densely-connected neural network" ))

  reticulate::use_virtualenv("myenv", required = T)

  if(!is.null(seed)) tf$random$set_seed(seed)
  if(is.null(init.p)) init.p=mean(Y_train[Y_train>=0]==1)


  model<-logistic.NN.build(X_train_nn,X_train_lin,X_train_add_basis, type, init.p, widths,filter.dim)
  if(!is.null(init.wb_path)) model <- load_model_weights_tf(model,filepath=init.wb_path)
  model %>% compile(
    optimizer="adam",
    loss = bce_loss,
    run_eagerly=T
  )

  if(!is.null(Y_valid)) checkpoint <- callback_model_checkpoint(paste0("model_bernoulli_checkpoint"), monitor = "val_loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch") else checkpoint <- callback_model_checkpoint(paste0("model_bernoulli_checkpoint"), monitor = "loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch")


  if(!is.null(Y_valid)){
    history <- model %>% fit(
      train.data, Y_train,
      epochs = n.ep, batch_size = batch.size,
      callback=list(checkpoint),
      validation_data=validation.data

    )
  }else{

    history <- model %>% fit(
      train.data, Y_train,
      epochs = n.ep, batch_size = batch.size,
      callback=list(checkpoint)
    )
  }
  print("Loading checkpoint weights")
  model <- load_model_weights_tf(model,filepath=paste0("model_bernoulli_checkpoint"))

  return(model)
}
#' @rdname logistic.NN
#' @export
#'
logistic.NN.predict=function(X_train, model)
{



  if(is.null(X_train)  ) stop("No predictors provided")

  X_train_nn=X_train$X_train_nn
  X_train_lin=X_train$X_train_lin
  X_train_add_basis=X_train$X_train_add_basis

  if(!is.null(X_train_nn) & !is.null(X_train_add_basis) & !is.null(X_train_lin) ) predictions<-model %>% predict(  list(X_train_lin,X_train_add_basis,X_train_nn))
  if(is.null(X_train_nn) & !is.null(X_train_add_basis) & !is.null(X_train_lin) )    predictions<-model %>% predict( list(X_train_lin,X_train_add_basis))
  if(!is.null(X_train_nn) & is.null(X_train_add_basis) & !is.null(X_train_lin) ) predictions<-model %>% predict( list(X_train_lin,X_train_nn))
  if(!is.null(X_train_nn) & !is.null(X_train_add_basis) & is.null(X_train_lin) ) predictions<-model %>% predict( list(X_train_add_basis,X_train_nn))
  if(is.null(X_train_nn) & is.null(X_train_add_basis) & !is.null(X_train_lin) )  predictions<-model %>% predict(list(X_train_lin))
  if(is.null(X_train_nn) & !is.null(X_train_add_basis) & is.null(X_train_lin) )   predictions<-model %>% predict( list(X_train_add_basis))
  if(!is.null(X_train_nn) & is.null(X_train_add_basis) & is.null(X_train_lin) )   predictions<-model %>% predict( list(X_train_nn))

  if(!is.null(X_train_add_basis))  gam.weights<-matrix(t(model$get_layer("add_p")$get_weights()[[1]]),nrow=dim(X_train_add_basis)[length(dim(X_train_add_basis))-1],ncol=dim(X_train_add_basis)[length(dim(X_train_add_basis))],byrow=T)

  if(!is.null(X_train_add_basis) & !is.null(X_train_lin)) return(list("predictions"=predictions, "lin.coeff"=c(model$get_layer("lin_p")$get_weights()[[1]]),"gam.weights"=gam.weights))
  if(is.null(X_train_add_basis) & !is.null(X_train_lin)) return(list("predictions"=predictions, "lin.coeff"=c(model$get_layer("lin_p")$get_weights()[[1]])))
  if(!is.null(X_train_add_basis) & is.null(X_train_lin)) return(list("predictions"=predictions,"gam.weights"=gam.weights))
  if(is.null(X_train_add_basis) & is.null(X_train_lin)) return(list("predictions"=predictions))


}
#'
#'
logistic.NN.build=function(X_train_nn,X_train_lin,X_train_add_basis, type, init.p, widths,filter.dim)
{
  #Additive input
  if(!is.null(X_train_add_basis))  input_add<- layer_input(shape = dim(X_train_add_basis)[-1], name = 'add_input_p')

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

    addBranchp <- input_add %>%
      layer_reshape(target_shape=c(dim(X_train_add_basis)[2:(n.dim.add-2)],prod(dim(X_train_add_basis)[(n.dim.add-1):n.dim.add]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_p',
                  weights=list(matrix(0,nrow=prod(dim(X_train_add_basis)[(n.dim.add-1):n.dim.add]),ncol=1)),use_bias = F)
  }
  if(!is.null(X_train_add_basis) & is.null(X_train_add_basis) ) {

    addBranchp <- input_add %>%
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


  #Apply link functions
  pBranchjoined <- pBranchjoined %>%
    layer_activation( activation = 'sigmoid')


  if(!is.null(X_train_nn) & !is.null(X_train_add_basis) & !is.null(X_train_lin) ) model <- keras_model(  inputs = c(input_lin,input_add,input_nn),   outputs = c(pBranchjoined),name="Bernoulli" )
  if(is.null(X_train_nn) & !is.null(X_train_add_basis) & !is.null(X_train_lin) )  model <- keras_model(  inputs = c(input_lin,input_add),   outputs = c(pBranchjoined),name="Bernoulli" )
  if(!is.null(X_train_nn) & is.null(X_train_add_basis) & !is.null(X_train_lin) ) model <- keras_model(  inputs = c(input_lin,input_nn),   outputs = c(pBranchjoined) ,name="Bernoulli")
  if(!is.null(X_train_nn) & !is.null(X_train_add_basis) & is.null(X_train_lin) )  model <- keras_model(  inputs = c(input_add,input_nn),   outputs = c(pBranchjoined) ,name="Bernoulli")
  if(is.null(X_train_nn) & is.null(X_train_add_basis) & !is.null(X_train_lin) )  model <- keras_model(  inputs = c(input_lin),   outputs = c(pBranchjoined) ,name="Bernoulli")
  if(is.null(X_train_nn) & !is.null(X_train_add_basis) & is.null(X_train_lin) )  model <- keras_model(  inputs = c(input_add),   outputs = c(pBranchjoined) ,name="Bernoulli")
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
