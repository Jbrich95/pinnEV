#' Build and train a partially-interpretable neural network for fitting a bGEV point-process model
#'

#' @param type A string defining the type of network to be built. If \code{type=="MLP"}, the network will have all densely connected layers; if \code{type=="CNN"},
#'  the network will have all convolutional layers. Defaults to an MLP. (Currently the same network is used for all parameters, may change in future versions)
#' @param Y_train,Y_test A 2 or 3 dimensional array of training or test real response values.
#' Missing values can be handled by setting corresponding entries to \code{Y_train} or \code{Y_test} to \code{-1e5}.
#' The first dimension should be the observation indices, e.g., time.
#'
#' If \code{type=="CNN"}, then \code{Y_train} and \code{Y_test} must have three dimensions with the latter two corresponding to an \eqn{M} by \eqn{N} regular grid of spatial locations.
#' If \code{Y_test==NULL}, no validation loss will be computed and the returned model will be that which minimises the training loss over \code{n.ep} epochs.
#'
#'@param u_train An array with the same dimension of \code{Y_train}. Gives the quantile above which the bGEV-PP model is fitted, see below. Note that \code{u_train} is applies to both \code{Y_train} and \code{Y_test}.
#' @param X_train_q A list of arrays corresponding to complementary subsets of the \eqn{d\geq 1} predictors which are used for modelling the location parameter \eqn{q_\alpha}. Must contain at least one of the following three named entries:\describe{
#' \item{\code{X_train_lin_q}}{A 3 or 4 dimensional array of "linear" predictor values. Same number of dimensions as \code{X_train_nn_1}. If \code{NULL}, a model without the linear component is built and trained.
#' The first 2/3 dimensions should be equal to that of \code{Y_train}; the last dimension corresponds to the chosen \eqn{l_1\geq 0} 'linear' predictor values.}
#' \item{\code{X_train_add_basis_q}}{A 4 or 5 dimensional array of basis function evaluations for the "additive" predictor values.
#' The first 2/3 dimensions should be equal to that of \code{Y_train}; the penultimate dimensions corresponds to the chosen \eqn{a_1\geq 0} 'linear' predictor values and the last dimension is equal to the number of knots used for estimating the splines. See example.
#' If \code{NULL}, a model without the additive component is built and trained.}
#' \item{\code{X_train_nn_q}}{A 3 or 4 dimensional array of "non-additive" predictor values.  If \code{NULL}, a model without the NN component is built and trained; if this is the case, then \code{type} has no effect.
#' The first 2/3 dimensions should be equal to that of \code{Y_train}; the last dimension corresponds to the chosen \eqn{d-l_1-a_1\geq 0} 'non-additive' predictor values.}
#' }
#' Note that \code{X_train_q} and \code{X_train_s} are the predictors for both \code{Y_train} and \code{Y_test}.
#' @param X_train_S Similarly to \code{X_train_s}, but for modelling the scale parameter \eqn{s_\beta>0}. Note that both \eqn{q_\beta} and \eqn{s_\beta} must be modelled as non-stationary in this version.
#' @param n.ep Number of epochs used for training. Defaults to 1000.
#' @param alpha,beta,p_a,p_b Constants associated with the bGEV distribution. Defaults to those used by Castro-Camilo, D., et al. (2021). Require \code{alpha >= p_b} and \code{beta/2 >= p_b}.
#' @param batch.size Batch size for stochastic gradient descent. If larger than \code{dim(Y_train)[1]}, i.e., the number of observations, then regular gradient descent used.
#' @param init.loc,init.spread,init.xi Sets the initial \eqn{q_\alpha,s_\beta} and \eqn{\xi} estimates across all dimensions of \code{Y_train}. Must be supplied.
#' @param widths Vector of widths/filters for hidden dense/convolution layers. Number of layers is equal to \code{length(widths)}. Defaults to (6,3).
#' @param filter.dim If \code{type=="CNN"}, this 2-vector gives the dimensions of the convolution filter kernel; must have odd integer inputs. Note that filter.dim=c(1,1) is equivalent to \code{type=="MLP"}. The same filter is applied for each hidden layer across all parameters with NN predictors.
#' @param seed Seed for random initial weights and biases.
#' @param loc.link A string defining the link function used for the location parameter, see \eqn{h_1} below. If \code{link=="exp"}, then \eqn{h_1=\exp(x)}; if \code{link=="identity"}, then \eqn{h_1(x)=x}.
#' @param model Fitted \code{keras} model. Output from \code{train_bGEVPP_NN}.

#' @details{
#' Consider a real-valued random variable \eqn{Y} and let \eqn{\mathbf{X}} denote a \eqn{d}-dimensional predictor set with observations \eqn{\mathbf{x}}.
#' For \eqn{i=1,2}, we define integers \eqn{l_i\geq 0,a_i \geq 0} and \eqn{0\leq l_i+a_i \leq d}, and let \eqn{\mathbf{X}^{(i)}_L, \mathbf{X}^{(i)}_A} and \eqn{\mathbf{X}^{(i)}_N} be distinct sub-vectors
#' of \eqn{\mathbf{X}}, with observations of each component denoted \eqn{\mathbf{x}^{(i)}_L, \mathbf{x}^{(i)}_A} and \eqn{\mathbf{x}^{(i)}_N}, respectively; the lengths of the sub-vectors are \eqn{l_i,a_i} and \eqn{d_i-l_i-a}, respectively.
#' For a fixed threshold \eqn{u(\mathbf{x})}, dependent on predictors, we model \eqn{Y|\mathbf{X}=\mathbf{x}\sim\mbox{bGEV-PP}(q_\alpha(\mathbf{x}),s_\beta(\mathbf{x}),\xi>0;u(\mathbf{x}))} with
#' \deqn{q_\alpha (\mathbf{x})=h_1[\eta^{(1)}_0+m^{(1)}_L\{\mathbf{x}^{(1)}_L\}+m^{(1)}_A\{x^{(1)}_A\}+m^{(1)}_N\{\mathbf{x}^{(1)}_N\}]} and
#' \deqn{s_\beta (\mathbf{x})=\exp[\eta^{(2)}_0+m^{(2)}_L\{\mathbf{x}^{(2)}_L\}+m^{(2)}_A\{x^{(2)}_A\}+m^{(2)}_N\{\mathbf{x}^{(2)}_N\}]}
#' where \eqn{h_1} is some link-function and \eqn{\eta^{(1)}_0,\eta^{(2)}_0} are constant intercepts. The unknown functions \eqn{m^{(1)}_L,m^{(2)}_L} and \eqn{m^{(1)}_A,m^{(2)}_A} are estimated using a linear function and spline, respectively, and are
#' both returned as outputs; \eqn{m^{(1)}_N,m^{(2)}_N} are estimated using neural networks (currently the same architecture is used for both). Note that \eqn{\xi>0} is fixed across all predictors; may change in future versions.
#'
#' The model is fitted by minimising the negative log-likelihood associated with the bGEV-PP model over \code{n.ep} training epochs.
#' Although the model is trained by minimising the loss evaluated for \code{Y_train}, the final returned model may minimise some other loss.
#' The current state of the model is saved after each epoch, using \code{keras::callback_model_checkpoint}, if the value of some criterion subcedes that of the model from the previous checkpoint; this criterion is the loss evaluated for validation/test set \code{Y_test} if \code{!is.null(Y_test)} and for \code{Y_train}, otherwise.
#'
#'}
#' @return \code{train_bGEVPP_NN} returns the fitted \code{model}.  \code{predict_bGEVPP_nn} is a wrapper for \code{keras::predict} that returns the predicted parameter estimates, and, if applicable, their corresponding linear regression coefficients and spline bases weights.
#'
#'#'
#'@references Castro-Camilo, Huser and Rue (2021),
#' (\href{https://doi.org/10.48550/arXiv.2106.13110}{doi})
#'
#' @examples
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
#' m_A = 0.1*X_train_add[,,,1]^2+0.2*X_train_add[,,,1]-0.1*X_train_add[,,,2]^3+0.5*X_train_add[,,,2]^2
#'
#' #Non-additive contribution - to be estimated by NN
#' m_N = exp(-3+X_train_nn[,,,2]+X_train_nn[,,,3])
#' +sin(X_train_nn[,,,1]-X_train_nn[,,,2])*(X_train_nn[,,,1]+X_train_nn[,,,2])
#'
#'theta=1+m_L+m_A+m_N #Identity link
#' #We simulate normal data and estimate the median, i.e., the 50% quantile or mean,
#' #as the form for this is known
#' Y=apply(theta,1:3,function(x) rnorm(1,mean=x,sd=2))

#'
#' #Create training and test, respectively.
#' #We mask 20% of the Y values and use this for validation/testing.
#' #Masked values must be set to -1e5 and are treated as missing whilst training
#'
#' mask_inds=sample(1:length(Y),size=length(Y)*0.8)
#'
#' Y_train<-Y_test<-Y #Create training and test, respectively.
#' Y_train[-mask_inds]=-1e5
#' Y_test[mask_inds]=-1e5
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
#' for( i in 1:dim(X_train_add)[4]) knots[i,]=quantile(X_train_add[,,,i],probs=seq(0,1,length=n.knot))
#'
#' X_train_add_basis<-array(dim=c(dim(X_train_add),n.knot))
#' for( i in 1:dim(X_train_add)[4]) {
#' for(k in 1:n.knot) {
#' X_train_add_basis[,,,i,k]= rad(x=X_train_add[,,,i],c=knots[i,k])
#' #Evaluate rad at all entries to X_train_add and for all knots
#' }}
#'
#' X_train=list("X_train_nn"=X_train_nn, "X_train_lin"=X_train_lin,"X_train_add_basis"=X_train_add_basis)
#'
#'#Build and train a two-layered "lin+GAM+NN" MLP
#' tau <- 0.5 # tau must be set as a global variable to pass it to the keras custom loss function
#' model<-train_quant_NN(Y_train, Y_test,X_train,  type="MLP",link="identity",tau=0.5,n.ep=50,
#'                       batch.size=50, widths=c(6,3))
#'
#' out<-predict_quant_nn(X_train,model)
#'hist(out$predictions) #Plot histogram of predicted quantiles

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
#'  points(knots[i,],rep(mean(plt.y),n.knot),col="red",pch=2) #Adds red triangles that denote knot locations
#'}

#'
#'
#'#To save model, run model %>% save_model_tf(paste0("model_",tau,"-quantile"))
#'#To load model, run model  <- load_model_tf(paste0("model_",tau,"-quantile"),
#'#custom_objects=list("tilted_loss"=tilted_loss))
#'
#' @rdname train_bGEVPP_NN
#' @export

train_bGEVPP_NN=function(Y_train, Y_test = NULL,X_train_q,X_train_s, u_train = NULL, type="MLP",link.loc="identity",
                        n.ep=100, batch.size=100,init.loc=NULL, init.spread=NULL,init.xi=NULL,widths=c(6,3), filter.dim=c(3,3),seed=NULL,
                        alpha=0.5,beta=0.5,p_a=0.05,p_b=0.2)
{




  if(is.null(X_train_q)  ) stop("No predictors provided for q_\alpha")
  if(is.null(X_train_s)  ) stop("No predictors provided for s_\beta")
  if(is.null(Y_train)) stop("No training response data provided")
  if(is.null(u_train)) stop("No threshold u_train provided")

  if(is.null(init.loc)  ) stop("Inital location estimate not provided")
  if(is.null(init.spread)  ) stop("Inital spread estimate not provided")
  if(is.null(init.xi)  ) stop("Inital shape estimate not provided")

  X_train_nn_q=X_train_q$X_train_nn_q
  X_train_lin_q=X_train_q$X_train_lin_q
  X_train_add_basis_q=X_train_q$X_train_add_basis_q

  if(!is.null(X_train_nn_q) & !is.null(X_train_add_basis_q) & !is.null(X_train_lin_q) ) {  test.data= list(X_train_lin_q,X_train_add_basis_q,X_train_nn_q); print("Defining lin+GAM+NN model for q_\alpha" );  if(!is.null(Y_test)) validation.data=list(list(lin_input_q=X_train_lin_q,add_input_q=X_train_add_basis_q,  nn_input_q=X_train_nn_q),Y_test)}
  if(is.null(X_train_nn_q) & !is.null(X_train_add_basis_q) & !is.null(X_train_lin_q) ) {   test.data= list(X_train_lin_q,X_train_add_basis_q); print("Defining lin+GAM model for q_\alpha" );  if(!is.null(Y_test)) validation.data=list(list(lin_input_q=X_train_lin_q,add_input_q=X_train_add_basis_q),Y_test)}
  if(!is.null(X_train_nn_q) & is.null(X_train_add_basis_q) & !is.null(X_train_lin_q) ) { test.data= list(X_train_lin_q,X_train_nn_q); print("Defining lin+NN model for q_\alpha" );  if(!is.null(Y_test)) validation.data=list(list(lin_input_q=X_train_lin_q, nn_input_q=X_train_nn_q),Y_test)}
  if(!is.null(X_train_nn_q) & !is.null(X_train_add_basis_q) & is.null(X_train_lin_q) ) {test.data= list(X_train_add_basis_q,X_train_nn_q); print("Defining GAM+NN model for q_\alpha" );  if(!is.null(Y_test)) validation.data=list(list(add_input_q=X_train_add_basis_q,  nn_input_q=X_train_nn_q),Y_test)}
  if(is.null(X_train_nn_q) & is.null(X_train_add_basis_q) & !is.null(X_train_lin_q) )   {test.data= list(X_train_lin_q); print("Defining fully-linear model for q_\alpha" );  if(!is.null(Y_test)) validation.data=list(list(lin_input_q=X_train_nn_q),Y_test)}
  if(is.null(X_train_nn_q) & !is.null(X_train_add_basis_q) & is.null(X_train_lin_q) )   {test.data= list(X_train_add_basis_q); print("Defining fully-additive model for q_\alpha" );  if(!is.null(Y_test)) validation.data=list(list(add_input_q=X_train_add_basis_q),Y_test)}
  if(!is.null(X_train_nn_q) & is.null(X_train_add_basis_q) & is.null(X_train_lin_q) )   {test.data= list(X_train_nn_q); print("Defining fully-NN model for q_\alpha" );  if(!is.null(Y_test)) validation.data=list(list( nn_input_q=X_train_nn),Y_test)}

  X_train_nn_s=X_train_s$X_train_nn_s
  X_train_lin_s=X_train_s$X_train_lin_s
  X_train_add_basis_s=X_train_s$X_train_add_basis_s

  if(!is.null(X_train_nn_s) & !is.null(X_train_add_basis_s) & !is.null(X_train_lin_s) ) {  test.data= c(test.data,list(X_train_lin_s,X_train_add_basis_s,X_train_nn_s)); print("Defining lin+GAM+NN model for s_\beta" );  if(!is.null(Y_test)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X_train_lin_s,add_input_s=X_train_add_basis_s,  nn_input_s=X_train_nn_s)),Y_test)}
  if(is.null(X_train_nn_s) & !is.null(X_train_add_basis_s) & !is.null(X_train_lin_s) ) {   test.data= c(test.data,list(X_train_lin_s,X_train_add_basis_s)); print("Defining lin+GAM model for s_\beta" );  if(!is.null(Y_test)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X_train_lin_s,add_input_s=X_train_add_basis_s)),Y_test)}
  if(!is.null(X_train_nn_s) & is.null(X_train_add_basis_s) & !is.null(X_train_lin_s) ) { test.data= c(test.data,list(X_train_lin_s,X_train_nn_s)); print("Defining lin+NN model for s_\beta" );  if(!is.null(Y_test)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X_train_lin_s, nn_input_s=X_train_nn_s)),Y_test)}
  if(!is.null(X_train_nn_s) & !is.null(X_train_add_basis_s) & is.null(X_train_lin_s) ) {test.data= c(test.data,list(X_train_add_basis_s,X_train_nn_s)); print("Defining GAM+NN model for s_\beta" );  if(!is.null(Y_test)) validation.data=list(c(validation.data[[1]],list(add_input_s=X_train_add_basis_s,  nn_input_s=X_train_nn_s)),Y_test)}
  if(is.null(X_train_nn_s) & is.null(X_train_add_basis_s) & !is.null(X_train_lin_s) )   {test.data= c(test.data,list(X_train_lin_s)); print("Defining fully-linear model for s_\beta" );  if(!is.null(Y_test)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X_train_nn_s)),Y_test)}
  if(is.null(X_train_nn_s) & !is.null(X_train_add_basis_s) & is.null(X_train_lin_s) )   {test.data= c(test.data,list(X_train_add_basis_s)); print("Defining fully-additive model for s_\beta" );  if(!is.null(Y_test)) validation.data=list(c(validation.data[[1]],list(add_input_s=X_train_add_basis_s)),Y_test)}
  if(!is.null(X_train_nn_s) & is.null(X_train_add_basis_s) & is.null(X_train_lin_s) )   {test.data= c(test.data,list(X_train_nn_s)); print("Defining fully-NN model for s_\beta" );  if(!is.null(Y_test)) validation.data=list(c(validation.data[[1]],list(nn_input_s=X_train_nn)),Y_test)}


  if(type=="CNN" & !is.null(X_train_nn)) print(paste0("Building ",length(widths),"-layer convolutional neural network with ", filter.dim[1]," by ", filter.dim[2]," filter" ))
  if(type=="MLP"  & !is.null(X_train_nn) ) print(paste0("Building ",length(widths),"-layer densely-connected neural network" ))

  reticulate::use_virtualenv("myenv", required = T)

  if(!is.null(seed)) tf$random$set_seed(seed)

  model<-build_bGEVPP_nn(X_train_nn_q,X_train_lin_q,X_train_add_basis_q,
                         X_train_nn_s,X_train_lin_s,X_train_add_basis_s,
                         u_train,type, init.loc,init.spread,init.xi, widths,filter.dim,link.loc,alpha,beta,p_a,p_b)

  model %>% compile(
    optimizer="adam",
    loss = bgev_PP_loss(alpha,beta,p_a,p_b),
    run_eagerly=T
  )

  if(!is.null(Y_test)) checkpoint <- callback_model_checkpoint(paste0("model_bGEVPP_checkpoint"), monitor = "val_loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch") else checkpoint <- callback_model_checkpoint(paste0("model_bGEVPP_checkpoint"), monitor = "loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch")


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

  print("Loading checkpoint weights")
  model <- load_model_weights_tf(model,filepath=paste0("model_bGEVPP_checkpoint"))


  return(model)
}
#' @rdname train_bGEVPP_NN
#' @export
#'
predict_bGEVPP_nn=function(X_train, model)
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

  if(!is.null(X_train_add_basis))  gam.weights<-matrix(t(model$get_layer("add_q")$get_weights()[[1]]),nrow=dim(X_train_add_basis)[length(dim(X_train_add_basis))-1],ncol=dim(X_train_add_basis)[length(dim(X_train_add_basis))],byrow=T)

  if(!is.null(X_train_add_basis) & !is.null(X_train_lin)) return(list("predictions"=predictions, "lin.coeff"=c(model$get_layer("lin_q")$get_weights()[[1]]),"gam.weights"=gam.weights))
  if(is.null(X_train_add_basis) & !is.null(X_train_lin)) return(list("predictions"=predictions, "lin.coeff"=c(model$get_layer("lin_q")$get_weights()[[1]])))
  if(!is.null(X_train_add_basis) & is.null(X_train_lin)) return(list("predictions"=predictions,"gam.weights"=gam.weights))

}
#'
#'
build_bGEVPP_nn=function(X_train_nn_q,X_train_lin_q,X_train_add_basis_q,
                         X_train_nn_s,X_train_lin_s,X_train_add_basis_s,
                         u_train,
                         type, init.loc,init.spread,init.xi, widths,filter.dim,link.loc,alpha,beta,p_a,p_b)
{
  #Additive inputs
  if(!is.null(X_train_add_basis_q))  input_add_q<- layer_input(shape = dim(X_train_add_basis_q)[-1], name = 'add_input_q')
  if(!is.null(X_train_add_basis_s))  input_add_s<- layer_input(shape = dim(X_train_add_basis_s)[-1], name = 'add_input_s')

  #NN input

  if(!is.null(X_train_nn_q))   input_nn_q <- layer_input(shape = dim(X_train_nn_q)[-1], name = 'nn_input_q')
  if(!is.null(X_train_nn_s))   input_nn_s <- layer_input(shape = dim(X_train_nn_s)[-1], name = 'nn_input_s')

  #Linear input

  if(!is.null(X_train_lin_q)) input_lin_q <- layer_input(shape = dim(X_train_lin_q)[-1], name = 'lin_input_q')
  if(!is.null(X_train_lin_s)) input_lin_s <- layer_input(shape = dim(X_train_lin_s)[-1], name = 'lin_input_s')

  #Threshold input

  if(lin.loc=="exp") init.loc=log(init.loc) else if(link.loc =="identity") init.loc=init.loc else stop("Invalid link function for location parameter")
  #NN towers
  if(!is.null(X_train_nn_q)){

    nunits=c(widths,1)
    n.layers=length(nunits)-1

    nnBranchq <- input_nn_q
    if(type=="MLP"){
      for(i in 1:n.layers){
        nnBranchq <- nnBranchq  %>% layer_dense(units=nunits[i],activation = 'relu',
                                                input_shape =dim(X_train_nn_q)[-1], name = paste0('nn_q_dense',i) )
      }
    }else if(type=="CNN"){
      for(i in 1:n.layers){
        nnBranchq <- nnBranchq  %>% layer_conv_2d(filters=nunits[i],activation = 'relu',kernel_size=c(filter.dim[1],filter.dim[2]), padding='same',
                                                  input_shape =dim(X_train_nn_q)[-1], name = paste0('nn_q_cnn',i) )
      }

    }

    nnBranchq <-   nnBranchq  %>%   layer_dense(units = nunits[n.layers+1], activation = "linear", name = 'nn_q_dense_final',
                                                weights=list(matrix(0,nrow=nunits[n.layers],ncol=1), array(init.loc)))

  }

  #Additive tower
  n.dim.add=length(dim(X_train_add_basis))
  if(!is.null(X_train_add_basis) & !is.null(X_train_add_basis) ) {

    addBranchq <- input_add %>%
      layer_reshape(target_shape=c(dim(X_train_add_basis)[2:(n.dim.add-2)],prod(dim(X_train_add_basis)[(n.dim.add-1):n.dim.add]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_q',
                  weights=list(matrix(0,nrow=prod(dim(X_train_add_basis)[(n.dim.add-1):n.dim.add]),ncol=1)),use_bias = F)
  }
  if(!is.null(X_train_add_basis) & is.null(X_train_add_basis) ) {

    addBranchq <- input_add %>%
      layer_reshape(target_shape=c(dim(X_train_add_basis)[2:(n.dim.add-2)],prod(dim(X_train_add_basis)[(n.dim.add-1):n.dim.add]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_q',
                  weights=list(matrix(0,nrow=prod(dim(X_train_add_basis)[(n.dim.add-1):n.dim.add]),ncol=1),array(init.q)),use_bias = T)
  }

  #Linear tower


  if(!is.null(X_train_lin) ) {
    n.dim.lin=length(dim(X_train_lin))

    if(is.null(X_train_nn) & is.null(X_train_add_basis )){
      linBranchq <- input_lin%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X_train_lin)[-1], name = 'lin_q',
                    weights=list(matrix(0,nrow=dim(X_train_lin)[n.dim.lin],ncol=1),array(init.q)),use_bias=T)
    }else{
      linBranchq <- input_lin%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X_train_lin)[-1], name = 'lin_q',
                    weights=list(matrix(0,nrow=dim(X_train_lin)[n.dim.lin],ncol=1)),use_bias=F)
    }
  }



  if(!is.null(X_train_nn) & !is.null(X_train_add_basis) & !is.null(X_train_lin) )  qBranchjoined <- layer_add(inputs=c(addBranchq,  linBranchq,nnBranchq))  #Add all towers
  if(is.null(X_train_nn) & !is.null(X_train_add_basis) & !is.null(X_train_lin) )  qBranchjoined <- layer_add(inputs=c(addBranchq,  linBranchq))  #Add GAM+lin towers
  if(!is.null(X_train_nn) & is.null(X_train_add_basis) & !is.null(X_train_lin) )  qBranchjoined <- layer_add(inputs=c(  linBranchq,nnBranchq))  #Add NN+lin towers
  if(!is.null(X_train_nn) & !is.null(X_train_add_basis) & is.null(X_train_lin) )  qBranchjoined <- layer_add(inputs=c(addBranchq,  nnBranchq))  #Add NN+GAM towers
  if(is.null(X_train_nn) & is.null(X_train_add_basis) & !is.null(X_train_lin) )  qBranchjoined <- linBranchq  #Just lin tower
  if(is.null(X_train_nn) & !is.null(X_train_add_basis) & is.null(X_train_lin) )  qBranchjoined <- addBranchq  #Just GAM tower
  if(!is.null(X_train_nn) & is.null(X_train_add_basis) & is.null(X_train_lin) )  qBranchjoined <- nnBranchq  #Just NN tower


  #Use exponential activation so sig > 0
  if(link=="exp") qBranchjoined <- qBranchjoined %>% layer_activation( activation = 'exponential') else if(link=="linear") qBranchjoined <- qBranchjoined %>% layer_activation( activation = 'identity')


  if(!is.null(X_train_nn) & !is.null(X_train_add_basis) & !is.null(X_train_lin) ) model <- keras_model(  inputs = c(input_lin,input_add,input_nn),   outputs = c(qBranchjoined),name=paste0("quantile") )
  if(is.null(X_train_nn) & !is.null(X_train_add_basis) & !is.null(X_train_lin) )  model <- keras_model(  inputs = c(input_lin,input_add),   outputs = c(qBranchjoined),name=paste0("quantile") )
  if(!is.null(X_train_nn) & is.null(X_train_add_basis) & !is.null(X_train_lin) ) model <- keras_model(  inputs = c(input_lin,input_nn),    outputs = c(qBranchjoined),name=paste0("quantile") )
  if(!is.null(X_train_nn) & !is.null(X_train_add_basis) & is.null(X_train_lin) )  model <- keras_model(  inputs = c(input_add,input_nn),   outputs = c(qBranchjoined),name=paste0("quantile") )
  if(is.null(X_train_nn) & is.null(X_train_add_basis) & !is.null(X_train_lin) )  model <- keras_model(  inputs = c(input_lin),    outputs = c(qBranchjoined),name=paste0("quantile") )
  if(is.null(X_train_nn) & !is.null(X_train_add_basis) & is.null(X_train_lin) )  model <- keras_model(  inputs = c(input_add),    outputs = c(qBranchjoined),name=paste0("quantile") )
  if(!is.null(X_train_nn) & is.null(X_train_add_basis) & is.null(X_train_lin) )  model <- keras_model(  inputs = c(input_nn),   outputs = c(qBranchjoined),name=paste0("quantile") )

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


  beta_dist=tfd_beta(concentration1 = c1,concentration0 = c2)
  p= beta_dist %>% tfd_cdf((y-a)/(b-a)*obsInds)

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

  beta_dist=tfd_beta(concentration1 = c1,concentration0 = c2)
  p= beta_dist %>% tfd_cdf(((y-a)/(b-a))*exceedInds)

  temp=(y-a)/(b-a) #Need to set values <0 and >1 to 0 and 1, otherwise function breaks
  temp=K$relu(temp)
  temp=1-temp
  temp=K$relu(temp)
  temp=1-temp
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

bgev_PP_loss <-function(alpha,beta,p_a,p_b){

loss<- function( y_true, y_pred) {

  K <- backend()

  c1=5
  c2=5

  u=y_pred[all_dims(),1]
  q_a=y_pred[all_dims(),2]
  s_b=y_pred[all_dims(),3]
  xi=y_pred[all_dims(),4]




  # Find inds of non-missing obs.  Remove missing obs, i.e., -1e5. This is achieved by adding an
  # arbitrarily large (<1e5) value to y_true and then taking the sign ReLu
  obsInds=K$sign(K$relu(y_true+1e4))

  #Find exceedance inds
  exceed=y_true-u
  exceedInds=K$sign(K$relu(exceed))


  a=Finverse(p_a,q_a,s_b,xi,alpha,beta)
  b=Finverse(p_b,q_a,s_b,xi,alpha,beta)
  b =b + (1-obsInds)
  s_b=s_b+(1-obsInds)

  #Use exceedance only only
  lam=lambda(y_true,q_a,s_b,xi,alpha,beta,a,b,p_a,p_b,c1,c2,obsInds,exceedInds)
  loglam=K$log(lam+(1-exceedInds))*exceedInds



  #Use all values of y_true i.e., non-exceedances + exceedances.

  LAM=-logH(u,q_a,s_b,xi,alpha,beta,a,b,p_a,p_b,c1,c2,obsInds) #1/12 as 12 obs per year
  return(-(
    K$sum(loglam)
    -(1/12)*K$sum(LAM)
  ))
}
return(loss)
}
