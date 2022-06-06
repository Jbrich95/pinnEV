#'blended-GEV PINN
#'
#' Build and train a partially-interpretable neural network for fitting a bGEV model
#'

#' @param type  string defining the type of network to be built. If \code{type=="MLP"}, the network will have all densely connected layers; if \code{type=="CNN"},
#'  the network will have all convolutional layers. Defaults to an MLP. (Currently the same network is used for all parameters, may change in future versions)
#' @param Y_train,Y_valid a 2 or 3 dimensional array of training or validation real response values.
#' Missing values can be handled by setting corresponding entries to \code{Y_train} or \code{Y_valid} to \code{-1e5}.
#' The first dimension should be the observation indices, e.g., time.
#'
#' If \code{type=="CNN"}, then \code{Y_train} and \code{Y_valid} must have three dimensions with the latter two corresponding to an \eqn{M} by \eqn{N} regular grid of spatial locations.
#' If \code{Y_valid==NULL}, no validation loss will be computed and the returned model will be that which minimises the training loss over \code{n.ep} epochs.
#'
#' @param X_train_q  list of arrays corresponding to complementary subsets of the \eqn{d\geq 1} predictors which are used for modelling the location parameter \eqn{q_\alpha}. Must contain at least one of the following three named entries:\describe{
#' \item{\code{X_train_lin_q}}{A 3 or 4 dimensional array of "linear" predictor values. Same number of dimensions as \code{X_train_nn_1}. If \code{NULL}, a model without the linear component is built and trained.
#' The first 2/3 dimensions should be equal to that of \code{Y_train}; the last dimension corresponds to the chosen \eqn{l_1\geq 0} 'linear' predictor values.}
#' \item{\code{X_train_add_basis_q}}{A 4 or 5 dimensional array of basis function evaluations for the "additive" predictor values.
#' The first 2/3 dimensions should be equal to that of \code{Y_train}; the penultimate dimensions corresponds to the chosen \eqn{a_1\geq 0} 'linear' predictor values and the last dimension is equal to the number of knots used for estimating the splines. See example.
#' If \code{NULL}, a model without the additive component is built and trained.}
#' \item{\code{X_train_nn_q}}{A 3 or 4 dimensional array of "non-additive" predictor values.  If \code{NULL}, a model without the NN component is built and trained; if this is the case, then \code{type} has no effect.
#' The first 2/3 dimensions should be equal to that of \code{Y_train}; the last dimension corresponds to the chosen \eqn{d-l_1-a_1\geq 0} 'non-additive' predictor values.}
#' }
#' Note that \code{X_train_q} and \code{X_train_s} are the predictors for both \code{Y_train} and \code{Y_valid}.
#' @param X_train_S similarly to \code{X_train_s}, but for modelling the scale parameter \eqn{s_\beta>0}. Note that both \eqn{q_\beta} and \eqn{s_\beta} must be modelled as non-stationary in this version.
#' @param n.ep number of epochs used for training. Defaults to 1000.
#' @param alpha,beta,p_a,p_b hyper-parameters associated with the bGEV distribution. Defaults to those used by Castro-Camilo, D., et al. (2021). Require \code{alpha >= p_b} and \code{beta/2 >= p_b}.
#' @param batch.size batch size for stochastic gradient descent. If larger than \code{dim(Y_train)[1]}, i.e., the number of observations, then regular gradient descent used.
#' @param init.loc,init.spread,init.xi sets the initial \eqn{q_\alpha,s_\beta} and \eqn{\xi\in(0,1)} estimates across all dimensions of \code{Y_train}. Overridden by \code{init.wb_path} if \code{!is.null(init.wb_path)}, but otherwise the initial parameters must be supplied.
#' @param init.wb_path filepath to a \code{keras} model which is then used as initial weights and biases for training the new model. The original model must have
#' the exact same architecture and trained with the same input data as the new model. If \code{NULL}, then initial weights and biases are random (with seed \code{seed}) but the
#' final layer has zero initial weights to ensure that the initial location, spread and shape estimates are \code{init.loc, init.spread} and \code{init.xi}, respectively,  across all dimensions.
#' @param widths vector of widths/filters for hidden dense/convolution layers. Number of layers is equal to \code{length(widths)}. Defaults to (6,3).
#' @param filter.dim if \code{type=="CNN"}, this 2-vector gives the dimensions of the convolution filter kernel; must have odd integer inputs. Note that filter.dim=c(1,1) is equivalent to \code{type=="MLP"}. The same filter is applied for each hidden layer across all parameters with NN predictors.
#' @param seed seed for random initial weights and biases.
#' @param loc.link string defining the link function used for the location parameter, see \eqn{h_1} below. If \code{link=="exp"}, then \eqn{h_1=\exp(x)}; if \code{link=="identity"}, then \eqn{h_1(x)=x}.
#' @param model fitted \code{keras} model. Output from \code{bGEVPP.NN.train}.

#'@name bGEV.NN

#' @details{
#' Consider a real-valued random variable \eqn{Y} and let \eqn{\mathbf{X}} denote a \eqn{d}-dimensional predictor set with observations \eqn{\mathbf{x}}.
#' For \eqn{i=1,2}, we define integers \eqn{l_i\geq 0,a_i \geq 0} and \eqn{0\leq l_i+a_i \leq d}, and let \eqn{\mathbf{X}^{(i)}_L, \mathbf{X}^{(i)}_A} and \eqn{\mathbf{X}^{(i)}_N} be distinct sub-vectors
#' of \eqn{\mathbf{X}}, with observations of each component denoted \eqn{\mathbf{x}^{(i)}_L, \mathbf{x}^{(i)}_A} and \eqn{\mathbf{x}^{(i)}_N}, respectively; the lengths of the sub-vectors are \eqn{l_i,a_i} and \eqn{d_i-l_i-a}, respectively.
#' For a fixed threshold \eqn{u(\mathbf{x})}, dependent on predictors, we model \eqn{Y|\mathbf{X}=\mathbf{x}\sim\mbox{bGEV}(q_\alpha(\mathbf{x}),s_\beta(\mathbf{x}),\xi)} for \eqn{\xi\in(0,1)} with
#' \deqn{q_\alpha (\mathbf{x})=h_1[\eta^{(1)}_0+m^{(1)}_L\{\mathbf{x}^{(1)}_L\}+m^{(1)}_A\{x^{(1)}_A\}+m^{(1)}_N\{\mathbf{x}^{(1)}_N\}]} and
#' \deqn{s_\beta (\mathbf{x})=\exp[\eta^{(2)}_0+m^{(2)}_L\{\mathbf{x}^{(2)}_L\}+m^{(2)}_A\{x^{(2)}_A\}+m^{(2)}_N\{\mathbf{x}^{(2)}_N\}]}
#' where \eqn{h_1} is some link-function and \eqn{\eta^{(1)}_0,\eta^{(2)}_0} are constant intercepts. The unknown functions \eqn{m^{(1)}_L,m^{(2)}_L} and
#' \eqn{m^{(1)}_A,m^{(2)}_A} are estimated using linear functions and splines, respectively, and are
#' both returned as outputs by \code{bGEV.NN.predict}; \eqn{m^{(1)}_N,m^{(2)}_N} are estimated using neural networks
#' (currently the same architecture is used for both parameters). Note that \eqn{\xi>0} is fixed across all predictors; this may change in future versions.
#'
#' For details of the bGEV distribution, see \code{help(pbGEV)}. 
#'
#' The model is fitted by minimising the negative log-likelihood associated with the bGEV model over \code{n.ep} training epochs.
#' Although the model is trained by minimising the loss evaluated for \code{Y_train}, the final returned model may minimise some other loss.
#' The current state of the model is saved after each epoch, using \code{keras::callback_model_checkpoint}, if the value of some criterion subcedes that of the model from the previous checkpoint; this criterion is the loss evaluated for validation set \code{Y_valid} if \code{!is.null(Y_valid)} and for \code{Y_train}, otherwise.
#'
#'}
#' @return \code{bGEV.NN.train} returns the fitted \code{model}.  \code{bGEV.NN.predict} is a wrapper for \code{keras::predict} that returns the predicted parameter estimates, and, if applicable, their corresponding linear regression coefficients and spline bases weights.
#'
#'@references
#' Castro-Camilo, D., Huser, R., and Rue, H. (2021), \emph{Practical strategies for GEV-based regression models for extremes}, arXiv.
#' (\href{https://doi.org/10.48550/arXiv.2106.13110}{doi})
#'
#' @examples
#'
#' # Build and train a simple MLP for toy data
#'
#' # Create  predictors
#' preds<-rnorm(128000)
#'
#' #Re-shape to a 4d array. First dimension corresponds to observations,
#' #last to the different components of the predictor set
#' dim(preds)=c(200,8,8,10) #We have ten predictors
#'
#' #Split predictors into linear, additive and nn. Different for the location and scale parameters.
#' X_train_nn_q=preds[,,,1:5] #Five nn predictors for q_\alpha
#' X_train_lin_q=preds[,,,6:7] #Two linear predictors for q_\alpha
#' X_train_add_q=preds[,,,8:10] #Three additive predictors for q_\alpha
#'
#' X_train_nn_s=preds[,,,1:3] #Three nn predictors for s_\beta
#' X_train_lin_s=preds[,,,4:8] #Five linear predictors for s_\beta
#' X_train_add_s=preds[,,,9:10] #Two additive predictors for s_\beta
#'
#'
#' # Create toy response data
#'
#' #Contribution to location parameter
#' #Linear contribution
#' m_L_1 = 0.3*X_train_lin_q[,,,1]+0.6*X_train_lin_q[,,,2]
#'
#' # Additive contribution
#' m_A_1 = 0.1*X_train_add_q[,,,1]^3+0.2*X_train_add_q[,,,1]-
#'   0.1*X_train_add_q[,,,2]^3+0.5*X_train_add_q[,,,2]^2-0.2*X_train_add_q[,,,3]^3
#'
#' #Non-additive contribution - to be estimated by NN
#' m_N_1 = exp(-3+X_train_nn_q[,,,2]+X_train_nn_q[,,,3])+
#'   sin(X_train_nn_q[,,,1]-X_train_nn_q[,,,2])*(X_train_nn_q[,,,1]+X_train_nn_q[,,,2])-
#'   cos(X_train_nn_q[,,,3]-X_train_nn_q[,,,4])*(X_train_nn_q[,,,2]+X_train_nn_q[,,,5])
#'
#' q_alpha=1+m_L_1+m_A_1+m_N_1 #Identity link
#'
#' #Contribution to scale parameter
#' #Linear contribution
#' m_L_2 = 0.2*X_train_lin_s[,,,1]+0.6*X_train_lin_s[,,,2]+0.1*X_train_lin_s[,,,3]-
#'   0.2*X_train_lin_s[,,,4]+0.5*X_train_lin_s[,,,5]
#'
#' # Additive contribution
#' m_A_2 = 0.1*X_train_add_s[,,,1]^2+0.2*X_train_add_s[,,,1]-0.2*X_train_add_s[,,,2]^2+
#'   0.1*X_train_add_s[,,,2]^3
#'
#' #Non-additive contribution - to be estimated by NN
#'m_N_2 = 0.25*exp(-3+X_train_nn_s[,,,2]+X_train_nn_s[,,,3])+
#' sin(X_train_nn_s[,,,1]-X_train_nn_s[,,,2])*(X_train_nn_s[,,,1]+X_train_nn_s[,,,2])
#'
#' s_beta=0.3*exp(-2+m_L_2+m_A_2+m_N_2) #Exponential link
#'
#' xi=0.1 # Set xi
#'
#' theta=array(dim=c(dim(s_beta),3))
#' theta[,,,1]=q_alpha; theta[,,,2] = s_beta; theta[,,,3]=xi
#' #We simulate data from the bGEV distribution
#'
#' Y=apply(theta,1:3,function(x) rbGEV(1,q_alpha=x[1],s_beta=x[2],xi=x[3]))
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
#' #the basis functions for each pre-specified knot and entry to X_train_add_q and X_train_add_s
#'
#'rad=function(x,c){ #Define a basis function. Here we use the radial bases
#'   out=abs(x-c)^2*log(abs(x-c))
#'   out[(x-c)==0]=0
#'  return(out)
#' }
#'
#'n.knot.q = 5; n.knot.s = 4 # set number of knots.
#'#Must be the same for each additive predictor,
#'#but can differ between the parameters q_\alpha and s_\beta
#'
#' #Get knots for q_\alpha predictors
#' knots.q=matrix(nrow=dim(X_train_add_q)[4],ncol=n.knot.q)
#'
#' #We set knots to be equally-spaced marginal quantiles
#' for( i in 1:dim(X_train_add_q)[4]) knots.q[i,]=quantile(X_train_add_q[,,,i],probs=seq(0,1,length=n.knot.q))
#'
#' #Evaluate radial basis functions for q_\alpha predictors
#' X_train_add_basis_q<-array(dim=c(dim(X_train_add_q),n.knot.q))
#' for( i in 1:dim(X_train_add_q)[4]) {
#'   for(k in 1:n.knot.q) {
#'     X_train_add_basis_q[,,,i,k]= rad(x=X_train_add_q[,,,i],c=knots.q[i,k])
#'     #Evaluate rad at all entries to X_train_add_q and for all knots
#'   }}
#'
#' #Get knots for s_\beta predictors
#' knots.s=matrix(nrow=dim(X_train_add_s)[4],ncol=n.knot.s)
#' for( i in 1:dim(X_train_add_s)[4]) knots.s[i,]=quantile(X_train_add_s[,,,i],probs=seq(0,1,length=n.knot.s))
#'
#' #Evaluate radial basis functions for s_\beta predictors
#' X_train_add_basis_s<-array(dim=c(dim(X_train_add_s),n.knot.s))
#' for( i in 1:dim(X_train_add_s)[4]) {
#'   for(k in 1:n.knot.s) {
#'     X_train_add_basis_s[,,,i,k]= rad(x=X_train_add_s[,,,i],c=knots.s[i,k])
#'     #Evaluate rad at all entries to X_train_add_q and for all knots
#'   }}
#'
#' #lin+GAM+NN models defined for both location and scale parameters
#' X_train_q=list("X_train_nn_q"=X_train_nn_q, "X_train_lin_q"=X_train_lin_q,
#'                "X_train_add_basis_q"=X_train_add_basis_q) #Predictors for q_\alpha
#' X_train_s=list("X_train_nn_s"=X_train_nn_s, "X_train_lin_s"=X_train_lin_s,
#'                "X_train_add_basis_s"=X_train_add_basis_s) #Predictors for s_\beta
#'
#'
#' #Fit the bGEV model
#' model<-bGEV.NN.train(Y_train, Y_valid,X_train_q,X_train_s, type="MLP",link.loc="identity",
#'                        n.ep=500, batch.size=50,init.loc=2, init.spread=2,init.xi=0.1,
#'                        widths=c(6,3),seed=1)
#' out<-bGEV.NN.predict(X_train_q=X_train_q,X_train_s=X_train_s,model)
#'
#' print("q_alpha linear coefficients: "); print(round(out$lin.coeff_q,2))
#' print("s_beta linear coefficients: "); print(round(out$lin.coeff_s,2))
#'
#' #To save model, run
#' #model %>% save_model_tf("model_bGEV")
#' #To load model, run
#' # model  <- load_model_tf("model_bGEV",
#' #custom_objects=list("bgev_loss_alpha__beta__p_a__p_b__c1__c2_"=bgev_loss))
#'
#' #Note that bGEV_loss() can take custom alpha,beta, p_a and p_b arguments if defaults not used
#'
#'
#' # Plot splines for the additive predictors
#'
#' #Location predictors
#' n.add.preds_q=dim(X_train_add_q)[length(dim(X_train_add_q))]
#' par(mfrow=c(1,n.add.preds_q))
#' for(i in 1:n.add.preds_q){
#'   plt.x=seq(from=min(knots.q[i,]),to=max(knots.q[i,]),length=1000)  #Create sequence for x-axis
#'
#'   tmp=matrix(nrow=length(plt.x),ncol=n.knot.q)
#'   for(j in 1:n.knot.q){
#'     tmp[,j]=rad(plt.x,knots.q[i,j]) #Evaluate radial basis function of plt.x and all knots
#'   }
#'   plt.y=tmp%*%out$gam.weights_q[i,]
#'   plot(plt.x,plt.y,type="l",main=paste0("q_alpha spline: predictor ",i),xlab="x",ylab="f(x)")
#'   points(knots.q[i,],rep(mean(plt.y),n.knot.q),col="red",pch=2)
#'   #Adds red triangles that denote knot locations
#'
#' }
#'
#' #Spread predictors
#' n.add.preds_s=dim(X_train_add_s)[length(dim(X_train_add_s))]
#' par(mfrow=c(1,n.add.preds_s))
#' for(i in 1:n.add.preds_s){
#'   plt.x=seq(from=min(knots.s[i,]),to=max(knots.s[i,]),length=1000)  #Create sequence for x-axis
#'
#'   tmp=matrix(nrow=length(plt.x),ncol=n.knot.s)
#'   for(j in 1:n.knot.s){
#'     tmp[,j]=rad(plt.x,knots.s[i,j]) #Evaluate radial basis function of plt.x and all knots
#'   }
#'   plt.y=tmp%*%out$gam.weights_s[i,]
#'   plot(plt.x,plt.y,type="l",main=paste0("s_beta spline: predictor ",i),xlab="x",ylab="f(x)")
#'   points(knots.s[i,],rep(mean(plt.y),n.knot.s),col="red",pch=2)
#'   #Adds red triangles that denote knot locations
#'
#' }
#'
#' @rdname bGEV.NN
#' @export

bGEV.NN.train=function(Y_train, Y_valid = NULL,X_train_q,X_train_s, type="MLP",link.loc="identity",
                         n.ep=100, batch.size=100,init.loc=NULL, init.spread=NULL,init.xi=NULL,
                         widths=c(6,3), filter.dim=c(3,3),seed=NULL,init.wb_path=NULL,
                         alpha=0.5,beta=0.5,p_a=0.05,p_b=0.2,c1=5,c2=5)
{
  
  
  
  
  if(is.null(X_train_q)  ) stop("No predictors provided for q_\alpha")
  if(is.null(X_train_s)  ) stop("No predictors provided for s_\beta")
  if(is.null(Y_train)) stop("No training response data provided")

  if(is.null(init.loc) & is.null(init.wb_path)  ) stop("Inital location estimate not provided")
  if(is.null(init.spread) & is.null(init.wb_path)   ) stop("Inital spread estimate not provided")
  if(is.null(init.xi)  & is.null(init.wb_path) ) stop("Inital shape estimate not provided")
  
  
  print(paste0("Creating bGEV model"))
  X_train_nn_q=X_train_q$X_train_nn_q
  X_train_lin_q=X_train_q$X_train_lin_q
  X_train_add_basis_q=X_train_q$X_train_add_basis_q
  
  
  if(!is.null(X_train_nn_q) & !is.null(X_train_add_basis_q) & !is.null(X_train_lin_q) ) {  train.data= list(X_train_lin_q,X_train_add_basis_q,X_train_nn_q); print("Defining lin+GAM+NN model for q_\alpha" );  if(!is.null(Y_valid)) validation.data=list(list(lin_input_q=X_train_lin_q,add_input_q=X_train_add_basis_q,  nn_input_q=X_train_nn_q),Y_valid)}
  if(is.null(X_train_nn_q) & !is.null(X_train_add_basis_q) & !is.null(X_train_lin_q) ) {   train.data= list(X_train_lin_q,X_train_add_basis_q); print("Defining lin+GAM model for q_\alpha" );  if(!is.null(Y_valid)) validation.data=list(list(lin_input_q=X_train_lin_q,add_input_q=X_train_add_basis_q),Y_valid)}
  if(!is.null(X_train_nn_q) & is.null(X_train_add_basis_q) & !is.null(X_train_lin_q) ) { train.data= list(X_train_lin_q,X_train_nn_q); print("Defining lin+NN model for q_\alpha" );  if(!is.null(Y_valid)) validation.data=list(list(lin_input_q=X_train_lin_q, nn_input_q=X_train_nn_q),Y_valid)}
  if(!is.null(X_train_nn_q) & !is.null(X_train_add_basis_q) & is.null(X_train_lin_q) ) {train.data= list(X_train_add_basis_q,X_train_nn_q); print("Defining GAM+NN model for q_\alpha" );  if(!is.null(Y_valid)) validation.data=list(list(add_input_q=X_train_add_basis_q,  nn_input_q=X_train_nn_q),Y_valid)}
  if(is.null(X_train_nn_q) & is.null(X_train_add_basis_q) & !is.null(X_train_lin_q) )   {train.data= list(X_train_lin_q); print("Defining fully-linear model for q_\alpha" );  if(!is.null(Y_valid)) validation.data=list(list(lin_input_q=X_train_lin_q),Y_valid)}
  if(is.null(X_train_nn_q) & !is.null(X_train_add_basis_q) & is.null(X_train_lin_q) )   {train.data= list(X_train_add_basis_q); print("Defining fully-additive model for q_\alpha" );  if(!is.null(Y_valid)) validation.data=list(list(add_input_q=X_train_add_basis_q),Y_valid)}
  if(!is.null(X_train_nn_q) & is.null(X_train_add_basis_q) & is.null(X_train_lin_q) )   {train.data= list(X_train_nn_q); print("Defining fully-NN model for q_\alpha" );  if(!is.null(Y_valid)) validation.data=list(list( nn_input_q=X_train_nn_q),Y_valid)}
  
  X_train_nn_s=X_train_s$X_train_nn_s
  X_train_lin_s=X_train_s$X_train_lin_s
  X_train_add_basis_s=X_train_s$X_train_add_basis_s
  
  if(!is.null(X_train_nn_s) & !is.null(X_train_add_basis_s) & !is.null(X_train_lin_s) ) {  train.data= c(train.data,list(X_train_lin_s,X_train_add_basis_s,X_train_nn_s)); print("Defining lin+GAM+NN model for s_\beta" );  if(!is.null(Y_valid)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X_train_lin_s,add_input_s=X_train_add_basis_s,  nn_input_s=X_train_nn_s)),Y_valid)}
  if(is.null(X_train_nn_s) & !is.null(X_train_add_basis_s) & !is.null(X_train_lin_s) ) {   train.data= c(train.data,list(X_train_lin_s,X_train_add_basis_s)); print("Defining lin+GAM model for s_\beta" );  if(!is.null(Y_valid)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X_train_lin_s,add_input_s=X_train_add_basis_s)),Y_valid)}
  if(!is.null(X_train_nn_s) & is.null(X_train_add_basis_s) & !is.null(X_train_lin_s) ) { train.data= c(train.data,list(X_train_lin_s,X_train_nn_s)); print("Defining lin+NN model for s_\beta" );  if(!is.null(Y_valid)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X_train_lin_s, nn_input_s=X_train_nn_s)),Y_valid)}
  if(!is.null(X_train_nn_s) & !is.null(X_train_add_basis_s) & is.null(X_train_lin_s) ) {train.data= c(train.data,list(X_train_add_basis_s,X_train_nn_s)); print("Defining GAM+NN model for s_\beta" );  if(!is.null(Y_valid)) validation.data=list(c(validation.data[[1]],list(add_input_s=X_train_add_basis_s,  nn_input_s=X_train_nn_s)),Y_valid)}
  if(is.null(X_train_nn_s) & is.null(X_train_add_basis_s) & !is.null(X_train_lin_s) )   {train.data= c(train.data,list(X_train_lin_s)); print("Defining fully-linear model for s_\beta" );  if(!is.null(Y_valid)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X_train_lin_s)),Y_valid)}
  if(is.null(X_train_nn_s) & !is.null(X_train_add_basis_s) & is.null(X_train_lin_s) )   {train.data= c(train.data,list(X_train_add_basis_s)); print("Defining fully-additive model for s_\beta" );  if(!is.null(Y_valid)) validation.data=list(c(validation.data[[1]],list(add_input_s=X_train_add_basis_s)),Y_valid)}
  if(!is.null(X_train_nn_s) & is.null(X_train_add_basis_s) & is.null(X_train_lin_s) )   {train.data= c(train.data,list(X_train_nn_s)); print("Defining fully-NN model for s_\beta" );  if(!is.null(Y_valid)) validation.data=list(c(validation.data[[1]],list(nn_input_s=X_train_nn_s)),Y_valid)}
  
  
  if(type=="CNN" & (!is.null(X_train_nn_q) | !is.null(X_train_nn_s)))print(paste0("Building ",length(widths),"-layer convolutional neural network with ", filter.dim[1]," by ", filter.dim[2]," filter" ))
  if(type=="MLP"  & (!is.null(X_train_nn_q) | !is.null(X_train_nn_s)) ) print(paste0("Building ",length(widths),"-layer densely-connected neural network" ))
  
  reticulate::use_virtualenv("myenv", required = T)
  
  if(!is.null(seed)) tf$random$set_seed(seed)
  
  model<-bGEV.NN.build(X_train_nn_q,X_train_lin_q,X_train_add_basis_q,
                         X_train_nn_s,X_train_lin_s,X_train_add_basis_s,
                        type, init.loc,init.spread,init.xi, widths,filter.dim,link.loc,alpha,beta,p_a,p_b, c1, c2)
  if(!is.null(init.wb_path)) model <- load_model_weights_tf(model,filepath=init.wb_path)
  
  model %>% compile(
    optimizer="adam",
    loss = bgev_loss(alpha,beta,p_a,p_b,c1,c2),
    run_eagerly=T
  )
  
  if(!is.null(Y_valid)) checkpoint <- callback_model_checkpoint(paste0("model_bGEV_checkpoint"), monitor = "val_loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch") else checkpoint <- callback_model_checkpoint(paste0("model_bGEV_checkpoint"), monitor = "loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch")
  
  
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
  model <- load_model_weights_tf(model,filepath=paste0("model_bGEV_checkpoint"))
  
  
  return(model)
}
#' @rdname bGEV.NN
#' @export
#'
bGEV.NN.predict=function(X_train_q,X_train_s, model)
{
  library(tensorflow)
  if(is.null(X_train_q)  ) stop("No predictors provided for q_\alpha")
  if(is.null(X_train_s)  ) stop("No predictors provided for s_\beta")
  
  
  
  X_train_nn_q=X_train_q$X_train_nn_q
  X_train_lin_q=X_train_q$X_train_lin_q
  X_train_add_basis_q=X_train_q$X_train_add_basis_q
  
  
  if(!is.null(X_train_nn_q) & !is.null(X_train_add_basis_q) & !is.null(X_train_lin_q) )   train.data= list(X_train_lin_q,X_train_add_basis_q,X_train_nn_q)
  if(is.null(X_train_nn_q) & !is.null(X_train_add_basis_q) & !is.null(X_train_lin_q) )   train.data= list(X_train_lin_q,X_train_add_basis_q)
  if(!is.null(X_train_nn_q) & is.null(X_train_add_basis_q) & !is.null(X_train_lin_q) )  train.data= list(X_train_lin_q,X_train_nn_q)
  if(!is.null(X_train_nn_q) & !is.null(X_train_add_basis_q) & is.null(X_train_lin_q) ) train.data= list(X_train_add_basis_q,X_train_nn_q)
  if(is.null(X_train_nn_q) & is.null(X_train_add_basis_q) & !is.null(X_train_lin_q) )   train.data= list(X_train_lin_q)
  if(is.null(X_train_nn_q) & !is.null(X_train_add_basis_q) & is.null(X_train_lin_q) )   train.data= list(X_train_add_basis_q)
  if(!is.null(X_train_nn_q) & is.null(X_train_add_basis_q) & is.null(X_train_lin_q) )   train.data= list(X_train_nn_q)
  
  X_train_nn_s=X_train_s$X_train_nn_s
  X_train_lin_s=X_train_s$X_train_lin_s
  X_train_add_basis_s=X_train_s$X_train_add_basis_s
  
  if(!is.null(X_train_nn_s) & !is.null(X_train_add_basis_s) & !is.null(X_train_lin_s) )   train.data= c(train.data,list(X_train_lin_s,X_train_add_basis_s,X_train_nn_s))
  if(is.null(X_train_nn_s) & !is.null(X_train_add_basis_s) & !is.null(X_train_lin_s) )   train.data= c(train.data,list(X_train_lin_s,X_train_add_basis_s))
  if(!is.null(X_train_nn_s) & is.null(X_train_add_basis_s) & !is.null(X_train_lin_s) )  train.data= c(train.data,list(X_train_lin_s,X_train_nn_s))
  if(!is.null(X_train_nn_s) & !is.null(X_train_add_basis_s) & is.null(X_train_lin_s) ) train.data= c(train.data,list(X_train_add_basis_s,X_train_nn_s))
  if(is.null(X_train_nn_s) & is.null(X_train_add_basis_s) & !is.null(X_train_lin_s) )   train.data= c(train.data,list(X_train_lin_s))
  if(is.null(X_train_nn_s) & !is.null(X_train_add_basis_s) & is.null(X_train_lin_s) )   train.data= c(train.data,list(X_train_add_basis_s))
  if(!is.null(X_train_nn_s) & is.null(X_train_add_basis_s) & is.null(X_train_lin_s) ) train.data= c(train.data,list(X_train_nn_s))
  
  
  predictions<-model %>% predict( train.data)
  predictions <- k_constant(predictions)
  pred.loc=k_get_value(predictions[all_dims(),1])
  pred.spread=k_get_value(predictions[all_dims(),2])
  pred.xi=k_get_value(predictions[all_dims(),3])
  
  if(!is.null(X_train_add_basis_q))  gam.weights_q<-matrix(t(model$get_layer("add_q")$get_weights()[[1]]),nrow=dim(X_train_add_basis_q)[length(dim(X_train_add_basis_q))-1],ncol=dim(X_train_add_basis_q)[length(dim(X_train_add_basis_q))],byrow=T)
  if(!is.null(X_train_add_basis_s))  gam.weights_s<-matrix(t(model$get_layer("add_s")$get_weights()[[1]]),nrow=dim(X_train_add_basis_s)[length(dim(X_train_add_basis_s))-1],ncol=dim(X_train_add_basis_s)[length(dim(X_train_add_basis_s))],byrow=T)
  
  out=list("pred.loc"=pred.loc,"pred.spread"=pred.spread,"pred.xi"=pred.xi)
  if(!is.null(X_train_lin_q) ) out=c(out,list("lin.coeff_q"=c(model$get_layer("lin_q")$get_weights()[[1]])))
  if(!is.null(X_train_lin_s) ) out=c(out,list("lin.coeff_s"=c(model$get_layer("lin_s")$get_weights()[[1]])))
  if(!is.null(X_train_add_basis_q) ) out=c(out,list("gam.weights_q"=gam.weights_q))
  if(!is.null(X_train_add_basis_s) ) out=c(out,list("gam.weights_s"=gam.weights_s))
  
  return(out)
  
}
#'
#'
bGEV.NN.build=function(X_train_nn_q,X_train_lin_q,X_train_add_basis_q,
                         X_train_nn_s,X_train_lin_s,X_train_add_basis_s,
                         type, init.loc,init.spread,init.xi, widths,filter.dim,link.loc,alpha,beta,p_a,p_b,c1,c2)
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
  
  #Create xi branch
  
  if(!is.null(X_train_nn_q)){
    xiBranch <- input_nn_q %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X_train_nn_q)[-1], trainable=F,
                                        weights=list(matrix(0,nrow=dim(X_train_nn_q)[length(dim(X_train_nn_q))],ncol=1),array(1,dim=c(1))), name = 'xi_dense') %>%
      layer_dense(units = 1 ,activation = 'sigmoid',use_bias = F,weights=list(matrix(qlogis(init.xi),nrow=1,ncol=1)), name = 'xi_activation')
  }else  if(!is.null(X_train_nn_s)){
    xiBranch <- input_nn_s %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X_train_nn_s)[-1], trainable=F,
                                        weights=list(matrix(0,nrow=dim(X_train_nn_s)[length(dim(X_train_nn_s))],ncol=1),array(1,dim=c(1))), name = 'xi_dense') %>%
      layer_dense(units = 1 ,activation = 'sigmoid',use_bias = F,weights=list(matrix(qlogis(init.xi),nrow=1,ncol=1)), name = 'xi_activation')
  }else  if(!is.null(X_train_lin_q)){
    xiBranch <- input_lin_q %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X_train_lin_q)[-1], trainable=F,
                                        weights=list(matrix(0,nrow=dim(X_train_lin_q)[length(dim(X_train_lin_q))],ncol=1),array(1,dim=c(1))), name = 'xi_dense') %>%
      layer_dense(units = 1 ,activation = 'sigmoid',use_bias = F,weights=list(matrix(qlogis(init.xi),nrow=1,ncol=1)), name = 'xi_activation')
  }else  if(!is.null(X_train_lin_s)){
    xiBranch <- input_lin_s %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X_train_lin_s)[-1], trainable=F,
                                        weights=list(matrix(0,nrow=dim(X_train_lin_s)[length(dim(X_train_lin_s))],ncol=1),array(1,dim=c(1))), name = 'xi_dense') %>%
      layer_dense(units = 1 ,activation = 'sigmoid',use_bias = F,weights=list(matrix(qlogis(init.xi),nrow=1,ncol=1)), name = 'xi_activation')
  }else if(!is.null(X_train_add_basis_q)){
    xiBranch <- input_add_q %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X_train_add_basis_q)[-1], trainable=F,
                                            weights=list(matrix(0,nrow=dim(X_train_add_basis_q)[length(dim(X_train_add_basis_q))],ncol=1),array(1,dim=c(1))), name = 'xi_dense') %>%
      layer_dense(units = 1 ,activation = 'sigmoid',use_bias = F,weights=list(matrix(qlogis(init.xi),nrow=1,ncol=1)), name = 'xi_activation')
  }else  if(!is.null(X_train_add_basis_s)){
    xiBranch <- input_add_s %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(X_train_add_basis_s)[-1], trainable=F,
                                            weights=list(matrix(0,nrow=dim(X_train_add_basis_s)[length(dim(X_train_add_basis_s))],ncol=1),array(1,dim=c(1))), name = 'xi_dense') %>%
      layer_dense(units = 1 ,activation = 'sigmoid',use_bias = F,weights=list(matrix(qlogis(init.xi),nrow=1,ncol=1)), name = 'xi_activation')
  }
  
  
  
  if(link.loc=="exp") init.loc=log(init.loc) else if(link.loc =="identity") init.loc=init.loc else stop("Invalid link function for location parameter")
  #NN towers
  
  #Location
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
  #Spread
  if(!is.null(X_train_nn_s)){
    
    nunits=c(widths,1)
    n.layers=length(nunits)-1
    
    nnBranchs <- input_nn_s
    if(type=="MLP"){
      for(i in 1:n.layers){
        nnBranchs <- nnBranchs  %>% layer_dense(units=nunits[i],activation = 'relu',
                                                input_shape =dim(X_train_nn_s)[-1], name = paste0('nn_s_dense',i) )
      }
    }else if(type=="CNN"){
      for(i in 1:n.layers){
        nnBranchs <- nnBranchs  %>% layer_conv_2d(filters=nunits[i],activation = 'relu',kernel_size=c(filter.dim[1],filter.dim[2]), padding='same',
                                                  input_shape =dim(X_train_nn_s)[-1], name = paste0('nn_s_cnn',i) )
      }
      
    }
    
    nnBranchs <-   nnBranchs  %>%   layer_dense(units = nunits[n.layers+1], activation = "linear", name = 'nn_s_dense_final',
                                                weights=list(matrix(0,nrow=nunits[n.layers],ncol=1), array(init.spread)))
    
  }
  #Additive towers
  #Location
  n.dim.add_q=length(dim(X_train_add_basis_q))
  if(!is.null(X_train_add_basis_q) & !is.null(X_train_add_basis_q) ) {
    
    addBranchq <- input_add_q %>%
      layer_reshape(target_shape=c(dim(X_train_add_basis_q)[2:(n.dim.add_q-2)],prod(dim(X_train_add_basis_q)[(n.dim.add_q-1):n.dim.add_q]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_q',
                  weights=list(matrix(0,nrow=prod(dim(X_train_add_basis_q)[(n.dim.add_q-1):n.dim.add_q]),ncol=1)),use_bias = F)
  }
  if(!is.null(X_train_add_basis_q) & is.null(X_train_add_basis_q) ) {
    
    addBranchq <- input_add_q %>%
      layer_reshape(target_shape=c(dim(X_train_add_basis_q)[2:(n.dim.add_q-2)],prod(dim(X_train_add_basis_q)[(n.dim.add_q-1):n.dim.add_q]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_q',
                  weights=list(matrix(0,nrow=prod(dim(X_train_add_basis_q)[(n.dim.add_q-1):n.dim.add_q]),ncol=1),array(init.loc)),use_bias = T)
  }
  #Location
  n.dim.add_s=length(dim(X_train_add_basis_s))
  if(!is.null(X_train_add_basis_s) & !is.null(X_train_add_basis_s) ) {
    
    addBranchs <- input_add_s %>%
      layer_reshape(target_shape=c(dim(X_train_add_basis_s)[2:(n.dim.add_s-2)],prod(dim(X_train_add_basis_s)[(n.dim.add_s-1):n.dim.add_s]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_s',
                  weights=list(matrix(0,nrow=prod(dim(X_train_add_basis_s)[(n.dim.add_s-1):n.dim.add_s]),ncol=1)),use_bias = F)
  }
  if(!is.null(X_train_add_basis_s) & is.null(X_train_add_basis_s) ) {
    
    addBranchs <- input_add_s %>%
      layer_reshape(target_shape=c(dim(X_train_add_basis_s)[2:(n.dim.add_s-2)],prod(dim(X_train_add_basis_s)[(n.dim.add_s-1):n.dim.add_s]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_s',
                  weights=list(matrix(0,nrow=prod(dim(X_train_add_basis_s)[(n.dim.add_s-1):n.dim.add_s]),ncol=1),array(init.spread)),use_bias = T)
  }
  #Linear towers
  
  #Location
  if(!is.null(X_train_lin_q) ) {
    n.dim.lin_q=length(dim(X_train_lin_q))
    
    if(is.null(X_train_nn_q) & is.null(X_train_add_basis_q )){
      linBranchq <- input_lin_q%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X_train_lin_q)[-1], name = 'lin_q',
                    weights=list(matrix(0,nrow=dim(X_train_lin_q)[n.dim.lin_q],ncol=1),array(init.loc)),use_bias=T)
    }else{
      linBranchq <- input_lin_q%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X_train_lin_q)[-1], name = 'lin_q',
                    weights=list(matrix(0,nrow=dim(X_train_lin_q)[n.dim.lin_q],ncol=1)),use_bias=F)
    }
  }
  #Spread
  if(!is.null(X_train_lin_s) ) {
    n.dim.lin_s=length(dim(X_train_lin_s))
    
    if(is.null(X_train_nn_s) & is.null(X_train_add_basis_s )){
      linBranchs <- input_lin_s%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X_train_lin_s)[-1], name = 'lin_s',
                    weights=list(matrix(0,nrow=dim(X_train_lin_s)[n.dim.lin_s],ncol=1),array(init.spread)),use_bias=T)
    }else{
      linBranchs <- input_lin_s%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X_train_lin_s)[-1], name = 'lin_s',
                    weights=list(matrix(0,nrow=dim(X_train_lin_s)[n.dim.lin_s],ncol=1)),use_bias=F)
    }
  }
  
  #Location
  if(!is.null(X_train_nn_q) & !is.null(X_train_add_basis_q) & !is.null(X_train_lin_q) )  qBranchjoined <- layer_add(inputs=c(addBranchq,  linBranchq,nnBranchq))  #Add all towers
  if(is.null(X_train_nn_q) & !is.null(X_train_add_basis_q) & !is.null(X_train_lin_q) )  qBranchjoined <- layer_add(inputs=c(addBranchq,  linBranchq))  #Add GAM+lin towers
  if(!is.null(X_train_nn_q) & is.null(X_train_add_basis_q) & !is.null(X_train_lin_q) )  qBranchjoined <- layer_add(inputs=c(  linBranchq,nnBranchq))  #Add nn+lin towers
  if(!is.null(X_train_nn_q) & !is.null(X_train_add_basis_q) & is.null(X_train_lin_q) )  qBranchjoined <- layer_add(inputs=c(addBranchq,  nnBranchq))  #Add nn+GAM towers
  if(is.null(X_train_nn_q) & is.null(X_train_add_basis_q) & !is.null(X_train_lin_q) )  qBranchjoined <- linBranchq  #Just lin tower
  if(is.null(X_train_nn_q) & !is.null(X_train_add_basis_q) & is.null(X_train_lin_q) )  qBranchjoined <- addBranchq  #Just GAM tower
  if(!is.null(X_train_nn_q) & is.null(X_train_add_basis_q) & is.null(X_train_lin_q) )  qBranchjoined <- nnBranchq  #Just nn tower
  
  #Spread
  if(!is.null(X_train_nn_s) & !is.null(X_train_add_basis_s) & !is.null(X_train_lin_s) )  sBranchjoined <- layer_add(inputs=c(addBranchs,  linBranchs,nnBranchs))  #Add all towers
  if(is.null(X_train_nn_s) & !is.null(X_train_add_basis_s) & !is.null(X_train_lin_s) )  sBranchjoined <- layer_add(inputs=c(addBranchs,  linBranchs))  #Add GAM+lin towers
  if(!is.null(X_train_nn_s) & is.null(X_train_add_basis_s) & !is.null(X_train_lin_s) )  sBranchjoined <- layer_add(inputs=c(  linBranchs,nnBranchs))  #Add nn+lin towers
  if(!is.null(X_train_nn_s) & !is.null(X_train_add_basis_s) & is.null(X_train_lin_s) )  sBranchjoined <- layer_add(inputs=c(addBranchs,  nnBranchs))  #Add nn+GAM towers
  if(is.null(X_train_nn_s) & is.null(X_train_add_basis_s) & !is.null(X_train_lin_s) )  sBranchjoined <- linBranchs  #Just lin tower
  if(is.null(X_train_nn_s) & !is.null(X_train_add_basis_s) & is.null(X_train_lin_s) )  sBranchjoined <- addBranchs  #Just GAM tower
  if(!is.null(X_train_nn_s) & is.null(X_train_add_basis_s) & is.null(X_train_lin_s) )  sBranchjoined <- nnBranchs  #Just nn tower
  
  #Apply link functions
  if(link.loc=="exp") qBranchjoined <- qBranchjoined %>% layer_activation( activation = 'exponential') else if(link.loc=="linear") qBranchjoined <- qBranchjoined %>% layer_activation( activation = 'identity')
  sBranchjoined <- sBranchjoined %>% layer_activation( activation = 'exponential')
  
  input=c()
  if(!is.null(X_train_lin_q) ) input=c(input,input_lin_q)
  if(!is.null(X_train_add_basis_q) ) input=c(input,input_add_q)
  if(!is.null(X_train_nn_q) ) input=c(input,input_nn_q)
  if(!is.null(X_train_lin_s) ) input=c(input,input_lin_s)
  if(!is.null(X_train_add_basis_s) ) input=c(input,input_add_s)
  if(!is.null(X_train_nn_s) ) input=c(input,input_nn_s)
  input=c(input)
  
  
  output <- layer_concatenate(c(qBranchjoined,sBranchjoined, xiBranch))
  
  model <- keras_model(  inputs = input,   outputs = output,name=paste0("bGEV"))
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

bgev_loss <-function(alpha=0.5,beta=0.5,p_a=0.05,p_b=0.2,c1=5,c2=5){
  
  loss<- function( y_true, y_pred) {
    
    library(tensorflow)
    K <- backend()
    
    
    q_a=y_pred[all_dims(),1]
    s_b=y_pred[all_dims(),2]
    xi=y_pred[all_dims(),3]
    
    
    
    
    # Find inds of non-missing obs.  Remove missing obs, i.e., -1e5. This is achieved by adding an
    # arbitrarily large (<1e5) value to y_true and then taking the sign ReLu
    obsInds=K$sign(K$relu(y_true+1e4))
    
  
    
    a=Finverse(p_a,q_a,s_b,xi,alpha,beta)
    b=Finverse(p_b,q_a,s_b,xi,alpha,beta)
    b =b + (1-obsInds)
    s_b=s_b+(1-obsInds)
    
    
    l1=logH(y_true,q_a,s_b,xi,alpha,beta,a,b,p_a,p_b,c1,c2,obsInds)
    l2=lambda(y_true,q_a,s_b,xi,alpha,beta,a,b,p_a,p_b,c1,c2,obsInds,obsInds) #use lambda functiom from bGEV_NN.R, but with exceedInds=obsInds
    
    l2=K$log(l2+(1-obsInds))*obsInds
    
    return( -K$sum(l1)
            -K$sum(l2) 
            )
  }
  return(loss)
}
