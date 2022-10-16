#'blended-GEV point process PINN
#'
#' Build and train a partially-interpretable neural network for fitting a bGEV point-process model
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
#'@param u an array with the same dimension as \code{Y.train}. Gives the threshold above which the bGEV-PP model is fitted, see below. Note that \code{u} is applied to both \code{Y.train} and \code{Y.valid}.
#' @param X.q  list of arrays corresponding to complementary subsets of the \eqn{d\geq 1} predictors which are used for modelling the location parameter \eqn{q_\alpha}. Must contain at least one of the following three named entries:\describe{
#' \item{\code{X.lin.q}}{A 3 or 4 dimensional array of "linear" predictor values. One more dimension than \code{Y.train}. If \code{NULL}, a model without the linear component is built and trained.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the last dimension corresponds to the chosen \eqn{l_1\geq 0} 'linear' predictor values.}
#' \item{\code{X.add.basis.q}}{A 4 or 5 dimensional array of basis function evaluations for the "additive" predictor values.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the penultimate dimensions corresponds to the chosen \eqn{a_1\geq 0} 'linear' predictor values and the last dimension is equal to the number of knots used for estimating the splines. See example.
#' If \code{NULL}, a model without the additive component is built and trained.}
#' \item{\code{X.nn.q}}{A 3 or 4 dimensional array of "non-additive" predictor values.  If \code{NULL}, a model without the NN component is built and trained; if this is the case, then \code{type} has no effect.
#' The first 2/3 dimensions should be equal to that of \code{Y.train}; the last dimension corresponds to the chosen \eqn{d-l_1-a_1\geq 0} 'non-additive' predictor values.}
#' }
#' Note that \code{X.q} and \code{X.s} are the predictors for both \code{Y.train} and \code{Y.valid}. If \code{is.null(X.q)}, then \eqn{q_\alpha} will be treated as fixed over the predictors.
#' @param X.s similarly to \code{X.q}, but for modelling the scale parameter \eqn{s_\beta>0}.  Note that we require at least one of \code{!is.null(X.q)} or \code{!is.null(X.s)}, otherwise the formulated model will be fully stationary and will not be fitted.
#' @param n.ep number of epochs used for training. Defaults to 1000.
#' @param alpha,beta,p_a,p_b,c1,c2 hyper-parameters associated with the bGEV distribution. Defaults to those used by Castro-Camilo, D., et al. (2021). Require \code{alpha >= p_b} and \code{beta/2 >= p_b}.
#' @param batch.size batch size for stochastic gradient descent. If larger than \code{dim(Y.train)[1]}, i.e., the number of observations, then regular gradient descent used.
#' @param init.loc,init.spread,init.xi sets the initial \eqn{q_\alpha,s_\beta} and \eqn{\xi\in(0,1)} estimates across all dimensions of \code{Y.train}. Overridden by \code{init.wb_path} if \code{!is.null(init.wb_path)}, but otherwise the initial parameters must be supplied.
#' @param init.wb_path filepath to a \code{keras} model which is then used as initial weights and biases for training the new model. The original model must have
#' the exact same architecture and trained with the same input data as the new model. If \code{NULL}, then initial weights and biases are random (with seed \code{seed}) but the
#' final layer has zero initial weights to ensure that the initial location, spread and shape estimates are \code{init.loc, init.spread} and \code{init.xi}, respectively,  across all dimensions.
#' @param widths vector of widths/filters for hidden dense/convolution layers. Number of layers is equal to \code{length(widths)}. Defaults to (6,3).
#' @param filter.dim if \code{type=="CNN"}, this 2-vector gives the dimensions of the convolution filter kernel; must have odd integer inputs. Note that filter.dim=c(1,1) is equivalent to \code{type=="MLP"}. The same filter is applied for each hidden layer across all parameters with NN predictors.
#' @param seed seed for random initial weights and biases.
#' @param loc.link string defining the link function used for the location parameter, see \eqn{h_1} below. If \code{link=="exp"}, then \eqn{h_1=\exp(x)}; if \code{link=="identity"}, then \eqn{h_1(x)=x}.
#' @param model fitted \code{keras} model. Output from \code{bGEVPP.NN.train}.
#' @param n_b number of observations per block, e.g., if observations correspond to months and the interest is annual maxima, then \code{n_b=12}.
#' @param S_lambda List of smoothing penalty matrices for the splines modelling the effects of \code{X.add.basis.q} and \code{X.add.basis.s} on their respective parameters; each element only used if \code{!is.null(X.add.basis.q)} and \code{!is.null(X.add.basis.s)}, respectively. If \code{is.null(S_lambda[[1]])}, then no smoothing penalty used for \code{!is.null(X.add.basis.q)}; similarly for the second element and \code{!is.null(X.add.basis.s)}. 

#'@name bGEVPP.NN

#' @details{
#' Consider a real-valued random variable \eqn{Y} and let \eqn{\mathbf{X}} denote a \eqn{d}-dimensional predictor set with observations \eqn{\mathbf{x}}.
#' For \eqn{i=1,2}, we define integers \eqn{l_i\geq 0,a_i \geq 0} and \eqn{0\leq l_i+a_i \leq d}, and let \eqn{\mathbf{X}^{(i)}_L, \mathbf{X}^{(i)}_A} and \eqn{\mathbf{X}^{(i)}_N} be distinct sub-vectors
#' of \eqn{\mathbf{X}}, with observations of each component denoted \eqn{\mathbf{x}^{(i)}_L, \mathbf{x}^{(i)}_A} and \eqn{\mathbf{x}^{(i)}_N}, respectively; the lengths of the sub-vectors are \eqn{l_i,a_i} and \eqn{d_i-l_i-a}, respectively.
#' For a fixed threshold \eqn{u(\mathbf{x})}, dependent on predictors, we model \eqn{Y|\mathbf{X}=\mathbf{x}\sim\mbox{bGEV-PP}(q_\alpha(\mathbf{x}),s_\beta(\mathbf{x}),\xi;u(\mathbf{x}))} for \eqn{\xi\in(0,1)} with
#' \deqn{q_\alpha (\mathbf{x})=h_1\{\eta^{(1)}_0+m^{(1)}_L(\mathbf{x}^{(1)}_L)+m^{(1)}_A(x^{(1)}_A)+m^{(1)}_N(\mathbf{x}^{(1)}_N)\}} and
#' \deqn{s_\beta (\mathbf{x})=\exp\{\eta^{(2)}_0+m^{(2)}_L(\mathbf{x}^{(2)}_L)+m^{(2)}_A(x^{(2)}_A)+m^{(2)}_N(\mathbf{x}^{(2)}_N)\}}
#' where \eqn{h_1} is some link-function and \eqn{\eta^{(1)}_0,\eta^{(2)}_0} are constant intercepts. The unknown functions \eqn{m^{(1)}_L,m^{(2)}_L} and
#' \eqn{m^{(1)}_A,m^{(2)}_A} are estimated using linear functions and splines, respectively, and are
#' both returned as outputs by \code{bGEVPP.NN.predict}; \eqn{m^{(1)}_N,m^{(2)}_N} are estimated using neural networks
#' (currently the same architecture is used for both parameters). Note that \eqn{\xi>0} is fixed across all predictors; this may change in future versions.
#'
#'Note that for sufficiently large \eqn{u} that \eqn{Y\sim\mbox{bGEV-PP}(q_\alpha,s_\beta,\xi;u)} implies that \eqn{\max_{i=1,\dots,n_b}\{Y_i\}\sim \mbox{bGEV}(q_\alpha,s_\beta,\xi)},
#'i.e., the \eqn{n_b}-block maxima of independent realisations of \eqn{Y} follow a bGEV distribution (see \code{help(pbGEV)}). The size of the block can be specified by the parameter \code{n_b}.
#'
#' The model is fitted by minimising the negative log-likelihood associated with the bGEV-PP model plus some smoothing penalty for the additive functions (determined by \code{S_lambda}; see Richards and Huser, 2022); training is performed over \code{n.ep} training epochs.
#' Although the model is trained by minimising the loss evaluated for \code{Y.train}, the final returned model may minimise some other loss.
#' The current state of the model is saved after each epoch, using \code{keras::callback_model_checkpoint}, if the value of some criterion subcedes that of the model from the previous checkpoint; this criterion is the loss evaluated for validation set \code{Y.valid} if \code{!is.null(Y.valid)} and for \code{Y.train}, otherwise.
#'
#'}
#' @return \code{bGEVPP.NN.train} returns the fitted \code{model}.  \code{bGEVPP.NN.predict} is a wrapper for \code{keras::predict} that returns the predicted parameter estimates, and, if applicable, their corresponding linear regression coefficients and spline bases weights.
#'
#'@references
#' Castro-Camilo, D., Huser, R., and Rue, H. (2021), \emph{Practical strategies for generalized extreme value-based regression models for extremes}, Environmetrics, e274.
#' (\href{https://doi.org/10.1002/env.2742}{doi})
#'
#' Richards, J. and Huser, R. (2022), \emph{A unifying partially-interpretable framework for neural network-based extreme quantile regression}. (\href{https://arxiv.org/abs/2208.07581}{arXiv:2208.07581}).
#'
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
#' #Other dimensions correspond to indices of predictors, e.g., a grid of locations. Can be just a 1D grid.
#' dim(preds)=c(200,10,10,8) 
#' #We have 200 observations of eight predictors on a 10 by 10 grid.

#'
#' #Split predictors into linear, additive and nn. Different for the location and scale parameters.
#'X.nn.q=preds[,,,1:4] #Four nn predictors for q_\alpha
#'X.lin.q=preds[,,,5:6] #Two additive predictors for q_\alpha
#'X.add.q=preds[,,,7:8] #Two additive predictors for q_\alpha
#'
#'X.nn.s=preds[,,,1:2] #Two nn predictors for s_\beta
#'X.lin.s=preds[,,,3] #One linear predictor for s_\beta
#'dim(X.lin.s)=c(dim(X.lin.s),1) #Change dimension so consistent
#'X.add.s=preds[,,,4] #One additive predictor for s_\beta
#'dim(X.add.s)=c(dim(X.add.s),1) #Change dimension so consistent
#'
#' # Create toy response data
#'
#' #Contribution to location parameter
#'#Linear contribution
#'m_L_1 = 0.3*X.lin.q[,,,1]+0.6*X.lin.q[,,,2]
#'
#'# Additive contribution
#'m_A_1 = 0.1*X.add.q[,,,1]^3+0.2*X.add.q[,,,1]-
#'  0.1*X.add.q[,,,2]^3+0.5*X.add.q[,,,2]^2
#'
#'#Non-additive contribution - to be estimated by NN
#'m_N_1 = 0.5*exp(-3+X.nn.q[,,,4]+X.nn.q[,,,1])+
#'  sin(X.nn.q[,,,1]-X.nn.q[,,,2])*(X.nn.q[,,,4]+X.nn.q[,,,2])-
#'  cos(X.nn.q[,,,4]-X.nn.q[,,,1])*(X.nn.q[,,,3]+X.nn.q[,,,1])
#'
#' q_alpha=1+m_L_1+m_A_1+m_N_1 #Identity link
#'
#' #Contribution to scale parameter
#'#Linear contribution
#'m_L_2 = 0.5*X.lin.s[,,,1]
#'
#'# Additive contribution
#'m_A_2 = 0.1*X.add.s[,,,1]^2+0.2*X.add.s[,,,1]
#'
#'#Non-additive contribution - to be estimated by NN
#'m_N_2 = 0.2*exp(-4+X.nn.s[,,,2]+X.nn.s[,,,1])+
#'  sin(X.nn.s[,,,1]-X.nn.s[,,,2])*(X.nn.s[,,,1]+X.nn.s[,,,2])
#'
#'s_beta=0.2*exp(m_L_2+m_A_2+m_N_2) #Exponential link
#'
#' xi=0.1 # Set xi
#'
#' theta=array(dim=c(dim(s_beta),3))
#' theta[,,,1]=q_alpha; theta[,,,2] = s_beta; theta[,,,3]=xi
#' #We simulate data from the extreme value point process model with u take as the 80% quantile
#'
#'#Gives the 80% quantile of Y
#'u<-apply(theta,1:3,function(x) qPP(prob=0.8,loc=x[1],scale=x[2],xi=x[3],re.par = T)) 
#'
#' #Simulate from re-parametrised point process model using same u as given above
#' Y=apply(theta,1:3,function(x) rPP(1,u.prob=0.8,loc=x[1],scale=x[2],xi=x[3],re.par=T)) 
#'
#' # Note that the point process model is only valid for Y > u. If Y < u, then rPP gives NA.
#' # We can set NA values to some c < u as these do not contribute to model fitting.
#' Y[is.na(Y)]=u[is.na(Y)]-1
#'
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
#' #the basis functions for each pre-specified knot and entry to X.add.q and X.add.s
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
#' knots.q=matrix(nrow=dim(X.add.q)[4],ncol=n.knot.q)
#'
#' #We set knots to be equally-spaced marginal quantiles
#' for( i in 1:dim(X.add.q)[4]){
#'  knots.q[i,]=quantile(X.add.q[,,,i],probs=seq(0,1,length=n.knot.q))
#'}
#'
#' #Evaluate radial basis functions for q_\alpha predictors
#' X.add.basis.q<-array(dim=c(dim(X.add.q),n.knot.q))
#' for( i in 1:dim(X.add.q)[4]) {
#'   for(k in 1:n.knot.q) {
#'     X.add.basis.q[,,,i,k]= rad(x=X.add.q[,,,i],c=knots.q[i,k])
#'     #Evaluate rad at all entries to X.add.q and for all knots
#'   }}
#'
#'   #'#Create smoothing penalty matrix for the two q_alpha additive functions
#' 
#'# Set smoothness parameters for two functions
#'  lambda = c(0.1,0.2) 
#'
#' S_lambda.q=matrix(0,nrow=n.knot.q*dim(X.add.q)[4],ncol=n.knot.q*dim(X.add.q)[4])
#'for(i in 1:dim(X.add.q)[4]){
#'  for(j in 1:n.knot.q){
#'   for(k in 1:n.knot.q){
#'      S_lambda.q[(j+(i-1)*n.knot.q),(k+(i-1)*n.knot.q)]=lambda[i]*rad(knots.q[i,j],knots.q[i,k])
#'   }
#' }
#'}
#'
#' #Get knots for s_\beta predictor
#' knots.s=matrix(nrow=dim(X.add.s)[4],ncol=n.knot.s)
#' for( i in 1:dim(X.add.s)[4]){
#'  knots.s[i,]=quantile(X.add.s[,,,i],probs=seq(0,1,length=n.knot.s))
#' }
#'
#' #Evaluate radial basis functions for s_\beta predictor
#' X.add.basis.s<-array(dim=c(dim(X.add.s),n.knot.s))
#' for( i in 1:dim(X.add.s)[4]) {
#'   for(k in 1:n.knot.s) {
#'     X.add.basis.s[,,,i,k]= rad(x=X.add.s[,,,i],c=knots.s[i,k])
#'     #Evaluate rad at all entries to X.add.q and for all knots
#'   }}
#'
#'#Create smoothing penalty matrix for the s_beta additive function
#' 
#'# Set smoothness parameter
#'  lambda = c(0.2) 
#'
#' S_lambda.s=matrix(0,nrow=n.knot.s*dim(X.add.s)[4],ncol=n.knot.s*dim(X.add.s)[4])
#'for(i in 1:dim(X.add.s)[4]){
#'  for(j in 1:n.knot.s){
#'   for(k in 1:n.knot.s){
#'      S_lambda.s[(j+(i-1)*n.knot.s),(k+(i-1)*n.knot.s)]=lambda[i]*rad(knots.s[i,j],knots.s[i,k])
#'   }
#' }
#'}
#'  
#'#Join in one list
#'S_lambda =list("S_lambda.q"=S_lambda.q, "S_lambda.s"=S_lambda.s)
#'
#' #lin+GAM+NN models defined for both location and scale parameters
#' X.q=list("X.nn.q"=X.nn.q, "X.lin.q"=X.lin.q,
#'                "X.add.basis.q"=X.add.basis.q) #Predictors for q_\alpha
#' X.s=list("X.nn.s"=X.nn.s, "X.lin.s"=X.lin.s,
#'                "X.add.basis.s"=X.add.basis.s) #Predictors for s_\beta
#'
#' #We treat u as fixed and known. In an application, u can be estimated using quant.NN.train.
#'
#' #Fit the bGEV-PP model using u. Note that training is not run to completion.
#' NN.fit<-bGEVPP.NN.train(Y.train, Y.valid,X.q,X.s, u, type="MLP",link.loc="identity",
#'                        n.ep=500, batch.size=50,init.loc=2, init.spread=2,init.xi=0.1,
#'                        widths=c(6,3),seed=1, n_b=12,S_lambda=S_lambda)
#' out<-bGEVPP.NN.predict(X.q=X.q,X.s=X.s,u=u,NN.fit$model)
#'
#' print("q_alpha linear coefficients: "); print(round(out$lin.coeff_q,2))
#' print("s_beta linear coefficients: "); print(round(out$lin.coeff_s,2))
#'
#' # Note that this is a simple example that can be run in a personal computer. 
#' # Whilst the q_alpha functions are well estimated, more data/larger n.ep are required for more accurate
#' # estimation of s_beta functions and xi
#'
#' #To save model, run
#' #model %>% NN.fit$save_model_tf("model_bGEVPP")
#' #To load model, run
#' # model  <- load_model_tf("model_bGEVPP",
#' #  custom_objects=list(
#' #    "bgev_PP_loss_alpha__beta__p_a__p_b__c1__c2__n_b__S_lambda___S_lambda_"=
#' #      bgev_PP_loss(n_b=12,S_lambda=S_lambda))
#' #        )
#'
#' #Note that bGEV_PP_loss() can take custom alpha,beta, p_a, p_b, c1 and c2 arguments if defaults not used.
#'
#'
#' # Plot splines for the additive predictors
#'
#' #Location predictors
#' n.add.preds_q=dim(X.add.q)[length(dim(X.add.q))]
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
#'   plot(plt.x,plt.y,type="l",main=paste0("s_beta spline: predictor ",i),xlab="x",ylab="f(x)")
#'   points(knots.s[i,],rep(mean(plt.y),n.knot.s),col="red",pch=2)
#'   #Adds red triangles that denote knot locations
#'
#' }
#'
#' @import reticulate tfprobability keras tensorflow
#'
#' @rdname bGEVPP.NN
#' @export

bGEVPP.NN.train=function(Y.train, Y.valid = NULL,X.q,X.s, u = NULL, type="MLP",link.loc="identity",
                        n.ep=100, batch.size=100,init.loc=NULL, init.spread=NULL,init.xi=NULL,
                        widths=c(6,3), filter.dim=c(3,3),seed=NULL,init.wb_path=NULL,
                        alpha=0.5,beta=0.5,p_a=0.05,p_b=0.2,c1=5,c2=5,n_b=1,
                        S_lambda=NULL)
{




  if(is.null(X.q) &  is.null(X.s)  ) stop("No predictors provided for q_alpha or s_beta: Stationary models are not permitted ")
  if(is.null(Y.train)) stop("No training response data provided")
  if(is.null(u)) stop("No threshold u provided")

  if(is.null(init.loc) & is.null(init.wb_path)  ) stop("Inital location estimate not provided")
  if(is.null(init.spread) & is.null(init.wb_path)   ) stop("Inital spread estimate not provided")
  if(is.null(init.xi)  & is.null(init.wb_path) ) stop("Inital shape estimate not provided")


  print(paste0("Creating bGEV-PP model with ",n_b,"-block maxima following bGEV"))
  X.nn.q=X.q$X.nn.q
  X.lin.q=X.q$X.lin.q
  X.add.basis.q=X.q$X.add.basis.q


  if(!is.null(X.nn.q) & !is.null(X.add.basis.q) & !is.null(X.lin.q) ) {  train.data= list(X.lin.q,X.add.basis.q,X.nn.q); print("Defining lin+GAM+NN model for q_\alpha" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_q=X.lin.q,add_input_q=X.add.basis.q,  nn_input_q=X.nn.q),Y.valid)}
  if(is.null(X.nn.q) & !is.null(X.add.basis.q) & !is.null(X.lin.q) ) {   train.data= list(X.lin.q,X.add.basis.q); print("Defining lin+GAM model for q_\alpha" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_q=X.lin.q,add_input_q=X.add.basis.q),Y.valid)}
  if(!is.null(X.nn.q) & is.null(X.add.basis.q) & !is.null(X.lin.q) ) { train.data= list(X.lin.q,X.nn.q); print("Defining lin+NN model for q_\alpha" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_q=X.lin.q, nn_input_q=X.nn.q),Y.valid)}
  if(!is.null(X.nn.q) & !is.null(X.add.basis.q) & is.null(X.lin.q) ) {train.data= list(X.add.basis.q,X.nn.q); print("Defining GAM+NN model for q_\alpha" );  if(!is.null(Y.valid)) validation.data=list(list(add_input_q=X.add.basis.q,  nn_input_q=X.nn.q),Y.valid)}
  if(is.null(X.nn.q) & is.null(X.add.basis.q) & !is.null(X.lin.q) )   {train.data= list(X.lin.q); print("Defining fully-linear model for q_\alpha" );  if(!is.null(Y.valid)) validation.data=list(list(lin_input_q=X.lin.q),Y.valid)}
  if(is.null(X.nn.q) & !is.null(X.add.basis.q) & is.null(X.lin.q) )   {train.data= list(X.add.basis.q); print("Defining fully-additive model for q_\alpha" );  if(!is.null(Y.valid)) validation.data=list(list(add_input_q=X.add.basis.q),Y.valid)}
  if(!is.null(X.nn.q) & is.null(X.add.basis.q) & is.null(X.lin.q) )   {train.data= list(X.nn.q); print("Defining fully-NN model for q_\alpha" );  if(!is.null(Y.valid)) validation.data=list(list( nn_input_q=X.nn.q),Y.valid)}
  if(is.null(X.nn.q) & is.null(X.add.basis.q) & is.null(X.lin.q) )   {train.data= list(); print("Defining stationary model for q_\alpha" );  if(!is.null(Y.valid)) validation.data=list(list( ),Y.valid)}
  
  S_lambda.q=S_lambda$S_lambda.q
  if(is.null(S_lambda.q)){print("No smoothing penalty used for q_\alpha")}
  if(is.null(X.add.basis.q)){S_lambda.q=NULL}
  
  X.nn.s=X.s$X.nn.s
  X.lin.s=X.s$X.lin.s
  X.add.basis.s=X.s$X.add.basis.s

  if(!is.null(X.nn.s) & !is.null(X.add.basis.s) & !is.null(X.lin.s) ) {  train.data= c(train.data,list(X.lin.s,X.add.basis.s,X.nn.s,u)); print("Defining lin+GAM+NN model for s_\beta" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X.lin.s,add_input_s=X.add.basis.s,  nn_input_s=X.nn.s,u_input=u)),Y.valid)}
  if(is.null(X.nn.s) & !is.null(X.add.basis.s) & !is.null(X.lin.s) ) {   train.data= c(train.data,list(X.lin.s,X.add.basis.s,u)); print("Defining lin+GAM model for s_\beta" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X.lin.s,add_input_s=X.add.basis.s,u_input=u)),Y.valid)}
  if(!is.null(X.nn.s) & is.null(X.add.basis.s) & !is.null(X.lin.s) ) { train.data= c(train.data,list(X.lin.s,X.nn.s,u)); print("Defining lin+NN model for s_\beta" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X.lin.s, nn_input_s=X.nn.s,u_input=u)),Y.valid)}
  if(!is.null(X.nn.s) & !is.null(X.add.basis.s) & is.null(X.lin.s) ) {train.data= c(train.data,list(X.add.basis.s,X.nn.s,u)); print("Defining GAM+NN model for s_\beta" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(add_input_s=X.add.basis.s,  nn_input_s=X.nn.s,u_input=u)),Y.valid)}
  if(is.null(X.nn.s) & is.null(X.add.basis.s) & !is.null(X.lin.s) )   {train.data= c(train.data,list(X.lin.s,u)); print("Defining fully-linear model for s_\beta" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(lin_input_s=X.lin.s,u_input=u)),Y.valid)}
  if(is.null(X.nn.s) & !is.null(X.add.basis.s) & is.null(X.lin.s) )   {train.data= c(train.data,list(X.add.basis.s,u)); print("Defining fully-additive model for s_\beta" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(add_input_s=X.add.basis.s,u_input=u)),Y.valid)}
  if(!is.null(X.nn.s) & is.null(X.add.basis.s) & is.null(X.lin.s) )   {train.data= c(train.data,list(X.nn.s,u)); print("Defining fully-NN model for s_\beta" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(nn_input_s=X.nn.s,u_input=u)),Y.valid)}
  if(is.null(X.nn.s) & is.null(X.add.basis.s) & is.null(X.lin.s) )   {train.data=  c(train.data,list(u)); print("Defining stationary model for s_\beta" );  if(!is.null(Y.valid)) validation.data=list(c(validation.data[[1]],list(u_input=u)),Y.valid)}

  S_lambda.s=S_lambda$S_lambda.s
  if(is.null(S_lambda.s)){print("No smoothing penalty used for s_\beta")}
  if(is.null(X.add.basis.s)){S_lambda.s=NULL}

  S_lambda =list("S_lambda.q"=S_lambda.q, "S_lambda.s"=S_lambda.s)
  
  if(type=="CNN" & (!is.null(X.nn.q) | !is.null(X.nn.s)))print(paste0("Building ",length(widths),"-layer convolutional neural network with ", filter.dim[1]," by ", filter.dim[2]," filter" ))
  if(type=="MLP"  & (!is.null(X.nn.q) | !is.null(X.nn.s)) ) print(paste0("Building ",length(widths),"-layer densely-connected neural network" ))

  reticulate::use_virtualenv("myenv", required = T)

  if(!is.null(seed)) tf$random$set_seed(seed)

  if(length(dim(u))!=length(dim(Y.train))+1) dim(u)=c(dim(u),1)
  model<-bGEVPP.NN.build(X.nn.q,X.lin.q,X.add.basis.q,
                         X.nn.s,X.lin.s,X.add.basis.s,
                         u,type, init.loc,init.spread,init.xi, widths,filter.dim,link.loc,alpha,beta,p_a,p_b)
  if(!is.null(init.wb_path)) model <- load_model_weights_tf(model,filepath=init.wb_path)

  model %>% compile(
    optimizer="adam",
    loss = bgev_PP_loss(alpha,beta,p_a,p_b,c1,c2,n_b,S_lambda=S_lambda),
    run_eagerly=T
    
  )

  if(!is.null(Y.valid)) checkpoint <- callback_model_checkpoint(paste0("model_bGEVPP_checkpoint"), monitor = "val_loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch") else checkpoint <- callback_model_checkpoint(paste0("model_bGEVPP_checkpoint"), monitor = "loss", verbose = 0,   save_best_only = TRUE, save_weights_only = TRUE, mode = "min",   save_freq = "epoch")

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
  model <- load_model_weights_tf(model,filepath=paste0("model_bGEVPP_checkpoint"))
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
#' @rdname bGEVPP.NN
#' @export
#'
bGEVPP.NN.predict=function(X.q,X.s,u, model)
{
    library(tensorflow)
  if(is.null(X.q) &  is.null(X.s)  ) stop("No predictors provided for q_alpha or s_beta: Stationary models are not permitted ")
  


  X.nn.q=X.q$X.nn.q
  X.lin.q=X.q$X.lin.q
  X.add.basis.q=X.q$X.add.basis.q


  if(!is.null(X.nn.q) & !is.null(X.add.basis.q) & !is.null(X.lin.q) )   train.data= list(X.lin.q,X.add.basis.q,X.nn.q)
  if(is.null(X.nn.q) & !is.null(X.add.basis.q) & !is.null(X.lin.q) )   train.data= list(X.lin.q,X.add.basis.q)
  if(!is.null(X.nn.q) & is.null(X.add.basis.q) & !is.null(X.lin.q) )  train.data= list(X.lin.q,X.nn.q)
  if(!is.null(X.nn.q) & !is.null(X.add.basis.q) & is.null(X.lin.q) ) train.data= list(X.add.basis.q,X.nn.q)
  if(is.null(X.nn.q) & is.null(X.add.basis.q) & !is.null(X.lin.q) )   train.data= list(X.lin.q)
  if(is.null(X.nn.q) & !is.null(X.add.basis.q) & is.null(X.lin.q) )   train.data= list(X.add.basis.q)
  if(!is.null(X.nn.q) & is.null(X.add.basis.q) & is.null(X.lin.q) )   train.data= list(X.nn.q)
  if(is.null(X.nn.q) & is.null(X.add.basis.q) & is.null(X.lin.q) )   train.data= list()
  
  X.nn.s=X.s$X.nn.s
  X.lin.s=X.s$X.lin.s
  X.add.basis.s=X.s$X.add.basis.s

  if(!is.null(X.nn.s) & !is.null(X.add.basis.s) & !is.null(X.lin.s) )   train.data= c(train.data,list(X.lin.s,X.add.basis.s,X.nn.s,u))
  if(is.null(X.nn.s) & !is.null(X.add.basis.s) & !is.null(X.lin.s) )   train.data= c(train.data,list(X.lin.s,X.add.basis.s,u))
  if(!is.null(X.nn.s) & is.null(X.add.basis.s) & !is.null(X.lin.s) )  train.data= c(train.data,list(X.lin.s,X.nn.s,u))
  if(!is.null(X.nn.s) & !is.null(X.add.basis.s) & is.null(X.lin.s) ) train.data= c(train.data,list(X.add.basis.s,X.nn.s,u))
  if(is.null(X.nn.s) & is.null(X.add.basis.s) & !is.null(X.lin.s) )   train.data= c(train.data,list(X.lin.s,u))
  if(is.null(X.nn.s) & !is.null(X.add.basis.s) & is.null(X.lin.s) )   train.data= c(train.data,list(X.add.basis.s,u))
  if(!is.null(X.nn.s) & is.null(X.add.basis.s) & is.null(X.lin.s) ) train.data= c(train.data,list(X.nn.s,u))
  if(is.null(X.nn.s) & is.null(X.add.basis.s) & is.null(X.lin.s) ) train.data= c(train.data,list(u))

    predictions<-model %>% predict( train.data)
    predictions <- k_constant(predictions)
    pred.loc=k_get_value(predictions[all_dims(),2])
    pred.spread=k_get_value(predictions[all_dims(),3])
    pred.xi=k_get_value(predictions[all_dims(),4])

  if(!is.null(X.add.basis.q))  gam.weights_q<-matrix(t(model$get_layer("add_q")$get_weights()[[1]]),nrow=dim(X.add.basis.q)[length(dim(X.add.basis.q))-1],ncol=dim(X.add.basis.q)[length(dim(X.add.basis.q))],byrow=T)
    if(!is.null(X.add.basis.s))  gam.weights_s<-matrix(t(model$get_layer("add_s")$get_weights()[[1]]),nrow=dim(X.add.basis.s)[length(dim(X.add.basis.s))-1],ncol=dim(X.add.basis.s)[length(dim(X.add.basis.s))],byrow=T)

    out=list("pred.loc"=pred.loc,"pred.spread"=pred.spread,"pred.xi"=pred.xi)
  if(!is.null(X.lin.q) ) out=c(out,list("lin.coeff_q"=c(model$get_layer("lin_q")$get_weights()[[1]])))
  if(!is.null(X.lin.s) ) out=c(out,list("lin.coeff_s"=c(model$get_layer("lin_s")$get_weights()[[1]])))
    if(!is.null(X.add.basis.q) ) out=c(out,list("gam.weights_q"=gam.weights_q))
    if(!is.null(X.add.basis.s) ) out=c(out,list("gam.weights_s"=gam.weights_s))

  return(out)

}
#'
#'
bGEVPP.NN.build=function(X.nn.q,X.lin.q,X.add.basis.q,
                         X.nn.s,X.lin.s,X.add.basis.s,
                         u,
                         type, init.loc,init.spread,init.xi, widths,filter.dim,link.loc,alpha,beta,p_a,p_b,c1,c2)
{
  #Additive inputs
  if(!is.null(X.add.basis.q))  input_add_q<- layer_input(shape = dim(X.add.basis.q)[-1], name = 'add_input_q')
  if(!is.null(X.add.basis.s))  input_add_s<- layer_input(shape = dim(X.add.basis.s)[-1], name = 'add_input_s')

  #NN input

  if(!is.null(X.nn.q))   input_nn_q <- layer_input(shape = dim(X.nn.q)[-1], name = 'nn_input_q')
  if(!is.null(X.nn.s))   input_nn_s <- layer_input(shape = dim(X.nn.s)[-1], name = 'nn_input_s')

  #Linear input

  if(!is.null(X.lin.q)) input_lin_q <- layer_input(shape = dim(X.lin.q)[-1], name = 'lin_input_q')
  if(!is.null(X.lin.s)) input_lin_s <- layer_input(shape = dim(X.lin.s)[-1], name = 'lin_input_s')

  #Threshold input
  input_u <- layer_input(shape = dim(u)[-1], name = 'u_input')

  #Create xi branch


  xiBranch <- input_u %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(u)[-1], trainable=F,
                                       weights=list(matrix(0,nrow=dim(u)[length(dim(u))],ncol=1),array(1,dim=c(1))), name = 'xi_stationary_dense') %>%
    layer_dense(units = 1 ,activation = 'sigmoid',use_bias = F,weights=list(matrix(qlogis(init.xi),nrow=1,ncol=1)), name = 'xi_activation')




  if(link.loc=="exp") init.loc=log(init.loc) else if(link.loc =="identity") init.loc=init.loc else stop("Invalid link function for location parameter")
  init.spread=log(init.spread)
  #NN towers

  #Location
  if(!is.null(X.nn.q)){

    nunits=c(widths,1)
    n.layers=length(nunits)-1

    nnBranchq <- input_nn_q
    if(type=="MLP"){
      for(i in 1:n.layers){
        nnBranchq <- nnBranchq  %>% layer_dense(units=nunits[i],activation = 'relu',
                                                input_shape =dim(X.nn.q)[-1], name = paste0('nn_q_dense',i) )
      }
    }else if(type=="CNN"){
      for(i in 1:n.layers){
        nnBranchq <- nnBranchq  %>% layer_conv_2d(filters=nunits[i],activation = 'relu',kernel_size=c(filter.dim[1],filter.dim[2]), padding='same',
                                                  input_shape =dim(X.nn.q)[-1], name = paste0('nn_q_cnn',i) )
      }

    }

    nnBranchq <-   nnBranchq  %>%   layer_dense(units = nunits[n.layers+1], activation = "linear", name = 'nn_q_dense_final',
                                                weights=list(matrix(0,nrow=nunits[n.layers],ncol=1), array(init.loc)))

  }
#Spread
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

    }

    nnBranchs <-   nnBranchs  %>%   layer_dense(units = nunits[n.layers+1], activation = "linear", name = 'nn_s_dense_final',
                                                weights=list(matrix(0,nrow=nunits[n.layers],ncol=1), array(init.spread)))

  }
  #Additive towers
  #Location
  n.dim.add_q=length(dim(X.add.basis.q))
  if(!is.null(X.add.basis.q) & !is.null(X.add.basis.q) ) {

    addBranchq <- input_add_q %>%
      layer_reshape(target_shape=c(dim(X.add.basis.q)[2:(n.dim.add_q-2)],prod(dim(X.add.basis.q)[(n.dim.add_q-1):n.dim.add_q]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_q',
                  weights=list(matrix(0,nrow=prod(dim(X.add.basis.q)[(n.dim.add_q-1):n.dim.add_q]),ncol=1)),use_bias = F)
  }
  if(!is.null(X.add.basis.q) & is.null(X.add.basis.q) ) {

    addBranchq <- input_add_q %>%
      layer_reshape(target_shape=c(dim(X.add.basis.q)[2:(n.dim.add_q-2)],prod(dim(X.add.basis.q)[(n.dim.add_q-1):n.dim.add_q]))) %>%
      layer_dense(units = 1, activation = 'linear', name = 'add_q',
                  weights=list(matrix(0,nrow=prod(dim(X.add.basis.q)[(n.dim.add_q-1):n.dim.add_q]),ncol=1),array(init.loc)),use_bias = T)
  }
  #Spread
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
                  weights=list(matrix(0,nrow=prod(dim(X.add.basis.s)[(n.dim.add_s-1):n.dim.add_s]),ncol=1),array(init.spread)),use_bias = T)
  }
  #Linear towers

  #Location
  if(!is.null(X.lin.q) ) {
    n.dim.lin_q=length(dim(X.lin.q))

    if(is.null(X.nn.q) & is.null(X.add.basis.q )){
      linBranchq <- input_lin_q%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X.lin.q)[-1], name = 'lin_q',
                    weights=list(matrix(0,nrow=dim(X.lin.q)[n.dim.lin_q],ncol=1),array(init.loc)),use_bias=T)
    }else{
      linBranchq <- input_lin_q%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X.lin.q)[-1], name = 'lin_q',
                    weights=list(matrix(0,nrow=dim(X.lin.q)[n.dim.lin_q],ncol=1)),use_bias=F)
    }
  }
  #Spread
  if(!is.null(X.lin.s) ) {
    n.dim.lin_s=length(dim(X.lin.s))

    if(is.null(X.nn.s) & is.null(X.add.basis.s )){
      linBranchs <- input_lin_s%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X.lin.s)[-1], name = 'lin_s',
                    weights=list(matrix(0,nrow=dim(X.lin.s)[n.dim.lin_s],ncol=1),array(init.spread)),use_bias=T)
    }else{
      linBranchs <- input_lin_s%>%
        layer_dense(units = 1, activation = 'linear',
                    input_shape =dim(X.lin.s)[-1], name = 'lin_s',
                    weights=list(matrix(0,nrow=dim(X.lin.s)[n.dim.lin_s],ncol=1)),use_bias=F)
    }
  }

  
  #Stationary towers
  
  #Location
  if(is.null(X.nn.q) & is.null(X.add.basis.q) & is.null(X.lin.q)) {
  
      statBranchq <- input_u %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(u)[-1], trainable=F,
                                                weights=list(matrix(0,nrow=dim(u)[length(dim(u))],ncol=1),array(1,dim=c(1))), name = 'q_stationary_dense1') %>%
        layer_dense(units = 1 ,activation = 'linear',use_bias = F,weights=list(matrix(array(init.loc),nrow=1,ncol=1)), name = 'q_stationary_dense2')
   
    
  }
  
  #Spread
  if(is.null(X.nn.s) & is.null(X.add.basis.s) & is.null(X.lin.s)) {
    
    
      statBranchs <- input_u %>% layer_dense(units = 1 ,activation = 'relu', input_shape =dim(u)[-1], trainable=F,
                                                weights=list(matrix(0,nrow=dim(u)[length(dim(u))],ncol=1),array(1,dim=c(1))), name = 's_stationary_dense1') %>%
        layer_dense(units = 1 ,activation = 'linear',use_bias = F,weights=list(matrix(array(init.spread),nrow=1,ncol=1)), name = 's_stationary_dense2')
   
    
  }
  
  #Combine towers
  
  
  #Location
  if(!is.null(X.nn.q) & !is.null(X.add.basis.q) & !is.null(X.lin.q) )  qBranchjoined <- layer_add(inputs=c(addBranchq,  linBranchq,nnBranchq),name="Combine_q_components")  #Add all towers
  if(is.null(X.nn.q) & !is.null(X.add.basis.q) & !is.null(X.lin.q) )  qBranchjoined <- layer_add(inputs=c(addBranchq,  linBranchq),name="Combine_q_components")  #Add GAM+lin towers
  if(!is.null(X.nn.q) & is.null(X.add.basis.q) & !is.null(X.lin.q) )  qBranchjoined <- layer_add(inputs=c(  linBranchq,nnBranchq),name="Combine_q_components")  #Add nn+lin towers
  if(!is.null(X.nn.q) & !is.null(X.add.basis.q) & is.null(X.lin.q) )  qBranchjoined <- layer_add(inputs=c(addBranchq,  nnBranchq),name="Combine_q_components")  #Add nn+GAM towers
  if(is.null(X.nn.q) & is.null(X.add.basis.q) & !is.null(X.lin.q) )  qBranchjoined <- linBranchq  #Just lin tower
  if(is.null(X.nn.q) & !is.null(X.add.basis.q) & is.null(X.lin.q) )  qBranchjoined <- addBranchq  #Just GAM tower
  if(!is.null(X.nn.q) & is.null(X.add.basis.q) & is.null(X.lin.q) )  qBranchjoined <- nnBranchq  #Just nn tower
  if(is.null(X.nn.q) & is.null(X.add.basis.q) & is.null(X.lin.q) )  qBranchjoined <- statBranchq  #Just stationary tower
  
  #Spread
  if(!is.null(X.nn.s) & !is.null(X.add.basis.s) & !is.null(X.lin.s) )  sBranchjoined <- layer_add(inputs=c(addBranchs,  linBranchs,nnBranchs),name="Combine_s_components")  #Add all towers
  if(is.null(X.nn.s) & !is.null(X.add.basis.s) & !is.null(X.lin.s) )  sBranchjoined <- layer_add(inputs=c(addBranchs,  linBranchs),name="Combine_s_components")  #Add GAM+lin towers
  if(!is.null(X.nn.s) & is.null(X.add.basis.s) & !is.null(X.lin.s) )  sBranchjoined <- layer_add(inputs=c(  linBranchs,nnBranchs),name="Combine_s_components")  #Add nn+lin towers
  if(!is.null(X.nn.s) & !is.null(X.add.basis.s) & is.null(X.lin.s) )  sBranchjoined <- layer_add(inputs=c(addBranchs,  nnBranchs),name="Combine_s_components")  #Add nn+GAM towers
  if(is.null(X.nn.s) & is.null(X.add.basis.s) & !is.null(X.lin.s) )  sBranchjoined <- linBranchs  #Just lin tower
  if(is.null(X.nn.s) & !is.null(X.add.basis.s) & is.null(X.lin.s) )  sBranchjoined <- addBranchs  #Just GAM tower
  if(!is.null(X.nn.s) & is.null(X.add.basis.s) & is.null(X.lin.s) )  sBranchjoined <- nnBranchs  #Just nn tower
  if(is.null(X.nn.s) & is.null(X.add.basis.s) & is.null(X.lin.s) )  sBranchjoined <- statBranchs  #Just stationary tower
  
  #Apply link functions
  if(link.loc=="exp") qBranchjoined <- qBranchjoined %>% layer_activation( activation = 'exponential', name = "q_activation") else if(link.loc=="identity") qBranchjoined <- qBranchjoined %>% layer_activation( activation = 'linear', name = "q_activation")
  sBranchjoined <- sBranchjoined %>% layer_activation( activation = 'exponential', name = "s_activation")

  input=c()
  if(!is.null(X.lin.q) ) input=c(input,input_lin_q)
  if(!is.null(X.add.basis.q) ) input=c(input,input_add_q)
  if(!is.null(X.nn.q) ) input=c(input,input_nn_q)
  if(!is.null(X.lin.s) ) input=c(input,input_lin_s)
  if(!is.null(X.add.basis.s) ) input=c(input,input_add_s)
  if(!is.null(X.nn.s) ) input=c(input,input_nn_s)
  input=c(input,input_u)


  output <- layer_concatenate(c(input_u,qBranchjoined,sBranchjoined, xiBranch),name="Combine_parameter_tensors")

  model <- keras_model(  inputs = input,   outputs = output,name=paste0("bGEV-PP"))
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

bgev_PP_loss <-function(alpha=0.5,beta=0.5,p_a=0.05,p_b=0.2,c1=5,c2=5,n_b=1,S_lambda=NULL){
  
  S_lambda.q=S_lambda$S_lambda.q;   S_lambda.s=S_lambda$S_lambda.s

  
  if(is.null(S_lambda.q) & is.null(S_lambda.s)){
    
loss<- function( y_true, y_pred) {

  library(tensorflow)
  K <- backend()

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
    -(1/n_b)*K$sum(LAM)
  ))
}
  }else if(!is.null(S_lambda.q) & !is.null(S_lambda.s)){
    
    loss<- function( y_true, y_pred) {
      
      library(tensorflow)
      K <- backend()
      
      u=y_pred[all_dims(),1]
      q_a=y_pred[all_dims(),2]
      s_b=y_pred[all_dims(),3]
      xi=y_pred[all_dims(),4]
      
      t.gam.weights.q=K$constant(t(model$get_layer("add_q")$get_weights()[[1]]))
      gam.weights.q=K$constant(model$get_layer("add_q")$get_weights()[[1]])
      S_lambda.q.tensor=K$constant(S_lambda.q)
      
      t.gam.weights.s=K$constant(t(model$get_layer("add_s")$get_weights()[[1]]))
      gam.weights.s=K$constant(model$get_layer("add_s")$get_weights()[[1]])
      S_lambda.s.tensor=K$constant(S_lambda.s)
      
      penalty = 0.5*K$dot(t.gam.weights.q,K$dot(S_lambda.q.tensor,gam.weights.q))+0.5*K$dot(t.gam.weights.s,K$dot(S_lambda.s.tensor,gam.weights.s))
      
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
      return(penalty-(
        K$sum(loglam)
        -(1/n_b)*K$sum(LAM)
      ))
    }
    
    }else if(is.null(S_lambda.q) & !is.null(S_lambda.s)){
      
      loss<- function( y_true, y_pred) {
        
        library(tensorflow)
        K <- backend()
        
        u=y_pred[all_dims(),1]
        q_a=y_pred[all_dims(),2]
        s_b=y_pred[all_dims(),3]
        xi=y_pred[all_dims(),4]
        
      
        t.gam.weights.s=K$constant(t(model$get_layer("add_s")$get_weights()[[1]]))
        gam.weights.s=K$constant(model$get_layer("add_s")$get_weights()[[1]])
        S_lambda.s.tensor=K$constant(S_lambda.s)
        
        penalty = 0.5*K$dot(t.gam.weights.s,K$dot(S_lambda.s.tensor,gam.weights.s))
        
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
        return(penalty-(
          K$sum(loglam)
          -(1/n_b)*K$sum(LAM)
        ))
      }
      
      }else if(!is.null(S_lambda.q) & is.null(S_lambda.s)){
        
        loss<- function( y_true, y_pred) {
          
          library(tensorflow)
          K <- backend()
          
          u=y_pred[all_dims(),1]
          q_a=y_pred[all_dims(),2]
          s_b=y_pred[all_dims(),3]
          xi=y_pred[all_dims(),4]
          
          
          t.gam.weights.q=K$constant(t(model$get_layer("add_q")$get_weights()[[1]]))
          gam.weights.q=K$constant(model$get_layer("add_q")$get_weights()[[1]])
          S_lambda.q.tensor=K$constant(S_lambda.q)
          
          
          penalty = 0.5*K$dot(t.gam.weights.q,K$dot(S_lambda.q.tensor,gam.weights.q))
          
          
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
          return(penalty-(
            K$sum(loglam)
            -(1/n_b)*K$sum(LAM)
          ))
        }
        
      }
    
    

return(loss)

}

