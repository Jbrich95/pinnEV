#' rm(list=ls())
#'  library(keras)#
#'  library(tfprobability)#
#'  reticulate::use_virtualenv("myenv", required = T)##
#'  library(keras)##
#'  sess = k_get_session()#
#'  sess$list_devices()##
#'  try(tfd_beta(1,1),silent=T) #Need to try tfprobability functions - Don't work first time but then work afterwards??#
#'  #If using Rstudio, need to change Python interpreter in global options to the virtual environment "myenv"##
#'  tf$random$set_seed(1)
#'
#'
#' # Build and train a simple MLP for toy data
#'
#' # Create  predictors
#' preds<-rnorm(12800)
#'
#' #Re-shape to a 4d array. First dimension corresponds to observations,
#' #last to the different components of the predictor set
#' dim(preds)=c(20,8,8,10) #We have ten predictors
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
#' m_A_1 = 0.1*X_train_add_q[,,,1]^3+0.2*X_train_add_q[,,,1]-0.1*X_train_add_q[,,,2]^3+0.5*X_train_add_q[,,,2]^2-0.2*X_train_add_q[,,,3]^3
#'
#' #Non-additive contribution - to be estimated by NN
#' m_N_1 = exp(-3+X_train_nn_q[,,,2]+X_train_nn_q[,,,3])+
#' sin(X_train_nn_q[,,,1]-X_train_nn_q[,,,2])*(X_train_nn_q[,,,1]+X_train_nn_q[,,,2])-
#' cos(X_train_nn_q[,,,3]-X_train_nn_q[,,,4])*(X_train_nn_q[,,,2]+X_train_nn_q[,,,5])
#'
#' q_alpha=3+m_L_1+m_A_1+m_N_1 #Identity link
#'
#' #Contribution to scale parameter
#' #Linear contribution
#' m_L_2 = 0.3*X_train_lin_s[,,,1]+0.6*X_train_lin_s[,,,2]+0.1*X_train_lin_s[,,,3]-0.2*X_train_lin_s[,,,4]+0.5*X_train_lin_s[,,,5]
#'
#' # Additive contribution
#' m_A_2 = 0.1*X_train_add_s[,,,1]^2+0.2*X_train_add_s[,,,1]-0.2*X_train_add_s[,,,2]^2+0.1*X_train_add_s[,,,2]^3
#'
#' #Non-additive contribution - to be estimated by NN
#' m_N_2 = exp(-3+X_train_nn_s[,,,2]+X_train_nn_s[,,,3])+
#'   sin(X_train_nn_s[,,,1]-X_train_nn_s[,,,2])*(X_train_nn_s[,,,1]+X_train_nn_s[,,,2])
#'
#' s_beta=exp(-1+m_L_2+m_A_2+m_N_2) #Exponential link
#'
#' xi=0.2 # Set xi
#'
#' theta=array(dim=c(dim(s_beta),3))
#' theta[,,,1]=q_alpha; theta[,,,2] = s_beta; theta[,,,3]=xi
#' Y=apply(theta,1:3,function(x) rnorm(1,mean=x[1],x[2])) #We simulate normal data and estimate the median, i.e., the 50% quantile or mean, as the form for this is known
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
#' #the basis functions for each pre-specified knot and entry to X_train_add_q and X_train_add_s
#'
#' rad=function(x,c){ #Define a basis function. Here we use the radial bases
#'   out=abs(x-c)^2*log(abs(x-c))
#'   out[(x-c)==0]=0
#'   return(out)
#' }
#'
#' n.knot.q = 5; n.knot.s = 4 # set number of knots. Must be the same for each additive predictor, but can differ between the parameters q_\alpha and s_\beta
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
#' for(k in 1:n.knot.q) {
#' X_train_add_basis_q[,,,i,k]= rad(x=X_train_add_q[,,,i],c=knots.q[i,k])
#' #Evaluate rad at all entries to X_train_add_q and for all knots
#' }}
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
#' X_train_q=list("X_train_nn_q"=X_train_nn_q, "X_train_lin_q"=X_train_lin_q,"X_train_add_basis_q"=X_train_add_basis_q) #Predictors for q_\alpha
#' X_train_s=list("X_train_nn_s"=X_train_nn_s, "X_train_lin_s"=X_train_lin_s,"X_train_add_basis_s"=X_train_add_basis_s) #Predictors for s_\beta
#'
#' #We first use train_quant_NN to estimate the 80% quantile whih will be used as u_train for train_bgevPP_NN
#' #A simple MLP with no linear or additive components is used
#'
#' model<-train_quant_NN(Y_train, Y_test,X_train=list("X_train_nn"=preds),  type="MLP",link="identity",tau=0.8,n.ep=1000,
#'                       batch.size=50, widths=c(6,3), seed=1)
#'
#' out<-predict_quant_nn(X_train=list("X_train_nn"=preds),model)
#' u_train <- out$predictions
#' print(mean(Y_train[Y_train> -1e5]>u_train[Y_train> -1e5]))
#'
#' #Fit the bGEV-PP using u_train as output from quantile estimation model
#' model2<-train_bGEVPP_NN(Y_train, Y_test,X_train_q,X_train_s, u_train, type="MLP",link.loc="identity",
#'                                  n.ep=500, batch.size=50,init.loc=2, init.spread=2,init.xi=0.1,
#'                                  widths=c(6,3),seed=1)
#' out2<-predict_bGEVPP_nn(X_train_q=X_train_q,X_train_s=X_train_s,u_train=u_train,model2)
#'
#'   print("q_alpha linear coefficients: "); print(round(out2$lin.coeff_q,2))
#'   print("s_beta linear coefficients: "); print(round(out2$lin.coeff_s,2))
#'
#' #To save model, run
#'
#' model %>% save_model_tf(paste0("model_bGEVPP"))
#' #'#To load model, run
#' model  <- load_model_tf(paste0("model_bGEVPP"), custom_objects=list("bgev_PP_loss_alpha__beta__p_a__p_b_"=bgev_PP_loss()))
#' #Note that bGEV_PP_loss() can take custom alpha,beta, p_a and p_b arguments
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
#'   plt.y=tmp%*%out2$gam.weights_q[i,]
#'   plot(plt.x,plt.y,type="l",main=paste0("q_alpha spline: predictor ",i),xlab="x",ylab="f(x)")
#'   points(knots.q[i,],rep(mean(plt.y),n.knot.q),col="red",pch=2) #Adds red triangles that denote knot locations
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
#'   plt.y=tmp%*%out2$gam.weights_s[i,]
#'   plot(plt.x,plt.y,type="l",main=paste0("s_beta spline: predictor ",i),xlab="x",ylab="f(x)")
#'   points(knots.s[i,],rep(mean(plt.y),n.knot.s),col="red",pch=2) #Adds red triangles that denote knot locations
#'
#' }
#'
