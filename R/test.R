# library(keras)
# library(tfprobability)
# reticulate::use_virtualenv("myenv", required = T)
#
# library(keras)
#
# sess = k_get_session()
# sess$list_devices()
#
# try(tfd_beta(1,1),silent=T) #Need to try tfprobability functions - Don't work first time but then work afterwards??
# #If using Rstudio, need to change Python interpreter in global options to the virtual environment "myenv"
#
# tf$random$set_seed(1)
# ##Build and train a simple MLP for toy data
# X_train_nn<-rnorm(5000); X_train_add<-rnorm(2000); X_train_lin<-rnorm(3000) # Create 'nn', 'additive' and 'linear' predictors
#
# dim(X_train_nn)=c(10,10,10,5) #Re-shape to a 4-d array. First dimension corresponds to observations, last to the different components of the predictor set
# dim(X_train_lin)=c(10,10,10,3)
# dim(X_train_add)=c(10,10,10,2)
#
# #Create toy response data
# m_L=0.3*X_train_lin[,,,1]+0.6*X_train_lin[,,,2]-0.2*X_train_lin[,,,3] # Linear contribution
# m_A=0.1*X_train_add[,,,1]^2+0.2*X_train_add[,,,1]-0.1*X_train_add[,,,2]^3+0.5*X_train_add[,,,2]^2 # Additive contribution
# m_N=exp(-3+X_train_nn[,,,2]+X_train_nn[,,,3])+sin(X_train_nn[,,,1]-X_train_nn[,,,2])*(X_train_nn[,,,1]+X_train_nn[,,,2]) #Non-additive contribution - to be estimated by NN
#
# p=0.5+0.5*tanh((m_L+m_A+m_N)/2) #Logistic link
# Y=apply(p,1:3,function(x) rbinom(1,1,x))
#
# #Create training and test, respectively. We mask 20% of the Y values and use this for validation/testing. Masked values must be set to -1e5 and are treated as missing whilst training
#
# mask_inds=sample(1:length(Y),size=length(Y)*0.8)
#
# Y_train<-Y_test<-Y #Create training and test, respectively.
# Y_train[-mask_inds]=-1e5
# Y_test[mask_inds]=-1e5
#
#
#
# To build a model with an additive component, we require an array of evaluations of the basis functions for each pre-specified knot and entry to X_train_add
#
# rad=function(x,c){ #Define a basis function. Here we use the radial bases
# out=abs(x-c)^2*log(abs(x-c))
# out[(x-c)==0]=0
# return(out)
# }
#
# n.knot = 5 # set number of knots. Must be the same for each additive predictor
# knots=matrix(nrow=dim(X_train_add)[4],ncol=n.knot)
#
# #We set knots to be equally-spaced marginal quantiles
# for( i in 1:dim(X_train_add)[4]) knots[i,]=quantile(X_train_add[,,,i],probs=seq(0,1,length=n.knot))
#
# X_train_add_basis<-array(dim=c(dim(X_train_add),n.knot))
# for( i in 1:dim(X_train_add)[4]) for(k in 1:n.knot)  X_train_add_basis[,,,i,k]= rad(x=X_train_add[,,,i],c=knots[i,k]) #Evaluate rad at all entries to X_train_add and for all knots
#
#
# X_train=list("X_train_nn"=X_train_nn, "X_train_lin"=X_train_lin,"X_train_add_basis"=NULL)
#
# #Build and train a two-layered "lin+GAM+NN" MLP
# model<-fit_bernoulli_nn(Y_train, Y_test,X_train,  type="MLP",n.ep=2000, batch.size=50,init.p=0.4, widths=c(6,3))
#
# out<-predict_bernoulli_nn(X_train,model)
# print(out$lin.coeff)
