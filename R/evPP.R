#' The extreme value point process
#'
#' Quantile function and random generation for the extreme value point process with location equal to \code{loc}, scale equal to \code{scale} and shape equal to \code{xi >= 0}.
#' Note that unlike similar functions in package \code{stats}, these functions accept only scalar inputs, rather than vectors, for the parameters.
#'
#'@param prob scalar probability.
#'@param loc  location parameter. If \code{re.par==FALSE}, then \code{loc} corresponds to \eqn{\mu}; otherwise, \code{loc} corresponds to \eqn{q_\alpha}.
#'@param scale  scale parameter. If \code{re.par==FALSE}, then \code{scale} corresponds to \eqn{\sigma}; otherwise, \code{scale} corresponds to \eqn{s_\beta}.
#'@param xi shape parameter. Require \code{xi >= 0}.
#'@param u.prob exceedance probability for threshold \eqn{u}.
#'@param alpha,beta hyper-parameters for the reparameterisation, see details. Defaults set both to \code{0.5}. Only used if \code{re.par==TRUE}.
#'@param n_b number of observations per block, e.g., if observations correspond to months and the interest is annual maxima, then \code{n_b=12}. See details.
#'@param tol tolerance for the numerical solver. Defaults to \code{1e-4}.
#'@param re.par logical; if \code{TRUE}, then the corresponding GEV used the alternative parameterisation.
#'@param qMax finite upper and lower bounds used by the numerical solver. If the absolute value of the output from \code{qPP} or maximum output from \code{rPP} is close to \code{qMax}, then \code{qMax} needs increasing at the cost of computation time. Defaults to \code{1e4}.
#'@param n number of replications.
#'
#'@name evPP

#'@details
#'Following Coles (2001), consider a sequence of independent random variables \eqn{Y_1,\dots,Y_n} with common distribution function \eqn{F}. For \eqn{n_b}-block-maxima \eqn{M_{n_b}=\max\{Y_1,\dots,Y_{n_b}\}};
#'if there exists sequences \eqn{\{a_{n} >0\}} and \eqn{\{b_{n}\}} such that
#'\deqn{\Pr\{(M_{n_b}-b_n)/a_n \leq z \}\rightarrow G(z)\;\; \mbox{as}\;\; n_b\rightarrow \infty,}
#'for non-degenerate \eqn{G}, then \eqn{G} is the generalised extreme value GEV\eqn{(\mu,\sigma,\xi)} distribution function, see \code{help{pbGEV}}.
#'If \eqn{\xi >0}, then \eqn{G} has finite lower-endpoint \eqn{z_-=\mu-\sigma/\xi}; if \eqn{\xi=0}, then the lower-endpoint is infinite.
#'
#'Assume that the above limit holds and \eqn{\xi \geq 0}. Then for any \eqn{u>z_-}, the sequence of point processes \eqn{N_n=\{(i/(n+1),(Y_i-b_n)/a_n): i =1,\dots,n\}}
#' converges on regions \eqn{(0,1)\times(u,\infty)} as \eqn{n\rightarrow \infty} to a Poisson point process with intensity measure
#'\eqn{\Lambda} of the form \eqn{\Lambda(A)=-(n/n_b)(t_2-t_1)\log G(z)}, where \eqn{A=[t_1,t_2]\times [z,\infty)}
#'for \eqn{ 0\leq t_1 \leq t_2 \leq 1}. We consider unit inter-arrival times and so set \eqn{t_2-t_1=1}. Here the functions \code{qPP} and \code{rPP} give the quantile function and random generation of \eqn{Y} assuming that
#'the Poisson process limit holds for \eqn{Y} above \eqn{u}. The threshold \eqn{u} is taken to be the \code{u.prob} quantile of \eqn{Y}.
#'
#'Castro-Camilo et al. (2021) propose a reparameterisation of the GEV distribution in terms of
#'  a location parameter \eqn{q_\alpha} for \eqn{\alpha\in(0,1)}, denoting the GEV \eqn{\alpha}-quantile, and a spread
#'  parameter \eqn{s_\beta=q_{1-\beta/2}-q_{\beta/2}} for \eqn{\beta\in(0,1)}; for the full mapping, see \code{help{pbGEV}}.
#'  If \code{re.par==TRUE}, then the input \code{loc} and \code{scale} correspond to \eqn{q_\alpha} and \eqn{s_\beta}, rather than \eqn{\mu} and \eqn{\sigma}.
#'
#' Distribution function inversion is performed numerically using the bisection method.
#'
#'
#' @return{
#' \code{qPP} gives the quantile function and \code{rPP} generates \code{n} random deviates. Any simulated values subceding the threshold \code{u} are treated as censored and set to \code{NA}.
#'
#'
#' }
#'
#'@examples
#'
#'set.seed(1)
#'loc<-3; scale<-4; xi<-0.2 #Parameter values
#'
#'u<-qPP(prob=0.9,loc,scale,xi) #Gives the 90% quantile of Y
#'
#'#Create 1000 realisations of Y with exceedance threshold equal to u.
#'#Note that the input to rPP is the exceedance probability u.prob, not the threshold itself
#'Y<-rPP(1000,u.prob=0.9,loc,scale,xi)
#'hist(Y)
#'#Note that values Y<u are censored and set to NA
#'
#'
#'
#'
#'

#'
#'@references Coles, S. G. (2001), \emph{An Introduction to Statistical Modeling of Extreme Values}. Volume 208, Springer.
#' (\href{https://doi.org/10.1007%2F978-1-4471-3675-0}{doi})
#'
#' Castro-Camilo, D., Huser, R., and Rue, H. (2021), \emph{Practical strategies for generalized extreme value-based regression models for extremes}, Environmetrics, e274.
#' (\href{https://doi.org/10.1002/env.2742}{doi})



#' @rdname evPP
#' @export
#'
qPP=function(prob,loc,scale,xi,n_b=1,re.par=F,alpha=0.5,beta=0.5,tol=1e-4,qMax=1e4){

  if(!re.par){ mu = loc; sig = scale }else if(re.par & xi > 0){
    mu=loc-scale*(l_r(alpha,xi)-1)/(l_r(1-beta/2,xi)-l_r(beta/2,xi)); sig=xi*scale/((l_r(1-beta/2,xi)-l_r(beta/2,xi)))
  }else if(re.par & xi == 0){
    mu=loc+scale*(l0_r(alpha))/(l0_r(beta/2)-l0_r(1-beta/2)); sig=scale/((l0_r(beta/2)-l0_r(1-beta/2)))

  }

  a=-qMax
  b=qMax


  diff=100
  while( diff > tol){
    if(1+xi*((a+b)/2-mu)/sig <= 0){bsup=b;binf=(a+b)/2}else{
      if(dpois(0,Lam_PP((a+b)/2,mu,sig,xi,n_b)) <prob){binf=(a+b)/2;bsup=b}
      if(dpois(0,Lam_PP((a+b)/2,mu,sig,xi,n_b))>=prob){bsup=(a+b)/2;binf=a}
    }
    a=binf
    b=bsup
    diff = b-a
  }
  return((a+b)/2)
}

#' @rdname evPP
#' @export
rPP=function(n,u.prob,loc,scale,xi,n_b=1,re.par=F,alpha=0.5,beta=0.5,tol=1e-4,qMax=1e4){

  if(!re.par){ mu = loc; sig = scale }else if(re.par & xi > 0){
    mu=loc-scale*(l_r(alpha,xi)-1)/(l_r(1-beta/2,xi)-l_r(beta/2,xi)); sig=xi*scale/((l_r(1-beta/2,xi)-l_r(beta/2,xi)))
  }else if(re.par & xi == 0){
    mu=loc+scale*(l0_r(alpha))/(l0_r(beta/2)-l0_r(1-beta/2)); sig=scale/((l0_r(beta/2)-l0_r(1-beta/2)))

  }
  U<-as.matrix(runif(n))
  U[U<u.prob]=NA

  X<-U
  u=qPP(u.prob,loc,scale,xi,n_b,re.par,alpha,beta,tol,qMax)
  for(j in which(!is.na(U))){
    exceedance=Ftinv(U[j],mu,sig,xi,u,Tmax=qMax-u,n_b=n_b,tol)
    X[j]= exceedance+u

  }
  return(X)
}

Lam_PP=function(x,mu,sig,xi,n_b=1){
  if(xi > 0) return(1/n_b*(1+xi*(x-mu)/sig)^(-1/xi)) else if(xi == 0) return(1/n_b*exp(-(x-mu)/sig))
}

l_r=function(a,xi){

  (-log(a))^(-xi)
}
l0_r = function(a){


  log(-log(a))
}




Ft=function(x,mu,sig,xi,thresh,n_b) 1-exp(Lam_PP(x+thresh,mu,sig,xi,n_b)-Lam_PP(thresh,mu,sig,xi,n_b))+dpois(0,Lam_PP(thresh,mu,sig,xi,n_b))
Ftinv=function(prob,mu,sig,xi,thresh,Tmax,n_b, tol=1e-4){
  a=0
  b=Tmax
  diff=100
  while( diff > tol){


    if(Ft((a+b)/2,mu,sig,xi,thresh,n_b)<=prob){binf=(a+b)/2;bsup=b}
    if(Ft((a+b)/2,mu,sig,xi,thresh,n_b)>=prob){bsup=(a+b)/2;binf=a}

    a=binf
    b=bsup

    diff = b - a
  }
  return((a+b)/2)
}


