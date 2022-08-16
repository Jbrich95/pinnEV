#' The blended-GEV distribution
#'
#' Distribution function, quantile function and random generation for the blended generalised extreme value (bGEV) distribution with location equal to \code{q_alpha}, spread equal to \code{s_beta} and shape equal to \code{xi}.
#' Note that unlike similar functions in package \code{stats}, these functions accept only scalar inputs, rather than vectors, for the parameters.
#'
#'@param y scalar quantile.
#'@param prob scalar probability.
#'@param q_alpha scalar location parameter.
#'@param s_beta scalar spread parameter.
#'@param xi scalar shape parameter.
#'@param alpha,beta,p_a,p_b,c1,c2 hyper-parameters for the bGEV distribution, see details. Defaults set to those proposed by Castro-Camilo et al. (2021).
#'@param log logical; if \code{TRUE}, probabilities are given as \code{log(prob)}.
#'@param n number of replications.
#'
#'@name bGEV
#'
#'
#' @details{
#' The GEV distribution function for real location \eqn{\mu} and scale \eqn{\sigma>0} is
#'\deqn{G(y|\mu,\sigma,\xi)=\exp[-\{1+\xi(y-\mu)/\sigma\}_+^{-1/\xi}]} for \eqn{\xi>0} and
#'  \deqn{G(y|\mu,\sigma,\xi)=\exp\{-\exp(-(y-\mu)/\sigma)\}} for \eqn{\xi=0}, where \eqn{\{x\}_+=\max\{0,x\}}. It can be re-parameterised in terms of
#'  a location parameter \eqn{q_\alpha} for \eqn{\alpha\in(0,1)}, denoting the GEV \eqn{\alpha}-quantile, and a spread
#'  parameter \eqn{s_\beta=q_{1-\beta/2}-q_{\beta/2}} for \eqn{\beta\in(0,1)}. This
#'is achieved using the following one-to-one mapping; if \eqn{\xi>0}, then
#'  \deqn{\mu=q_\alpha-s_\beta(l_{\alpha,\xi}-1)/(l_{1-\beta/2,\xi}-l_{\beta/2,\xi})} and
#'  \deqn{\sigma=\xi s_\beta/(l_{1-\beta/2,\xi}-l_{\beta/2,\xi})} where \eqn{l_{x,\xi}=(-\log(x))^{-\xi}}; if \eqn{\xi=0}, then
#'  \deqn{\mu=q_\alpha+s_\beta l_{\alpha}/(l_{\beta/2}-l_{1-\beta/2})} and
#'  \deqn{\sigma=s_\beta/(l_{\beta/2}-l_{1-\beta/2})} where \eqn{l_{x}=\log(-\log(x))}.
#'
#'
#' By Castro-Camilo et al. (2021), the blended-GEV has distribution function
#' \deqn{ F(y|q_\alpha,s_\beta,\xi,a,b)=G(y|\tilde{q}_\alpha,\tilde{s}_\beta,\xi=0)^{1-p(y;a,b)}G(y|_\alpha,s_\beta,\xi)^{p(y;a,b)},}
#' for real \eqn{q_\alpha}, \eqn{s_\beta>0} and \eqn{\xi>0}. The weight function \eqn{p} is defined by \eqn{p(y;a,b)=F_{beta}((y-a)/(b-a)|c_1,c_2),} the distribution function
#' of a beta random variable with shape parameters \eqn{c_1>3,c_2>3}. For continuity of \eqn{G}, we set \eqn{a=G^{-1}(p_a|q_\alpha,s_\beta,\xi)} and
#' \eqn{b=G^{-1}(p_b|q_\alpha,s_\beta,\xi)}
#'for small \eqn{0<p_a<p_b<1} and let \eqn{\tilde{q}_\alpha=a-(b-a)(l_\alpha-l_{p_a})/(l_{p_a}-l_{p_b})} and
#'\eqn{\tilde{s}_\beta=(b-a)(l_{\beta/2}-l_{1-\beta/2})/(l_{p_a}-l_{p_b})}.
#'}
#' @return{
#' \code{pbGEV} gives the distribution function; \code{qbGEV} gives the quantile function; \code{rbGEV} generates random deviates.
#' }
#'
#'
#'@references
#' Castro-Camilo, D., Huser, R., and Rue, H. (2021), \emph{Practical strategies for generalized extreme value-based regression models for extremes}, Environmetrics, e274.
#' (\href{https://doi.org/10.1002/env.2742}{doi})
#'


#'
#' @rdname bGEV
#' @export
#'
 pbGEV=function(y,q_alpha,s_beta,xi,alpha=0.5,beta=0.5,p_a=0.05,p_b=0.2,c1=5,c2=5,log=F){


a=Finverse_r(p_a,q_alpha,s_beta,xi,alpha,beta)
b=Finverse_r(p_b,q_alpha,s_beta,xi,alpha,beta)

#Upper tail
z1=(y-q_alpha)/(s_beta/(l_r(1-beta/2,xi)-l_r(beta/2,xi)))+l_r(alpha,xi)
z1=max(z1,0)

t1=(z1)^(-1/xi)

#Weight


p= pbeta((y-a)/(b-a),c1,c2)

#Lower tail
q_a_tilde=a-(b-a)*(l0_r(alpha)-l0_r(p_a))/(l0_r(p_a)-l0_r(p_b))
s_b_tilde=(b-a)*(l0_r(beta/2)-l0_r(1-beta/2))/(l0_r(p_a)-l0_r(p_b))

z2=(y-q_a_tilde)/(s_b_tilde/(l0_r(beta/2)-l0_r(1-beta/2)))-l0_r(alpha)


t2=exp(-z2)

if(log == F){
  out=exp(p*(-t1)+(1-p)*(-t2))
  if(p==0){out = exp((1-p)*(-t2))}else if(p==1){out=exp(p*(-t1))}
}else if(log==T){
  out=p*(-t1)+(1-p)*(-t2)
  if(p==0){out = (1-p)*(-t2)}else if(p==1){out=p*(-t1)}
}
return(out)
}

 #'@import stats
 
 #' @rdname bGEV
 #' @export
 #'
qbGEV=function(prob,q_alpha,s_beta,xi,alpha=0.5,beta=0.5,p_a=0.05,p_b=0.2,c1=5,c2=5){

  if(prob < p_b){
    func=function(x){
      prob-pbGEV(x,q_alpha,s_beta,xi,alpha,beta,p_a,p_b,c1,c2)
    }
    return( uniroot(f=func,interval=c(-1e8,1e8),tol=1e-5 )$root)
  }else{
    #Not using tensors

    return(Finverse_r(prob,q_alpha,s_beta,xi,alpha,beta))

  }
}


l_r=function(a,xi){

  (-log(a))^(-xi)
}
l0_r = function(a){


  log(-log(a))
}

Finverse_r = function(x,q_alpha,s_beta,xi,alpha,beta){


  ( (-log(x))^(-xi)-l_r(alpha,xi))*s_beta/(l_r(1-beta/2,xi)-l_r(beta/2,xi))+q_alpha
}

#' @rdname bGEV
#' @export
#'
rbGEV=function(n,q_alpha,s_beta,xi,alpha=0.5,beta=0.5,p_a=0.05,p_b=0.2,c1=5,c2=5){
    out <- rep(0,n)
    for(i in 1:n) out[i]=qbGEV(runif(1),q_alpha,s_beta,xi,alpha,beta,p_a,p_b,c1,c2)
    return(out)

}
