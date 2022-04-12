#' Train a partially-interpretable neural network to fit a logistic regression model
#'
#' Model is fitted by minimisation of binary cross-entropy loss over \code{n.ep} epochs.
#'
#' @param type A string defining the type of network to be built. If \code{type=="MLP"}, the network will have all densely connected layers; if \code{type=="CNN"}, the network will have all convolutional layers.
#' @param Y_train,Y_test A 2 or 3 dimensional array of training or test response values.
#' The first dimension should be the observation indices, e.g., time.
#' If \code{type=="CNN"}, then \code{Y_train} and \code{Y_test} must have three dimensions with the latter two corresponding to an \eqn{M} by \eqn{N} regular grid of spatial locations.
#' If \code{Y_test==NULL}, no validation loss will be computed and the returned model will be that which minimises the training loss over \code{n.ep} epochs.
#'
#' @param X_train_nn A 3 or 4 dimensional array of "non-additive" predictor values. The first three dimensions should be equal to that of \code{Y_train}; the last dimension corresponds to the chosen \eqn{d-l-a} 'non-additive' predictor values.
#' @param X_train_lin A 3 or 4 dimensional array of "linear" predictor values. The first three dimensions should be equal to that of \code{Y_train}; the last dimension corresponds to the chosen \eqn{l} 'linear' predictor values.
#' @param X_train_add_basis A 4 or 5 dimensional array of basis function evaluations for the "additive" predictor values.
#' @param n.ep Number of epochs used for training
#'
#' @return Returns the fitted model which minimises some loss over the specified number of epochs; if \code{!is.null(Y_test)}, minimises the validation loss and minmises the training loss, otherwise.
#'
#'
#' @examples

#'
#' @rdname fit_bern_NN
#' @export

fit_bernoulli_nn=function(Y_train,X_train_nn,X_train_lin,X_train_add_basis, Y_test = NULL, type=c("MLP","CNN"),
                          n.ep=1000)
{

}


