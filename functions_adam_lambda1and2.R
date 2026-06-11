library(Matrix)
library(parallel)


#### SETUP

#edges: matrix (i, j), one row per edge (with self loop)
#X: E x N feature matrix
#n: number of nodes (genes)

setup_network <- function(edges, X, n) {
  E <- nrow(edges) #number of edges
  N <- ncol(X) #number of features
  edges <- as.matrix(edges)
  
  #sparse numeric matrix, n x n
  #cell [i, j] = 1 when edge present from gene i to j, and 0 otherwise.
  edge_presence <- sparseMatrix(
    i = edges[, 1], j = edges[, 2], x = 1,
    dims = c(n, n)
  )
  
  list(
    edges = edges,
    X = X,
    edge_presence = edge_presence,
    n = n,
    E = E,
    N = N
  )
}

################################################################################
################################################################################


#### EQUATION 1, ACTIVATION MATRIX A

eq1_activation <- function(w, net) {
  a_edges <- 1 / (1 + exp(-as.numeric(net$X %*% w)))   # the equation, E length numeric vector, entry k= activation strength of edge k, no. 0-1. For eq6
  A <- sparseMatrix(
    i = net$edges[, 1], j = net$edges[, 2], x = a_edges,  # values from a_edges put into a matrix, A[i, j]= activation of edge (i, j)
    dims = c(net$n, net$n)
  )
  output_eq1 <- list(A = A, a_edges = a_edges)
  return(output_eq1)
}


################################################################################
################################################################################


#### EQUATION 2, TRANSITION MATRIX Q
#sparse n x n matrix Q where Q[i, j] = a_ij / r[i]

eq2_transition <- function(A, net) {
  r <- rowSums(A) #total activation from each row (gene)
  Q <- Diagonal(x = 1 /(r)) %*% A  #1/r[i] (1/total activation of each gene) on the diagonal of a matrix. Then divides by A
  return(Q)
}


################################################################################
################################################################################


#### EQUATION 3 RWR
#returns m x n dense matrix.

eq3_rwr <- function(P0, Q, alpha = 0.5, rtol = 1e-5, atol = 1e-8, max_iter = 100) {
  P <- P0 #initialises P to starting position P0
  for (t in seq_len(max_iter)) {   #iteration loop
    P_new <- (1 - alpha) * as.matrix(P %*% Q) + alpha * P0  #the equation
    if (all(abs(P_new - P) <= atol + (rtol * abs(P)))) {  #convergence check, each entry of P tol relative to its scale
      return(P_new) #converged, return current P
    }
    P <- P_new #not converged, update P
  }
  return(P) #hit max_iter, return current P
}


################################################################################
################################################################################


#### EQUATION 4, OBJECTIVE J
#scalar value of J(w, v)

eq4_objective <- function(w, v, P, y, lambda1, lambda2) {
  m <- length(y)   #number of tumours
  resid <- y - as.numeric(P %*% v)  #calculation of residual  
  J <- lambda1 * sum(abs(w)) + lambda2 * sum(abs(v)) + (1 / (2 * m)) * sum(resid^2)   #the equation 
  return(J)
}

################################################################################
################################################################################


#### EQUATION 6 dA/dw

eq6_dA_dw <- function(a_edges, net) {
  logistic_slope <- a_edges * (1 - a_edges) # length E vector, a*(1-a) part of equation for every edge
  dA_dw <- net$X * logistic_slope # E x N, each row of net$X*logistic_slope row (corresponding). The equation.
  return(dA_dw)
}


################################################################################
################################################################################


#### EQUATION 7 dQ/dw_l, NBS2-STYLE

eq7_dQ_dwl <- function(dA_dw_l, A, net) {  # dA_dw_l E length vector that is given by 'solver' later
  dA_dwl_mat <- sparseMatrix( # scatters dA_dw_l in a n x n matrix
    i = net$edges[, 1], j = net$edges[, 2], x = as.numeric(dA_dw_l), # takes the corresponding i,j values from edges, and puts the dA_dw_l values in place
    dims = c(net$n, net$n))
  r <- rowSums(A) # total activation from gene i
  r_dA <- rowSums(dA_dwl_mat) # total rate of change if the activation from gene i wrt feature l
  num <- dA_dwl_mat * r - A * r_dA # equation numerator
  dQ_dwl <- num / (r^2)  # the equation
  return(dQ_dwl)
}


################################################################################
################################################################################


#### EQUATION 9 dP/dw_l

eq9_dP_dwl <- function(P, Q, dQ_dwl, alpha = 0.5, rtol = 1e-5, atol = 1e-8, max_iter = 100) {
  m <- nrow(P) # number of tumours
  n <- ncol(P) # number of genes
  dP <- matrix(0, m, n) # initialise gradient matrix to 0
  P_dQ <- as.matrix(P %*% dQ_dwl) # constant inside loop, computed once outside loop
  for (t in seq_len(max_iter)) {
    dP_new <- (1 - alpha) * (as.matrix(dP %*% Q) + P_dQ)  # equation
    if (max(abs(dP_new - dP) - rtol * abs(dP)) <= atol) { # tolerance check
      return(dP_new) # if converged, return this
    }
    dP <- dP_new   # not converged, update for next iteration
  }
  return(dP)  # hit max_iter, return current dP
}


################################################################################
################################################################################


#### EQUATION 10 dJ/dw_l

eq10_dJ_dwl <- function(w_l, dP_dwl, v, y, P, lambda1) {                         
  m <- length(y) #number of tumours 
  resid <- y - as.numeric(P %*% v)    #same as in eq4
  dJ_dwl <- sign(w_l) * lambda1 - (1 / m) * as.numeric(t(dP_dwl %*% v) %*% resid) #the equation
  return(dJ_dwl)
}


################################################################################
################################################################################


compute_dJ_dw <- function(w, v, A, Q, P, y, net, dA_dw, alpha, lambda1, rtol, atol, max_iter, n_cores = 1) { #ncores can be changed if testing only this function
  solver <- function(l) {   
    dA_dw_l <- dA_dw[, l]   #E length vector, lth feature column (also used in eq7)
    dQ_dwl <- eq7_dQ_dwl(dA_dw_l, A, net) #eq7
    dP_dwl <- eq9_dP_dwl(P, Q, dQ_dwl, alpha, rtol, atol, max_iter) #eq9
    dJ_dwl <- eq10_dJ_dwl(w[l], dP_dwl, v, y, P, lambda1) #eq10
    return(dJ_dwl)
  }
  
  if (n_cores > 1 && .Platform$OS.type != "windows") { #parallel on Mac/Linux
    grads <- mclapply(seq_len(net$N), solver, mc.cores = n_cores)
  } else {   #serial for Windows or n_cores=1, mclapply only works on mac or linux. look into using one thing that works for all (Parlapply).
    grads <- lapply(seq_len(net$N), solver) #serial
  }
  
  dJ_dw <- unlist(grads)
  return(dJ_dw) #list, each element single number, numeric vector N length, entry l is dJ_dwl
}


################################################################################
################################################################################


#### EQUATION 13 dJ/dv  
# length n numeric vector, entry i is dJ/dv_i, goes straight into Adam update for v

eq13_dJ_dv <- function(v, P, y, lambda2) {
  m <- length(y) #number of tumours
  resid <- y - as.numeric(P %*% v) #same as in eq4 and eq10
  dJ_dv <- sign(v) * lambda2 - (1 / m) * as.numeric(t(P) %*% resid) #the equation
  return(dJ_dv)
}


################################################################################
################################################################################


#### EQUATION 15 dJ/dv_i
#scalar form of eq13, only for cross checking
#eq13 will be used - computes all gradients at once, faster than calling eq15 n times

eq15_dJ_dvi <- function(i, v, P, y, lambda2) {
  m <- length(y)  #number of tumours
  resid <- y - as.numeric(P %*% v)   #same as in eq4, eq10, eq13
  dJ_dvi <- sign(v[i]) * lambda2 - (1 / m) * as.numeric(t(P[, i]) %*% resid) #the equation. Same concept, just for single genes
  return(dJ_dvi)
}


################################################################################
################################################################################


#### ADAM

# initialisation, called once at the start separately for w and v e.g. state_w <- init_adam(net$N)

init_adam <- function(size) { # the size of whats being optimised, w (N) and v (n)
  state <- list(m = rep(0, size), v = rep(0, size), t = 0L) # m: first moment estimate, v:second moment estimate. both set to 0. t is timestep counter.
  return(state)
}

### Adam update step, instead of eq12, 14

# theta: parameter vector being updated (w or v), grad: eq 10/13, state: from init_adam

adam_step <- function(theta, grad, state, eta = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
  state$t <- state$t + 1L # counts how many updates have happened
  state$m <- beta1 * state$m + (1 - beta1) * grad  # update m, state$m becomes a smoothed version of the gradient
  state$v <- beta2 * state$v + (1 - beta2) * grad^2  # same for v
  m_hat <- state$m / (1 - beta1^state$t) # bias correction, since states start at 0, MAs are biased toward 0 (early iterations).
  v_hat <- state$v / (1 - beta2^state$t)   # correction divides by (1 - beta^t) to undo this. Imp when t is small
  theta <- theta - eta * m_hat / (sqrt(v_hat) + eps) # Adam update, each theta entry gets a learning rate based on gradient history
  out <- list(theta = theta, state = state)
  return(out) # return updated parameter vector and updated state for next iteration
}


################################################################################
################################################################################


rwr_L1and2_train <- function(P0, edges, X, y,
                                 alpha = 0.5, lambda1 = 0.01, lambda2 = 0.01,
                                 eta = 0.001, beta1 = 0.9, beta2 = 0.999,
                                 eps = 1e-8, max_outer = 500, conv_threshold_outer = 1e-5,
                                 rtol = 1e-5, atol = 1e-8, max_iter = 100,
                                 n_cores = max(1, detectCores() - 1)) {
  n <- ncol(P0) # set n (number of genes) from P0
  net <- setup_network(edges, X, n) # build net
  w <- rep(0, net$N) # initialise feature weights to 0
  v <- rep(0, n) # initialise gene weights to 0
  
  state_w <- init_adam(net$N)  # Adam states
  state_v <- init_adam(n)
  
  J_history <- numeric(max_outer) # J_history set to max outer iterations, for tracking
  
  for (iter in seq_len(max_outer)) {
    act <- eq1_activation(w, net)  # forward pass, eq1
    A <- act$A # eq1
    a_edges <- act$a_edges # eq1
    Q <- eq2_transition(A, net) # eq2
    P <- eq3_rwr(P0, Q, alpha, rtol, atol, max_iter) # eq3
    
    J_history[iter] <- eq4_objective(w, v, P, y, lambda1, lambda2) # record current J for this iter
    
    dA_dw <- eq6_dA_dw(a_edges, net) # eq6 using a_edges, E x N grad. of activation for this iteration
    
    # parallel computation of eq7, 9, 10 
    dJ_dw <- compute_dJ_dw(w, v, A, Q, P, y, net, dA_dw, alpha, lambda1,
                           rtol, atol, max_iter, n_cores = n_cores)
    dJ_dv <- eq13_dJ_dv(v, P, y, lambda2)  # eq13 
    
    upd_w <- adam_step(w, dJ_dw, state_w, eta, beta1, beta2, eps) # adam updates for w
    upd_v <- adam_step(v, dJ_dv, state_v, eta, beta1, beta2, eps) # adam updates for v
    w_new <- upd_w$theta # new w stored as updated theta
    state_w <- upd_w$state # state updated and saved for next iter
    v_new <- upd_v$theta # same for v
    state_v <- upd_v$state
    
    delta <- max(abs(w_new - w), abs(v_new - v)) # parameter convergence check, takes largest value so both have to converge for loop to stop
    w <- w_new # rename w (overwriting for next iter or result)
    v <- v_new # rename v
    
    if (delta < conv_threshold_outer) { # convergence check, if BOTH converged:
      J_history <- J_history[seq_len(iter)] # crop the J_history to the num of iterations that have happened
      break
    }
  }
  
  result <- list( # if not converged, continue until max_outer
    w = w,
    v = v,
    A = A,
    Q = Q,
    P = P,
    J_history = J_history,
    iterations = length(J_history),
    lambda1 = lambda1,
    lambda2 = lambda2,
    net = net
  )
  return(result)
}