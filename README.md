------------------------------------------------------------------------

editor_options: markdown: wrap: 72 ---

# Title

##### By Una Milovanovic

##### 15/05/2026

## Introduction

## Algorithm Summary

## **Repository Contents**

- **functions_and_adam.R:** Main implementation of the algorithm; defines all of the functions and the main wrapper.

- **testing_functions_and_adam.R:** Tests the application of the functions.

- **README.md:** This file; a description of the background and contents for further use.

- ***Gradient descent algorithm equation sheet?***

## Requirements

- **R:** Version 4.0.0 or later

- **R Packages:**

  - Matrix

  - parallel

- **Operating system:** macOS or Linux is required for parallel execution. If a Windows device is used, the code will automatically use serial execution.

## Installation

There are no prior installation steps required, only loading of the listed R packages.

## Quick Start

``` R
source("functions_and_adam.R")

#Inputs:
#P0: m x n binary mutation matrix (tumours x genes)
#edges: E x 2 integer matrix of edge endpoints
#X: E x N numeric matrix of edge features 
#y: length-m numeric vector of tumour phenotypes (eg growth rates)

result <- rwr_lasso_train(
  P0 = P0,
  edges = edges,
  X = X,
  y = y,
  alpha = 0.5,    
  lambda = 0.01,   
  eta = 0.001,  
  max_outer = 500
)

#Results
result$w  #learned feature weights (length N)
result$v  #learned gene weights (length n)

plot(result$J_history, type = "l",
     xlab = "Iteration", ylab = "Objective J")
```

## Inputs

- **P0** (numeric matrix, m x n): A binary mutation profile where the rows are tumours, and the columns are genes. 1 = mutant, 0 = wild type.

- **edges** (integer matrix, E x 2): Each row is an edge (i, j), where i and j are node indices. Self loops (i, i) must be included.

- **X** (numeric matrix, E x N): The edge feature matrix where row k corresponds to the kth edge in **edges**. The self loop indicator should be one of the (N) features.

- **y** (numeric vector, length m): The continuous phenotype to be predicted (tumour growth rate). There is one value per tumour.

## Outputs

- **w** (numeric vector, length N): The learned edge weights.

- **v** (numeric vector, length n): The learned gene weights.

- **A** (sparse matrix, n x n): The final activation matrix (refer to Equation 1).

- **Q** (sparse matrix, n x n): The final transition matrix (refer to Equation 2).

- **P** (numeric matrix, m x n): The final propagated mutation profile (refer to Equation 3).

- **J_history** (numeric vector): Objective function value at each iteration. Used mainly for checking convergence.

- **iterations** (integer): Number of iterations run by the program.

- **net** (list): The internal network list by **setup_network**

## Function Reference

Each equation corresponds to one function. The wrapper function **rwr_lasso_train** is the only function that would be called.

**Table 1 \| Summary function table**

| Function | Purpose | Inputs | Outputs |
|--------------|-----------------|------------------------------|-------------|
| **setup_network** | Pre-computation | edges, X, n | Net bundle |
| **eq1_activation** | Eq. 1 | w, net | A |
| **eq2_transition** | Eq. 2 | A, net | Q |
| **eq3_rwr** | Eq. 3 | P0, Q, alpha, conv_threshold, max_iter | P |
| **eq4_objective** | Eq. 4 | w, v, P, y, lambda | J |
| **eq6_dA_dwl** | Eq. 6 | l, A, net | dA/dwl |
| **eq7_dQ_dwl** | Eq. 7 | dA, A | dQ/dwl |
| **eq9_dP_dwl** | Eq. 9 | P, Q, dQ_dwl, alpha, conv_threshold, max_iter | dP/dwl |
| **eq10_dJ_dwl** | Eq. 10 | wl, dP, v, y, P, lambda | dJ/dwl |
| **eq13_dJ_dv** | Eq. 13 | v, P, y, lambda | dJ/dv |
| **compute_dJ_dw** | Combines Eq. 6,7,9,10 | All of the above | dJ/dw |
| **init_adam** | Adam initialisation | Parameter size | state list with m, v, t |
| **adam_step** | Adam update | Parameters, gradient, state | updated theta and state |
| **rwr_lasso_train** | Main training loop | All data inputs | Results list |

#### 1. Setting up the Network

``` r
setup_network <- function(edges, X, n) {
  E <- nrow(edges)
  N <- ncol(X)
  edges <- as.matrix(edges)
  
  edge_presence <- sparseMatrix(
    i = edges[, 1], j = edges[, 2], x = 1,
    dims = c(n, n)
  )
  
  X_l_mat <- lapply(seq_len(N), function(l) {                                   
    sparseMatrix(
      i = edges[, 1], j = edges[, 2], x = X[, l],
      dims = c(n, n)
    )
  })
  
  list(
    edges = edges,
    X = X,
    X_l_mat = X_l_mat,
    edge_presence = edge_presence,
    n = n,
    E = E,
    N = N
  )
}
```

The setup function runs once before training, to build the sparse representations of the network so that the following functions don't need to recompute them for every iteration.

- **edge_presence** (sparse numeric matrix, n x n): Binary matrix where 1 = edge present and 0 = no edge present. Takes vectors of row indices (**i**), column indices (**j**) and values (**x**), and builds the sparse matrix.

- **X_l_mat** (list of N sparse matrices): For each feature column **l** of **X**, it scatters into a n x n matrix. Considers edge presence, **X_l_mat[[1]][i, j] = x_ijl** at edge positions, 0 elsewhere.

- **The returned list**: places everything into one object so **net** can be used moving forward (refer to Outputs section), instead of separate arguments.

#### 2. Equation 1 - Activation Matrix A

``` r
eq1_activation <- function(w, net) {                                            
  a_edges <- 1 / (1 + exp(-as.numeric(net$X %*% w)))                            
  sparseMatrix(                                                                 
    i = net$edges[, 1], j = net$edges[, 2], x = a_edges,
    dims = c(net$n, net$n)
  )
}
```

This step calculates the activation score for every edge, given the (current) feature weights **w**, and scatters these into an n x n matrix, with **A** activations at edge features.

#### 3. Equation 2 - Transition Matrix Q

``` r
 eq2_transition <- function(A, net) {
  r <- rowSums(A)                                                               
  Diagonal(x = 1 / r) %*% A                                                    
}
```

This step row-normalises **A** in order to produce a transition matrix **Q**, where each of the rows sums to 1.

- **r** (numeric vector, length n): Entry i is Σ_k a_ik, the row sum of row **i**.

#### 4. Equation 3 - Random Walk with Restart (RWR)

``` r
eq3_rwr <- function(P0, Q, alpha = 0.5, conv_threshold = 1e-6, max_iter = 100) { 
  P <- P0                                                                       
  for (t in seq_len(max_iter)) {                                                
    P_new <- (1 - alpha) * as.matrix(P %*% Q) + alpha * P0                      
    if (max(abs(P_new - P)) < conv_threshold) {                                 
      return(P_new)
    }
    P <- P_new                                                                  
  }
  P                                                                             
}
```

This step is iterative, it runs the RWR until **P** stops changing.

First, **P0** initialises **P** to **P**$^{0}$. Then, the iteration 'for' loop begins, and **P_new** is calculated through the RWR update equation. The convergence of **P** is assessed, if the difference between this **P_new** value and **P** is below the pre-set convergence threshold, convergence has occurred, and the value is returned. Otherwise, **P_new** is renamed to **P**, and the loop begins again. The loop will continue until either convergence is reached, or the preset maximum number of iterations has been hit.

#### 5. Equation 4 - Objective Function J

``` r
eq4_objective <- function(w, v, P, y, lambda) {
  m <- length(y)                                                                
  resid <- y - as.numeric(P %*% v)                                              
  lambda * sum(abs(w)) + lambda * sum(abs(v)) + (1 / (2 * m)) * sum(resid^2)    
}
```

This step evaluates the LASSO objective function **J**.

- **m** represents the number of tumours.

#### 6. Equation 5 - Objective Function J in Summative Form

This function is identical to Equation 4, in summation form. It is written with nested loops rather than matrix operations. The outputs of Equation 4 and Equation 5 are the same. This is primarily used for cross-checking. Equation 4 is the preferred version since the loops in Equation 5 are significantly slower.

``` r
eq5_objective_sum <- function(w, v, P, y, lambda) {
  m <- length(y)                                                                
  n <- length(v)                                                                
  total <- 0                                                                    
  for (c in seq_len(m)) {                                                       
    pred <- 0
    for (i in seq_len(n)) {
      pred <- pred + P[c, i] * v[i]
    }
    total <- total + (y[c] - pred)^2
  }
  lambda * sum(abs(w)) + lambda * sum(abs(v)) + (1 / (2 * m)) * total           
}  
```

- **m** represents the number of tumours; **n** represents the number of genes.

The outer loop computes the prediction for tumour **c** by summing **P[c, i] \* v[i]**, over all genes **i**. Then, the squared residual is added to this sum, and renamed **total**. After the loops, the calculation of **J** is done, using the final **total** value.

#### 7. Equation 6 - dA/dwl

``` r
eq6_dA_dwl <- function(l, A, net) {
  Xl <- net$X_l_mat[[l]]                                                        
  XlA <- Xl * A                                                                 
  XlA - XlA * A                                                                 
  #gradient calculation, from eq.6. XlA(1-A) rewritten to XlA-XlA*A (every non edge is 0, not 1)
}
```

This step computes the gradient **dA/dwl**, of the activation matrix **A** with respect to one feature weight **wl**.

It takes the **l**th feature scatter from **X_l_mat**, stored in **net**, and multiplies this with the activation matrix **A**. Next, the result undergoes Equation 6, with a slight alteration to allow for the retention of a sparse matrix.

#### 8. Equation 7 - dQ/dwl

``` r
eq7_dQ_dwl <- function(dA_dwl, A) {              
  r <- rowSums(A)                                                               
  r_dA <- rowSums(dA_dwl)                                                       
  num <- Diagonal(x = r) %*% dA_dwl - Diagonal(x = r_dA) %*% A                  
  Diagonal(x = 1 / (r^2)) %*% num                                               
}
```

This step computes the full gradient **dQ/dwl,** of the transition matrix **Q** with respect to one feature weight **wl**.

- **r** (numeric vector, length n): Entry i is Σ_k a_ik, the row sum of row **i**.

- **r_dA** (numeric vector, length n): Entry i is Σ_k da_ik/dw_l. This is the derivative of **r** with respect to **wl**

Diagonal multiplication is used here to follow the equation since the Diagonal() form is the standard sparse pattern, it allows for preservation of this sparcitiy and is more reliable (for sparse matrices) than simple multiplication.

#### 9. Equation 9 - dP/dWl (Combines Equation 8 and 9)

``` r
eq9_dP_dwl <- function(P, Q, dQ_dwl, alpha = 0.5, conv_threshold = 1e-6, max_iter = 100) {
  m <- nrow(P)                                                                  
  n <- ncol(P)                                                                  
  dP <- matrix(0, m, n)                                                         
  P_dQ <- as.matrix(P %*% dQ_dwl)                                               
  for (t in seq_len(max_iter)) {
    dP_new <- (1 - alpha) * (as.matrix(dP %*% Q) + P_dQ)                        
    if (max(abs(dP_new - dP)) < conv_threshold) {                               
      return(dP_new)
    }
    dP <- dP_new                                                                
  }
  dP                                                                            
}
```

Similar loop construction as for Section 4 (Equation 3). This step iteratively solves for **dP/dwl**, until convergance.

First, **P_dQ** is computed by multiplying **P** and **dQ_dwl**. This is done outside of the loop, since it remains constant throughout the iterations.

An initial **dP** starting matrix is created, and once the loop is entered, **dP_new** is calculated using the equation. The convergence of **dP** is assessed, if the difference between this **dP_new** value and d**P** is below the pre-set convergence threshold, convergence has occurred, and the value is returned. Otherwise, **dP_new** is renamed to **dP**, and the loop begins again. The loop will continue until either convergence is reached, or the preset maximum number of iterations has been hit.

#### 10. Equation 10 - dJ/dwl

``` r
eq10_dJ_dwl <- function(w_l, dP_dwl, v, y, P, lambda) {                         
  m <- length(y)                                                                
  resid <- y - as.numeric(P %*% v)                                              
  sign(w_l) * lambda - (1 / m) * as.numeric(t(dP_dwl %*% v) %*% resid)          
}
```

This step computes the gradient **dJ/dwl**, of **J** with respect to one feature weight **wl**. The output is scalar.

- **m** represents the number of tumours.

- The residual **resid (**y - Pv) is calculated, the same way as in Equation 4.

*Note.* The t() function 'transpose' is used to turn the m × 1 column vector into 1 × m row vector, so that the matrix dimensions are consistent.

#### 11. Equation 13 - dJ/dv

``` r
eq13_dJ_dv <- function(v, P, y, lambda) {
  m <- length(y)                                                                
  resid <- y - as.numeric(P %*% v)                                              
  sign(v) * lambda - (1 / m) * as.numeric(t(P) %*% resid)                       #the equation
}
```

This step computes the full gradient vector **dJ/dv**, of **J** with respect to the gene weight weight **v**, in one matrix operation. A loop is not necessary for this equation as **v** is a vector, and therefore its gradient is also a vector.

- **m** represents the number of tumours.

- **resid**: residual, same calculation as in Equations 4 and 10.

*Note.* The t() function 'transpose' is used to turn the m × 1 column vector into 1 × m row vector, so that the matrix dimensions are consistent.

#### 12. Equation 15 - dJ/dvi

Scalar form of Equation 13.

``` R
eq15_dJ_dvi <- function(i, v, P, y, lambda) {
  m <- length(y)                                                                
  resid <- y - as.numeric(P %*% v)
  sign(v[i]) * lambda - (1 / m) * as.numeric(t(P[, i]) %*% resid)               
}
```

This step computes the gradient **dJ/dv** with respect to gene **vi**. The output is scalar, the **i**th entry of Equation 13.

- **m** represents the number of tumours.

- **resid**: residual, same calculation as in Equations 4, 10 and 13.

This function's purpose is only for cross checking. Equation 13 will be used in general, since it computes all n gradients in one matrix operation, and this is much faster than calling Equation 15 n times.

#### 13. Adam Optimiser 

The Adam optimiser code replaces Equations 12 and 13 from the Gradient Descent Algorithm document. Adam uses adaptive per-parameter learning rates, and converges in fewer iterations.

*Note.* The functions for Equations 12 and 13 are provided in the functions_and_adam.R script, for reference, but are not utilised.

####      13a. Adam Initialisation 

``` R
init_adam <- function(size) {                                                   #size - how many parameters optimising (for w - N, for v - n)
  list(m = rep(0, size), v = rep(0, size), t = 0L)
}
```

This step creates the state object which Adam uses to track between iterations. This is called once at the start, separately for **w** and for **v** (see Section 15).

- **m**: first moment estimate, exponentially weighted moving average of the gradient, starts at zero.

- **v**: second moment estimate, exponentially weighted moving average of the squared gradient, starts at zero.

- **t**: timestep count, starts at 0, tracks how many Adam updates have happened.

####      13b. Adam Update Step

``` R
adam_step <- function(theta, grad, state,                                       
                      eta = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
  state$t <- state$t + 1L           #increment counter
  state$m <- beta1 * state$m + (1 - beta1) * grad      #update m, state$m becomes a smoothed version of the gradient
  state$v <- beta2 * state$v + (1 - beta2) * grad^2    #same for v
  m_hat <- state$m / (1 - beta1^state$t)      
  v_hat <- state$v / (1 - beta2^state$t)        
  theta <- theta - eta * m_hat / (sqrt(v_hat) + eps)    #Adam update, each theta entry gets a learning rate based on gradient history
  list(theta = theta, state = state)       #return updated parameter vector and updated state for next iteration 
}
```

This is specifically what replaces the original Equations 12 and 14.

- **theta**: parameter vector being updated (**w** or **v**).

- **grad**: Gradient of **J** with respect to **theta**, computed externally in Equation 10 or 13.

- **state**: Adam state object, from **init_adam**.

- **eta, beta1, beta2, eps**: Standard Adam hyperparameters (Kingma and Ba, 2014). Please see 'Hyperparameters' section for more information.

The **m_hat** and **v_hat** functions work towards bias correction. Since states start at 0, the moving averages are biased toward 0, especially in early iterations. The correction divides by (1 - beta\^t) to balance this.

Both the updated parameters **theta** and the updated states **state** are carried onto the next iteration.

#### 14. Parallel Gradient Computation for all Features

``` R
compute_dJ_dw <- function(w, v, A, Q, P, y, net, alpha, lambda,
                          rwr_conv_threshold, rwr_max, n_cores = 1) {
  solver <- function(l) {                                                       
    dA_dwl <- eq6_dA_dwl(l, A, net)
    dQ_dwl <- eq7_dQ_dwl(dA_dwl, A)
    dP_dwl <- eq9_dP_dwl(P, Q, dQ_dwl, alpha, rwr_conv_threshold, rwr_max)
    eq10_dJ_dwl(w[l], dP_dwl, v, y, P, lambda)
  }
  
  if (n_cores > 1 && .Platform$OS.type != "windows") {                         
    grads <- mclapply(seq_len(net$N), solver, mc.cores = n_cores)
  } else {
    grads <- lapply(seq_len(net$N), solver)
  }
  unlist(grads)
}
```

This step computes the full gradient vector **dJ/dw**, length N, with one entry per feature. The code runs in parrallel.

The **solver** function takes a feature index **l** and runs the full chain (Equation 6, 7, 9, 10) to compute **dJ/dwl**. This is possible to be done in parallel, since the calls are independent.

- **mclapply**: The parallel version of lapply. It splits the N feaure indices across **mc.cores** solver processes that run simultaneously. If your Mac device has 8 cores, the function will use 7 of those, and therefore it will compute 7 times faster than in serial.

- **unlist(grads)**: Returns a list, as a numeric vector for Adam.

*Note.* The program works for Mac/Lenux devices by default, allowing the parallel process to work. However, if a Windows device is used, the code will automatically run in serial.

#### 15. The Wrapper Function 

This is the full training algorithm. One iteration consists of a forward pass, gradient computation and Adam updates.

``` R
rwr_lasso_train <- function(P0, edges, X, y,
                            alpha = 0.5,
                            lambda = 0.01,
                            eta = 0.001,                                        
                            beta1 = 0.9,
                            beta2 = 0.999,
                            eps = 1e-8,
                            max_outer = 500,                                    
                            conv_threshold_outer = 1e-5,
                            rwr_conv_threshold   = 1e-6,
                            rwr_max = 100,
                            n_cores = max(1, detectCores() - 1)) {
  n   <- ncol(P0)
  net <- setup_network(edges, X, n)                                             
  
  #initialise w and v to 0 
  w <- rep(0, net$N)
  v <- rep(0, n)
  
  #creates one Adam state for each parameter vector
  state_w <- init_adam(net$N)
  state_v <- init_adam(n)
  
  J_history <- numeric(max_outer)
  
  for (iter in seq_len(max_outer)) {                                            
    A <- eq1_activation(w, net)                                                 
    Q <- eq2_transition(A, net)
    P <- eq3_rwr(P0, Q, alpha, rwr_conv_threshold, rwr_max)
    
    J_history[iter] <- eq4_objective(w, v, P, y, lambda)                        
    
    dJ_dw <- compute_dJ_dw(w, v, A, Q, P, y, net, alpha, lambda,                
                           rwr_conv_threshold, rwr_max, n_cores = n_cores)
    dJ_dv <- eq13_dJ_dv(v, P, y, lambda)
    
    upd_w <- adam_step(w, dJ_dw, state_w, eta, beta1, beta2, eps)               
    upd_v <- adam_step(v, dJ_dv, state_v, eta, beta1, beta2, eps)               
    w_new <- upd_w$theta;  state_w <- upd_w$state
    v_new <- upd_v$theta;  state_v <- upd_v$state
    
    delta <- max(abs(w_new - w), abs(v_new - v))   #convergence checks
    w <- w_new                                                                  
    v <- v_new
    
    if (delta < conv_threshold_outer) {                                         
      J_history <- J_history[seq_len(iter)]
      break
    }
  }
  
  list(
    w = w,
    v = v,
    A = A,
    Q = Q,
    P = P,
    J_history = J_history,
    iterations = length(J_history),
    net = net
  )
}
```

- **P0**, **edges**, **X**, **y**: Data inputs to be supplied.

- **alpha** = 0.5: RWR restart probability. (Zhang et. al., 2018).

- **lambda** = 0.01: Lasso regularisation strength. Should be fine-tuned.

- **eta** = 0.001: Adam learning rate. Suggested starting point from Kingma & Ba (2014). Can be fine-tuned.

- **beta1** = 0.9, **beta2** = 0.999, **eps** = 1e-8: Adam momentum and stability parameters. Defaults (Kingma & Ba, 2014).

- **max_outer** = 500: Iteration cap, manually selected.

- **conv_threshold_outer** = 1e-5: convergence threshold on the largest weight change.

- **rwr_tol** = 1e-6, **rwr_max** = 100: inner loop convergence parameters for RWR and gradient solves.

- **n_cores** = **detectCores() - 1**: parallel cores allowed to be used for gradient computation.

Inside the loop:

1.  Forward pass; compute **A, Q, P** from the current **w**. Reused in this iteration.

2.  Record **J** in **J_history**.

3.  **dJ_dw** computed via the parallel block, **dJ_dv** via the single matrix expression in Equation 13.

4.  One Adam step is applied to each parameter vector.

5.  Convergence check, iterations stop if convergance threshold is met, or maximum iterations have been completed.

The returned list provides:

- **w** and **v,** the learned weights of the features and the genes.

-  **A, Q, P**, the final state of the forward-pass matrices.

- **J_history**, which stores the trajectory of the objective iterations. This is useful for plotting and visualising the convergence.

- **net**, the network bundle

## Hyperparameters

**Table 2 \| Hyperparameter description table.**

| Hyperparameter | Default Value | Purpose |
|------------------|-------------|------------------------------------------|
| **alpha** | 0.5 | RWR restart probability. The higher the number,  more of the original signal is retained. This matches the Zhang et. al (2018) default value. |
| **lambda** | 0.01 | L1 penalty strength. Larger values produce sparser signitures. This value should be fine-tuned using the validation data. |
| **eta** | 0.001 | Adam learning rate. Suggested starting value, can be fine tuned to a maximum value of 0.1 (Kingma and Ba, 2014). |
| **beta1** | 0.9 | Adam first moment decay. Default (Kingma and Ba, 2014). |
| **beta2** | 0.999 | Adam second moment decay. Default (Kingma and Ba, 2014). |
| **eps** | 1e-8 | Adam numerical stability constant. Default (Kingma and Ba, 2014). |
| **max_outer** | 500 | Maximum outer iterations. Can be changed. |
| **conv_threshold_outer** | 1e-5 | Convergence threshold on the largest absolute weight change. Can be changed. |
| **rwr_conv_threshold** | 1e-8 | Convergence threshold for the inner RWR iteration. Can be changed. |
| **rwr_max** | 100 | Maximum inner RWR iterations. Can be changed. |

## 

## References

Kingma, D.P. and Ba, J. (2014) Adam: a method for stochastic optimization. arXiv:1412.6980.

Zhang, W., Ma, J. and Ideker, T. (2018) ‘Classifying tumors by supervised network propagation’, *Bioinformatics*, 34(13), pp. i484–i493. Available at: <https://doi.org/10.1093/bioinformatics/bty247.>  
