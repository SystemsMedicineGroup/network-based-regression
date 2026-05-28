# Title

##### By Una Milovanovic

##### 15/05/2026

## Introduction

## Algorithm Summary

## **Repository Contents**

- **functions_and_adam.R:** Main implementation of the algorithm; defines all of the functions and the main wrapper.

- **testing_functions_and_adam.R:** Tests the application of the functionres.

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

``` r
result <- rwr_lasso_train(
  P0 = P0,
  edges = edges,
  X = X,
  y = y,
  alpha = 0.5,
  lambda = 0.01,
  eta = 0.01,
  max_outer = 200,
  n_cores = 1 #set to the amount of cores on your device -1
)

#Results
result$w  #learned feature weights (length N)
result$v  #learned gene weights (length n)

plot(result$J_history, type = "l", xlab = "Iteration", ylab = "Objective J")
```

## Inputs

- **P0** (numeric matrix, m x n): A binary mutation profile where the rows are tumours, and the columns are genes. 1 = mutant, 0 = wild type.

- **edges** (integer matrix, E x 2): Each row is an edge (i, j), where i and j are node indices. E.g. if row one is (5,12), there is an edge from gene 5 to 12. Self loops (i, i) must be included.

- **X** (numeric matrix, E x N): The edge feature matrix. It has one row per edge and one column per feature. The self loop indicator should be one of the (N) features.

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
|-----------------|-----------------|---------------------|-----------------|
| **setup_network** | Pre-computation | edges, X, n | Net bundle |
| **eq1_activation** | Eq. 1 | w, net | A |
| **eq2_transition** | Eq. 2 | A, net | Q |
| **eq3_rwr** | Eq. 3 | P0, Q, alpha, rtol, atol, max_iter | P |
| **eq4_objective** | Eq. 4 | w, v, P, y, lambda | J |
| **eq6_dA_dw** | Eq. 6 | a_edges, net | dA/dw |
| **eq7_dQ_dwl** | Eq. 7 | dA_dw_l, A, net | dQ/dwl |
| **eq9_dP_dwl** | Eq. 9 | P, Q, dQ_dwl, alpha, rtol, atol, max_iter | dP/dwl |
| **eq10_dJ_dwl** | Eq. 10 | wl, dP, v, y, P, lambda | dJ/dwl |
| **compute_dJ_dw** | Combines Eq. 6,7,9,10 | All of the above | dJ/dw |
| **eq13_dJ_dv** | Eq. 13 | v, P, y, lambda | dJ/dv |
| **init_adam** | Adam initialisation | Parameter size | state list with m, v, t |
| **adam_step** | Adam update | Parameters, gradient, state | updated theta and state |
| **rwr_lasso_train** | Main training loop | All data inputs | Results list |

#### 1. Setting up the Network

``` r
setup_network <- function(edges, X, n) {
  E <- nrow(edges)
  N <- ncol(X)
  edges <- as.matrix(edges)
  
  #Sparse edge presence (n x n, binary)
  #edge_presence[i, j] = 1 if (i, j) is an edge, 0 otherwise. Only non zero entries stored internally (sparse)
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
```

The **setup_network** function runs once before training, taking the raw inputs (edge list, the feature matrix, and the number of genes), and collects them into a single object **net**.The following functions in this code use **net** as an argument, and take the required inputs from it forward.

- **E**: The number of edges (number of rows in the edge list **edges**)

- **N**: The number of features (number of columns in the feature matrix **X**)

- **edge_presence** (sparse numeric matrix, n x n): Binary matrix where cell **[i, j] = 1** if there is an edge present from gene **i** to **j**, and **0** otherwise. The dimensions are set by **dims = c(n, n)** as the edge list might not mention every gene, since some genes may not have any edges, but are still present in the network.

- **The returned list**: Places everything into one object so **net** can be used moving forward (refer to Outputs section), instead of separate arguments.

#### 2. Equation 1 - Activation Matrix A

``` r
eq1_activation <- function(w, net) {                                            
  a_edges <- 1 / (1 + exp(-as.numeric(net$X %*% w)))                            
  A <- sparseMatrix(                                                            
    i = net$edges[, 1], j = net$edges[, 2], x = a_edges,
    dims = c(net$n, net$n)
  )
  output_eq1 <- list(A = A, a_edges = a_edges)   
  return(output_eq1)                                               
}
```

This step is the first part of the forward pass. It computes the activation score for every edge, given the (current) feature weights **w**, and scatters these into an n x n matrix **A**, where **A[i, j]** is a number between 0 and 1, representing the strength of the edge from gene **i** and **j**. Both the per-edge vector and the n x n matrix are returned.

- **a_edges**: Calculates the whole equation, which is an E length numeric vector, in which entry **k** is the activation strength of edge **k**, as a number from 0 to 1. This is computed for further use in equation 6.

- **A**: This takes the values from **a_edges** and orders them into the (i, j) positions given by **edges**. **A[i, j]** equals the activation of edge **(i, j)** if it exists, and 0 if it doesn't. **sparseMatrix** is used since n x n is huge, but only a percentage of this are non-zero values, therefore we only store these and their coordinates. This is used later in equation 2 and equation 7.

#### 3. Equation 2 - Transition Matrix Q

``` r
 eq2_transition <- function(A, net) {
  r <- rowSums(A)                                                               
  Q <- Diagonal(x = 1 / (r + 1e-8)) %*% A 
  return(Q)
}
```

This is the second step of the forward pass. The activation matrix **A** is used in order to produce a transition matrix **Q**, where each of the rows sums to 1. Because **Q** is going to be used in equation 3 for a random walk with restart function, the entries of each row need to be probabilities, and the probabilites need to add up to 1.

For every cell **[i, j]**, the activation of edge **(i, j)** is divided by the sum of all activations leaving node **i**.

- **r** (numeric vector, length n): This is the total activation flowing out of node **i**.
- **Diagonal(...)**: This builds a sparse diagonal matrix with 1/r[i] on the diagonal and 0 elsewhere, and then multiplies this by **A**. This is the same as dividing each row by its own total, but needs to be done this way as otherwise R would do column wise recycling and divide the wrong axis, giving the wrong answer.
- 1e-8 is added to **r** in order to ensure that in case any row sum is exactly 0 (can only be possible if there is an error in your setup or if activations are very small), the code does not crash, it just instead divides by 1e-8. This value is so small that it will not affect the results.

#### 4. Equation 3 - Random Walk with Restart (RWR)

``` r
eq3_rwr <- function(P0, Q, alpha = 0.5, rtol = 1e-5, atol = 1e-8, max_iter = 100) { 
  P <- P0                                                                       
  for (t in seq_len(max_iter)) {                                                
    P_new <- (1 - alpha) * as.matrix(P %*% Q) + alpha * P0                      
    if (max(abs(P_new - P) - rtol * abs(P)) <= atol) {                          
      return(P_new)
    }
    P <- P_new                                                                  
  }
  return(P)                                                                             
}
```

This is the third step of the forward pass. It takes the initial gene expression matrix **P0** and smooths it across the network using **Q**. The process is iterative, it runs the RWR until **P** stops changing.

First, **P0** initialises **P** to **P**$^{0}$. Then, the iteration 'for' loop begins, and **P_new** is calculated through the RWR update equation. The convergence of **P** is assessed using a relative tolerance check that scales the allowed change between **P** and **P_new** according to the size of each entry, so both large and small values converge proportionally (each entry of **P** gets a tolerance which is appropriate and proportional to its scale). If the check passes, convergence has occurred and the current **P_new** value is returned. Otherwise, **P_new** is renamed to **P**, and the loop repeats again. The loop process continues until either convergence is reached, or the preset maximum number of iterations has been hit.

#### 5. Equation 4 - Objective Function J

``` r
eq4_objective <- function(w, v, P, y, lambda) {
  m <- length(y)                                                                
  resid <- y - as.numeric(P %*% v)                                              
  J <- lambda * sum(abs(w)) + lambda * sum(abs(v)) + (1 / (2 * m)) * sum(resid^2) 
  return(J)
}
```

This step evaluates the LASSO objective function **J**. The optimiser later will try to minimise this **J** value, as **J** is the sum of the LASSO penalty on the feature weights, the gene weights, and the squared error data fit.

- **m**: This represents the number of tumours.
- **resid** (m x 1 vector, length m): This computes the residual vector, the actual growth rate minus the predicted growth rate of a tumour.
- **J**: This is the main equation, computing the J value.

#### 6. Equation 5 - Objective Function J in Summative Form

This function is identical to Equation 4, in summation form. It is written with nested loops rather than matrix operations. The outputs of Equation 4 and Equation 5 should be the same. This is primarily used for cross-checking. Equation 4 is the preferred version since the loops in Equation 5 are significantly slower.

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
  J_summation <- lambda * sum(abs(w)) + lambda * sum(abs(v)) + (1 / (2 * m)) * total
  return(J_summation)
}  
```

- **m**: Represents the number of tumours
- **n**: Represents the number of genes.

The inner loop computes the prediction for tumour **c** by summing **P[c, i] \* v[i]**, over all genes **i**. The outer loop takes the squared residual for that tumour, and adds it to **total**. After the loops, the equation is computed to give **J_summation**.

#### 7. Equation 6 - dA/dw

``` r
eq6_dA_dw <- function(a_edges, net) {                                   
  logistic_slope <- a_edges * (1 - a_edges)                                     
  dA_dw <- net$X * logistic_slope                                               
  return(dA_dw)
}
```

This is the first step of the backward pass. This computes the gradient **dA/dw**, of the activation matrix **A** with respect to all features. The result is an E x N table where every value of **dAij/dwl** is stored. Row **k** of **dA/dw** is the gradient of edge **k**'s activation wrt every feature. Column **l** of **dA/dw** is the gradient of every edge's activation wrt feature **l**, in the edge-list order.

- **logistic_slope**: This is the a \* (1 - a) part of the equation, done for every edge, element-wise. The result is an E length vector, representing the per-edge logistic slope. This is done first since it can be re-used, and this is better than computing it separately **N** times.

- **dA_dw**: Each row of **net\$X** gets multiplied by the **logistic_slope** row corresponding to it. The result is an E x N matrix where **[k, l]** is **X[k, l] \* logistic_slope[k]**. This is the equation, computed for every edge-feature pair sumultaneously.

#### 8. Equation 7 - dQ/dwl

``` r
eq7_dQ_dwl <- function(dA_dw_l, A, net) {              
  dA_dwl_mat <- sparseMatrix(                                                       
    i = net$edges[, 1], j = net$edges[, 2], x = as.numeric(dA_dw_l),
    dims = c(net$n, net$n))
  r <- rowSums(A)                                                               
  r_dA <- rowSums(dA_dwl_mat)                                                       
  num <- dA_dwl_mat * r - A * r_dA                                                  
  dQ_dwl <- num / (r^2)  
  return(dQ_dwl)
}
```

This is the second step of the backward-pass. It computes the full gradient **dQ/dwl,** of the transition matrix **Q** with respect to one feature weight **wl**. It essentially describes how **Q** changes when **wl** changes.

- **dA_dwl_mat**: This step scatters the length E vector of **dA/dwl** and places each entry at the correct position in a sparse n x n matrix (with 0 everywhere else). This step is similar to the one done when computing **A**.

- **r** (numeric vector, length n): This is the total activation flowing out of node **i**.

- **r_dA** (numeric vector, length n): This represents the total rate of change of the activation flowing out of node **i**, wrt feature **l**.

- **num**: This is the numerator of the equation.

#### 9. Equation 9 - dP/dwl

``` r
eq9_dP_dwl <- function(P, Q, dQ_dwl, alpha = 0.5, rtol = 1e-5, atol = 1e-8, max_iter = 100) { 
  m <- nrow(P)                                                                  
  n <- ncol(P)                                                                  
  dP <- matrix(0, m, n)                                                         
  P_dQ <- as.matrix(P %*% dQ_dwl)                                               
  for (t in seq_len(max_iter)) {
    dP_new <- (1 - alpha) * (as.matrix(dP %*% Q) + P_dQ)                        
    if (max(abs(dP_new - dP) - rtol * abs(dP)) <= atol) {                       
      return(dP_new)
    }
    dP <- dP_new                                                                
  }
  return(dP)                                                                            
}
```

This is the third step of the backward pass. Similar loop construction as for Section 4 (Equation 3). This step iteratively solves for **dP/dwl**, until convergance.

- **m**: Number of tumours.

- **n**: Number of genes.

- **dP**: A m x n matrix of all 0s. This is the starting point for the iteration.

First, **P_dQ** is computed by multiplying **P** and **dQ_dwl** (from equation 7). This is done outside of the loop, since it remains constant throughout the iterations.

As mentioned, an initial **dP** starting matrix is created, and once the loop is entered, **dP_new** is calculated using the equation. The convergence of **dP** is assessed the same way as in Equation 3. If convergence has occurred the **dP** value is returned. Otherwise, **dP_new** is renamed to **dP**, and the loop begins again. The loop will continue until either convergence is reached, or the preset maximum number of iterations has been hit.

#### 10. Equation 10 - dJ/dwl

``` r
eq10_dJ_dwl <- function(w_l, dP_dwl, v, y, P, lambda) {                         
  m <- length(y)                                                                
  resid <- y - as.numeric(P %*% v)                                              
  dJ_dwl <- sign(w_l) * lambda - (1 / m) * as.numeric(t(dP_dwl %*% v) %*% resid) 
  return(dJ_dwl)
}
```

This step is another part of the backward pass. It computes the gradient **dJ/dwl**, of **J** with respect to one feature weight **wl**. The output is scalar.

- **m**: Number of tumours.

- The residual **resid (**y - Pv) is calculated, the same way as in Equation 4.

- **dJ_dwl**: The full equation. The t() function 'transpose' is used since **dP_dwl** is m × n , **v** is n × 1. Matrix \* vector gives a m × 1 column vector. Transposing it poduces a 1 x m row vector, which can be multiplied by **resid**, giving a 1 x 1 scalar.

#### 11. Parallel Gradient Computation for all Features

``` r
compute_dJ_dw <- function(w, v, A, Q, P, y, net, dA_dw, alpha, lambda, rtol, atol, max_iter, n_cores = 1) {
  solver <- function(l) {                                                       
    dA_dw_l <- dA_dw[, l]                                                       
    dQ_dwl <- eq7_dQ_dwl(dA_dw_l, A, net)
    dP_dwl <- eq9_dP_dwl(P, Q, dQ_dwl, alpha, rtol, atol, max_iter)
    dJ_dwl <- eq10_dJ_dwl(w[l], dP_dwl, v, y, P, lambda)
    return(dJ_dwl)
  }
  if (n_cores > 1 && .Platform$OS.type != "windows") {                          
    grads <- mclapply(seq_len(net$N), solver, mc.cores = n_cores)
  } else {                                                                     
    grads <- lapply(seq_len(net$N), solver)
  }
  dJ_dw <- unlist(grads)
  return(dJ_dw)
}
```

This step computes the full backwards pass for **w**. It computes the gradient vector **dJ/dw**, length N, with one entry per feature, to update the weights. The code runs in parrallel.

- **rtol, atol, max_iter**: These are the convergence parameters for eq9.

- **n_cores**: This is the number of CPU cores to use for parallel computation. The default 1 = serial.

- The **solver** function is the per-feature gradient computation. It takes a feature index **l** and runs the full backward chain for that feature (Equation 6, 7, 9, 10) to compute a sungle scalar **dJ/dwl**.

- The **if** statement: There are two possible conditions, for the code to run in parallel and for the code to run in serial.

  - The code will run in parallel if you are on a Mac/Linux device, with **n_cores \> 1**; **mclapply** splits the N feaure indices across **mc.cores** solver processes that run simultaneously. If your device has 8 cores, the function will use 7 of those, and therefore it will compute 7 times faster than in serial (implemented in the full training loop.

  - The code will run in serial if you are on a Windows device, or **n_cores = 1.** This essentially means that plain **lapply** is used, and it calls **solver** once for each **l** from 1 to N in order, with no parallelism.

- **unlist(grads)**: Returns a list, where each element is a single number. Unlist makes this a numeric vector, length N, where entry **l** is **dJ_dwl**.

#### 12. Equation 13 - dJ/dv

``` r
eq13_dJ_dv <- function(v, P, y, lambda) {
  m <- length(y)                                                                
  resid <- y - as.numeric(P %*% v)                                              
  dJ_dv <- sign(v) * lambda - (1 / m) * as.numeric(t(P) %*% resid) 
  return(dJ_dv)
}
```

This step computes the full gradient vector **dJ/dv**, of **J** with respect to the gene weight weight **v**. The output is a n length vector. This gradient is direct, and is computed as a single matrix-vector operation.

- **m**: Number of tumours.

- The residual **resid (**y - Pv) is calculated, the same way as in Equation 4 and 10.

- **dJ_dv**: The full equation. The t() function 'transpose' is used to turn **P** (m x n) into an n x m matrix. Multiplying this transposed **P** by **resid** (m x 1) produces a n x 1 product. This gets rid of the m dimension (which is tumours), otherwise the multiplication would not be possible. Each entry shows how that gene's contribution correlates with the current prediction errors across all tumours. This is then flattened into a n length vector, for the rest of the equation.

#### 13. Equation 15 - dJ/dvi

Scalar form of Equation 13.

``` r
eq15_dJ_dvi <- function(i, v, P, y, lambda) {
  m <- length(y)                                                                
  resid <- y - as.numeric(P %*% v)
  dJ_dvi <- sign(v[i]) * lambda - (1 / m) * as.numeric(t(P[, i]) %*% resid)
  return(dJ_dvi)
}
```

This step computes the gradient **dJ/dv** with respect to gene **i**. The output is the scalar version of equation 13, a single number, which is the gradient for gene **i**.

- **m**: Number of tumours.

- The residual **resid (**y - Pv) is calculated, the same way as in Equation 4, 10 and 13.

- **dJ_dvi**: This is the equation again. Instead of computing for all genes at once, this is done for the **i**th entry of vector **v** and the **i**th column of **P** (length m). Once again the transpose function is used, to turn the length m column **P[ ,i]** into a 1 x m row, so that multiplying it by **resid** (m x 1), gives a 1 x 1 scalar. The 1 x 1 matrix is flattened to a plain scalar for the rest of the calculation.

This function's purpose is only for cross checking. Equation 13 will be used in general, since it computes all n gradients in one matrix operation, and this is much faster than calling Equation 15 n times.

#### 14. Adam Optimiser

The Adam optimiser code replaces Equations 12 and 14 from the Gradient Descent Algorithm document, it takes the gradients and produces parameter updates.

*Note.* The functions for Equations 12 and 14 are provided in the functions_and_adam.R script, for reference, but are not utilised.

#### 14a. Adam Initialisation

``` r
init_adam <- function(size) {                                                   
  state <- list(m = rep(0, size), v = rep(0, size), t = 0L)
  return(state)
}
```

This step creates the state object which Adam uses to track between iterations. This is called once at the start, separately for **w** and for **v** (see Section 15). The two states track the gradients independently, since the parameter vectors are updated independently.

- **m**: first moment estimate, exponentially weighted moving average of the gradient, starts at 0.

- **v**: second moment estimate, exponentially weighted moving average of the squared gradient, starts at 0.

- **t**: timestep count, starts at 0, tracks how many Adam updates have happened.

#### 14b. Adam Update Step

``` r
adam_step <- function(theta, grad, state, eta = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
  state$t <- state$t + 1L           
  state$m <- beta1 * state$m + (1 - beta1) * grad      
  state$v <- beta2 * state$v + (1 - beta2) * grad^2    
  m_hat <- state$m / (1 - beta1^state$t)      
  v_hat <- state$v / (1 - beta2^state$t)        
  theta <- theta - eta * m_hat / (sqrt(v_hat) + eps)    
  out <- list(theta = theta, state = state)  
  return(out)
}
```

This is specifically what replaces the original Equations 12 and 14. This is called once per outer iteration, per parameter vector.

- What adam_step requires

  - **theta**: parameter vector being updated (**w** or **v**).

  - **grad**: Gradient of **J** with respect to **theta**, computed externally in Equation 10 or 13.

  - **state**: Adam state object, from **init_adam**.

  - **eta, beta1, beta2, eps**: Standard Adam hyperparameters (Kingma and Ba, 2014). Please see 'Hyperparameters' section for more information.

- **state\$t**: This tracks how many updates have happened.

- **state\$m**: Updating the first moment. The new value is a combination of the old vaalue and the current **grad**. With **beta1 = 0.9**, 90% of the new **m** comes from the old **m**, and 10% comes from the current gradient (new information).

- **state\$v**: The same concept as above, but uses **beta2** and this is applied to the squared gradient. With **beta2 = 0.999**, 99.9% of the new **v** comes from the old **v**, while only 0.1% from the new.

- The **m_hat** and **v_hat** values work towards bias correction. Since states start at 0, the moving averages are biased toward 0, especially in early iterations. The correction divides by (1 - beta\^t) to balance this.

- **theta \<- theta - ...** : This is the Adam update rule, where each paameter's smoothed gradient is divided by its smoothed magnitude (m_hat/sqrt(v_hat)), and scaled by learning rate **eta**. This is subtracted from curent **theta** to give the updated **theta**.

Both the updated parameters **theta** and the updated states **state** are carried onto the next iteration.

#### 15. The Wrapper Function

This is the full training algorithm. One iteration consists of a forward pass, gradient computation and Adam updates.

``` r
rwr_lasso_train <- function(P0, edges, X, y, alpha = 0.5, lambda = 0.01, eta = 0.001, 
                            beta1 = 0.9, beta2 = 0.999,eps = 1e-8, max_outer = 500, 
                            conv_threshold_outer = 1e-5, rtol = 1e-5, atol = 1e-8,
                            max_iter = 100, n_cores = max(1, detectCores() - 1)) {
  n   <- ncol(P0)
  net <- setup_network(edges, X, n)                                             
  w <- rep(0, net$N)     
  v <- rep(0, n)                               
  state_w <- init_adam(net$N)                                                   
  state_v <- init_adam(n)
  
  J_history <- numeric(max_outer)                                               
  
  for (iter in seq_len(max_outer)) {                                            
    act <- eq1_activation(w, net)                                               
    A <- act$A
    a_edges <- act$a_edges                                                      
    Q <- eq2_transition(A, net)
    P <- eq3_rwr(P0, Q, alpha, rtol, atol, max_iter)
    
    J_history[iter] <- eq4_objective(w, v, P, y, lambda)                        
    
    dA_dw <- eq6_dA_dw(a_edges, net)                                            
    dJ_dw <- compute_dJ_dw(w, v, A, Q, P, y, net, dA_dw, alpha, lambda, rtol, atol, max_iter,
                           n_cores = n_cores)
    dJ_dv <- eq13_dJ_dv(v, P, y, lambda)                                        
    
    upd_w <- adam_step(w, dJ_dw, state_w, eta, beta1, beta2, eps)
    upd_v <- adam_step(v, dJ_dv, state_v, eta, beta1, beta2, eps)
    w_new <- upd_w$theta  
    state_w <- upd_w$state
    v_new <- upd_v$theta
    state_v <- upd_v$state
    
    delta <- max(abs(w_new - w), abs(v_new - v))                                
    w <- w_new
    v <- v_new
    
    if (delta < conv_threshold_outer) {
      J_history <- J_history[seq_len(iter)]                                     
      break
    }
  }
  result <- list(
    w = w,
    v = v,
    A = A,
    Q = Q,
    P = P,
    J_history = J_history,
    iterations = length(J_history),
    net = net
  )
  return(result)
}
```

This is the main training loop, combining all of the previous code into what is actually used for training. it takes the raw inputs and runs the gradient descent with Adam, and returns the 'trained' parameters. This is the only function which will be called directly for training (the others can be used for cross-checking).

The required inputs for this are:

- **P0**: The initial gene expression matrix (raw input data).

- **edges**: The 2 column matrix of edges, with one row per edge (includes self loops).

- **X**: The E x N feature matrix, with one row per edge and one feature per column.

- **y**: The (m length) vector of the observed tumour growth rates.

- Hyperparameter information for **alpha**, **lambda, eta**, **beta1**, **beta2** and **eps** can be found in the 'Hyperparameter' section of this file.

- **max_outer** = 500: Iteration cap, manually selected.

- **conv_threshold_outer** = 1e-5: convergence threshold on the largest parameter change.

- **rtol**, **atol**, **max_iter**: These are the tolerance values used in equation 3 and equation 9.

- **n_cores** = **detectCores() - 1**: Used for parallelism. The code essentially ensures all available cores on the device are used except 1 (1 is left for the machine to continue working while the code runs).

Firstly, initialisation takes place:

The number of genes is taken from **P0**, and stored as **n**, and **net** is built. Next, both parameters are set to 0 (**w** and **v**). Next, the Adam states are initialised, and **J_history** is set as the **max_outer** value (this is used to track the iterations further on).

Inside the loop, a maximum of **max_outer** (in this case 500) iterations occur. Each iteration does the full forward pass, backward pass, parameter update and convergence check.

1.  Forward pass; compute **A, a_edges** (for eq 6)**, Q, P** from the current **w**.

2.  Record **J** in **J_history**, using equation 4. This computes and stores the current loss, as a diagnostic feature, to see how it is progressing over the iterations.

3.  **dA_dw** is computed using **a_edges** from the forward pass. This is the E x N gradient of activation matrix for this iteration.

4.  **dJ_dw** is computed via the parallel block from section 11.

5.  **dJ_dv** is computed using Equation 13.

6.  Next, the Adam updates happen. For each parameter, Adam is run separately. These are stored in **w_new** and **v_new**. The state values are also updated.

7.  Then, convergence is checked. **delta** is calculated as the largest single parameter change in this iteration, for either vector. **w** and **v** are updated. The iterations stop if convergance threshold is met (if **delta** is smaller than the set **conv_threshold_outer**), or maximum iterations have been completed.

8.  **J_history** is cropped to capture only the iterations that have actually been compleeted (if the loop stops before **max_iter** has been hit)

The returned list provides:

- **w** and **v,** the learned parameter weights of the features and the genes.

- **A, Q, P**, the final state of the forward-pass matrices.

- **J_history**, which stores the trajectory of the objective iterations. This is useful for plotting and visualising the convergence.

- **net**, the network bundle.

## Hyperparameters

**Table 2 \| Hyperparameter description table.**

| Hyperparameter | Default Value | Purpose |
|-------------------|-------------------|------------------------------------|
| **alpha** | 0.5 | RWR restart probability. The higher the number, more of the original signal is retained. This matches the Zhang et. al (2018) default value. |
| **lambda** | 0.01 | L1 penalty strength. Larger values produce sparser signitures. This value should be fine-tuned using the validation data. |
| **eta** | 0.001 | Adam learning rate. Suggested starting value, can be fine tuned to a maximum value of 0.1 (Kingma and Ba, 2014). |
| **beta1** | 0.9 | Adam first moment decay. Default (Kingma and Ba, 2014). |
| **beta2** | 0.999 | Adam second moment decay. Default (Kingma and Ba, 2014). |
| **eps** | 1e-8 | Adam numerical stability constant. Default (Kingma and Ba, 2014). |
| **max_outer** | 500 | Maximum outer iterations. Can be changed. |
| **rtol** | 1e-5 | Relative tolerance for inner RWR convergence (Equations 3 and 9). Can be changed. |
| **atol** | 1e-8 | Absolute tolerance for inner RWR convergence (Equations 3 and 9). Can be changed. |
| **max_iter** | 100 | Maximum inner RWR iterations. Can be changed. |

## 

## References

Kingma, D.P. and Ba, J. (2014) Adam: a method for stochastic optimization. arXiv:1412.6980.

Zhang, W., Ma, J. and Ideker, T. (2018) ‘Classifying tumors by supervised network propagation’, *Bioinformatics*, 34(13), pp. i484–i493. Available at: <https://doi.org/10.1093/bioinformatics/bty247.>  
