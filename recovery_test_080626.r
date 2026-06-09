
setwd("~/Desktop/LIFE 702 - Intro to Research/Coding")  
source("functions_and_adam.R")
library(Matrix)
library(parallel)

set.seed(1)


##### 1. Setting up the network and features
# 3 genes, edges between 1 -> 2, 1 -> 3
# 1 -> 2 important, there is a feature there, and large gene weight on gene 2, and large feature weight on edge
# 2 -> 3 unimportant, no feature


n <- 3
edges <- rbind(c(1,1), c(2,2), c(3,3), # self-loops
               c(1,2), # 1 -> 2  (with feature)
               c(1,3)) # 1 -> 3  (no feature)
E <- nrow(edges)
edges

X <- matrix(0, nrow = E, ncol = 1)  
X[4, 1] <- 1 # one feature, only on edge 1->2
X


##### 2. Setting the true parameters and P0

w_true <- 4  # BIG effect 
v_true <- c(0, 5, 0) # gene 2 only
alpha  <- 0.5  

m  <- 500
P0 <- matrix(runif(m * n), nrow = m, ncol = n) # uniform[0,1], all sampled together
# P0 values are not 1s and 0s... 
# will make a script doing this too

net <- setup_network(edges, X, n)

A_true <- eq1_activation(w_true, net) # equation 1
Q_true <- eq2_transition(A_true$A, net) # equation 2
Q_true
P_true <- eq3_rwr(P0, Q_true, alpha) # equation 3
y_true <- as.numeric(P_true %*% v_true) # getting the true growth rate 


################################################################################
################################################################################


##### Gradient descent as normal 
# to check parameter recovery 

RWR_normal <- rwr_lasso_train(P0 = P0, edges = edges, X = X, y = y_true,
                            alpha = alpha, lambda = 1e-4, eta = 0.01,
                            max_outer = 2000, conv_threshold_outer = 1e-7,
                            n_cores = 1)

print(w_true) # 4
print(RWR_normal$w) # -2.514011

print(v_true) # 0 5 0
print(RWR_normal$v) # 1.85961364 4.86674214 0.08622599

# gene 2 is correctly picked out as the driver
# v3 almost zero, gene 3 is correctly dropped
# BUT gene 1 has picked up signal where it shouldn't have (because of w?)


cor(y_true, as.numeric(RWR_normal$P %*% RWR_normal$v)) #0.9997768

# the model is still predicting growth rate correctly though given P and v 


################################################################################
################################################################################

##### Goodness of fit (y) and parameter plots 

par(mfrow = c(1, 2))

##### goodness of fit (growth rate prediction )

y_pred <- as.numeric(RWR_normal$P %*% RWR_normal$v)
r2 <- 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
rmse <- sqrt(mean((y_true - y_pred)^2))
coefs <- coef(lm(y_true ~ y_pred)) # intercept, slope 
intercept <- coefs[1]
slope <- coefs[2]

# the plot 

plot(y_pred, y_true, xlab = "Predicted Growth Rate", ylab = "Observed (true) Growth Rate",
     xlim = c(0, 6.5), ylim = c(0, 6.5), main = "Goodness of Fit (y)", 
     pch = 16, cex = 0.5, col = rgb(0, 0, 1, 0.4))
abline(0, 1, col = "red", lwd = 2)

legend("bottomright", legend = "y = x", col = "red", lwd = 2, lty = 1, bty = "n")

legend("topleft", legend = c(paste0("R2 = ", round(r2, 4)),
                  paste0("RMSE = ", round(rmse, 4)),
                  paste0("Slope = ", round(slope, 4)),
                  paste0("Intercept = ", round(intercept, 4))))


##### parameter recovery plot

true_par <- c(w = w_true, v1 = v_true[1], v2 = v_true[2], v3 = v_true[3])
rec_par  <- c(w = RWR_normal$w, v1 = RWR_normal$v[1], v2 = RWR_normal$v[2], v3 = RWR_normal$v[3])

# colors for the parameters, w in orange, v in  blue
par_cols <- c(w = "darkorange", v1 = "blue", v2 = "blue", v3 = "blue")

plot(true_par, rec_par, xlab = "True Parameter Value", ylab = "Recovered Parameter Value",
     main = "Parameter Recovery", pch = 16, cex = 1.2, col = par_cols,
     xlim = c(-3, 6.5), ylim = c(-3, 6.5))
abline(0, 1, col = "red", lwd = 2)
text(true_par, rec_par, labels = names(true_par), pos = 4, cex = 0.8)

legend("topleft", legend = c("w (feature weight)", "v (gene weights)"),
       pch = c(16, 16), col = c("darkorange", "blue"))

legend("bottomright", legend = "y = x", col = "red", lwd = 2, lty = 1, bty = "n")


################################################################################
################################################################################


#### To fix v1 at 0 I need to change rwr_lasso train slightly 
# The only change is 0ing value of v1 in lines 135, 145, 162, 168 


rwr_lasso_train_fixv <- function(P0, edges, X, y, fix_v1 = integer(0),
                                 alpha = 0.5, lambda = 0.0001, eta = 0.01,
                                 beta1 = 0.9, beta2 = 0.999, eps = 1e-8,
                                 max_outer = 2000, conv_threshold_outer = 1e-7,
                                 rtol = 1e-5, atol = 1e-8, max_iter = 100,
                                 n_cores = 1) {
  n   <- ncol(P0)
  net <- setup_network(edges, X, n)
  w <- rep(0, net$N)
  v <- rep(0, n)
  v[fix_v1] <- 0
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
    dJ_dw <- compute_dJ_dw(w, v, A, Q, P, y, net, dA_dw, alpha, lambda,
                           rtol, atol, max_iter, n_cores = n_cores)
    dJ_dv <- eq13_dJ_dv(v, P, y, lambda)
    dJ_dv[fix_v1] <- 0  # no gradient for gene 1
    
    upd_w <- adam_step(w, dJ_dw, state_w, eta, beta1, beta2, eps)
    upd_v <- adam_step(v, dJ_dv, state_v, eta, beta1, beta2, eps)
    w_new <- upd_w$theta; state_w <- upd_w$state
    v_new <- upd_v$theta; state_v <- upd_v$state
    v_new[fix_v1] <- 0  # force 0 for gene 1
    
    delta <- max(abs(w_new - w), abs(v_new - v))
    w <- w_new; v <- v_new
    if (delta < conv_threshold_outer) { J_history <- J_history[seq_len(iter)]; break }
  }
  list(w = w, v = v, A = A, Q = Q, P = P, J_history = J_history, iterations = length(J_history), net = net)
}



################################################################################
################################################################################


# force v1 = 0, fit only v2, v3 
# using rwr_lasso_train_fixv from above 


RWR_fix_v1 <- rwr_lasso_train_fixv(P0 = P0, edges = edges, X = X, y = y_true,
                                fix_v1 = 1L, # fixes the first gene v to 0, as enforced above 
                                alpha = alpha, lambda = 1e-4, eta = 0.01,
                                max_outer = 4000, conv_threshold_outer = 1e-8,
                                n_cores = 1)

print(w_true) # 4
print(RWR_fix_v1$w) # 2.783354

# better than before 

print(v_true) # 0 5 0
print(RWR_fix_v1$v) # 0 5.012737235 0.006418316

# also better than before. Signal does not go towards gene 1 (of course), but 
# goes to gene 3 even less 

print(cor(y_true, as.numeric(RWR_fix_v1$P %*% RWR_fix_v1$v))) # 0.9999891

# slightly better 


################################################################################
################################################################################


##### Goodness of fit (y) and parameter plots 

par(mfrow = c(1, 2))

##### goodness of fit (growth rate prediction)

y_pred1 <- as.numeric(RWR_fix_v1$P %*% RWR_fix_v1$v)
r21 <- 1 - sum((y_true - y_pred1)^2) / sum((y_true - mean(y_true))^2)
rmse1 <- sqrt(mean((y_true - y_pred1)^2))
coefs1 <- coef(lm(y_true ~ y_pred1)) # intercept, slope 
intercept1 <- coefs1[1]
slope1 <- coefs1[2]

# the plot 

plot(y_pred1, y_true, xlab = "Predicted Growth Rate", ylab = "Observed (true) Growth Rate",
     xlim = c(0, 6.5), ylim = c(0, 6.5), main = "Goodness of Fit (Fixed v1)", 
     pch = 16, cex = 0.5, col = rgb(0, 0, 1, 0.4))
abline(0, 1, col = "red", lwd = 2)

legend("bottomright", legend = "y = x", col = "red", lwd = 2, lty = 1, bty = "n")

legend("topleft", legend = c(paste0("R2 = ", round(r21, 4)),
                             paste0("RMSE = ", round(rmse1, 4)),
                             paste0("Slope = ", round(slope1, 4)),
                             paste0("Intercept = ", round(intercept1, 4))))


##### parameter recovery plot

rec_par1 <- c(w = RWR_fix_v1$w, v1 = RWR_fix_v1$v[1], v2 = RWR_fix_v1$v[2], v3 = RWR_fix_v1$v[3])

# colors for the parameters, w in orange, v in  blue
par_cols <- c(w = "darkorange", v1 = "blue", v2 = "blue", v3 = "blue")

plot(true_par, rec_par1, xlab = "True Parameter Value", ylab = "Recovered Parameter Value",
     main = "Parameter Recovery (Fixed v1)", pch = 16, cex = 1.2, col = par_cols,
     xlim = c(-0.5, 6.5), ylim = c(-0.5, 6.5))
abline(0, 1, col = "red", lwd = 2)
label_pos <- c(w = 4, v1 = 3, v2 = 4, v3 = 1)
text(true_par, rec_par1, labels = names(true_par), pos = label_pos, cex = 0.8)

legend("topleft", legend = c("w (feature weight)", "v (gene weights)"),
       pch = c(16, 16), col = c("darkorange", "blue"))

legend("bottomright", legend = "y = x", col = "red", lwd = 2, lty = 1, bty = "n")

