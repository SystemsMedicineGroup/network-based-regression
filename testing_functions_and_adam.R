setwd("~/Desktop/LIFE 702 - Intro to Research/Coding")
source("functions_and_adam.R")

#This is just to see if the functions do what theyre supposed to and that everything makes sense 
#I will also create a network and try to recover the weights
library(Matrix)
library(parallel)


n <- 4  #number of genes (nodes)
edges <- rbind(
  c(1, 1), #self loop
  c(2, 2), #self loop
  c(3, 3), #self loop
  c(4, 4), #self loop
  c(1, 2), #1 -> 2
  c(2, 3),  #2 -> 3
  c(3, 4)  #3 -> 4
)

E <- nrow(edges) #no. edges
E #7
N <- 2  #no. features

set.seed(42)
#edge feature matrix E x N (7 x 2) One row per edge, one col per feature
X <- matrix(c(1, 0.5, 1, 0.3, 1, 0.7, 1, 0.1, 0, 0.8, 0, 0.4, 0, 0.9),
            ncol = N, byrow = TRUE)
X
#here, col 1 is self loop indicator (so 1=self loop)
#col 2 

net <- setup_network(edges, X, n)


####setup net check
net$n    #4
net$E   # 7
net$N   #2
dim(net$edges)   #7 2
dim(net$X) #also 7 2
net$edge_presence 
length(net$edge_presence@x) #no. entries not 0 (7)
length(net$edge_presence) #total entries (16) n x n

#all good.

####eq1
#w = 0, every activation should = 0.5

w_zero <- rep(0, net$N)
act_zero <- eq1_activation(w_zero, net)

class(act_zero) #list                                                                
names(act_zero) #everything in the list, so A, a_edges                                                                
length(act_zero$a_edges) #7 (E)                                                       
dim(act_zero$A)  #4 4   

act_zero$a_edges #all are 0.5
act_zero$A

#with non 0 w, check edge 1 (self-loop gene 1)
#features (1, 0.5)
w_test_eq1 <- c(0.5, -0.3)
act_test <- eq1_activation(w_test_eq1, net)
act_test$a_edges[1]                                                          
1 / (1 + exp(-0.35))  #same    

act_test$a_edges
act_test$A

act_test$A[1, 3] #0, non edge                                                    

#all good.

####eq2
Q_test <- eq2_transition(act_test$A, net)
Q_test

rowSums(Q_test) #all rows sum to 1                                                   

#all good.

####eq3 
P0_test <- matrix(c(1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0), nrow = 3, byrow = TRUE)
m <- nrow(P0_test)

P_test <- eq3_rwr(P0_test, Q_test, alpha = 0.5)
P_test

dim(P_test)                                                                    

#alpha = 1 so 100% restart w no propagation
P_alpha1 <- eq3_rwr(P0_test, Q_test, alpha = 1.0)
max(abs(P_alpha1 - P0_test)) #P = P0, good. Dif is 0.                                                  


####larger network
set.seed(50)
n <- 50  #genes
m <- 20 #tumours
N <- 10 #features

self_loops <- cbind(seq_len(n), seq_len(n))

n_random_edges <- 300
random_i <- sample(seq_len(n), n_random_edges, replace = TRUE) #picks 300 random i genes, reps allowed
random_j <- sample(seq_len(n), n_random_edges, replace = TRUE) #picks 300 random j genes, reps allowed
random_edges <- cbind(random_i[random_i != random_j], random_j[random_i != random_j])

n_random_edges <- 300
random_edges <- cbind(
  sample(seq_len(n), n_random_edges, replace = TRUE), #picks 300 random i genes, reps allowed
  sample(seq_len(n), n_random_edges, replace = TRUE) #picks 300 random j genes, reps allowed
)
random_edges <- random_edges[random_edges[, 1] != random_edges[, 2], ] 
#filters out self-loops from here, puts everything else into 2col matrix

edges <- rbind(self_loops, random_edges) #puts self loops in
edges <- unique(edges) #gets rid of duplicates
colnames(edges) <- c("i", "j")
edges
E <- nrow(edges)
E #346 edges inc self loops

X <- matrix(rnorm(E * (N - 1)), E, N - 1) #making the feature matrix randomly from normal dist. N-1 bc we need one col for self loops below
self_loop_feat <- as.integer(edges[, 1] == edges[, 2]) #adding self loop (col 1 edges = col 2 edges)
X <- cbind(X, self_loop_feat) #putting them together
X

P0 <- matrix(rbinom(m * n, 1, 0.3), m, n) #m x n (1000) random P0 values generated from binomial dist, 30% chance of being 1, put in a matrix
y <- rnorm(m, mean = 1, sd = 0.3) #m (20) random values from normal dist, mean 1, stdv 0.3
y

#all good.

####eq1,2,3
net <- setup_network(edges, X, n)
w_test <- rnorm(N, sd = 0.5) #N (10) random values from normal dist, stdv 0.5
v_test <- rnorm(n, sd = 0.5) #n (50) random values from normal dist, stdc 0.5

act <- eq1_activation(w_test, net)
A <- act$A
A
a_edges <- act$a_edges
a_edges
Q <- eq2_transition(A, net)
Q
rowSums(Q)
P <- eq3_rwr(P0, Q, alpha = 0.5)
P

dim(A) #50 50 (n x n)                                                                        
dim(Q)  #50 50 (n x n)                                                                        
dim(P)  #20 50 (m x n)                                                                       

max(abs(rowSums(Q) - 1)) #Q row sums should add to 1. rowsums-1= basically 0 (8.550872e-09)

#Eq 1, 2, 3 all good. 

####checking eq4, does eq4 = eq5

J4 <- eq4_objective(w_test, v_test, P, y, lambda = 0.1)
J5 <- eq5_objective_sum(w_test, v_test, P, y, lambda = 0.1)
J4
J5

#all good. same. (2.152989)

#larger lambda, bigger J
J4_big <- eq4_objective(w_test, v_test, P, y, lambda = 1.0)
J4_big #19.04158                                                                

#all good.

####eq6 

dA_dw <- eq6_dA_dw(a_edges, net)
dim(dA_dw) #364 10 (E x N)  
dA_dw

#manual calc [k, l] = X[k, l] * a_edges[k] * (1 - a_edges[k])
slope <- a_edges * (1 - a_edges)
dA_dw_manual <- net$X * slope
max(abs(dA_dw - dA_dw_manual)) #should be 0, all good.                                       


#all good.

####eq7
#dQ/dwl, n x n

dA_dw_l <- dA_dw[, 1]
dQ_dwl <- eq7_dQ_dwl(dA_dw_l, A, net)
dim(dQ_dwl) #50 50 (n x n)                                                                    

#property: rows of Q sum to 1 (constant), so rows of dQ/dwl should sum to 0
row_sums_dQ <- rowSums(dQ_dwl)
row_sums_dQ #all rowsums are basically 0

#all good.


####eq9
#dP/dwl, m x n dense matrix

dP_dwl <- eq9_dP_dwl(P, Q, dQ_dwl, alpha = 0.5)

dim(dP_dwl) #20 50 (m x n)    
dP_dwl

#dQ = 0, dP goes to 0
zero_dQ <- Matrix(0, n, n, sparse = TRUE) #makes sparse matrix with only 0s
dP_zero <- eq9_dP_dwl(P, Q, zero_dQ, alpha = 0.5)
dP_zero #all 0

#all good

####eq10
dJ_dwl_test <- eq10_dJ_dwl(w_test[1], dP_dwl, v_test, y, P, lambda = 0.1)
dJ_dwl_test

length(dJ_dwl_test) #1 (scalar)                                                   

#all good.


####check if eq13=eq15
grad_full <- eq13_dJ_dv(v_test, P, y, lambda = 0.1)
grad_full
grad_scalar <- numeric(n)
for (i in seq_len(n)) {
  grad_scalar[i] <- eq15_dJ_dvi(i, v_test, P, y, lambda = 0.1)
}
grad_scalar

grad_full[4] #-0.1067779
grad_scalar[4] #-0.1067779 same

length(grad_full) #50 (n)                                                           
length(grad_scalar) #same                                                      

abs(grad_full - grad_scalar)  #0 eq13=eq15                                            

#all good.


####compute_dJ_dw
#serial run
dA_dw_full <- eq6_dA_dw(a_edges, net)
dJ_dw_serial <- compute_dJ_dw(w_test, v_test, A, Q, P, y, net, dA_dw_full,
                              alpha = 0.5, lambda = 0.1,
                              rtol = 1e-5, atol = 1e-8, max_iter = 100,
                              n_cores = 1) #uses 1 core so serial

length(dJ_dw_serial) #10 (N)                                                         
dJ_dw_serial

dJ_dw_parallel <- compute_dJ_dw(w_test, v_test, A, Q, P, y, net, dA_dw_full,
                                  alpha = 0.5, lambda = 0.1,
                                  rtol = 1e-5, atol = 1e-8, max_iter = 100,
                                  n_cores = 2)
dJ_dw_parallel
abs(dJ_dw_serial - dJ_dw_parallel)  #all 0, all good. 

#all good

####full
#J should decrease over time

result <- rwr_lasso_train(
  P0 = P0,
  edges = edges,
  X = X,
  y = y,
  alpha = 0.5,
  lambda = 0.01,
  eta = 0.01,
  max_outer = 200,
  n_cores = 3
)

result 

result$iterations #200, hit max                                                    
length(result$w) #10 (N)     
result$w
length(result$v) #50 (n) 
result$v

head(result$J_history, 5) #first 5 J
tail(result$J_history, 5) #last 5 J  
#J decreases over iterations

result$J_history[1] #first J 0.4943949                                                  
result$J_history[length(result$J_history)] #last J (in this case 200th) 0.05848343

#plot J
plot(result$J_history, type = "l", xlab = "Iteration", ylab = "J")

