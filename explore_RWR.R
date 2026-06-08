# Construct a network with 3 genes with 2 edges only
# 
# Gene 1 -> Gene 2 
# Gene 1 -> Gene 3
# 
# The objective is to evaluate explore the effects of the random walk with 
# restart transformation


rm(list=ls())
library(lhs)      # Latin Hypercube
library(ggplot2)  # 2D scatter plot
library(plotly)   # 3D plot

################################################################################
# 1. The sum of the scores is not changed by RWR
################################################################################
# Specify the initial condition for the 3 genes
P0 = c(1, 0, 0)
# P0 = c(0, 1, 0)

# Specify activity for the edges 1->2 and 1->3
a12 = 10
a13 = 1

# Construct the adjacency matrix
Q = matrix(c(1, a12, a13,
             0, 1, 0,
             0, 0, 1),
           byrow = TRUE,
           nrow = 3)
# Degree-normalise the adjacency matrix
Q[1,] = Q[1,]/sum(Q[1,])
Q[2,] = Q[2,]/sum(Q[2,])
Q[3,] = Q[3,]/sum(Q[3,])
Q

# Set the value of alpha for restart probability
alpha = 0.3

# Calculate the propagated profile (P_RWR) using the analytic solution
P_RWR = alpha * P0 %*% solve(diag(3) - (1 - alpha)*Q)

# Display the initial and the propagated profiles for comparison
P0
P_RWR

# Display the sum of scores for the initial and the propagated profiles
sum(P0)
sum(P_RWR)

# RWR changes the distribution of scores. The total score of all nodes remain 
# the same. 

################################################################################
# Explore how the distribution of scores change with Q: 
# 1) Sample the initial scores using a 3-dimensional Latin Hypercube
# 2) Apply the same Q for all P0 vectors
# 3) Scatter plot: P0 v P_RWR
# 4) 3D plot: P0 v P_RWR
################################################################################

# For reproducibility
set.seed(42)

# Generate a sample for the 3 genes
num_samples <- 1000
P0 <- randomLHS(n = num_samples, k = 3)

# For simplicity in comparison, normalise the sum of each row to 1
P0 = P0/rowSums(P0)

# Apply RWR with Q specified in the previous section
P_RWR = alpha * P0 %*% solve(diag(3) - (1 - alpha)*Q)

# Collect the results into a data frame for the 2D scatter plot
out = data.frame(P0_1 = P0[,1],
                 P0_2 = P0[,2],
                 P0_3 = P0[,3],
                 P_RWR_1 = P_RWR[,1],
                 P_RWR_2 = P_RWR[,2],
                 P_RWR_3 = P_RWR[,3])

# Scatter plot
ggplot() +
  geom_point(data = out,
             aes(x = P0_1, y = P_RWR_1, col = as.factor("Node 1")), size = 1) +
  geom_point(data = out,
             aes(x = P0_2, y = P_RWR_2, col = as.factor("Node 2")), size = 1) +
  geom_point(data = out,
             aes(x = P0_3, y = P_RWR_3, col = as.factor("Node 3")), size = 1) +
  scale_x_continuous(expression(P[0]),
                     limits=c(0,1),
                     breaks = seq(0,1,0.2),
                     minor_breaks = seq(0,1,0.1)) +
  scale_y_continuous(expression(P[RWR]),
                     limits=c(0,1),
                     breaks = seq(0,1,0.2),
                     minor_breaks = seq(0,1,0.1)) +
  guides(col=guide_legend(title=NULL))+
  labs(title = expression(P[0]*" v "*P[RWR])) + # Graph Title
  theme_bw()+
  theme(legend.position = c(0.1, 0.85),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 15, hjust = 0.5),
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 12),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12))

# Check the range of 3 genes after RWR
range(P_RWR[,1])
range(P_RWR[,2])
range(P_RWR[,3])


# Collect the results into a data frame for the 3D scatter plot
dat2plotly = data.frame(P1 = c(P0[,1], P_RWR[,1]),
                        P2 = c(P0[,2], P_RWR[,2]),
                        P3 = c(P0[,3], P_RWR[,3]),
                        Type = c(rep("P0", num_samples), rep("P_RWR", num_samples)))


fig <- plot_ly(dat2plotly, x = ~P1, y = ~P2, z = ~P3, 
               color = ~Type, colors = c('#BF3030', '#2A4B7C'), size = 1)
# colors = c('#BF3030', '#2A4B7C', '#32B54A')
fig <- fig %>% add_markers()
fig <- fig %>% layout(scene = list(xaxis = list(title = 'Gene 1'),
                                   yaxis = list(title = 'Gene 2'),
                                   zaxis = list(title = 'Gene 3')))
fig

# All points are in the same plane, due to the normalisation at line 71.
# If the step at line 71 is skipped, you should expect P0 to distribute randomly
# in a cube, and P_RWR to distribute in a parallelepiped.

