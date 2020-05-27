# Embed and generate samples

args <- commandArgs(trailingOnly = TRUE)

if(args[1] == "mnist"){
  data_type <- "variant"
  source("scripts/model_mnist.R")
} else if(args[1] == "invariance"){
  data_type <- "invariant"
  source("scripts/model_mnist_invariance.R")
} else if(args[1] == "lex"){
  data_type <- "lex"
  source("scripts/model_mnist_lex.R")
}


fix_point_idx <- args[2] #Index of "fixated point"

sample_to_idx <- args[3] #Index of point to path to

iter <- args[4] #Training run to restore

rm(R) # Freeing up some space from memory
# Load in mnist data
load("data/mnist/train.RDa")
train <- train$x[1:5000,]

fix_point <- matrix(train[fix_point_idx,], ncol = 784)

sample_to <- matrix(train[sample_to_idx,], ncol = 784)

library(FNN)
nn_to_fix_point <- knnx.index(train, query = fix_point, k = 2)[2]

#' Initialize session
session <- tf$Session()
session$run(tf$global_variables_initializer())
# Get parameters
saver <- tf$train$Saver()
load_file <- paste("results/mnist/",data_type,"_mnist_iteration",iter, sep = "") 
saver$restore(session, load_file)

# Embed at fixated point

A <- tf$Variable(diag(D), dtype = float_type)
qr <- tf$qr(A)
Q <- qr$q

optimizer_rotation <- tf$train$AdamOptimizer(learning_rate = 0.05)

latent_neighbors <- tf$constant(latents$v_par$mu[c(fix_point,nn_to_fix_point),]) # 2xd

manifold_path <- sample_gp_marginal(model, x_batch = seq_d(latent_neighbors, 30)) # 30 x WIS x d
delta_z <- latent_neighbors[,2] - latent_neighbors[,1] #  
manifold_path <- tf$matmul
manifold_path <- tf$cumsum()
rmse <- tf$sqrt( tf$reduce_mean( tf$square( train[nn_to_fix_point,] - Q)))
train_rotation <- optimizer_rotation$minimize(rmse, var_list = A)
rmse <- 
for(i in 1:500){
  
}