# Write function that takes in location and returns E[det{J^t J}].

measure_from_metric <- function(model,location){
  J <- sample_gp_marginal(model, x_batch = location, samples = 100, joint_cov = 100) # Sx1x out_dim x in_dim
  J <- tf$squeeze(J) #Sx out_dim x in_dim
  L <- tf$tile(model$L_scale_matrix[NULL,,], as.integer(c(100,1,1))) #SxDxout_dim
  J <- tf$matmul(L,J) # S x D x in_dim --- This is Jacobian
  out <- tf$linalg$det(tf$matmul(J,J, transpose_a = TRUE)) # S
  out <- tf$reduce_mean(out)
  return(out)
}

# Script begins
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

my_frame <- expand.grid(seq(-2.5,2.5,length.out = 48),seq(-2.5,2.5,length.out = 48))
my_frame$E <- rep(0, 48^2)


