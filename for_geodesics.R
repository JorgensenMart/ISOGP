# Write function that takes in location and returns E[det{J^t J}].

metric <- function(model,location){
  J <- sample_gp_marginal(model, x_batch = location, samples = 100, joint_cov = TRUE) # Sx1x out_dim x in_dim
  J <- tf$squeeze(J) #Sx out_dim x in_dim
  L <- tf$tile(model$L_scale_matrix[NULL,,], as.integer(c(100,1,1))) #SxDxout_dim
  J <- tf$matmul(L,J) # S x D x in_dim --- This is Jacobian
  out <- tf$matmul(J,J,transpose_a = TRUE) # S x in_dim x in_dim
  out <- tf$reduce_mean(out, axis = 0L) # in_dim x in_dim
  #out <- tf$linalg$det(tf$matmul(J,J, transpose_a = TRUE)) # S
  #out <- tf$reduce_mean(out)
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

iter <- args[2]

my_frame <- expand.grid(seq(1,3,by = 0.05),seq(-1,1,by = 0.05))
#my_frame$E <- rep(0, 56^2)

#' Initialize session
session <- tf$Session()
session$run(tf$global_variables_initializer())
# Get parameters
saver <- tf$train$Saver()
load_file <- paste("results/mnist/",data_type,"_mnist_iteration",iter, sep = "")
saver$restore(session, load_file)

load("data/mnist/5pca.RDa")
model$L_scale_matrix <- tf$constant(sqrt(max(var(z))) * W, dtype = float_type)
rm(W)

rm(R) # Freeing up some space from memory

#for(i in 1:48^2){
#  location <- tf$expand_dims(tf$constant(matrix(my_frame[i,1:2],ncol = 2), dtype = float_type), 0L)
#  my_frame$E[i] <- session$run(measure_from_metric(model, location = location))
#  print(i)
#}
K <- length(my_frame[,1])
my_array <- array(rep(NA, K*4), dim = c(K,2,2))

place_idx <- tf$placeholder(shape = c(2),tf$float64)
meanJtJ <- metric(model,tf$expand_dims(place_idx, 0L))
session$run(tf$global_variables_initializer())
for(i in 1:K){
  #location <- tf$expand_dims(tf$constant(matrix(my_frame[i,1:2],ncol = 2), dtype = float_type), 0L)
  my_batch <- dict(place_idx = matrix(my_frame[i,1:2],ncol = 2))
  meanJtJ_out <- session$run(meanJtJ, feed_dict = my_batch)
  my_array[i,,] <- meanJtJ_out
  print(i)
}

library(R.matlab)
writeMat("results/mnist/to_mat.mat", locations = as.matrix(my_frame), JtJ = my_array)
#fn <- paste("results/mnist/",data_type,"_Edet",".csv", sep = "")
#write.csv(my_frame, file = fn)
