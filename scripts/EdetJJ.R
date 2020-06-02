# Write function that takes in location and returns E[det{J^t J}].

measure_from_metric <- function(model,location){
  J <- sample_gp_marginal(model, x_batch = location, samples = 100, joint_cov = TRUE) # Sx1x out_dim x in_dim
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

iter <- args[2]
K = 55
my_frame <- expand.grid(seq(-2.5,3,length.out = K),seq(-2.4,2.4,length.out = K))
my_frame$E <- rep(0, K^2)

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

for(i in 1:K^2){
  location <- tf$expand_dims(tf$constant(matrix(my_frame[i,1:2],ncol = 2), dtype = float_type), 0L)
  my_frame$E[i] <- session$run(measure_from_metric(model, location = location))
  print(i)
}

fn <- paste("results/mnist/",data_type,"_Edet",".csv", sep = "")
write.csv(my_frame, file = fn)

mean_embedding <- session$run(latents$v_par$mu)
fn <- paste("results/mnist/",data_type,"_means",".csv", sep = "")
write.csv(mean_embedding, file = fn)
