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
fix_point_idx <- as.integer(args[2]) #Index of "fixated point"

sample_to_idx <- as.integer(args[3]) #Index of point to path to

iter <- as.integer(args[4]) #Training run to restore

#' Initialize session
session <- tf$Session()
session$run(tf$global_variables_initializer())
# Get parameters
saver <- tf$train$Saver()
load_file <- paste("results/mnist/",data_type,"_mnist_iteration",iter, sep = "")
saver$restore(session, load_file)


rm(R) # Freeing up some space from memory
# Load in mnist data
load("data/mnist/train.RDa")
train <- train$x[1:5000,]

fix_point <- matrix(train[fix_point_idx,], ncol = 784)
f <- tf$constant(fix_point, dtype = float_type)
sample_to <- matrix(train[sample_to_idx,], ncol = 784)
f_to <- tf$constant(sample_to, dtype = float_type)

library(FNN)
nn_to_fix_point <- knnx.index(train, query = fix_point, k = 2)[2]
y <- tf$constant(matrix(train[nn_to_fix_point,], ncol = 784))


# Embed at fixated point

#A <- tf$Variable(rnorm((D+1)*(D)/2,0,1), dtype = float_type)
A <- tf$Variable(tf$contrib$distributions$fill_triangular_inverse(diag(D)))
B <- tf$contrib$distributions$fill_triangular(A)
qr <- tf$qr(B)
Q <- qr$q

optimizer_rotation <- tf$train$AdamOptimizer(learning_rate = 0.001)

latent_neighbors <- latents$v_par$mu[c(fix_point_idx,nn_to_fix_point),] # 2xd

manifold_path <- sample_gp_marginal(model, x_batch = seq_d(latent_neighbors, 30)[1:30,], joint_cov = TRUE) # 1x30 x WIS x d
delta_z <- latent_neighbors[2,] - latent_neighbors[1,] # d x 1
delta_z <- delta_z / tf$constant(30, dtype = delta_z$dtype)
#z_norm <- tf$norm(delta_z) / tf$constant(30, dtype = delta_z$dtype)



delta_z <- tf$tile(delta_z[NULL,NULL,,NULL], as.integer(c(1,30,1,1))) #1x30xdx1
manifold_path <- tf$matmul(manifold_path,delta_z)[1,,,] # 30 x WIS x 1
#manifold_path <- manifold_path * z_norm

yhat <- tf$matmul(model$L_scale_matrix, tf$cumsum(manifold_path, axis = as.integer(0))[30,,]) # D x 1
yhat <- tf$matmul(Q,yhat)
rse <- tf$sqrt( tf$reduce_sum( tf$square( y - tf$clip_by_value(f + tf$transpose(yhat),0,1)) ) )

# Works (?) until here

train_rotation <- optimizer_rotation$minimize(rse, var_list = A)

session$run(tf$global_variables_initializer()) # Initialize new variables

#Plotting
show_digit <- function(arr784, col=gray(1:12/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

for(i in 1:500){
  print(session$run(rse))
  session$run(train_rotation)
  im1 <- session$run(tf$clip_by_value(f + tf$transpose(yhat), 0, 1))
  if(i %% 10 == 0){
    show_digit(im1)
  }
}

pdf(file = "results/mnist/out_digit.pdf")
show_digit(im1)
dev.off()

latent_neighbors <- latents$v_par$mu[c(fix_point_idx,sample_to_idx),]
trajectory <- sample_gp_marginal(model, seq_d(latent_neighbors,100)[1:100,], joint_cov = TRUE) # 1x100x WIS x d

delta_z <- latent_neighbors[2,] - latent_neighbors[1,] # d x 1
delta_z <- delta_z / tf$constant(100, dtype = delta_z$dtype)
#z_norm <- tf$norm(delta_z) / tf$constant(30, dtype = delta_z$dtype)



delta_z <- tf$tile(delta_z[NULL,NULL,,NULL], as.integer(c(1,100,1,1))) #1x100xdx1
trajectory <- tf$matmul(trajectory,delta_z)[1,,,] # 100 x WIS x 1
#manifold_path <- manifold_path * z_norm

trajectory <- tf$matmul(tf$tile(model$L_scale_matrix[NULL,,], as.integer(c(100,1,1))), 
                  tf$cumsum(trajectory, axis = as.integer(0))) # 100 x D x 1
trajectory <- tf$matmul(tf$tile(Q[NULL,,], as.integer(c(100,1,1))),trajectory) # 100 x D x 1

for(i in seq(10,100,by=10)){
  out <- session$run(tf$clip_by_value(f + tf$transpose(trajectory[i,,],0,1)))
  fn <- paste("results/mnist/trace",i,data_type,".pdf",sep = "")
  pdf(file = fn)
  show_digit(out)
  dev.off()  
}

# Plot 100 images on trajectory