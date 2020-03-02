#' Test script
#' 

#' Downloading the swissroll with 1600 observations
swissroll <- read.table('http://people.cs.uchicago.edu/~dinoj/manifold/swissroll.dat')

#' Parameters and more
N <- 1600 # Number of observations
m <- 100 # Number of inducing points
D <- 3 # Ambient dimension / data dimension
d <- 2 # Latent dimension

z <- matrix(rnorm(N*d,0,1), ncol = d) # Initial value of latent points

A <- as.matrix(dist(swissroll))

cut_off <- median(A) # A bad choice

#' R is the distance matrix with the censored values replaced with the cutoff 
R <- matrix(rep(1,N^2), ncol = N)
R[which(A < cut_off, arr.ind = TRUE)] <- A[which(A < cut_off, arr.ind = TRUE)]
R[which(A >= cut_off, arr.ind = TRUE)] <- cut_off * R[which(A >= cut_off, arr.ind = TRUE)]

model <- make_gp_model(kern.type = "ARD",
                       input = z,
                       num_inducing = m,
                       in_dim = d, out_dim = D,
                       is.WP = TRUE, deg_free = d) # Should be unconstrained Wishart to generate Dxd matrices

latents <- tf$Variable(z, dtype = tf$float32)

p <- 50 # Number of points in batch

I_batch <- tf$placeholder(dtype = tf$int32, shape = as.integer(p*(p-1)/2,2)) #Nx2

z_batch <- tf$transpose(tf$gather(latents, I_batch), as.integer(c(0,2,1))) # Nx2xd
dist_batch <- tf$gather_nd(R, I_batch) # N,

# check that batches match reality

trainer <- tf$train$AdamOptimizer(learning_rate = 0.01) 

driver <- censored_nakagami(model, z_batch, dist_batch, cut_off, number_of_interpolants = 10, samples = 15)
loss <- tf$reduce_mean(driver) # Add KL terms
optimizer <- trainer$minimize(-loss)

#' Initialize session
session <- tf$Session()
session$run(tf$global_variables_initializer())

#' Training

iterations <- 100

for(i in 1:iterations){
  I <- sample(N, p, replace = FALSE) - 1 # Index of selected points in sample (tensorflow uses 0-indexing)
  batch_dict <- dict(I_batch = batch_to_pairs(I))
  session$run(optimizer, feed_dict = batch_dict)
}