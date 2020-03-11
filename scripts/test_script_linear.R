#' Test script
#' 
sapply(paste("functions/",list.files("functions/"), sep = ""), source)

#' Parameters and more
N <- 300 # Number of observations
m <- 100 # Number of inducing points
D <- 1 # Ambient dimension / data dimension
d <- 1 # Latent dimension

z_true = runif(N,-1,1)
x <- matrix(2*z_true, ncol = D)
#x <- cbind(x,x)

A <- as.matrix(dist(x))

B <- svd(t(x) %*% x)
W <- B$u[,1:d] # "Jacobian"
z <- x %*% B$u[,1:d] # PCA latents
# Scaling of A should also depend on dimensions, for comparisions with z

#z <- matrix(rnorm(N*d,0,1), ncol = d) # Initial value of latent points
#z <- cmdscale(A, k = d) # Makes a swirl (sub-optimal ?)
#z <- scale(z)
cut_off <- 1.07 # Seems like a good choice (based on TDA)
# Should be more automated

#' R is the distance matrix with the censored values replaced with the cutoff 
R <- matrix(rep(1,N^2), ncol = N)
R[which(A < cut_off, arr.ind = TRUE)] <- A[which(A < cut_off, arr.ind = TRUE)]
R[which(A >= cut_off, arr.ind = TRUE)] <- cut_off * R[which(A >= cut_off, arr.ind = TRUE)]


model <- make_gp_model(kern.type = "ARD",
                       input = z,
                       num_inducing = m,
                       in_dim = d, out_dim = D,
                       is.WP = TRUE, deg_free = d) # Should be unconstrained Wishart to generate Dxd matrices

model$kern$ARD$ls <- tf$Variable(rep(log(exp(5)-1),d))
model$kern$ARD$var <- tf$Variable(1, constraint = constrain_pos)

model$v_par$mu <- tf$Variable(aperm(array(rep(W,m), c(D,d,m)), perm = c(3,1,2)), dtype = tf$float32)
model$v_par$chol <- model$v_par$chol

rm(A) # Remove A from memory
rm(W)

latents <- make_gp_model(kern.type = "white",
                         input = z,
                         num_inducing = N,
                         in_dim = d, out_dim = d,
                         variational_is_diag = TRUE)


latents$kern$white$noise <- tf$constant(1, dtype = tf$float32) # GP hyperparameter is not variable here
#latents$v_par$v_x <- tf$Variable(z, dtype = tf$float32) # Latents to be marginalized
latents$v_par$mu <- tf$Variable(z, dtype = tf$float32)
latents$v_par$chol <- 1e-4 *latents$v_par$chol
# Make smarter inizialization of z

p <- 10 # Number of points in batch

I_batch <- tf$placeholder(dtype = tf$int32, shape = as.integer(c(p*(p-1)/2,2))) #{p(p-1)/2}x2

z_batch <- tf$gather(latents$v_par$mu, I_batch) +
  tf$gather(tf$transpose(latents$v_par$chol), I_batch) * 
  tf$random_normal(as.integer(c(p*(p-1)/2,2,d)))
# z_batch is (mini-batch) sampled from q(z)

dist_batch <- float_32(tf$gather_nd(R, I_batch)) # N,
# check that batches match reality

trainer <- tf$train$AdamOptimizer(learning_rate = 0.01)

driver <- censored_nakagami(model, z_batch, dist_batch, cut_off, number_of_interpolants = 10, samples = 15)
an_arc_length <- arc_length(model, z_batch[1,,])
loss <- tf$reduce_mean(driver) #- compute_kl(model) / as.double(N) - compute_kl(latents) / as.double(N) # Add K_q for latents
optimizer <- trainer$minimize(-loss)

#' Initialize session
session <- tf$Session()
session$run(tf$global_variables_initializer())

#' Training

iterations <- 2000

J <- sample(N, p, replace = FALSE) - 1 # Validation batch
test_batch <- dict(I_batch = batch_to_pairs(J))
for(i in 1:iterations){
  I <- sample(N, p, replace = FALSE) - 1 # Index of selected points in sample (tensorflow uses 0-indexing)
  batch_dict <- dict(I_batch = batch_to_pairs(I))
  session$run(optimizer, feed_dict = batch_dict)
  if(i %% 5 == 0){
    print(session$run(loss, feed_dict = test_batch))
    print(session$run(model$kern$ARD, feed_dict = test_batch))
    print(session$run(driver, feed_dict = test_batch))
    print(session$run(z_batch, feed_dict = test_batch))
    #plot(session$run(latents$v_par$mu, feed_dict = batch_dict))
  }
}

saver <- tf$train$Saver()
saver$save(session, "my_model")
