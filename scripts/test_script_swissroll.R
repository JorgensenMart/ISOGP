#' Test script
#' 
sapply(paste("functions/",list.files("functions/"), sep = ""), source)

load("data/swissroll.Rda") # Loads swissroll

#' Parameters and more
N <- 2000 # Number of observations
m <- 100 # Number of inducing points
D <- 3 # Ambient dimension / data dimension
d <- 2 # Latent dimension
float_type = tf$float64

swiss <- scale(swissroll)
A <- as.matrix(dist(swiss))

B <- svd(t(swiss) %*% swiss)
W <- B$u[,1:d] # "Jacobian"
z <- swiss %*% B$u[,1:d] # PCA latents

cut_off <- 0.4 # Should be more automated

#' R is the distance matrix with the censored values replaced with the cutoff 
R <- matrix(rep(1,N^2), ncol = N)
R[which(A < cut_off, arr.ind = TRUE)] <- A[which(A < cut_off, arr.ind = TRUE)]
R[which(A >= cut_off, arr.ind = TRUE)] <- cut_off * R[which(A >= cut_off, arr.ind = TRUE)]

prior_mean <- function(s){ # This makes prior mean "diagonal"
  N <- s$get_shape()$as_list()[1]
  a <- tf$constant(W, dtype = float_type)
  a <- tf$tile(a[NULL,,], as.integer(c(N,1,1)))
  return(a)
}

model <- make_gp_model(kern.type = "ARD",
                       input = z,
                       num_inducing = m,
                       in_dim = d, out_dim = D,
                       is.WP = TRUE, deg_free = d,
                       mf = prior_mean, float_type = float_type) # Should be unconstrained Wishart to generate Dxd matrices

model$kern$ARD$ls <- tf$Variable(rep(log(exp(2)-1),d), dtype = float_type)
model$kern$ARD$var <- tf$Variable(2, constraint = constrain_pos, dtype = float_type)

model$v_par$mu <- tf$Variable(aperm(array(rep(W,m), c(D,d,m)), perm = c(3,1,2)), dtype = float_type)

rm(A) # Remove A from memory
rm(swissroll)

latents <- make_gp_model(kern.type = "white",
                         input = z,
                         num_inducing = N,
                         in_dim = d, out_dim = d,
                         variational_is_diag = TRUE,
                         float_type = float_type)


latents$kern$white$noise <- tf$constant(1, dtype = float_type) # GP hyperparameter is not variable here
latents$v_par$mu <- tf$Variable(z, dtype = float_type)
latents$v_par$chol <- tf$Variable(matrix( rep(1e-3, d*N), ncol = N  ), dtype = float_type, constraint = constrain_pos)

I_batch <- tf$placeholder(tf$int32, shape(NULL,2L))

z_batch <- tf$transpose(tf$gather(latents$v_par$mu, I_batch), as.integer(c(0,2,1))) +
  tf$transpose(tf$gather(tf$transpose(latents$v_par$chol), I_batch), as.integer(c(0,2,1))) * 
  tf$random_normal(tf$shape(tf$transpose(tf$gather(latents$v_par$mu, I_batch), as.integer(c(0,2,1)))), dtype = float_type)


dist_batch <- tf$cast(tf$gather_nd(R, I_batch), dtype = float_type) # N,


trainer <- tf$train$AdamOptimizer(learning_rate = 0.005)
reset_trainer <- tf$variables_initializer(trainer$variables())

driver <- censored_nakagami(model, z_batch, dist_batch, cut_off, number_of_interpolants = 10, samples = 15)
llh <- tf$reduce_mean(driver)
KL <- compute_kl(model) / tf$constant(N, dtype = float_type) #+ compute_kl(latents) / tf$constant(N, dtype = float_type)

optimizer_model <- trainer$minimize( - (llh - KL), var_list = list(model$kern$ARD, model$v_par$v_x, model$v_par$mu, model$v_par$chol))
optimizer_latents <- trainer$minimize( - (llh - KL), var_list = list(latents$v_par$mu, latents$v_par$chol))

#' Initialize session
session <- tf$Session()
session$run(tf$global_variables_initializer())

saver <- tf$train$Saver()
#' Training

iterations <- 50000
p <- 70

J <- sample(N, p, replace = FALSE) - 1 # Validation batch
test_batch <- dict(I_batch = batch_to_pairs(J))
#idx <- kNN_for_each(swiss, k = 30)
Switch = TRUE
for(i in 1:iterations){
  # Training
  if(i < 3000){
    Switch = TRUE
  } else{
    if( i %% 200 == 0){
      if(Switch == TRUE){
        Switch = FALSE
        session$run(reset_trainer)
      } else{
        Switch = TRUE
        session$run(reset_trainer)
      }
    }
  }
  if(Switch == TRUE){
    I <- sample(N, p) - 1
    batch_dict <- dict(I_batch = batch_to_pairs(I))
    session$run(optimizer_model, feed_dict = batch_dict)
  } else{
    I <- sample(N,p) - 1
    batch_dict <- dict(I_batch = batch_to_pairs(I))
    session$run(optimizer_latents, feed_dict = batch_dict)
  }
  
  # Printing
  if(i %% 50 == 0){
    printllh <- session$run(llh, feed_dict = test_batch)
    printkl <- session$run(KL, feed_dict = test_batch)
    cat("Iteration:", i, "\n")
    cat("Log likelihood:", printllh, "KL:", printkl, "\n")
    cat("ELBO:", printllh - printkl, "\n")
  }
  if(i %% 1000 == 0){ # Save a model every 1000 iterations
    filename <- paste("results/swissroll/swissroll_iteration",i, sep = "")
    saver$save(session, "swissroll_iter")
  }
}
