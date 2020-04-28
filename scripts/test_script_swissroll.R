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

#B <- svd(t(swiss) %*% swiss)
#W <- B$u[,1:d] # "Jacobian"
#z <- swiss %*% B$u[,1:d] # PCA latents
library(vegan)
Z <- isomap(A, ndim = d, k = 5) # Initializes with isomap
z <- Z$points

cut_off <- 0.4 # Should be more automated

#' R is the distance matrix with the censored values replaced with the cutoff 
R <- matrix(rep(1,N^2), ncol = N)
R[which(A < cut_off, arr.ind = TRUE)] <- A[which(A < cut_off, arr.ind = TRUE)]
R[which(A >= cut_off, arr.ind = TRUE)] <- cut_off * R[which(A >= cut_off, arr.ind = TRUE)]

#prior_mean <- function(s){ # This makes prior mean "diagonal"
#  N <- s$get_shape()$as_list()[1]
#  a <- tf$constant(W, dtype = float_type)
#  a <- tf$tile(a[NULL,,], as.integer(c(N,1,1)))
#  return(a)
#}

model <- make_gp_model(kern.type = "ARD",
                       input = z,
                       num_inducing = m,
                       in_dim = d, out_dim = D,
                       is.WP = TRUE, deg_free = d,
                       #mf = prior_mean, 
                       float_type = float_type) # Should be unconstrained Wishart to generate Dxd matrices

model$kern$ARD$ls <- tf$Variable(rep(log(exp(2)-1),d), dtype = float_type)
model$kern$ARD$var <- tf$Variable(2, constraint = constrain_pos, dtype = float_type)

#model$v_par$mu <- tf$Variable(aperm(array(rep(W,m), c(D,d,m)), perm = c(3,1,2)), dtype = float_type)

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

warm_start_model <- tf$placeholder(dtype = float_type, shape = c())
warm_start_latents <- tf$placeholder(dtype = float_type, shape = c())

trainer <- tf$train$AdamOptimizer(learning_rate = 0.01)
reset_trainer <- tf$variables_initializer(trainer$variables())

driver <- censored_nakagami(model, z_batch, dist_batch, cut_off, number_of_interpolants = 10, samples = 15)
llh <- tf$reduce_mean(driver)
KL <- warm_start_model * compute_kl(model) / tf$constant(N*(N-1)/2, dtype = float_type) + warm_start_latents * compute_kl(latents) / tf$constant(N*(N-1)/2, dtype = float_type)

optimizer_model <- trainer$minimize( - (llh - KL), var_list = list(model$kern$ARD, model$v_par$v_x, model$v_par$mu, model$v_par$chol))
optimizer_latents <- trainer$minimize( - (llh - KL), var_list = list(latents$v_par$mu, latents$v_par$chol))

#' Initialize session
session <- tf$Session()
session$run(tf$global_variables_initializer())

saver <- tf$train$Saver()
#' Training

iterations <- 50000
p <- 50

J <- sample(N, p, replace = FALSE) - 1 # Validation batch
test_batch <- dict(I_batch = batch_to_pairs(J), warm_start_model = 1, warm_start_latents = 1)
idx <- kNN_for_each(swiss, k = 10)
Switch = TRUE
warm_up <- 3000
for(i in 1:iterations){
  # Training
  if(i < warm_up){
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
    if( i < warm_up){
      I <- sample(N, p) - 1
      warm_start_latents_ <- 0
      warm_start_model_ <- min(1, i/2000)
    } else{
      I <- local_sampler(idx, psu = 10, ssu = 5) - 1
      warm_start_latents_ <- min(1, (i - warm_up)/20000)
      warm_start_model_ <- min(1, i/2000)
    }
    batch_dict <- dict(I_batch = batch_to_pairs(I), 
                       warm_start_latents = warm_start_latents_, warm_start_model = warm_start_model_)
    session$run(optimizer_model, feed_dict = batch_dict)
  } else{
    #I <- sample(N,p) - 1
    I <- local_sampler(idx, psu = 5, ssu = 9) - 1
    warm_start_latents_ <- min(1, (i - warm_up)/20000)
    warm_start_model_ <- min(1, i/2000)
    batch_dict <- dict(I_batch = batch_to_pairs(I), 
                       warm_start_latents = warm_start_latents_, warm_start_model = warm_start_model_)
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
    filename <- paste("results/swissroll/loc_sampler_swissroll_iteration",i, sep = "")
    saver$save(session, filename)
  }
}
