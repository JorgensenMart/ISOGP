#' Test script
#' 
sapply(paste("functions/",list.files("functions/"), sep = ""), source)

#' Parameters and more
N <- 20 # Number of observations
m <- 50 # Number of inducing points
D <- 2 # Ambient dimension / data dimension
d <- 2 # Latent dimension
float_type = tf$float64

grid <- matrix(c(seq(1,N, by = 1),rep(0,N)), ncol = D)
#grid <- scale(grid)
A <- as.matrix(dist(grid))

B <- svd(t(grid) %*% grid)

z <- matrix(rnorm(N*d,0,1), ncol = d)

W <- B$u
#z <- grid %*% W
cut_off <- 1.7 # 
# Should be more automated

#' R is the distance matrix with the censored values replaced with the cutoff 
R <- matrix(rep(1,N^2), ncol = N)
R[which(A < cut_off, arr.ind = TRUE)] <- A[which(A < cut_off, arr.ind = TRUE)]
R[which(A >= cut_off, arr.ind = TRUE)] <- cut_off * R[which(A >= cut_off, arr.ind = TRUE)]

#prior_mean <- function(s){ # This makes prior mean "diagonal"
#  N <- s$get_shape()$as_list()[1]
#  if(model$is.WP == FALSE){
#    a <- tf$ones(shape(N,D), dtype = float_type)
#  } else{
#    a <- tf$constant(W, dtype = float_type)
#    a <- tf$tile(a[NULL,,], as.integer(c(N,1,1)))
#  }
#  return(a)
#}
  
model <- make_gp_model(kern.type = "ARD",
                       input = z,
                       num_inducing = m,
                       in_dim = d, out_dim = D,
                       is.WP = TRUE, deg_free = d,
                       #mf = prior_mean, 
                       float_type = float_type) # Should be unconstrained Wishart to generate Dxd matrices

model$kern$ARD$ls <- tf$Variable(rep(log(exp(3)-1),d), dtype = float_type)
model$kern$ARD$var <- tf$Variable(2, constraint = constrain_pos, dtype = float_type)

model$v_par$mu <- tf$Variable(aperm(array(rep(W,m), c(D,d,m)), perm = c(3,1,2)), dtype = float_type)
model$v_par$chol <- tf$Variable(array(rep(sqrt(0.1), D*d*((m*m - m) / 2 + m)) , c(D,d,((m*m - m) / 2 + m))), dtype = float_type)
#model$v_par$chol <- sqrt(0.1)*model$v_par$chol

rm(A) # Remove A from memory
rm(B)


latents <- make_gp_model(kern.type = "white",
                         input = z,
                         num_inducing = N,
                         in_dim = d, out_dim = d,
                         variational_is_diag = TRUE,
                         float_type = float_type)


latents$kern$white$noise <- tf$constant(1, dtype = float_type) # GP hyperparameter is not variable here
#latents$v_par$v_x <- tf$Variable(z, dtype = tf$float32) # Latents to be marginalized
latents$v_par$mu <- tf$Variable(z, dtype = float_type)
latents$v_par$chol <- tf$Variable(matrix( rep(1e-3, d*N), ncol = N  ), dtype = float_type , constraint = constrain_pos)
# Make smarter inizialization of z

#model$v_par$v_x <- latents$v_par$mu

#I_batch <- tf$placeholder(dtype = tf$int32, shape = as.integer(c(p*(p-1)/2,2))) #{p(p-1)/2}x2
I_batch <- tf$placeholder(tf$int32, shape(NULL,2L))

z_batch <- tf$transpose(tf$gather(latents$v_par$mu, I_batch), as.integer(c(0,2,1))) +
  tf$transpose(tf$gather(tf$transpose(latents$v_par$chol), I_batch), as.integer(c(0,2,1))) * 
  tf$random_normal(tf$shape(tf$transpose(tf$gather(latents$v_par$mu, I_batch), as.integer(c(0,2,1)))), dtype = float_type)
# z_batch is (mini-batch) sampled from q(z) # Px

dist_batch <- tf$cast(tf$gather_nd(R, I_batch), dtype = float_type) # N,
# check that batches match reality

trainer <- tf$train$AdamOptimizer(learning_rate = 0.01, beta1 = 0.9)
reset_trainer <- tf$variables_initializer(trainer$variables())

driver <- censored_nakagami(model, z_batch, dist_batch, cut_off, number_of_interpolants = 10, samples = 20)
loss <- tf$reduce_mean(driver)  - compute_kl(model) / tf$constant(N*(N-1)/2, dtype = float_type) #- compute_kl(latents) / as.double(N)# Add K_q for latents

#grad <- trainer$compute_gradients(-loss)
#capped_grap <- tf$clip_by_value(grad, -10,10)

optimizer_lat <- trainer$minimize(-loss, var_list = list(latents$v_par$mu, latents$v_par$chol))
optimizer_model <- trainer$minimize(-loss, var_list = list(model$kern$ARD, model$v_par$v_x, model$v_par$mu, model$v_par$chol))

#' Initialize session
session <- tf$Session()
session$run(tf$global_variables_initializer())

#' Training

iterations <- 200000
p <- min(N,30)

J <- sample(N, N, replace = FALSE) - 1 # Validation batch
test_batch <- dict(I_batch = batch_to_pairs(J))
idx <- kNN_for_each(grid, k = 3)
Switch = TRUE
for(i in 1:iterations){
  # Training
  if( i %% 200 == 0){
    if(Switch == TRUE){
      Switch = FALSE
      session$run(reset_trainer)
    } else{
      if( i %% 200 == 0){
      Switch = TRUE
      session$run(reset_trainer)
      }
    }
  }
  if(Switch == TRUE){
    I <- sample(N, p) - 1
    #I <- local_sampler(idx, psu = 2, ssu = 2) - 1
    batch_dict <- dict(I_batch = batch_to_pairs(I))
    session$run(optimizer_model, feed_dict = batch_dict)
    #print(session$run(driver, feed_dict = batch_dict))
  } else{
    #I <- local_sampler(idx, psu = 2, ssu = 3) - 1
    I <- sample(N,p) - 1
    batch_dict <- dict(I_batch = batch_to_pairs(I))
    session$run(optimizer_lat, feed_dict = batch_dict)
  }
  
  # Printing
  if(i %% 20 == 0){
    cat("Iterations:", i, "\n")
    cat("ELBO:", session$run(loss, feed_dict = test_batch), "\n")
    plot(session$run(latents$v_par$mu),type = 'o-')
    cat("Switch:", Switch, "\n")
    #print(session$run(model$v_par$mu))
  }
}

#saver <- tf$train$Saver()
#saver$save(session, "my_model")