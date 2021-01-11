#MNIST
sapply(paste("functions/",list.files("functions/"), sep = ""), source)

#' Parameters and more
N <- 1440 # Number of observations
m <- 100 # Number of inducing points
D <- 16384 # Ambient dimension / data dimension
WIS = 5
d <- 2 # Latent dimension
float_type = tf$float64
#################### Not needed ?
load("data/coil20/dist_object.RDa") # This loads in the pairwise distances (A)
A <- as.matrix(A)
#####################

load("data/coil20/init_location.RDa") # Loads in the init locations (initialized with isomap)
z <- (z-colMeans(z))/sqrt(max(var(z))) # Scaling here
####################### z could be scaled ?
cut_off <- 22
#######################
#' R is the distance matrix with the censored values replaced with the cutoff 
R <- matrix(rep(1,N^2), ncol = N)
R[which(A < cut_off, arr.ind = TRUE)] <- A[which(A < cut_off, arr.ind = TRUE)]
R[which(A >= cut_off, arr.ind = TRUE)] <- cut_off * R[which(A >= cut_off, arr.ind = TRUE)]

rm(A) # Remove A from memory
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
                       wis_factor = WIS, 
                       float_type = float_type) # Should be unconstrained Wishart to generate Dxd matrices

model$kern$ARD$ls <- tf$Variable(rep(log(exp(2)-1),d), dtype = float_type)
model$kern$ARD$var <- tf$Variable(2, constraint = constrain_pos, dtype = float_type)

#model$v_par$mu <- tf$Variable(aperm(array(rep(W,m), c(D,d,m)), perm = c(3,1,2)), dtype = float_type)



latents <- make_gp_model(kern.type = "white",
                         input = z,
                         num_inducing = N,
                         in_dim = d, out_dim = d,
                         variational_is_diag = TRUE,
                         float_type = float_type)


latents$kern$white$noise <- tf$constant(1, dtype = float_type) # GP hyperparameter is not variable here
latents$v_par$mu <- tf$Variable(z, dtype = float_type)
latents$v_par$chol <- tf$Variable(matrix( rep(1e-6, d*N), ncol = N  ), dtype = float_type, constraint = constrain_pos)

I_batch <- tf$placeholder(tf$int32, shape(NULL,2L))

z_batch <- tf$transpose(tf$gather(latents$v_par$mu, I_batch), as.integer(c(0,2,1))) +
  tf$transpose(tf$gather(tf$transpose(latents$v_par$chol), I_batch), as.integer(c(0,2,1))) * 
  tf$random_normal(tf$shape(tf$transpose(tf$gather(latents$v_par$mu, I_batch), as.integer(c(0,2,1)))), dtype = float_type)

dist_batch <- tf$cast(tf$gather_nd(R, I_batch), dtype = float_type) # N,

warm_start_model <- tf$placeholder(dtype = float_type, shape = c())
warm_start_latents <- tf$placeholder(dtype = float_type, shape = c())

trainer <- tf$train$AdamOptimizer(learning_rate = 3e-3)
reset_trainer <- tf$variables_initializer(trainer$variables())

driver <- censored_nakagami(model, z_batch, dist_batch, cut_off, number_of_interpolants = 8L, samples = 20L)
llh <- tf$reduce_mean(driver)
KL <- warm_start_model * compute_kl(model) / tf$constant(N*(N-1)/2, dtype = float_type) + warm_start_latents * compute_kl(latents) / tf$constant(N*(N-1)/2, dtype = float_type)

optimizer_model <- trainer$minimize( - (llh - KL), var_list = list(model$kern$ARD, 
                                                                   model$v_par$v_x, 
                                                                   model$v_par$mu, model$v_par$chol))
optimizer_latents <- trainer$minimize( - (llh - KL), var_list = list(latents$v_par$mu, latents$v_par$chol))