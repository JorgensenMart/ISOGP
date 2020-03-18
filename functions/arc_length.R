## Function that computes arc-length of functions

#' Input: GP-model and embedded latent "end points" that are linearly interpolated.
#' 
#' Output: Empirical first and second moments of squared arc-length.

arc_length <- function(model, z_end, number_of_interpolants = 10, samples = 15){
  zs <- seq_d(z_end, number_of_interpolants = number_of_interpolants) # Function in utils
  jitter = 1e-5
  K_MM <- build_kernel_matrix(model,model$v_par$v_x,model$v_par$v_x,equals = TRUE) + jitter*tf$eye(as.integer(model$v_par$num_inducing))
  C_MM <- tf$cholesky(K_MM)
  K_q = list(Kmm = K_MM, Kmmchol = C_MM)
  js <- sample_gp_marginal(model, zs[1:number_of_interpolants,], samples = samples, 
                           joint_cov = TRUE, 
                           K_q = K_q) 
  #' js is SxNxDxd
  delta_z <- z_end[2,] - z_end[1,]
  l_z <- tf$norm(delta_z) # Distance between points
  delta_z <- tf$tile(delta_z[NULL,NULL,,NULL], as.integer(c(samples,number_of_interpolants,1,1)) ) #SxNxdx1
  
  numeric_integrator <- function(js,zs){
    out <- tf$matmul(js,delta_z) # SxNxDx1 samples interpolants D 1
    out <- tf$norm(out, axis = as.integer(c(-2,-1))) # Euclidean norm
    out <- 1 / number_of_interpolants * out # Assuming equidistant interpolants
    out <- tf$reduce_sum(out, axis = as.integer(1)) # Size: Samples
    return(out)
  }
  
  out <- numeric_integrator(js,zs) # length of curve (s)
  #' Make it return moments of s^2
  #' 
  #m <- tf$reduce_mean(tf$square(out)); O <- m - tf$reduce_mean(out)^2
  O <- tf$reduce_mean(tf$square(out)); var <- tf$reduce_mean(tf$square(tf$square(out))) - O^2; # Variance of second moment
  m <- O^2 / var
  L <- list(m = m, O = O)
  return(L)
}

