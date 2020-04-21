#' Function that returns the censored Nakagami log-likelihood
#' 
#'  Input: Batches, cut-off
#'  Output: Value of each pair

censored_nakagami <- function(model,z_batch, dist_batch, cut_off, number_of_interpolants = 10, samples = 15){
  #' Input: z_batch is Nx2x(latent dim) (list of endpoints)
  #' d_batch is Nx1
  #' cutoff is the threshold value
  jitter = 1e-4 # This is big jitter, but numerical issue somewhere?
  #v_x <- tf$cast(model$v_par$v_x, dtype = tf$float64)
  v_x <- model$v_par$v_x
  K_MM <- build_kernel_matrix(model,v_x,v_x,equals = TRUE) + tf$constant(jitter, dtype = v_x$dtype) * tf$eye(as.integer(model$v_par$num_inducing), dtype = v_x$dtype)
  ## Go to float64 and back
    #K_MM <- tf$cast(K_MM, dtype = tf$float64)
    C_MM <- tf$cholesky(K_MM)
    
    #K_MM <- tf$cast(K_MM, dtype = tf$float32)
    #C_MM <- tf$cast(C_MM, dtype = tf$float32)
  ##
  K_q = list(Kmm = K_MM, Kmmchol = C_MM)
  
  #' Define uncensored and censored functions
  uncensored <- function(z,d){
    L <- arc_length(model, z,
                    number_of_interpolants = number_of_interpolants, samples = samples, K_q = K_q)
    m <- tf$maximum(L$m, 0.51); O <- tf$maximum(L$O,0.001); # Restrict m larger than 0.5
    res <- nakagami_pdf(d,m,O)
    return(res)
  }
  censored <- function(z,d){
    L <- arc_length(model, z,
                    number_of_interpolants = number_of_interpolants, samples = samples, K_q = K_q)
    m <- tf$maximum(L$m, 0.51); O <- tf$maximum(L$O,0.001); # Resctrict m larger than 0.5
    res <- nakagami_cdf(d,m,O) #Tail probability # cdf is tail
    return(res)
  }
  
  #' Function based on condition
  censored_llh <- function(z_and_d){
    z <- z_and_d[[1]]; d <- tf$squeeze(z_and_d[[2]])
    res <- tf$cond( d == cut_off,
             true_fn = function(){return(censored(z,d))},
             false_fn = function(){return(uncensored(z,d))} ) # Making it callable
    return(res)
  }
  
  #' Compute for each element
  out <- tf$map_fn(censored_llh, c(z_batch,dist_batch), dtype = z_batch$dtype,
                   parallel_iterations = 32L) # Holds likelihood value on each batch element
  return(out)
}

nakagami_pdf <- function(x,m,O){
  sig <- O / m # Reparametrize
  res <- - tf$lgamma(m) - m * tf$log(sig) + (tf$constant(2, dtype = m$dtype)*m - tf$constant(1, dtype = m$dtype)) * tf$log(x) - tf$square(x) / sig 
  return(res)
  # Returns log-pdf
}

nakagami_cdf <- function(x,m,O){
  #res <- tf$log( 1 - tf$math$igamma(m, (m / O) * x^2) )
  res <- tf$log(tf$constant(1, dtype = m$dtype) - tf$math$igamma(m, (m / O) * x^2) + tf$constant(1e-8, dtype = m$dtype))
  return(res)
  # Return log-cdf
}
