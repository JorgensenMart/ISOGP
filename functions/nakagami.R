#' Function that returns the censored Nakagami log-likelihood
#' 
#'  Input: Batches, cut-off
#'  Output: Value of each pair

censored_nakagami <- function(model,z_batch, dist_batch, cut_off, number_of_interpolants = 10, samples = 15){
  #' Input: z_batch is Nx2x(latent dim) (list of endpoints)
  #' d_batch is Nx1
  #' cutoff is the threshold value
  
  #' Define uncensored and censored functions
  uncensored <- function(z,d){
    L <- arc_length(model, z,
                    number_of_interpolants = number_of_interpolants, samples = samples)
    m <- L$m; O <- L$O;
    res <- nakagami_pdf(d,m,O)
    return(res)
  }
  censored <- function(z,d){
    L <- arc_length(model, z,
                    number_of_interpolants = number_of_interpolants, samples = samples)
    m <- L$m; O <- L$O;
    res <- nakagami_cdf(d,m,O) #Tail probability # cdf is tail
    return(res)
  }
  
  #' Function based on condition
  censored_llh <- function(z_and_d){
    z <- tf$transpose(z_and_d[[1]]); d <- tf$squeeze(z_and_d[[2]])
    res <- tf$cond( d == cut_off,
             true_fn = function(){return(censored(z,d))},
             false_fn = function(){return(uncensored(z,d))} ) # Making it callable
    return(res)
  }
  
  #' Compute for each element
  out <- tf$map_fn(censored_llh, c(z_batch,dist_batch), dtype = tf$float32) # Holds likelihood value on each batch element
  return(out)
} 

nakagami_pdf <- function(x,m,O){
  sig <- O / m # Reparametrize
  res <- - tf$lgamma(m) - m * tf$log(sig) + (2*m - 1) * tf$log(x) - tf$square(x) / sig 
  return(res)
  # Returns log-pdf
}

nakagami_cdf <- function(x,m,O){
  #res <- tf$log( 1 - tf$math$igamma(m, (m / O) * x^2) )
  res <- tf$log( 1 - tf$math$igamma(m, (m / O) * x^2) + 1e-10)
  return(res)
  # Return log-cdf
}
