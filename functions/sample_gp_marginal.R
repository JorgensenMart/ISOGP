sample_gp_marginal <- function(model, x_batch, samples  = 1, K_q = NULL, K_MN = NULL, joint_cov = FALSE){
  if(joint_cov == FALSE){
    L <- get_mu_and_var(x_batch, model, K_q = K_q, K_MN = K_MN)
    mu <- L$mean; var <- L$var #NxDxNU
    if(length(mu$get_shape()$as_list()) == 3){
      w <- tf$random_normal(shape(samples,
                                  mu$get_shape()$as_list()[1],
                                  mu$get_shape()$as_list()[2],
                                  mu$get_shape()$as_list()[3]),
                            mean = 0,
                            stddev = 1) #SxNxDxNU
      out <- mu + tf$sqrt(var)*w 
    }
    else if(model$is.WP == TRUE && model$constrain_deg_free == TRUE){
      var <- tf$expand_dims(var,as.integer(2))
      w <- tf$random_normal(shape(samples,
                                  mu$get_shape()$as_list()[1],
                                  mu$get_shape()$as_list()[2],
                                  model$deg_free),
                            mean = 0,
                            stddev = 1) #SxNxDxNU
      mu <- tf$tile(mu[,,NULL], as.integer(c(1,1,model$deg_free)))
      out <- mu + tf$sqrt(var)*w 
    }
    else{
      w <- tf$random_normal(shape(samples,
                                  mu$get_shape()$as_list()[1],
                                  mu$get_shape()$as_list()[2]),
                            mean = 0,
                            stddev = 1) #SxNxD
      out <- mu + tf$sqrt(var)*w 
    }
  } else{
    L <- get_mu_and_var(x_batch, model, K_q = K_q, K_MN = K_MN, joint_cov = TRUE)
    mu <- L$mean; #NxD or NxDxNU
    var <- L$var #DxNxN or DxNUxNxN
    if(length(mu$get_shape()$as_list()) == 3){
      w <- tf$random_normal(shape(samples,
                                  mu$get_shape()$as_list()[2],
                                  mu$get_shape()$as_list()[3],
                                  mu$get_shape()$as_list()[1],
                                  1),
                            mean = 0,
                            stddev = 1) #SxDxNUxNx1
      jitter = 1e-5
      chol <- tf$cholesky(var + jitter*tf$eye(mu$get_shape()$as_list()[1], dtype = tf$float32)) # DxNUxNxN
      chol <- tf$tile(chol[NULL,,,,], as.integer(c(samples,1,1,1,1))) # SxDxNUxNxN
      out <- tf$matmul(chol,w) # SxDxNUxNx1
      out <- mu + tf$transpose(out, as.integer(c(0,3,1,2,4)))[,,,,1] #SxNxDxNU
    }
    else if(model$is.WP == TRUE && model$constrain_deg_free == TRUE){
      var <- tf$expand_dims(var,as.integer(1)) # Dx1xNxN
      var <- tf$tile(var, as.integer(c(1,model$deg_free,1,1))) #DxNUxNxN
      w <- tf$random_normal(shape(samples,
                                  mu$get_shape()$as_list()[2],
                                  mu$get_shape()$as_list()[3],
                                  mu$get_shape()$as_list()[1],
                                  1),
                            mean = 0,
                            stddev = 1) #SxDxNUxNx1
      jitter = 1e-5
      chol <- tf$cholesky(var + jitter*tf$eye(mu$get_shape()$as_list()[1], dtype = tf$float32)) # DxNUxNxN
      chol <- tf$tile(chol[NULL,,,,], as.integer(c(samples,1,1,1,1))) # SxDxNUxNxN
      out <- tf$matmul(chol,w) # SxDxNUxNx1
      out <- mu + tf$transpose(out, as.integer(c(0,3,1,2,4)))[,,,,1] #SxNxDxNU
    }
    else{
      w <- tf$random_normal(shape(samples,
                                  mu$get_shape()$as_list()[2],
                                  mu$get_shape()$as_list()[1],
                                  1),
                            mean = 0,
                            stddev = 1) #SxDxNx1
      jitter = 1e-5
      chol <- tf$cholesky(var + jitter*tf$eye(mu$get_shape()$as_list()[1], dtype = tf$float32)) # DxNxN
      chol <- tf$tile(chol[NULL,,,], as.integer(c(samples,1,1,1))) # SxDxNxN
      out <- tf$matmul(chol,w) # SxDxNx1
      out <- mu + tf$transpose(out, as.integer(c(0,2,1,3)))[,,,1] #SxNxD
    }
  }
  return(out)
}
