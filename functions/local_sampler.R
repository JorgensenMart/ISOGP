#' A locality sampler
#' 
#' First function that returns K-NN for each point

kNN_for_each <- function(data, k){
  require(FNN)
  idx <- get.knnx(data, query = data, k = k)$nn.index
  return(idx)
}

local_sampler <- function(idx, psu = 50, ssu = 1){
  N <- length(idx[,1]); K <- length(idx[1,])
  psu = min(psu, N)
  ssu = min(ssu, K)
  
  first_samp <- sample(N, psu, replace = FALSE)
  second_samp <- sample(K, ssu, replace = FALSE)
  
  samp <- idx[first_samp, second_samp]
  
  samp <- unique(as.vector(samp))
  return(samp)
}
