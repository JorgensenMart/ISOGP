# MNIST Lexicographic

library(vegan)
library(parallel)

load("data/mnist/train.RDa")
y <- train$y[1:5000]
x <- train$x[1:5000,]
N <- 5000

A <- matrix(rep(0,N*N), ncol = N)
cutoff <- 30

lex_dist <- function(i,j){
  if(y[i] == y[j]){
    return(min(2*cutoff,norm(matrix(x[i,], ncol = 28) - matrix(x[j,], ncol = 28))))
  }
  else{
    return(cutoff)
  }
}

for(i in 1:(N-1)){
  time0 <- Sys.time()
  cat("Iteration:",i,"of",N,"\n")
  out <- mclapply((i+1):N,lex_dist, i = i, mc.cores = getOption("mc.cores", 32L))
  out <- unlist(out, use.names = FALSE)
  A[i,(i+1):N] = out
  time_end <- Sys.time()
  print(time_end - time0)
}

A <- A + t(A)

save(A, file = "data/mnist/dist_object_lex.RDa")

z <- cmdscale(A, k = 2)

save(z, file = "data/mnist/init_location_lex.RDa")

pdf(file = "init_lex.pdf")
plot(z)
dev.off()