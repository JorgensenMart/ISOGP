#TDA Playing around

library(TDA)

swissroll <- read.table('http://people.cs.uchicago.edu/~dinoj/manifold/swissroll.dat')

A <- dist(scale(swissroll))

diagram <- ripsDiag(A, maxdimension = 1, 
                    maxscale = 1.5, 
                    dist = "arbitrary", library = "Dionysus")



function(c_deaths){
  out <- rep(NaN,11)
  for(k in 5:15){
  k <- 12
  c_outliers <- lofactor(c_deaths, k = k)
  S <- sort(c_outliers, TRUE)[1:10]
  plot(1:10,S)
  R <- runif(100,S[10],S[1])
  counter <- function(r){ return(sum(S > r))}
  H <- sapply(R, counter)
  out[k-4] <- 
  }
}