#Isomap
library(RDRToolbox)
A <- as.matrix(dist(scale(swissroll)))


im <- Isomap(swissroll, dims = 2, k = 10)

im_ <- isomap(B, epsilon = 0.4, ndim = 2)

swissroll <- SwissRoll()