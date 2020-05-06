library(vegan)

load("data/mnist/dist_object_invariance.RDa")

Z <- isomap(A, ndim = 2, k = 5)

z <- Z$points

save(z, file = "data/mnist/init_location_invariance.RDa")