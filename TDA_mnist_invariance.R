library(TDA)

load("data/mnist/dist_object_invariance.RDa")
diagram <- ripsDiag(A, maxdimension = 0, 
                    maxscale = 10,
                    dist = "arbitrary", library = "Dionysus")
pdf(file = "plots/TDA_mnist_invariance.pdf")
plot(diagram$diagram)
dev.off()
