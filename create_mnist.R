# Load MNIST

load_mnist <- function(){
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train <<- load_image_file('data/mnist/train-image-idx3-ubyte')
  #test <<- load_image_file('mnist/t10k-images-idx3-ubyte')
  
  train$y <<- load_label_file('data/mnist/train-labels-idx1-ubyte')
  #test$y <<- load_label_file('mnist/t10k-labels-idx1-ubyte')  
}


show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

library(OpenImageR)

find_invariant_distance <- function(idx1,idx2){
  image1 <- matrix(subtrain$x[idx1,], nrow = 28)
  image2 <- matrix(subtrain$x[idx2,], nrow = 28)
  dist <- Inf
  out <- mclapply(seq(0,355, by =5),rotateImage, image = image2, mc.cores = getOption("mc.cores", 4L))
  disttoimage1 <- function(im2){norm(image1 - im2)}
  norms <- mclapply(out,disttoimage1, mc.cores = getOption("mc.cores", 4L))
  norms <- unlist(norms, use.names = FALSE)
  return(min(norms))
}

load_mnist()

subtrain <- list(n = 5000, x = train$x[1:5000,]/255, y = train$y[1:5000])
rm(train)

plot_image <- function(idx){
  image(matrix(255*subtrain$x[idx,], nrow = 28)[,28:1], col = gray(1:12/12))
}

N <- subtrain$n
A <- matrix(rep(0,N*N), ncol = N)

for(i in 1:(N-1)){
  time0 <- Sys.time()
  cat("Iteration:",i,"of",N,"\n")
  out <- mclapply((i+1):N,find_invariant_distance, idx1 = i, mc.cores = getOptions("mc.cores", 16L))
  out <- unlist(out, use.names = FALSE)
  A[i,(i+1):N] = out
  time_end <- Sys.time()
  print(time_end - time0)
}
A
A <- A + t(A)
save(A, file = "data/mnist/dist_object_invariance.RDa")

