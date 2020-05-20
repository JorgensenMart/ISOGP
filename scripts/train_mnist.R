#Train mnist

source("scripts/model_mnist.R") # This builds the model

#' Initialize session
session <- tf$Session()
session$run(tf$global_variables_initializer())

load("data/mnist/5pca.RDa")
model$L_scale_matrix <- tf$constant(sqrt(max(var(z))) * W, dtype = float_type)
rm(W)
saver <- tf$train$Saver()
#' Training

iterations <- 100000
p <- 40

# For plotting:
source("functions/plot_at_iteration.R")

J <- sample(N, p, replace = FALSE) - 1 # Validation batch
test_batch <- dict(I_batch = batch_to_pairs(J), warm_start_model = 1, warm_start_latents = 1)
idx <- kNN_for_each(swiss, k = 6)
Switch = TRUE
warm_up <- 3000
for(i in 1:iterations){
  # Training
  if(i < warm_up){
    Switch = TRUE
  } else{
    if( i %% 200 == 0){
      if(Switch == TRUE){
        Switch = FALSE
        session$run(reset_trainer)
      } else{
        Switch = TRUE
        session$run(reset_trainer)
      }
    }
  }
  if(Switch == TRUE){
    if( i < warm_up){
      I <- sample(N, p) - 1
      warm_start_latents_ <- 0
      warm_start_model_ <- min(1, i/2000)
    } else{
      I <- local_sampler(idx, psu = 20, ssu = 2) - 1
      warm_start_latents_ <- min(1, (i - warm_up)/10000)
      warm_start_model_ <- min(1, i/2000)
    }
    batch_dict <- dict(I_batch = batch_to_pairs(I), 
                       warm_start_latents = warm_start_latents_, warm_start_model = warm_start_model_)
    session$run(optimizer_model, feed_dict = batch_dict)
  } else{
    #I <- sample(N,p) - 1
    I <- local_sampler(idx, psu = 10, ssu = 4) - 1
    warm_start_latents_ <- min(1, (i - warm_up)/20000) * (p/N) # This scaling to avoid contraction (?)
    warm_start_model_ <- min(1, i/2000)
    batch_dict <- dict(I_batch = batch_to_pairs(I), 
                       warm_start_latents = warm_start_latents_, warm_start_model = warm_start_model_)
    session$run(optimizer_latents, feed_dict = batch_dict)
  }
  
  # Printing
  if(i %% 50 == 0){
    printllh <- session$run(llh, feed_dict = test_batch)
    printkl <- session$run(KL, feed_dict = test_batch)
    cat("Iteration:", i, "\n")
    cat("Log likelihood:", printllh, "KL:", printkl, "\n")
    cat("ELBO:", printllh - printkl, "\n")
  }
  if(i %% 1000 == 0){ # Save a model every 1000 iterations
    new_z <- session$run(latents$v_par$mu)
    plot_at_iteration(new_z, i, "variant")
    filename <- paste("results/mnist/variant_mnist_iteration",i, sep = "")
    saver$save(session, filename)
  }
}