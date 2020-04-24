# Build model
library(tensorflow)
tf$reset_default_graph()
source("model_swissroll.R")

saver <- tf$train$Saver()

iter <- 3000

session <- tf$Session()
load_file <- paste("results/swissroll/swissroll_iteration",iter, sep = "")  
saver$restore(session, load_file)

z_lat <- session$run(latents$v_par$mu)
plot(z_lat)
