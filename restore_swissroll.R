# Build model
library(tensorflow)
tf$reset_default_graph()
source("model_swissroll.R")
session <- tf$Session()
saver <- tf$train$Saver()

iter <- 27000

load_file <- paste("results/swissroll/loc_sampler_swissroll_iteration",iter, sep = "")  
saver$restore(session, load_file)

ZZ <- session$run(latents$v_par$mu)

library(ggplot2)
R <- rowSums(swiss[,c(1,3)]^2)
ZZ <- data.frame(ZZ)
ZZ <- cbind(ZZ,R)
out <- ggplot(data = ZZ, aes(x = ZZ[,1], y = ZZ[,2], color = ZZ[,3])) + 
  geom_point() +
  xlim(-9,9) + ylim(-4,4) + title(paste(iter))

out