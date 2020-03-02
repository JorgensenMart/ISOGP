#' Test script for arc-length
#' 
source("functions/arc_length.R"); 
source("functions/make_gp_model.R")
source("functions/util.R")
source("functions/sample_gp_marginal.R")
source("functions/get_mu_and_var.R")
source("functions/build_kernel_matrix.R")
library(tensorflow)

N <- 250; d <- 1; D <- 1
z <- matrix(rnorm(N*d,0,1), ncol = d)

model <- make_gp_model(kern.type = "lin",
                       input = z,
                       num_inducing = 100,
                       in_dim = d, out_dim = D,
                       is.WP = TRUE, deg_free = d) # Should be unconstrained Wishart to generate Dxd matrices


session <- tf$Session()
init <- tf$global_variables_initializer()
session$run(init)

z_end <- tf$constant(matrix(z[1:2,]), dtype = tf$float32)
A <- arc_length(model,z_end)
session$run(z_end)
session$run(A)

zs <- seq_d(z_end, 10)
J <- sample_gp_marginal(model, zs, samples = 10, joint_cov = TRUE)