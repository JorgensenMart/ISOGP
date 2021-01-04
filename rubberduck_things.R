devtools::install_github("jlmelville/coil20")
library(coil20)

coil20 <- download_coil20(verbose = TRUE)

rubberduck_ind <- startsWith(rownames(coil20), "1_")
rubberduck <- coil20[rubberduck_ind,]

dist_rubber <- dist(rubberduck)

pca_rubber <- prcomp(x = datmat, center = TRUE)