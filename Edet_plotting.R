df <- read.csv("results/mnist/variant_Edet.csv")

library(ggplot2)

df$E[df$E > 7.5] <- 7.5

out <- ggplot(df, aes(x = Var1, y = Var2, fill = E)) + 
  geom_raster() + 
  #scale_fill_gradient2(midpoint = 3, low = "white",mid = "blue", high = "darkblue")
  scale_fill_gradientn(colours = topo.colors(50))
out
