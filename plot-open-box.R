Z <- isomap(A, ndim = 2, k = 5)

## 

library(ggplot2)

new_df <- data.frame(x = Z$points[,1], y = Z$points[,2], z = res$Plate)

out <- ggplot(data = new_df, aes(x = x, y = y)) + 
  geom_point(aes(colour = factor(z))) +
  scale_color_manual(values = c("1" = "red", "2" = "blue", "3" = "green", "4" = "purple", "5" = "cyan", "6" = "orange")) +
  xlab("") + ylab("") +
  theme(panel.background = element_rect(fill = "white"),
        axis.text = element_text(size = 18),
        legend.position = "none")
out


library(TDA)

RD <- ripsDiag(res[,1:3], maxdimension = 1, maxscale = 0.2)
plot(RD$diagram, barcode = TRUE, main = "Barcode")
