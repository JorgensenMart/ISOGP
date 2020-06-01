# Swissroll plotting

library(ggplot2)
library(RColorBrewer)
R <- rowSums(swiss[,c(1,3)]^2)
ZZ <- data.frame(ZZ)
ZZ <- cbind(ZZ,R)
out <- ggplot(data = ZZ, aes(x = ZZ[,1], y = ZZ[,2], color = ZZ[,3])) + 
  geom_point() + scale_color_gradient2(midpoint = 2, low = "cyan", mid = "darkblue", high = "red") +
  xlab("") + ylab("") +
  theme(panel.background = element_rect(fill = "white"),
        axis.text = element_text(size = 26))

#z <- data.frame(z)
#z <- cbind(z,R)

out2 <- ggplot(data = z, aes(x = z[,1], y = z[,2], color = z[,3])) + 
  geom_point() + scale_color_gradient2(midpoint = 2, low = "cyan", mid = "darkblue", high = "red") + 
  theme(panel.background = element_rect(fill = "white"))
out2

out3 <- ggplot(data = I, aes(x = I[,1], y = I[,2], color = I[,3])) + 
  geom_point()
out2