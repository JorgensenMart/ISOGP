df <- read.csv("results/mnist/lex_Edet.csv")
df2 <- read.csv("results/mnist/lex_means.csv")
library(ggplot2)

df$E[df$E > 6] <- 6

load("data/mnist/train.RDa")
integ <- train$y[1:5000]
rm(train)

df2$integ <- integ
out <- ggplot(data = NULL) + 
  geom_raster(data = df, aes(x = Var1, y = Var2, fill = E)) + 
  scale_fill_gradient(low = "black", high = "white") +
  geom_point(data = df2, aes(x = V1, y = V2, color = as.factor(integ)), size = 1.7) +
  xlim(c(-3,3)) + ylim(c(-3,3)) + xlab("") + ylab("") +
  theme(panel.background = element_rect(fill = "white"),
        axis.text = element_text(size = 30), axis.ticks = element_line(linetype = 3),
        )
out
