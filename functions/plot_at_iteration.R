library(ggplot2)

plot_at_iteration <- function(new_z, iteration, method){
load("data/mnist/train.RDa")
y <- train$y[1:5000]
rm(train)

my_data_frame <- data.frame(x1 = new_z[,1], x2 = new_z[,2], Integer = as.factor(y))

out_plot <- ggplot(data = my_data_frame, aes(x = x1, y = x2, color = Integer)) +
  geom_point(size=0.9) + 
  xlab("") + ylab("") +
  theme_minimal()

out_plot

my_file = paste("results/mnist/", method, iteration,".pdf", sep = "")
ggsave(my_file, device = "pdf")
}
