library(ggplot2)

plot_at_iteration_coil20 <- function(new_z, iteration){
  my_data_frame <- data.frame(x1 = new_z[,1], x2 = new_z[,2])
  my_data_frame$object <- rep(1:20, each = 72)
  out_plot <- ggplot(data = my_data_frame, aes(x = x1, y = x2, col = as.factor(object))) +
    geom_point(size=0.9) + 
    xlab("") + ylab("") +
    theme_minimal()
  
  out_plot
  
  my_file = paste("results/coil20/coil20",iteration,".pdf", sep = "")
  ggsave(my_file, device = "pdf")
}