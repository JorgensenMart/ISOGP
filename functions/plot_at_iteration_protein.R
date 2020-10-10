library(ggplot2)

plot_at_iteration_protein <- function(new_z, iteration){
  my_data_frame <- data.frame(x1 = new_z[,1], x2 = new_z[,2])
  
  out_plot <- ggplot(data = my_data_frame, aes(x = x1, y = x2)) +
    geom_point(size=0.9) + 
    xlab("") + ylab("") +
    theme_minimal()
  
  out_plot
  
  my_file = paste("results/protein/protein",iteration,".pdf", sep = "")
  ggsave(my_file, device = "pdf")
}
