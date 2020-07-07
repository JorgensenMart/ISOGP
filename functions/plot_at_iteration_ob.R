library(ggplot2)

plot_at_iteration_ob <- function(new_z, iteration, method){
  load("data/open-box/open-box-data.RDa")

  new_df <- data.frame(x = new_z[,1], y = new_z[,2], z = res$Plate)
  
  out <- ggplot(data = new_df, aes(x = x, y = y)) + 
    geom_point(aes(colour = factor(z))) +
    scale_color_manual(values = c("1" = "red", "2" = "blue", "3" = "green", "4" = "purple", "5" = "cyan", "6" = "orange")) +
    xlab("") + ylab("") +
    theme(panel.background = element_rect(fill = "white"),
          axis.text = element_text(size = 18),
          legend.position = "none")
  out
  
  my_file = paste("results/open-box/", method, iteration,".pdf", sep = "")
  ggsave(my_file, device = "pdf")
}
