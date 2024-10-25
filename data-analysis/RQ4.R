# RQ4 Inference Time

library(ggplot2)
library(dplyr)
library(rstatix)
library(effsize)
library(gridExtra)

data <- read.csv("run_table.csv")

summary(data)

# box plot
ggplot(data, aes(x = quantization_type, y = Inference.Time, fill = quantization_type)) +
  geom_boxplot() +
  labs(title = "Inference Time by Quantization Type", 
       x = "Quantization Type", 
       y = "Inference Time (sec)") +
  theme_minimal() +
  theme(legend.position = "none", 
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 8),
        plot.title = element_text(hjust = 0.5, size = 14), # Title font size
        axis.title.y = element_text(size = 11), # Y-axis title font size
        axis.text.y = element_text(size = 11),
        axis.title.x = element_text(size = 11), # Y-axis title font size
        axis.text.x = element_text(size = 11)) 

str(data)
data$quantization_type <- trimws(data$quantization_type)

plot_by_quantization <- function(quant_type) {
  df <- subset(data, quantization_type == quant_type)
  
  # Calculate average for each task
  avg_data <- aggregate(`Inference.Time` ~ task_name, data = df, mean)
  
  ggplot(df, aes(x = task_name, y = `Inference.Time`, color = task_name)) +
    geom_point(size = 2, show.legend = FALSE, position = position_jitter(width = 0.2, height = 0)) +  # Jitter points to avoid overlap
    geom_point(data = avg_data, aes(x = task_name, y = `Inference.Time`), color = "black", shape = 4, size = 5) +  # Cross marker for average
    geom_text(data = avg_data, aes(x = task_name, y = `Inference.Time`, label = round(`Inference.Time`, 2)), vjust = -1, color = "black") +  # Show average values
    labs(title = quant_type, x = "Task Name", y = "Inference Time (sec)") +
    theme_minimal() +
    theme(legend.position = "right", 
          legend.title = element_text(size = 12),
          legend.text = element_text(size = 8),
          plot.title = element_text(hjust = 0.5, size = 14), # Title font size
          axis.title.y = element_text(size = 11), # Y-axis title font size
          axis.text.y = element_text(size = 11),
          axis.title.x = element_text(size = 11), # Y-axis title font size
          axis.text.x = element_text(angle = 45, hjust = 1, size=11)) 
}

p1 <- plot_by_quantization('4-bit')
p2 <- plot_by_quantization('8-bit')
p3 <- plot_by_quantization('32-bit')

grid.arrange(p1, p2, p3, ncol = 3)

# Normality Check
data %>%
  group_by(quantization_type) %>%
  summarise(shapiro_test = shapiro_test(Inference.Time)) %>%
  pull(shapiro_test)


results <- data %>%
  group_by(quantization_type) %>%
  mutate(
    log_Inference_Time = log(Inference.Time + 1),  # Log transformation to handle zero values
    sqrt_Inference_Time = sqrt(Inference.Time)     # Square root transformation
  ) %>%
  group_by(quantization_type) %>% 
  summarise(
    log_normality_p_value = shapiro_test(log_Inference_Time)$p.value,
    sqrt_normality_p_value = shapiro_test(sqrt_Inference_Time)$p.value
  )

results

# Kruskal-Wallis Test
kruskal_test_result <- kruskal.test(Inference.Time ~ quantization_type, data = data)

kruskal_test_result


# Cliff's Delta
delta_32_8 <- cliff.delta(data$Inference.Time[data$quantization_type == "32-bit"],
                          data$Inference.Time[data$quantization_type == "8-bit"])

# Calculate Cliff's Delta for 32-bit vs 4-bit
delta_32_4 <- cliff.delta(data$Inference.Time[data$quantization_type == "32-bit"],
                          data$Inference.Time[data$quantization_type == "4-bit"])

# Calculate Cliff's Delta for 8-bit vs 4-bit
delta_8_4 <- cliff.delta(data$Inference.Time[data$quantization_type == "8-bit"],
                         data$Inference.Time[data$quantization_type == "4-bit"])

# Print the results
print(list("32-bit vs 8-bit" = delta_32_8,
           "32-bit vs 4-bit" = delta_32_4,
           "8-bit vs 4-bit" = delta_8_4))
