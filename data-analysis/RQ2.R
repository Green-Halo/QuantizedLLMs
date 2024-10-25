# RQ2: Accuracy

library(ggplot2)
library(dplyr)
library(rstatix)
library(effsize)

data <- read.csv("run_table.csv")

summary(data)

# box plot: quantization type
ggplot(data, aes(x = quantization_type, y = Accuracy, fill = quantization_type)) +
  geom_boxplot() +
  labs(title = "Accuracy by Quantization Type", 
       x = "Quantization Type", 
       y = "Accuracy (%)") +
  theme_minimal() +
  theme(legend.position = "none", 
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 8),
        plot.title = element_text(hjust = 0.5, size = 14), # Title font size
        axis.title.y = element_text(size = 11), # Y-axis title font size
        axis.text.y = element_text(size = 11),
        axis.title.x = element_text(size = 11), # Y-axis title font size
        axis.text.x = element_text(size = 11)) 


energy_stats <- data %>%
  group_by(quantization_type) %>%
  summarise(
    mean_energy = mean(`GPU.Energy`, na.rm = TRUE),
    std_energy = sd(`GPU.Energy`, na.rm = TRUE)
  )

energy_stats

# bar plot: task+quantization type
unique(mean_accuracy$task_name)

mean_accuracy <- data %>%
  group_by(quantization_type, task_name) %>%
  summarise(mean_accuracy = mean(Accuracy, na.rm = TRUE))

# Create the bar plot
ggplot(mean_accuracy, aes(x = quantization_type, y = mean_accuracy, fill = task_name)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
  geom_text(aes(label = sprintf("%.2f", mean_accuracy)), vjust = -0.5, position = position_dodge(width = 0.9)) +
  labs(title = "Comparison of Accuracy by Quantization Type and Task", 
       x = "Quantization Type", 
       y = "Mean Accuracy") +
  theme_minimal() +
  theme(legend.position = "bottom", 
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 8),
        plot.title = element_text(hjust = 0.5, size = 14), # Title font size
        axis.title.y = element_text(size = 11), # Y-axis title font size
        axis.text.y = element_text(size = 11),
        axis.title.x = element_text(size = 11), # Y-axis title font size
        axis.text.x = element_text(size = 11)) 



# Normality Check
shapiro_results <- data %>%
  group_by(quantization_type) %>%
  summarise(shapiro_stat = shapiro_test(Accuracy)$statistic,
            shapiro_p_value = shapiro_test(Accuracy)$p.value)

shapiro_results


# Kruskal-Wallis Test
kruskal_test_result <- kruskal.test(Accuracy ~ quantization_type, data = data)

kruskal_test_result



delta_32_8 <- cliff.delta(data$Accuracy[data$quantization_type == "32-bit"],
                          data$Accuracy[data$quantization_type == "8-bit"])

# Calculate Cliff's Delta for 32-bit vs 4-bit
delta_32_4 <- cliff.delta(data$Accuracy[data$quantization_type == "32-bit"],
                          data$Accuracy[data$quantization_type == "4-bit"])

# Calculate Cliff's Delta for 8-bit vs 4-bit
delta_8_4 <- cliff.delta(data$Accuracy[data$quantization_type == "8-bit"],
                         data$Accuracy[data$quantization_type == "4-bit"])

# Print the results
print(list("32-bit vs 8-bit" = delta_32_8,
           "32-bit vs 4-bit" = delta_32_4,
           "8-bit vs 4-bit" = delta_8_4))
