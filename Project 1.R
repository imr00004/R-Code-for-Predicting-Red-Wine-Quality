#Load necessary libraries
library(readr)
library(dplyr)
library(ggplot2)
library(corrplot)
library(caret)
library(randomForest)

rm(list=ls())

#Read the data 
wine_data <- read.csv("/Users/isabellareeser/Downloads/wine+quality/winequality-red.csv", sep = ";")

#Explore the data
str(wine_data)
summary(wine_data)

#Correlation plot
cor_matrix <- cor(wine_data)
corrplot(cor_matrix, method = "color", type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)

#Distribution of wine quality
ggplot(wine_data, aes(x = quality)) +
  geom_bar(fill = "darkred") +
  labs(title = "Distribution of Wine Quality", x = "Quality", y = "Count")

#Implementing linear regression

#Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(wine_data$quality, p = 0.8, list = FALSE)
train_data <- wine_data[train_index, ]
test_data <- wine_data[-train_index, ]

#Train the linear regression model
lm_model <- lm(quality ~ ., data = train_data)

#Summary of the model
summary(lm_model)

# Make predictions on the test set
predictions <- predict(lm_model, newdata = test_data)

#Calculate RMSE
rmse <- sqrt(mean((test_data$quality - predictions)^2))

#Calculate R-squared
r_squared <- 1 - sum((test_data$quality - predictions)^2) / sum((test_data$quality - mean(test_data$quality))^2)

#Print performance metrics
cat("RMSE:", rmse, "\n")
cat("R-squared:", r_squared, "\n")

#Plot predicted vs actual values
ggplot(data.frame(actual = test_data$quality, predicted = predictions), aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Predicted vs Actual Wine Quality", x = "Actual Quality", y = "Predicted Quality")

#Implement Random Forest model
set.seed(123)
rf_model <- randomForest(quality ~ ., data = train_data, ntree = 500, importance = TRUE)

#Make predictions with Random Forest model
rf_predictions <- predict(rf_model, newdata = test_data)

#Calculate RMSE for Random Forest
rf_rmse <- sqrt(mean((test_data$quality - rf_predictions)^2))

#Calculate R-squared for Random Forest
rf_r_squared <- 1 - sum((test_data$quality - rf_predictions)^2) / sum((test_data$quality - mean(test_data$quality))^2)

#Print performance metrics for Random Forest
cat("Random Forest RMSE:", rf_rmse, "\n")
cat("Random Forest R-squared:", rf_r_squared, "\n")

#Compare model performances
model_comparison <- data.frame(
  Model = c("Linear Regression", "Random Forest"),
  RMSE = c(rmse, rf_rmse),
  R_squared = c(r_squared, rf_r_squared)
)
print(model_comparison)

#Visualizations

#Feature Importance Plot for Random Forest
importance_df <- as.data.frame(importance(rf_model))
importance_df$Feature <- rownames(importance_df)
importance_df <- importance_df[order(importance_df$`%IncMSE`, decreasing = TRUE), ]

ggplot(importance_df, aes(x = reorder(Feature, `%IncMSE`), y = `%IncMSE`)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  coord_flip() +
  labs(title = "Feature Importance in Random Forest Model",
       x = "Features", y = "% Increase in MSE") +
  theme_minimal()

#Actual vs Predicted Plot for both models
plot_data <- data.frame(
  Actual = test_data$quality,
  LM_Predicted = predictions,
  RF_Predicted = rf_predictions
)

ggplot(plot_data, aes(x = Actual)) +
  geom_point(aes(y = LM_Predicted, color = "Linear Regression"), alpha = 0.5) +
  geom_point(aes(y = RF_Predicted, color = "Random Forest"), alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, color = "black", linetype = "dashed") +
  labs(title = "Actual vs Predicted Wine Quality",
       x = "Actual Quality", y = "Predicted Quality") +
  scale_color_manual(values = c("Linear Regression" = "blue", "Random Forest" = "red")) +
  theme_minimal() +
  theme(legend.title = element_blank())

#Distribution of Errors for both models
error_data <- data.frame(
  Model = rep(c("Linear Regression", "Random Forest"), each = nrow(test_data)),
  Error = c(test_data$quality - predictions, test_data$quality - rf_predictions)
)

ggplot(error_data, aes(x = Error, fill = Model)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution of Prediction Errors",
       x = "Prediction Error", y = "Density") +
  theme_minimal()












