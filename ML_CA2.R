# Q2
X = read.csv(file="C:/Users/Padraig/Desktop/DSA4_SEM2/Eric/CA2/Q2_X.csv", header=TRUE)
X.valid = read.csv(file="C:/Users/Padraig/Desktop/DSA4_SEM2/Eric/CA2/Q2_Xvalid.csv", header=TRUE)
Y = read.csv(file="C:/Users/Padraig/Desktop/DSA4_SEM2/Eric/CA2/Q2_Y.csv", header=FALSE, stringsAsFactors = TRUE)[,1]
Y.valid = read.csv(file="C:/Users/Padraig/Desktop/DSA4_SEM2/Eric/CA2/Q2_Yvalid.csv", header=FALSE, stringsAsFactors=TRUE)[,1]
                                          
library(caret)

# Set seed for reproducibility


##############################################################################
# (1)
set.seed(6041)
# Define the training control
ctrl <- trainControl(method = "cv", number = 5)

# train knn model
knn_model <- train(X, Y, method = "knn", trControl = ctrl)

# print optimal k
cat("Optimal k:", knn_model$bestTune$k, "\n")
knn_pred <- predict(knn_model, X)
pred_table <- table(knn_pred, Y)
acc <- sum(diag(pred_table))/sum(pred_table)
acc

# print average test-set accuracy which is the 
cat("Average test-set accuracy:", mean(knn_model$resample[,1]))



#############################################################################
# (2)
set.seed(6041)
rfe_ctrl <- rfeControl(functions = rfFuncs,
                       method = "cv",
                       number = 5,
                       verbose = FALSE)
rf_model <- rfe(X, Y, method = "rf",
                trControl = ctrl,
                tuneLength = 5,
                rfeControl = rfe_ctrl)

rf_model$optVariables


cat("Validation Set Prediction Accuracy: ", mean(rf_model$resample[,2]))

##############################################################################################################
# (3)
set.seed(6041)
# Generate predictions on the validation set
rf_pred <- predict(rf_model, X.valid)

# Convert rf_pred$pred to a factor with the same levels as Y
rf_pred$pred <- factor(rf_pred$pred, levels = levels(Y.valid))

# Create confusion matrix
pred_table_rf <- table(rf_pred$pred, Y.valid)


accuracy <- sum(diag(pred_table_rf))/sum(pred_table_rf)

accuracy


###############################################################################################################
# (3)

set.seed(6041)
ctrl <- trainControl(method = "cv", number = 5, classProbs = FALSE)
nnet_fit <- train(x = X, y = Y, method = "nnet", trControl = ctrl, tuneLength = 10)

# Extract the optimal number of neurons in the input layer
optimal_input_size <- nnet_fit$bestTune$size[1]

# Print the optimal architecture
cat("Optimal input layer size:", optimal_input_size)

# Generate predictions for the test data
nnet_pred <- predict(nnet_fit, newdata = X.valid)

# Print the validation set prediction accuracy
nnet_acc <- mean(nnet_pred == Y.valid)
cat("Validation set prediction accuracy:", nnet_acc, "\n")


# Generate predictions for the validation set using the neural network classifier
pred <- predict(nnet_fit, newdata = X.valid)

# Tabulate the predicted class labels against the true class labels
pred_table <- table(pred, Y.valid)

# Calculate the validation set prediction accuracy
accuracy <- sum(diag(pred_table)) / sum(pred_table)

accuracy





######################################################################################################################
# Q3


# Load mtcars dataset
data(mtcars)

# Not sure Whats Right
############################################################################################
# Option 1
# With MPG vs Without MPG


# Remove "mpg" variable
mtcars_2 <- mtcars[, -1]

# Perform PCA on the remaining variables
pca <- prcomp(mtcars_2, scale. = TRUE)

# Extract the first eigenvector
eigenvector1 <- pca$rotation[, 1]

eigenvector1 <- c(0, eigenvector1)
# Perform PCA on mtcars with "mpg" variable
pca_mpg <- prcomp(mtcars, scale. = TRUE)

# Extract the first eigenvector
eigenvector1_mpg <- pca_mpg$rotation[, 1]

# Compare the two eigenvectors
cbind(eigenvector1_mpg, eigenvector1)

eigenvector1_mpg - eigenvector1


# Get the proportion of variance explained by each PC
pve <- pca_mpg$sdev^2/sum(pca_mpg$sdev^2)

# Create a data frame with variable names and PVE for each PC
pve_df <- data.frame(Variable = colnames(mtcars), PVE = pve)

##################################################################################################
# Option 2
# Scaled vs Not Scaled

# Remove "mpg" variable
mtcars_2 <- mtcars[, -1]

# Perform PCA on the remaining variables
pca <- prcomp(mtcars_2, scale. = TRUE)

# Extract the first eigenvector
eigenvector1 <- pca$rotation[, 1]

# eigenvector1 <- c(0, eigenvector1)
# Perform PCA on mtcars with "mpg" variable
pca_mpg <- prcomp(mtcars_2, scale. = FALSE)

# Extract the first eigenvector
eigenvector1_mpg <- pca_mpg$rotation[, 1]

# Compare the two eigenvectors
cbind(eigenvector1_mpg, eigenvector1)

eigenvector1_mpg - eigenvector1


# Get the proportion of variance explained by each PC
pve <- pca$sdev^2/sum(pca$sdev^2)
cat("Sum of Varinace Explained by the top 5 Variables", sum(head(pca$sdev^2, 5))/sum(pca$sdev^2) * 100, "% \n")

# Create a data frame with variable names and PVE for each PC
pve_df <- data.frame(Variable = colnames(mtcars_2), PVE = pve)
pve_df

# Look at the top 5 variables
head(pve_df, n = 5)

# Fit a linear regression model with the top 5 variables as predictors
model <- lm(mpg ~ cyl + disp + hp + drat + wt, data = mtcars)

# Print the model summary
summary(model)


















































































