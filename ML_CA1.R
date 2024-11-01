############################################################################################################
# Name: Padraig O'Brien
# Student Number: 119378796
# Program: ST4061
###########################################################################################################

library(MASS)
library(caret)
ca.train = read.csv("C:/Users/Padraig/Downloads/ca_train.csv",stringsAsFactors=TRUE)
ca.test = read.csv("C:/Users/Padraig/Downloads/ca_test.csv",stringsAsFactors=TRUE)

# Q1
# Prepare data
predictors <- names(ca.train)[2:ncol(ca.train)]
response <- "y"

# Fit the LDA model
lda_model <- lda(as.formula(paste(response, "~", paste(predictors, collapse="+"))), data = ca.train)

# Predict on test data
lda_preds <- predict(lda_model, newdata = ca.test)
lda_preds_factor <- as.factor(lda_preds$class)
test_y <- factor(ca.test$y, levels = unique(ca.train$y))
confusion_matrix <- confusionMatrix(lda_preds_factor, test_y, positive = "Yes")

print(confusion_matrix$table)

#####################################################################################################

qda.fit <- qda(as.formula(paste(response, "~", paste(predictors, collapse="+"))), data = ca.train)

qda.pred <- predict(qda.fit, newdata = ca.test[, -1])$class

qda_cm <- confusionMatrix(qda.pred, ca.test$y, positive = "Yes")

print(qda_cm$table)

#######################################################################################################

specificity_lda <- confusion_matrix$byClass["Specificity"]
cat("Specificity of LDA Model: ", specificity_lda)

specificity_qda <- qda_cm$byClass["Specificity"]
cat("Specificity of QDA Model: ", specificity_qda)

specificity_diff <- abs(specificity_qda-specificity_lda)
cat("Difference in Specificity: ", specificity_diff)


###########################################################################################################################


library(ISLR) # for the data
library(gbm)
library(randomForest) 

x.train = Khan$xtrain
x.test = Khan$xtest
y.train = as.factor(Khan$ytrain)
y.test = as.factor(Khan$ytest)

# A
table(y.train)
table(y.test)

# Create bar plots for y.train and y.test
par(mfrow=c(1,2))
barplot(table(y.train), main="y.train", xlab="Classes", ylab="Frequency", col="blue")
barplot(table(y.test), main="y.test", xlab="Classes", ylab="Frequency", col="red")


#B

#C
#Set seed for reproducibility
set.seed(4061)

# Fit a random forest to the training data
rf.model <- randomForest(x = x.train, y = y.train)

# Predict the class labels of the test data
y.pred <- predict(rf.model, newdata = x.test)

# Print the confusion matrix
confusion_matrix <- table(y.test, y.pred)
print(confusion_matrix)
confusion_matrix <-  as.matrix(confusion_matrix)

#D
# Generate predictions for the test data
y.pred <- predict(rf.model, newdata = x.test)

# Calculate the test set prediction accuracy
accuracy <- mean(y.pred == y.test)
cat("Accuracy Score:", accuracy, "\n")


#E
# Extract variable importance measures from the random forest model
var.importance <- importance(rf.model)

# Identify the features with importance greater than 0.4
important.features <- rownames(var.importance[var.importance > 0.4, , drop = FALSE])

# Print the important features
cat("Important Features: ", paste(important.features, collapse = ", "), "\n")


#F


#G
set.seed(4061)
df.train = as.data.frame(cbind(x.train, y.train))
options(warn = 0)
gbm.model = gbm(as.factor(y.train)~., data = df.train)
gb.pred = predict(gbm.model, newdata = as.data.frame(x.test), type = 'response')

gb.pred = colnames(gb.pred)[apply(gb.pred, 1, which.max)]
result = data.frame(y.test, gb.pred)

tb.gbm = table(gb.pred, y.test)
gbm_acc <- sum(diag(tb.gbm))/sum(tb.gbm)
cat("GBM Prediction Accuracy: ", gbm_acc, " ")

# End of Assignment script
####################################################################################################################


