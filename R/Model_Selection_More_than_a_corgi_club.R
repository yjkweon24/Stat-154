## ----setup, include=FALSE------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)
library(rpart) #decision tree
library(rpart.plot) # plotting decision tree
library(mlr, warn.conflicts = T) #missing values imputation
library(missForest, warn.conflicts = T) #missing values imputation
library(mi, warn.conflicts = T) #missing values imputation
library(mice, warn.conflicts = T) #missing values imputation
library(VIM) #finding patterns of missing values
library(Hmisc) #missing values imputation
library(lattice)
library(arules) #discretize
library(lme4) #dummfiy
library(tree)
library(caret)
library(ROCR, warn.conflicts = T) # ROC curve
library(pROC, warn.conflicts = T) # Get the optimal threshold
library(randomForest)
library(dplyr)
library(C50) #boosted tree
library(bst) #boosted tree
library(plyr) #boosted tree
library(gbm) #boosted tree
library(ggplot2)
options(scipen=999) #get rid of scientific notation 

## ------------------------------------------------------------------------
set.seed(100)
newtrain2 <- read.csv("../data/cleandata/newtrain2.csv", header = T)
newtest2 <- read.csv("../data/cleandata/newtest2.csv", header = T)



#Change to binary digit
combined <- rbind(newtrain2, newtest2)
combined$income <- as.numeric(combined$income) - 1

## ------------------------------------------------------------------------
set.seed(100)



#from classification 
final.auc1



#Getting predicted >50K of income probabilities 
tree_prob <- predict(final.auc1, newdata = newtest2, type = "prob")[, 2]
tree_prediction <- prediction(tree_prob, newtest2$income)
tree_performance <- ROCR::performance(tree_prediction, measure = "tpr", x.measure = "fpr")



#Plot ROC curve 
plot(tree_performance, main = "ROC curve")
abline(a = 0, b = 1, lty = 2)



#Calculate AUC
tree.auc <- ROCR::performance(tree_prediction, measure="auc")@y.values[[1]]
tree.auc



#==============================================================



#from bagged tree
final.auc2



#Getting predicted >50K of income probabilities 
tunned.bag.rf_prob <- predict(final.auc2, newdata = newtest2,
                     type = "prob")[, 2]
tunned.bag.rf_prediction <- prediction(tunned.bag.rf_prob, newtest2$income)
tunned.bag.rf_performance <- ROCR::performance(tunned.bag.rf_prediction,
                                               measure = "tpr",
                                               x.measure = "fpr")



#Plot ROC curve 
plot(tunned.bag.rf_performance, main = "ROC curve")
abline(a = 0, b = 1, lty = 2)



#Calculate AUC
tunned.bag.rf.auc <- ROCR::performance(tunned.bag.rf_prediction,
                                   measure = "auc")@y.values[[1]]
tunned.bag.rf.auc



#==============================================================



#from random forest
final.auc3



#Getting predicted >50K of income probabilities 
tunned.rf_prob <- predict(final.auc3, newdata = newtest2, 
                            type = "prob")[, 2]
tunned.rf_prediction <- prediction(tunned.rf_prob, newtest2$income)
tunned.rf_performance <- ROCR::performance(tunned.rf_prediction, measure = "tpr", x.measure = "fpr")



#Plot ROC curve 
plot(tunned.rf_performance, main = "ROC curve")
abline(a = 0, b = 1, lty = 2)



#Calculate AUC
tunned.rf.auc <- ROCR::performance(tunned.rf_prediction,
                                     measure = "auc")@y.values[[1]]
tunned.rf.auc



#==============================================================



#from boosted
final.auc4



#ROC curve - testing
pos5 <- c()
pos5 <- predict(final.auc4, newdata = combined[32403:48598, -44], n.trees = 800, type = "response")
predicts5 <- prediction(pos5, combined[32403:48598, 44])
roc5 <- ROCR::performance(predicts5, measure = "tpr", x.measure = "fpr")
plot(roc5, main = "ROC curve")
abline(0, 1, col = "red")
auc5 <- ROCR::performance(predicts5, measure = "auc")
auc5@y.values

## ------------------------------------------------------------------------
set.seed(100)



tree_class <- predict(final.auc1, newdata = newtest2, type = "class")
confusionMatrix(tree_class, newtest2$income)



#==============================================================



tunned.bag.rf_class <- predict(final.auc2, newdata = newtest2,
                     type = "class")
confusionMatrix(tunned.bag.rf_class, newtest2$income)



#==============================================================



tunned.rf_class <- predict(final.auc3, newdata = newtest2, 
                            type = "class")
confusionMatrix(tunned.rf_class, newtest2$income)



#==============================================================



boosted_class <- predict.gbm(final.auc4, 
                             newdata = combined[32403:48598, -44],
                n.trees = 800, type = "response")
boosted_class <- ifelse(boosted_class > 0.5, ">50K", "<=50K")
confusionMatrix(boosted_class, newtest2$income)

## ------------------------------------------------------------------------
set.seed(100)



#Plot ROC curve 
plot(tree_performance, main="ROC curve", col = "blue")   # classification
plot(tunned.bag.rf_performance, add = T, col = "red")  # bagged
plot(tunned.rf_performance, add = T, col = "green") # random forest
plot(roc5, add = T) # boosted
abline(a = 0, b = 1, lty = 2)
legend("bottomright", legend = c("Classification", "Bagged",
                                 "Boosted","Random Forest"),
       col=c("blue", "red", "black", "green"), lwd = 3, cex = .5, horiz = TRUE)



AUC.final <- data.frame(tree.auc, tunned.bag.rf.auc, tunned.rf.auc,
                        boosted.auc = auc5@y.values[[1]])



AUC.final[, order(AUC.final)]

## ------------------------------------------------------------------------
set.seed(100)



#from classification 
final.auc1



#Getting predicted >50K of income probabilities 
tree_prob <- predict(final.auc1, newdata = newtest2, type = "prob")[, 2]
tree_prediction <- prediction(tree_prob, newtest2$income)
tree_performance <- ROCR::performance(tree_prediction, measure = "tpr", x.measure = "tnr")



#Plot ROC curve 
plot(tree_performance, main = "TPR v.s. TNR")
abline(a = 1, b = -1, lty = 2)



#==============================================================



#from bagged tree
final.auc2



#Getting predicted >50K of income probabilities 
tunned.bag.rf_prob <- predict(final.auc2, newdata = newtest2,
                     type = "prob")[, 2]
tunned.bag.rf_prediction <- prediction(tunned.bag.rf_prob, newtest2$income)
tunned.bag.rf_performance <- ROCR::performance(tunned.bag.rf_prediction,
                                               measure = "tpr",
                                               x.measure = "tnr")



#Plot ROC curve 
plot(tunned.bag.rf_performance, main="TPR v.s. TNR")
abline(a = 1, b = -1, lty = 2)



#==============================================================



#from random forest
final.auc3



#Getting predicted >50K of income probabilities 
tunned.rf_prob <- predict(final.auc3, newdata = newtest2, 
                            type = "prob")[, 2]
tunned.rf_prediction <- prediction(tunned.rf_prob, newtest2$income)
tunned.rf_performance <- ROCR::performance(tunned.rf_prediction, measure = "tpr", x.measure = "tnr")



#Plot ROC curve 
plot(tunned.rf_performance, main = "TPR v.s. TNR")
abline(a = 1, b = -1, lty = 2)



#==============================================================



#from boosted
final.auc4



#ROC curve - testing
pos5 <- c()
pos5 <- predict(final.auc4, newdata = combined[32403:48598, -44], n.trees = 800, type = "response")
predicts5 <- prediction(pos5, combined[32403:48598, 44])
roc5 <- ROCR::performance(predicts5, measure = "tpr", x.measure = "tnr")
plot(roc5, main="TPR v.s. TNR")
abline(a = 1, b = -1, col = "red")



plot(tree_performance, main = "TPR v.s. TNR - AUC selection", col = "blue")
plot(tunned.bag.rf_performance, col = "red", add = TRUE)
plot(tunned.rf_performance, col = "green", add = TRUE)
plot(roc5, add = TRUE)
abline(a = 1, b = -1, lty = 2)
legend("bottomleft", legend = c("Classification", "Bagged",
                                 "Boosted","Random Forest"),
       col=c("blue", "red", "black", "green"), lwd = 3, cex = .5, horiz = TRUE)

## ------------------------------------------------------------------------
set.seed(100)



#from classification 
#final.thres1
info_prob <- predict(final.thres1.half, newdata = newtest2, type = "prob")[, 2]



#Test accuracy rate by using default cutoff 0.5
prunned.info.accuracy <- mean((info_prob > 0.5) == (newtest2$income == ">50K"))
cat("Accuracy classification :  ", prunned.info.accuracy, "\n")



#==============================================================




#from bagged tree 
#final.thres2  # bag.rforest$learner.model
tunned.bag.rf_prob <- predict(final.thres2.half, newdata = newtest2,
                     type = "prob")[, 2]



#Test accuracy rate by using default cutoff 0.5
tunned.bagged.accuracy <- mean((tunned.bag.rf_prob > 0.5) == (newtest2$income == ">50K"))
cat("Accuracy Bagged :  ", tunned.bagged.accuracy, "\n")



#==============================================================



#from random forest
#final.thres3  # untunned.forest$learner.model
untunned.rf_prob <- predict(final.thres3.half, newdata = newtest2,
                            type = "prob")[, 2]



#Test accuracy rate by using default cutoff 0.5
rf.untunned.accuracy <- mean((untunned.rf_prob > 0.5) == (newtest2$income == ">50K"))
cat("Accuracy Random Forest :  ", rf.untunned.accuracy, "\n")



#==============================================================



#from boosting
#final.thres4
e <- predict(final.thres4, newdata = combined[32403:48598, -44], n.trees = 800, type = "response")
e1 <- (e > 0.5)
e2 <- mean(e1 == combined[32403:48598, 44])
cat("Accuracy Boosted :  ", e2, "\n")

## ------------------------------------------------------------------------
set.seed(100)



classification_class2 <- predict(final.thres1.half$finalModel, newdata = newtest2, type = "class")
confusionMatrix(classification_class2, newtest2$income)



#==============================================================



tunned.bag.rf_class2 <- predict(final.thres2.half, newdata = newtest2,
                     type = "class")
confusionMatrix(tunned.bag.rf_class2, newtest2$income)



#==============================================================



untunned.rf_class2 <- predict(final.thres3.half, newdata = newtest2,
                            type = "class")
confusionMatrix(untunned.rf_class2, newtest2$income)



#==============================================================



boosted_class2 <- predict(final.thres4, newdata = combined[32403:48598, -44], n.trees = 800,
                          type = "response")
boosted_class2 <- ifelse(boosted_class2 > 0.5, ">50K", "<=50K")
confusionMatrix(boosted_class2, newtest2$income)

## ------------------------------------------------------------------------
set.seed(100)



#from classification 
final.thres1.half



#Getting predicted >50K of income probabilities 
tree_prob2 <- predict(final.thres1.half, newdata = newtest2, 
                      type = "prob")[, 2]
tree_prediction2 <- prediction(tree_prob2, newtest2$income)
tree_performance2 <- ROCR::performance(tree_prediction2,
                                      measure = "tpr", x.measure = "tnr")



#Plot ROC curve 
plot(tree_performance2, main = "TPR v.s. TNR")
abline(a = 1, b = -1, lty = 2)



#==============================================================



#from bagged tree
final.thres2.half



#Getting predicted >50K of income probabilities 
tunned.bag.rf_prob2 <- predict(final.thres2.half, newdata = newtest2,
                     type = "prob")[, 2]
tunned.bag.rf_prediction2 <- prediction(tunned.bag.rf_prob2, newtest2$income)
tunned.bag.rf_performance2 <- ROCR::performance(tunned.bag.rf_prediction2,
                                               measure = "tpr",
                                               x.measure = "tnr")



#Plot ROC curve 
plot(tunned.bag.rf_performance2, main = "TPR v.s. TNR")
abline(a = 1, b = -1, lty = 2)



#==============================================================



#from random forest
final.thres3.half



#Getting predicted >50K of income probabilities 
untunned.rf_prob3 <- predict(final.thres3.half, newdata = newtest2, 
                            type = "prob")[, 2]
untunned.rf_prediction3 <- prediction(untunned.rf_prob3, newtest2$income)
untunned.rf_performance3 <- ROCR::performance(untunned.rf_prediction3,
                                             measure = "tpr", x.measure = "tnr")



#Plot ROC curve 
plot(untunned.rf_performance3, main = "TPR v.s. TNR")
abline(a = 1, b = -1, lty = 2)



#==============================================================



#from boosted
final.thres4

#ROC curve - testing
pos5b <- c()
pos5b <- predict(final.thres4, newdata = combined[32403:48598, -44], n.trees = 800,
                type = "response")
predicts5b <- prediction(pos5b, combined[32403:48598, 44])
roc5b <- ROCR::performance(predicts5b, measure = "tpr", x.measure = "tnr")
plot(roc5b, main = "TPR v.s. TNR")
abline(a = 1, b = -1, col = "red")



#Combine into one graph
plot(tree_performance2, main = "TPR v.s. TNR - Accuracy selection", 
     col = "blue")
plot(tunned.bag.rf_performance2, col = "red", add = TRUE)
plot(untunned.rf_performance3, col = "green", add = TRUE)
plot(roc5b, add = TRUE)
abline(a = 1, b = -1, lty = 2)
legend("bottomleft", legend = c("Classification", "Bagged",
                                 "Boosted","Random Forest"),
       col=c("blue", "red", "black", "green"), lwd = 3, cex = .5, horiz = TRUE)

