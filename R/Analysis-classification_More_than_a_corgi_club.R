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
library(ggplot2)
options(scipen=999) #get rid of scientific notation 

## ------------------------------------------------------------------------
set.seed(100)
newtrain2 <- read.csv("../data/cleandata/newtrain2.csv", header = T)
newtest2 <- read.csv("../data/cleandata/newtest2.csv", header = T)
str(newtrain2)
str(newtest2)

## ------------------------------------------------------------------------
set.seed(100)



#Create a baseline Classification tree using gini index criterion using random cp
tree <- rpart(income ~., data = newtrain2, method = "class",
              parms = list(split = 'gini'), control = rpart.control(minsplit = 5, cp = 0.0001,
                                                                    maxdepth = 5))



#Visualization of the tree
rpart.plot(tree)



#Pick the optimal tuning parameter
cp <- tree$cptable[which.min(tree$cptable[, "xerror"]), "CP"]
cp   #0.0002603489
# this optimal cp is the same as the default cp that we used in rpart function



#Prune the tree using the optimal cp
treepruned <- prune(tree, cp = cp)
#Treepruned object
treepruned



#Information by cp cross-validation results 
printcp(treepruned)
plotcp(treepruned)



#summary information
summary(treepruned, digits = 3)



#Variable importance
varimp <- varImp(treepruned)
varimp



#Visualization of variable importance 
varimp <- data.frame(varimp, name = rownames(varimp))
ggplot(varimp, aes(x = reorder(name, -Overall), y = Overall)) + 
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(color = "blue", size = 6, angle = 90))



#Visualization of pruned tree
rpart.plot(treepruned)



#predicted income class from pruned tree object on train dataset
treepred1 <- predict(treepruned, newdata = newtrain2, type = "class")



#Confusion matrix - train dataset
confusion1 <- confusionMatrix(newtrain2$income, treepred1)
confusion1



#Training accuracy rate
(confusion1$table[1, 1] + confusion1$table[2, 2]) / sum(confusion1$table)

treepred2 <- predict(treepruned, newdata = newtest2, type = "class")



#Confusion matrix - test dataset
confusion2 <- confusionMatrix(newtest2$income, treepred2)
confusion2



#Misclassification Rate of prunned tree on test dataset
(confusion2$table[1, 2] + confusion2$table[2, 1]) / sum(confusion2$table)



#Accuracy Rate of prunned tree on test dataset
(confusion2$table[1, 1] +confusion2$table[2, 2]) / sum(confusion2$table)



#ROC Curve: https://stackoverflow.com/questions/30818188/roc-curve-in-r-using-rpart-package
#Baseline model's ROC curve
#Getting predicted >50K of income probabilities 
tree_prob <- predict(tree, newdata = newtest2, type = "prob")[, 2]
tree_prediction <- prediction(tree_prob, newtest2$income)
tree_performance <- ROCR::performance(tree_prediction, measure = "tpr", x.measure = "fpr")



#Plot ROC curve 
plot(tree_performance, main = "ROC curve")
abline(a = 0, b = 1, lty = 2)



#Calculate AUC
tree.auc <- ROCR::performance(tree_prediction, measure = "auc")@y.values[[1]]
tree.auc



#Pick the best threshold
str(tree_performance)
cutoffs <- data.frame(cut = tree_performance@alpha.values[[1]], 
                      fpr = tree_performance@x.values[[1]], 
                      tpr = tree_performance@y.values[[1]])
head(cutoffs)
roc <- pROC::roc(newtest2$income, tree_prob)
threshold <- coords(roc, "best", ret = "threshold")
cat("The best threshold is :  " , threshold, "\n")



#Get accuracy rate of testset data using the optimal threshold  ****
confusionMatrix(tree_prob > threshold, newtest2$income == ">50K")



#Pruned model's ROC curve
#Getting predicted >50K of income probabilities 
pruned_prob <- predict(treepruned, newdata = newtest2, type = "prob")[, 2]
pruned_prediction <- prediction(pruned_prob, newtest2$income)
pruned_performance <- ROCR::performance(pruned_prediction, measure = "tpr", x.measure = "fpr")



#Plot ROC curve 
plot(pruned_performance, main = "ROC curve")
abline(a = 0, b = 1, lty = 2)



#Calculate AUC
pruned.auc <- ROCR::performance(pruned_prediction,
                                measure = "auc")@y.values[[1]]
pruned.auc



#Pick the best threshold
str(pruned_performance)
cutoffs <- data.frame(cut = pruned_performance@alpha.values[[1]], 
                      fpr = pruned_performance@x.values[[1]], 
                      tpr = pruned_performance@y.values[[1]])
head(cutoffs)
roc <- pROC::roc(newtest2$income, pruned_prob)
threshold <- coords(roc, "best", ret = "threshold")
cat("The best threshold is :  " , threshold, "\n")



#Get accuracy rate of testset data using the optimal threshold  ****
confusionMatrix(pruned_prob > threshold, newtest2$income == ">50K")

## ------------------------------------------------------------------------
set.seed(100)



trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

# Training the Decision Tree classifier with criterion as gini index
dtree_fit <- caret::train(income ~., data = newtrain2,
                          method = "rpart",
                   parms = list(split = "gini"),
                   trControl = trctrl,
                   tuneLength = 10)
dtree_fit



#Tuning parameter - cp
dtree_fit$bestTune



#The model we selected by using the optimal cp we got
dtree_fit$finalModel



#Plot classification tree 
prp(dtree_fit$finalModel, box.palette = "Reds", tweak = 0.8, 
    fallen.leaves = FALSE, faclen = 0, extra = 1)



#Variable importance
varimp2 <- varImp(dtree_fit$finalModel)
varimp2



#Visualization of variable importance 
varimp2 <- data.frame(varimp2, name = rownames(varimp2))
ggplot(varimp2, aes(x = reorder(name, -Overall), y = Overall)) + 
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(color = "blue", size = 6, angle = 90))



#Predicted income class from the finalmodel tree object on train dataset
treepred3 <- predict(dtree_fit$finalModel, newdata = newtrain2, type = "class")



#Confusion matrix - train dataset
confusion3 <- confusionMatrix(newtrain2$income, treepred3)
confusion3



#Training accuracy rate
(confusion3$table[1, 1] + confusion3$table[2, 2]) / sum(confusion3$table)



#Predicted income class from the finalmodel tree object on test dataset
treepred4 <- predict(dtree_fit$finalModel, newdata = newtest2, type = "class")



#Confusion matrix - test dataset
confusion4 <- confusionMatrix(newtest2$income, treepred4)
confusion4



#Misclassification Rate of finalmodel tree on test dataset
(confusion4$table[1, 2] + confusion4$table[2, 1]) / sum(confusion4$table)



#Accuracy Rate of finalmodel tree on test dataset
(confusion4$table[1, 1] + confusion4$table[2, 2]) / sum(confusion4$table)



#Getting predicted >50K of income probabilities 
gini_prob <- predict(dtree_fit, newdata = newtest2, type = "prob")[, 2]
gini_prediction <- prediction(gini_prob, newtest2$income)
gini_performance <- ROCR::performance(gini_prediction, measure = "tpr", x.measure = "fpr")



#ROC Curve  : https://stackoverflow.com/questions/30818188/roc-curve-in-r-using-rpart-package
#Plot ROC curve 
plot(gini_performance, main = "ROC curve")
abline(a = 0, b = 1, lty = 2)



#Calculate AUC
gini.auc <- ROCR::performance(gini_prediction, measure = "auc")@y.values[[1]]
gini.auc



#Pick the best threshold
str(gini_performance)
cutoffs <- data.frame(cut = gini_performance@alpha.values[[1]], 
                      fpr = gini_performance@x.values[[1]], 
                      tpr = gini_performance@y.values[[1]])
head(cutoffs)
roc <- pROC::roc(newtest2$income, gini_prob)
threshold <- coords(roc, "best", ret = "threshold")
cat("The best threshold is :  ", threshold, "\n")



#Get accuracy rate of testset data using the optimal threshold  ****
confusionMatrix(gini_prob > threshold, newtest2$income == ">50K")



#====================================================================



#Training the Decision Tree classifier with criterion as information gain(cross entropy)
set.seed(100)
dtree_fit_info <- caret::train(income ~., data = newtrain2, method = "rpart",
                   parms = list(split = "information"),
                   trControl = trctrl,
                   tuneLength = 10)
dtree_fit_info



#Tuning parameter - cp
dtree_fit_info$bestTune



#The model we selected by using the optimal cp we got
dtree_fit_info$finalModel



#Plot classification tree
prp(dtree_fit_info$finalModel, box.palette = "Blues", tweak = 1.2, extra = 1)



#Variable importance
varimp3 <- varImp(dtree_fit_info$finalModel)
varimp3



#Visualization of variable importance 
varimp3 <- data.frame(varimp3, name = rownames(varimp3))
ggplot(varimp2, aes(x = reorder(name, -Overall), y = Overall)) + 
  geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(color = "blue", size = 6, angle = 90))



#Predicted income class from the finalmodel tree object on train dataset
treepred <- predict(dtree_fit_info$finalModel, newdata = newtrain2, type = "class")



#Confusion matrix - train dataset
confusion <- confusionMatrix(newtrain2$income, treepred)
confusion



#Training accuracy rate
(confusion$table[1, 1] + confusion$table[2, 2]) / sum(confusion$table)



#Predicted income class from the finalmodel tree object on test dataset
treepred1 <- predict(dtree_fit_info$finalModel, newdata = newtest2, type = "class")



#Confusion matrix - test dataset
confusion1 <- confusionMatrix(newtest2$income, treepred1)
confusion1



#Misclassification Rate of finalmodel tree on test dataset
(confusion1$table[1, 2] + confusion1$table[2, 1]) / sum(confusion1$table)



#Accuracy Rate of finalmodel tree on test dataset
(confusion1$table[1, 1] + confusion1$table[2, 2]) / sum(confusion1$table)



#Getting predicted >50K of income probabilities 
info_prob <- predict(dtree_fit_info, newdata = newtest2, type = "prob")[, 2]
info_prediction <- prediction(info_prob, newtest2$income)
info_performance <- ROCR::performance(info_prediction, measure = "tpr", x.measure = "fpr")



#ROC Curve: https://stackoverflow.com/questions/30818188/roc-curve-in-r-using-rpart-package
#Plot ROC curve 
plot(info_performance, main = "ROC curve")
abline(a = 0, b = 1, lty = 2)



#Calculate AUC
info.auc <- ROCR::performance(info_prediction, measure = "auc")@y.values[[1]]
info.auc



#Pick the best threshold======================= not for accuracy
str(info_performance)
cutoffs <- data.frame(cut = info_performance@alpha.values[[1]], 
                      fpr = info_performance@x.values[[1]], 
                      tpr = info_performance@y.values[[1]])
head(cutoffs)
roc <- pROC::roc(newtest2$income, info_prob)
threshold <- coords(roc, "best", ret = "threshold")
cat("The best threshold is :  ", threshold, "\n")



#Get accuracy rate of testset data using the optimal threshold  ****
confusionMatrix(info_prob > threshold, newtest2$income == ">50K")

## ------------------------------------------------------------------------
set.seed(100)
#Compare ROC curve 
plot(pruned_performance, main = "ROC curve", col = "blue")
plot(gini_performance, add = TRUE, col = "red")
plot(tree_performance, add = TRUE, col = "green")
plot(info_performance, add = TRUE)
abline(a = 0, b = 1, lty = 2)
legend("bottomright", legend = c("Pruned - 1st method", "Tunned - 2nd method",
                                 "Tunned - 3rd method", "unprunned"),
       col = c("blue", "red", "black", "green"), lwd = 3, cex = .45, horiz = TRUE)


## ------------------------------------------------------------------------
set.seed(100)
thresholds <- seq(from = 0.001, 0.999, 0.001)
accuracy <- c()



#Using train dataset to check new accuracy driven by  new threshold
gini_prob.train <- predict(dtree_fit, newdata = newtrain2, 
                           type = "prob")[, 2]



#Tuned by gini index splitting criterion model
for(i in 1:length(thresholds)){
  accuracy[i] <- mean((gini_prob.train > thresholds[i]) ==
                        (newtrain2$income == ">50K"))
}



#Threshold which give maximum accuracy
thres1 <- which.max(accuracy) * 0.001
thres1



#plot of accuracy vs thresholds
threstable <- data.frame(thresholds, accuracy)
ggplot(threstable, aes(x = thresholds, y = accuracy)) + geom_point()
  


#Get accuracy rate of testset data using the optimal threshold
confusionMatrix(gini_prob > thres1, newtest2$income == ">50K")

#Test accuracy rate by using optimal threshold
prunned.gini.accuracy <- mean((gini_prob > thres1) == (newtest2$income == ">50K"))

#Test accuracy rate by using default threshold(0.5)
prunned.gini.accuracy.half <- mean((gini_prob > 0.5) == (newtest2$income == ">50K"))



#==================================================================



#Using train dataset to check new accuracy driven by  new threshold
info_prob.train <- predict(dtree_fit_info, newdata = newtrain2, 
                           type = "prob")[, 2]



#Tuned by gini index splitting criterion model
for(i in 1:length(thresholds)){
  accuracy[i]  <- mean((info_prob.train > thresholds[i]) ==
                        (newtrain2$income == ">50K"))
}



#Threshold which give maximum accuracy
thres2 <- which.max(accuracy) * 0.001
thres2



#plot of accuracy vs thresholds
threstable <- data.frame(thresholds, accuracy)
ggplot(threstable, aes(x = thresholds, y = accuracy)) + geom_point()
  


#Get accuracy rate of testset data using the optimal threshold
confusionMatrix(info_prob > thres2, newtest2$income == ">50K")



#Test accuracy rate by using optimal threshold
prunned.info.accuracy <- mean((info_prob > thres2) == (newtest2$income == ">50K"))

#Test accuracy rate by using default threshold(0.5)
prunned.info.accuracy.half <- mean((info_prob > 0.5) == (newtest2$income == ">50K"))

#==================================================================



#Using train dataset to check new accuracy driven by new threshold
tree_prob.train <- predict(tree, newdata = newtrain2, type = "prob")[, 2]



#Tuned by gini index splitting criterion model
for(i in 1:length(thresholds)){
  accuracy[i] <- mean((tree_prob.train > thresholds[i]) ==
                        (newtrain2$income == ">50K"))
}



#Threshold which give maximum accuracy
thres3 <- which.max(accuracy) * 0.001
thres3



#plot of accuracy vs thresholds
threstable <- data.frame(thresholds, accuracy)
ggplot(threstable, aes(x = thresholds, y = accuracy)) + geom_point()
  


#Get accuracy rate of testset data using the optimal threshold
confusionMatrix(tree_prob > thres3, newtest2$income == ">50K")



#Test accuracy rate by using optimal threshold
unprunned.accuracy <- mean((tree_prob > thres3) == (newtest2$income == ">50K"))



#Test accuracy rate by using optimal threshold
unprunned.accuracy.half <- mean((tree_prob > 0.5) == (newtest2$income == ">50K"))



#==================================================================



#Using train dataset to check new accuracy driven by new threshold
pruned_prob.train <- predict(treepruned, newdata = newtrain2, 
                             type = "prob")[, 2]



#Tuned by gini index splitting criterion model
for(i in 1:length(thresholds)){
  accuracy[i] <- mean((pruned_prob.train > thresholds[i]) ==
                        (newtrain2$income == ">50K"))
}



#Threshold which give maximum accuracy
thres4 <- which.max(accuracy) * 0.001
thres4



#plot of accuracy vs thresholds
threstable <- data.frame(thresholds, accuracy)
ggplot(threstable, aes(x = thresholds, y = accuracy)) + geom_point()
  


#Get accuracy rate of testset data using the optimal threshold
confusionMatrix(pruned_prob > thres4, newtest2$income == ">50K")



#Test accuracy rate by using optimal threshold
prunned.accuracy <- mean((pruned_prob > thres4) == (newtest2$income == ">50K"))

prunned.accuracy.half <- mean((pruned_prob > 0.5) == (newtest2$income == ">50K"))

## ------------------------------------------------------------------------
set.seed(100)
#Compare AUC
auc <- data.frame(pruned.auc, info.auc, gini.auc, tree.auc)
auc[, order(auc)]



#Pick the model with the largest AUC - unprunned tree
final.auc1 <- tree


#Compare Accuracy - optimal threshold
accuracy.tree.df <- data.frame(unprunned.accuracy, prunned.accuracy,
                          prunned.gini.accuracy, prunned.info.accuracy)
accuracy.tree.df[, order(accuracy.tree.df)]



#Pick the model with the highest Accuracy - prunned.info.accuracy
final.thres1 <- dtree_fit_info



#Compare Accuracy - default threshold (0.5) 
accuracy.tree.df.half <- data.frame(unprunned.accuracy.half,
                                    prunned.accuracy.half,
                                    prunned.gini.accuracy.half,
                                    prunned.info.accuracy.half)

accuracy.tree.df.half[, order(accuracy.tree.df.half)] 



#Pick the model with the highest Accuracy - - prunned.info.accuracy
final.thres1.half <- dtree_fit_info

