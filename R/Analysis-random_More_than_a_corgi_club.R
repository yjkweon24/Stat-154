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
options(scipen=999) #get rid of scientific notation 

## ------------------------------------------------------------------------
set.seed(100)
newtrain2 <- read.csv("../data/cleandata/newtrain2.csv", header = T)
newtest2 <- read.csv("../data/cleandata/newtest2.csv", header = T)
str(newtrain2)
str(newtest2)

## ------------------------------------------------------------------------
set.seed(100)



#=============================================================



#Create a task
traintask <- makeClassifTask(data = newtrain2, target = "income", positive = ">50K")
testtask <- makeClassifTask(data = newtest2, target = "income", positive = ">50K")



#Brief view of trainTask
traintask



#For deeper View
str(getTaskData(traintask))



#=============================================================



#Make a random forest learner
rf <- makeLearner("classif.randomForest", predict.type = "response",
                  par.vals = list(ntree = 50L, importance = TRUE))



#To check the performance, set up a validation strategy
#set 3 fold cross validation
rdesc <- makeResampleDesc("CV", iters = 3L)



r2 <- resample(learner = rf, task = traintask, resampling = rdesc, 
               measures = list(tpr,fpr,fnr,tnr,acc), show.info = TRUE)



#Show true positive rate, false positive rate, false negative rate, false positive rate, and accuracy rate from random forest model
r2



#Aggr. Result: tpr.test.mean=0.623,fpr.test.mean=0.0598,fnr.test.mean=0.377,tnr.test.mean=0.94,acc.test.mean=0.865



#Internally, random forest uses a cutoff of 0.5  --> 
#if a particular unseen observation has a probability higher than 0.5, it will be classified as >50K.
#In random forest, we have the option to customize the internal cutoff. As the false negative rate is very high now, we'll increase the cutoff for negative classes (<=50K) and accordingly reduce it for positive classes (>50K). Then, train the model again.



#Evaluating by using new cutoff
rf$par.vals <- list(ntree = 50L, importance = TRUE, cutoff = c(0.53, 0.47))
r3 <- resample(learner = rf, task = traintask, resampling = rdesc, 
              measures = list(tpr,fpr,fnr,tnr,acc), show.info = TRUE)



#Show true positive rate, false positive rate, false negative rate, false positive rate, and accuracy rate from random forest model
r3



#Aggr. Result: tpr.test.mean=0.651,fpr.test.mean=0.0683,fnr.test.mean=0.349,tnr.test.mean=0.932,acc.test.mean=0.865    ---> we can see that false negative rate is decreased even though the accuracy rate stays the same. I have tried cutoff = c(0.6, 0.4), cutoff = c(0.7, 0.3) but they all gave lower accuracy late.



#========================================================================



#Random Forest tuning



#Train a old untunned model
untunnedforest <- mlr::train(rf, traintask)



#Let's see how the test classification error changes as we increase the number of trees for untunned model   ( #number of trees VS test classification error)

rf.untunned_ind <- predict(untunnedforest$learner.model, newtrain2, 
                    predict.all = T)$individual

head(rf.untunned_ind,2)
n <- dim(rf.untunned_ind)[1]
m <- dim(rf.untunned_ind)[2] / 2
predicted_ind <- c()
misclass.ind <- c()

for(i in 1:m){   # number of tree
  for(j in 1:n){
    predicted_ind[j] <- names(which.max(table(rf.untunned_ind[j, 1:i*2-1])))
  }
  misclass.ind[i] <- mean(predicted_ind != newtrain2$income)
}

rf.untunned.df <- data.frame(misclass.ind, ntree = seq(1, 49, 2))

ggplot(rf.untunned.df, aes(x = ntree, y = misclass.ind)) + geom_line() +
  ggtitle("Number of trees vs Misclassification rate in training dataset - untunned random forest model")



#======================== Let's actually tune the hyperparameters

getParamSet(rf)



#Specifying the search space for hyperparameters
rf_params <- makeParamSet(makeIntegerParam("mtry", lower = 2, upper = 10),
                       makeIntegerParam("nodesize", lower = 10, upper = 50),
                       makeIntegerParam("ntree", lower = 3, upper = 100)
                       )



#Set validation strategy
rdesc <- makeResampleDesc("CV", iters = 3L)



#Set optimization technique
rf_ctrl <- makeTuneControlRandom(maxit = 5L)



#Start Hypertuning the parameters
rf_tune <- tuneParams(learner = rf, task = traintask, resampling = rdesc,
                   measures = list(acc), par.set = rf_params,
                   control = rf_ctrl, show.info = TRUE)



#Optimal hypertuned parameters
rf_tune$x



#Accuracy rate from Cross Validation
rf_tune$y



#Use hyperparameters for modeling
rf_tree <- setHyperPars(rf, par.vals = rf_tune$x)



#Train a model
rforest <- mlr::train(rf_tree, traintask)
getLearnerModel(rforest)



#========================================================================



#Let's see how the test classification error changes as we increase the number of trees for tunned model  ( #number of trees VS test classification error)



rf.tunned_ind <- predict(rforest$learner.model, newtrain2, 
                    predict.all = T)$individual
head(rf.tunned_ind,2)
n <- dim(rf.tunned_ind)[1]
m <- ceiling(dim(rf.tunned_ind)[2] / 2)
predicted_ind <- c()
misclass.ind <- c()

for(i in 1:m){   # number of tree
  for(j in 1:n){
    predicted_ind[j] <- names(which.max(table(rf.tunned_ind[j, 1:i*2-1])))
  }
  misclass.ind[i] <- mean(predicted_ind != newtrain2$income)
}

rf.tunned.df <- data.frame(misclass.ind, ntree = seq(1, 80, 2))

ggplot(rf.untunned.df, aes(x = ntree, y = misclass.ind)) + geom_line() +
  ggtitle("Number of trees vs Misclassification rate in training dataset - tunned random forest model")



#========================================================================



#***Make plots for random forest model



#Variable importance statistics
varImpPlot(rforest$learner.model)
importance(rforest$learner.model)

## ------------------------------------------------------------------------
set.seed(100)
# ** Plot (top) subset of random forest tree
plot(rforest$learner.model)
#getTree(rforest$learner.model, k = 10, labelVar = TRUE)



# ** Make predictions on training dataset
rfclass1 <- predict(rforest, traintask)



#Confusion matrix on training dataset
confusionMatrix(rfclass1$data$response, rfclass1$data$truth)



#Make random forest plots on training dataset
plot(rfclass1$data$response, newtrain2$income)
abline(0, 1)



#Training accuracy rate
1 - mean(rfclass1$data$response != newtrain2$income)



#Make predictions on test dataset
rfclass2 <- predict(rforest, testtask)



#Confusion matrix on test dataset
confusionMatrix(rfclass2$data$response, rfclass2$data$truth)



#Make random forest plots on test dataset
plot(rfclass2$data$response, newtest2$income)
abline(0,1)



#Test accuracy rate
1 - mean(rfclass2$data$response != newtest2$income)

## ------------------------------------------------------------------------
set.seed(100)



#ROC Curve: https://stackoverflow.com/questions/30818188/roc-curve-in-r-using-rpart-package
#Untunned random forest model
#Getting predicted >50K of income probabilities 
untunned.forest <- mlr::train(rf, traintask)
untunned.rf_prob <- predict(untunned.forest$learner.model,
                            newdata = newtest2, type = "prob")[, 2]
untunned.rf_prediction <- prediction(untunned.rf_prob, newtest2$income)
untunned.rf_performance <- ROCR::performance(untunned.rf_prediction, measure = "tpr",
                                             x.measure = "fpr")



#Plot ROC curve 
plot(untunned.rf_performance, main = "ROC curve")
abline(a = 0, b = 1, lty = 2)



#Calculate AUC
untunned.rf.auc <- ROCR::performance(untunned.rf_prediction,
                                     measure = "auc")@y.values[[1]]
untunned.rf.auc



#=====================================================================



#Tunned random forest model
#Getting predicted >50K of income probabilities 
tunned.rf_prob <- predict(rforest$learner.model, newdata = newtest2,
                     type = "prob")[, 2]
tunned.rf_prediction <- prediction(tunned.rf_prob, newtest2$income)
tunned.rf_performance <- ROCR::performance(tunned.rf_prediction, measure = "tpr", x.measure = "fpr")



#Plot ROC curve 
plot(tunned.rf_performance, main = "ROC curve")
abline(a = 0, b = 1, lty = 2)



#Calculate AUC
tunned.rf.auc <- ROCR::performance(tunned.rf_prediction,
                                   measure="auc")@y.values[[1]]
tunned.rf.auc

## ------------------------------------------------------------------------
set.seed(100)



#Compare ROC curve 
plot(tunned.rf_performance, main = "ROC curve", col = "blue")
plot(untunned.rf_performance, add = TRUE, col = "red")
abline(a = 0, b = 1, lty = 2)
legend("bottomright", legend = c("Tunned", "Untunned"), col = c("blue", "red"),
       lwd=3, cex=.8, horiz = TRUE)


## ------------------------------------------------------------------------
set.seed(100)



thresholds <- seq(from = 0.001, 0.999, 0.001)
accuracy <- c()



#==================================================================



#Using train dataset to check new accuracy driven by  new threshold
untunned.rf_prob.train <- predict(untunned.forest$learner.model,
                            newdata = newtrain2, type = "prob")[, 2]



#Tuned by gini index splitting criterion model
for(i in 1:length(thresholds)){
  accuracy[i] <- mean((untunned.rf_prob.train > thresholds[i]) ==
                        (newtrain2$income == ">50K"))
}



#Threshold which give maximum accuracy
thres1 <- which.max(accuracy) * 0.001
thres1



#plot of accuracy vs thresholds
threstable <- data.frame(thresholds, accuracy)
ggplot(threstable, aes(x = thresholds, y = accuracy)) + geom_point()
  


#Get confusion matrix of testset data using the optimal threshold
confusionMatrix(untunned.rf_prob > thres1, newtest2$income == ">50K")



#Test accuracy rate by using optimal threshold
rf.untunned.accuracy <- mean((untunned.rf_prob > thres1) == (newtest2$income == ">50K"))



#compare the test accuracy by using default threshold (0.5)
rf.untunned.accuracy.half <- mean((untunned.rf_prob > 0.5) == (newtest2$income == ">50K"))




#==================================================================



#Using train dataset to check new accuracy driven by  new threshold
tunned.rf_prob.train <- predict(rforest$learner.model,
                            newdata = newtrain2, type = "prob")[, 2]



#Tuned by gini index splitting criterion model
for(i in 1:length(thresholds)){
  accuracy[i] <- mean((tunned.rf_prob.train > thresholds[i]) ==
                        (newtrain2$income == ">50K"))
}



#Threshold which give maximum accuracy
thres2 <- which.max(accuracy) * 0.001
thres2



#plot of accuracy vs thresholds
threstable <- data.frame(thresholds, accuracy)
ggplot(threstable, aes(x = thresholds, y = accuracy)) + geom_point()
  


#Get confusion matrix of testset data using the optimal threshold
confusionMatrix(tunned.rf_prob > thres2, newtest2$income == ">50K")



#Test accuracy rate by using optimal threshold
rf.tunned.accuracy <- mean((tunned.rf_prob > thres2) == (newtest2$income == ">50K"))



#compare the test accuracy by using default threshold (0.5)
rf.tunned.accuracy.half <- mean((tunned.rf_prob > 0.5) == (newtest2$income == ">50K"))

## ------------------------------------------------------------------------
set.seed(100)



#Compare AUC
auc <- data.frame(tunned.rf.auc, untunned.rf.auc)
auc[, order(auc)]



#Pick the model with the largest AUC
final.auc3 <- rforest$learner.model



#Compare Accuracy - optimal threshold
accuracy.random.df <- data.frame(rf.tunned.accuracy, rf.untunned.accuracy)
accuracy.random.df[, order(accuracy.random.df)]



#Pick the model with the highest Accuracy 
final.thres3 <- rforest$learner.model



#Compare Accuracy - default threshold(0.5)
accuracy.random.df.half <- data.frame(rf.tunned.accuracy.half,
                                      rf.untunned.accuracy.half)
accuracy.random.df.half[, order(accuracy.random.df.half)]



#Pick the model with the largest Accuracy 
final.thres3.half <- untunned.forest$learner.model

