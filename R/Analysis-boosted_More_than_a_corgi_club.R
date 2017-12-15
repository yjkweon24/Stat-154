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
newtrain2 <- read.csv("../data/cleandata/newtrain2.csv", header = T)
newtest2 <- read.csv("../data/cleandata/newtest2.csv", header = T)
str(newtrain2)
str(newtest2)

## ------------------------------------------------------------------------
set.seed(100)



#Change to binary digit
combined <- rbind(newtrain2, newtest2)
combined$income <- as.numeric(combined$income) - 1



#First model
boosting1 <- gbm(income ~., data = combined[1:32402, ], distribution = "bernoulli",
                 n.trees = 5000, interaction.depth = 5)
summary(boosting1)
varImp(boosting1, numTrees = 5000)



#Test error of the first model
set.seed(100)
testerror1 <- c()
thresh <- 0.5
for(i in 1:500){
  #If I do not type = "response", they will give you logit output.
  yhat <- predict(boosting1, newdata = combined[32403:48598, -44], n.trees = (10 * i),
                  type = "response")
  yhat <- (yhat > thresh)
  testerror1[i] <- mean(yhat != combined[32403:48598, 44])
}
plot(testerror1)



#ROC curve - testing
pos1 <- c()
pos1 <- predict(boosting1, newdata = combined[32403:48598, -44], n.trees = 5000, type = "response")
predicts1 <- prediction(pos1, combined[32403:48598, 44])
roc1 <- ROCR::performance(predicts1, measure = "tpr", x.measure = "fpr")
plot(roc1)
abline(0, 1, col = "red")
auc1 <- ROCR::performance(predicts1, measure = "auc")
auc1@y.values



#Train error of the first model
set.seed(100)
trainerror1 <- c()
thresh <- 0.5
for(i in 1:500){
  yhat <- predict(boosting1, newdata = combined[1:32402, -44], n.trees = (10 * i), type = "response")
  yhat <- (yhat > thresh)
  trainerror1[i] <- mean(yhat != combined[1:32402, 44])
}
plot(trainerror1)



#ROC curve - training
pos1b <- c()
pos1b <- predict(boosting1, newdata = combined[1:32402, -44], n.trees = 5000, type = "response")
predicts1b <- prediction(pos1b, combined[1:32402, 44])
roc1b <- ROCR::performance(predicts1b, measure = "tpr", x.measure = "fpr")
plot(roc1b)
abline(0, 1, col = "red")
auc1b <- ROCR::performance(predicts1b, measure = "auc")
auc1b@y.values



#Second model
set.seed(100)
boosting2 <- gbm(income ~., data = combined[1:32402, ], distribution = "bernoulli", n.trees = 2000,
                interaction.depth = 5)
summary(boosting2)
varImp(boosting2, numTrees = 2000)



#Test error of the second model
set.seed(100)
testerror2 <- c()
thresh <- 0.5
for(i in 1:200){
  yhat <- predict(boosting2, newdata = combined[32403:48598, -44], n.trees = (10 * i),
                  type = "response")
  yhat <- (yhat > thresh)
  testerror2[i] <- mean(yhat != combined[32403:48598, 44])
}
plot(testerror2)



#ROC curve - testing
pos2 <- c()
pos2 <- predict(boosting2, newdata = combined[32403:48598, -44], n.trees = 2000, type = "response")
predicts2 <- prediction(pos2, combined[32403:48598, 44])
roc2 <- ROCR::performance(predicts2, measure = "tpr", x.measure = "fpr")
plot(roc2)
abline(0, 1, col = "red")
auc2 <- ROCR::performance(predicts2, measure = "auc")
auc2@y.values



#Train error of the second model
set.seed(100)
trainerror2 <- c()
thresh <- 0.5
for(i in 1:200){
  yhat <- predict(boosting2, newdata = combined[1:32402, -44], n.trees = (10 * i), type = "response")
  yhat <- (yhat > thresh)
  trainerror2[i] <- mean(yhat != combined[1:32402, 44])
}
plot(trainerror2)



#ROC curve - training
pos2b <- c()
pos2b <- predict(boosting2, newdata = combined[1:32402, -44], n.trees = 2000, type = "response")
predicts2b <- prediction(pos2b, combined[1:32402, 44])
roc2b <- ROCR::performance(predicts2b, measure = "tpr", x.measure = "fpr")
plot(roc2b)
abline(0, 1, col = "red")
auc2b <- ROCR::performance(predicts2b, measure = "auc")
auc2b@y.values



#Third model
set.seed(100)
boosting3 <- gbm(income ~., data = combined[1:32402, ], distribution = "bernoulli", n.trees = 5000,
                interaction.depth = 3)
summary(boosting3)
varImp(boosting3, numTrees = 5000)



#Test error of the third model
set.seed(100)
testerror3 <- c()
thresh <- 0.5
for(i in 1:500){
  yhat <- predict(boosting3, newdata = combined[32403:48598, -44], n.trees = (10 * i),
                  type = "response")
  yhat <- (yhat > thresh)
  testerror3[i] <- mean(yhat != combined[32403:48598, 44])
}
plot(testerror3)



#ROC curve - testing
pos3 <- c()
pos3 <- predict(boosting3, newdata = combined[32403:48598, -44], n.trees = 5000, type = "response")
predicts3 <- prediction(pos3, combined[32403:48598, 44])
roc3 <- ROCR::performance(predicts3, measure = "tpr", x.measure = "fpr")
plot(roc3)
abline(0, 1, col = "red")
auc3 <- ROCR::performance(predicts3, measure = "auc")
auc3@y.values



#Train error of the third model
set.seed(100)
trainerror3 <- c()
thresh <- 0.5
for(i in 1:500){
  yhat <- predict(boosting3, newdata = combined[1:32402, -44], n.trees = (10 * i), type = "response")
  yhat <- (yhat > thresh)
  trainerror3[i] <- mean(yhat != combined[1:32402, 44])
}
plot(trainerror3)



#ROC curve - training
pos3b <- c()
pos3b <- predict(boosting3, newdata = combined[1:32402, -44], n.trees = 5000, type = "response")
predicts3b <- prediction(pos3b, combined[1:32402, 44])
roc3b <- ROCR::performance(predicts3b, measure = "tpr", x.measure = "fpr")
plot(roc3b)
abline(0, 1, col = "red")
auc3b <- ROCR::performance(predicts3b, measure = "auc")
auc3b@y.values



#Fourth model
set.seed(100)
boosting4 <- gbm(income ~., data = combined[1:32402, ], distribution = "bernoulli", n.trees = 5000,
                interaction.depth = 3, shrinkage = 0.2)
summary(boosting4)
varImp(boosting4, numTrees = 5000)



#Test error of the fourth model
set.seed(100)
testerror4 <- c()
thresh <- 0.5
for(i in 1:500){
  yhat <- predict(boosting4, newdata = combined[32403:48598, -44], n.trees = (10 * i),
                  type = "response")
  yhat <- (yhat > thresh)
  testerror4[i] <- mean(yhat != combined[32403:48598, 44])
}
plot(testerror4)



#ROC curve - testing
pos4 <- c()
pos4 <- predict(boosting4, newdata = combined[32403:48598, -44], n.trees = 150, type = "response")
predicts4 <- prediction(pos4, combined[32403:48598, 44])
roc4 <- ROCR::performance(predicts4, measure = "tpr", x.measure = "fpr")
plot(roc4)
abline(0, 1, col = "red")
auc4 <- ROCR::performance(predicts4, measure = "auc")
auc4@y.values



#Train error of the fourth model
set.seed(100)
trainerror4 <- c()
thresh <- 0.5
for(i in 1:500){
  yhat <- predict(boosting4, newdata = combined[1:32402, -44], n.trees = (10 * i), type = "response")
  yhat <- (yhat > thresh)
  trainerror4[i] <- mean(yhat != combined[1:32402, 44])
}
plot(trainerror4)



#ROC curve - training
pos4b <- c()
pos4b <- predict(boosting4, newdata = combined[1:32402, -44], n.trees = 5000, type = "response")
predicts4b <- prediction(pos4b, combined[1:32402, 44])
roc4b <- ROCR::performance(predicts4b, measure = "tpr", x.measure = "fpr")
plot(roc4b)
abline(0, 1, col = "red")
auc4b <- ROCR::performance(predicts4b, measure = "auc")
auc4b@y.values



#Fifth model
set.seed(100)
boosting5 <- gbm(income ~., data = combined[1:32402, ], distribution = "bernoulli", n.trees = 5000,
                interaction.depth = 3, shrinkage = 0.1)
summary(boosting5)
varImp(boosting5, numTrees = 5000)



#Test error of the fifth model
set.seed(100)
testerror5 <- c()
thresh <- 0.5
for(i in 1:500){
  yhat <- predict(boosting5, newdata = combined[32403:48598, -44], n.trees = (10 * i),
                  type = "response")
  yhat <- (yhat > thresh)
  testerror5[i] <- mean(yhat != combined[32403:48598, 44])
}
plot(testerror5)



#ROC curve - testing
pos5 <- c()
pos5 <- predict(boosting5, newdata = combined[32403:48598, -44], n.trees = 800, type = "response")
predicts5 <- prediction(pos5, combined[32403:48598, 44])
roc5 <- ROCR::performance(predicts5, measure = "tpr", x.measure = "fpr")
plot(roc5)
abline(0, 1, col = "red")
auc5 <- ROCR::performance(predicts5, measure = "auc")
auc5@y.values



#Train error of the fifth model
set.seed(100)
trainerror5 <- c()
thresh <- 0.5
for(i in 1:500){
  yhat <- predict(boosting5, newdata = combined[1:32402, -44], n.trees = (10 * i), type = "response")
  yhat <- (yhat > thresh)
  trainerror5[i] <- mean(yhat != combined[1:32402, 44])
}
plot(trainerror5)



#ROC curve - training
pos5b <- c()
pos5b <- predict(boosting5, newdata = combined[1:32402, -44], n.trees = 5000, type = "response")
predicts5b <- prediction(pos5b, combined[1:32402, 44])
roc5b <- ROCR::performance(predicts5b, measure = "tpr", x.measure = "fpr")
plot(roc5b)
abline(0, 1, col = "red")
auc5b <- ROCR::performance(predicts5b, measure = "auc")
auc5b@y.values



#ROC and AUC combined testing 
plot(roc1, type = "l", col = "red")
par(new = TRUE)
plot(roc2, type = "l", col = "green")
par(new = TRUE)
plot(roc3, type = "l", col = "blue")
par(new = TRUE)
plot(roc4, type = "l", col = "black")
par(new = TRUE)
plot(roc5, type = "l", col = "yellow",
     main = "model1: red, model2: green, model3: blue, model4: black, model5: yellow")

paste("AUC for model 1 is", round(auc1@y.values[[1]], 5))
paste("AUC for model 2 is", round(auc2@y.values[[1]], 5))
paste("AUC for model 3 is", round(auc3@y.values[[1]], 5))
paste("AUC for model 4 is", round(auc4@y.values[[1]], 5))
paste("AUC for model 5 is", round(auc5@y.values[[1]], 5))



#ROC and AUC combined training 
plot(roc1b, type = "l", col = "red")
par(new = TRUE)
plot(roc2b, type = "l", col = "green")
par(new = TRUE)
plot(roc3b, type = "l", col = "blue")
par(new = TRUE)
plot(roc4b, type = "l", col = "black")
par(new = TRUE)
plot(roc5b, type = "l", col = "yellow",
     main = "model1: red, model2: green, model3: blue, model4: black, model5: yellow")

paste("AUC for model 1 is", round(auc1b@y.values[[1]], 5))
paste("AUC for model 2 is", round(auc2b@y.values[[1]], 5))
paste("AUC for model 3 is", round(auc3b@y.values[[1]], 5))
paste("AUC for model 4 is", round(auc4b@y.values[[1]], 5))
paste("AUC for model 5 is", round(auc5b@y.values[[1]], 5))



#Partial dependence plots
variables <- c("Married.civ.spouse", "education.num", "age", "capital.gain",
               "hours.per.week", "capital.loss")
par(mfrow = c(2, 3))
for(i in 1:6){
  plot(boosting1, i = variables[i])
}

for(i in 1:6){
  plot(boosting2, i = variables[i])
}

for(i in 1:6){
  plot(boosting3, i = variables[i])
}

for(i in 1:6){
  plot(boosting4, i = variables[i])
}

for(i in 1:6){
  plot(boosting5, i = variables[i])
}



#Check imbalance
table(combined$income)
11443 / 48598 #23.5%
37155 / 48598 #76.5%

## ------------------------------------------------------------------------
set.seed(100)
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
boostingtrain <- caret::train(income~., data = newtrain2, method = "gbm", metric = "Accuracy", trControl = trctrl)

summary(boostingtrain)
boostingtrain
boostingtrain$bestTune
boostingtrain$results
boostingtrain$finalModel
boostingtrain$resample
boostingtrain$resampledCM
boostingtrain$perfNames



#Optimal model
# boostingoptimal <- gbm(income ~., data = combined[1:32402, ], distribution = "bernoulli", n.trees = 150,
#                 interaction.depth = 3, shrinkage = 0.1)
# summary(boostingoptimal)
# varImp(boostingoptimal, numTrees = 150)

#Test error of the optimal model
# testerroroptimal <- c()
# thresh <- 0.5
# for(i in 1:15){
#   yhat <- predict(boostingtrain, newdata = combined[32403:48598, -44], n.trees = (10 * i), type = "prob")
#   yhat <- (yhat > thresh)
#   testerroroptimal[i] <- mean(yhat != combined[32403:48598, 44])
# }
# plot(testerroroptimal)



#ROC curve - testing
set.seed(100)
posopt <- c()
posopt <- predict(boostingtrain, newdata = combined[32403:48598, -44], n.trees = 150, type = "prob")
predictsopt <- prediction(posopt[, 2], combined[32403:48598, 44])
rocopt <- ROCR::performance(predictsopt, measure = "tpr", x.measure = "fpr")
plot(rocopt)
abline(0, 1, col = "red")
aucopt <- ROCR::performance(predictsopt, measure = "auc")
aucopt@y.values



#ROC and AUC combined testing 
plot(roc1, type = "l", col = "red")
par(new = TRUE)
plot(roc2, type = "l", col = "green")
par(new = TRUE)
plot(roc3, type = "l", col = "blue")
par(new = TRUE)
plot(roc4, type = "l", col = "black")
par(new = TRUE)
plot(roc5, type = "l", col = "yellow")
par(new = TRUE)
plot(rocopt, type = "l", col = "purple",
     main = "1: red, 2: green, 3: blue, 4: black, 5: yellow, trained: purple")



#Train error of the optimal model
# trainerroropt <- c()
# thresh <- 0.5
# for(i in 1:500){
#   yhat <- predict(boostingoptimal, newdata = combined[1:32402, -44], n.trees = (10 * i), type = "response")
#   yhat <- (yhat > thresh)
#   trainerroropt[i] <- mean(yhat != combined[1:32402, 44])
# }
# plot(trainerroropt)



#ROC curve - training
pos5opt <- c()
pos5opt <- predict(boostingtrain, newdata = combined[1:32402, -44], n.trees = 150, type = "prob")
predicts5opt <- prediction(pos5opt[, 2], combined[1:32402, 44])
roc5opt <- ROCR::performance(predicts5opt, measure = "tpr", x.measure = "fpr")
plot(roc5opt)
abline(0, 1, col = "red")
auc5opt <- ROCR::performance(predicts5opt, measure = "auc")
auc5opt@y.values



# boosting <- C50::C5.0(newtrain2[, -45], newtrain2[, 45], trials = 10) #boosting iteration = 10
# summary(boosting)
# 
# classes <- predict(boosting, newtest2[, -45], type = "class")
# table(classes, newtest2[, 45])
# 
# acc <- sum(classes == newtest2[, 45]) / length(newtest2[, 45])
# acc



# https://github.com/topepo/caret/blob/master/RegressionTests/Code/C5.0.R 
# 
# cctrl1 <- trainControl(method = "cv", number = 3, returnResamp = "all",
#                        classProbs = TRUE, 
#                        summaryFunction = twoClassSummary)
# cctrl2 <- trainControl(method = "LOOCV",
#                        classProbs = TRUE, summaryFunction = twoClassSummary)
# cctrl3 <- trainControl(method = "none",
#                        classProbs = TRUE, summaryFunction = twoClassSummary)
# cctrlR <- trainControl(method = "cv", number = 3, returnResamp = "all",
#                        classProbs = TRUE, 
#                        search = "random")
# 
# y <- as.numeric(newtrain2$income) - 1 
# test_class_cv_model <- train(newtrain2[, -45], y, 
#                               method = "C5.0", 
#                               trControl = cctrl1,
#                               metric = "ROC", 
#                               control = C50::C5.0Control(seed = 1),
#                               preProc = c("center", "scale"))

## ---- echo = F, eval = F-------------------------------------------------
## thresholds <- seq(from = 0.001, 0.999, 0.001)
## 
## error <- c() #misclassfication for the first model
## for(i in 1:length(thresholds)){
##   pr <- c()
##   pr <- ifelse(predict(boosting1, newdata = combined[1:32402, -44], n.trees = 5000, type = "response")
##                > (0.001 * i), 1, 0)
##   error[i] <- mean(pr != combined[1:32402, 44])
## }
## 
## 
## 
## error2 <- c() #misclassfication for the second model
## for(i in 1:length(thresholds)){
##   pr <- c()
##   pr <- ifelse(predict(boosting2, newdata = combined[1:32402, -44], n.trees = 2000, type = "response")
##                > (0.001 * i), 1, 0)
##   error2[i] <- mean(pr != combined[1:32402, 44])
## }
## 
## 
## 
## error3 <- c() #misclassfication for the third model
## for(i in 1:length(thresholds)){
##   pr <- c()
##   pr <- ifelse(predict(boosting3, newdata = combined[1:32402, -44], n.trees = 5000, type = "response")
##                > (0.001 * i), 1, 0)
##   error3[i] <- mean(pr != combined[1:32402, 44])
## }
## 
## 
## 
## error4 <- c() #misclassfication for the fourth model
## for(i in 1:length(thresholds)){
##   pr <- c()
##   pr <- ifelse(predict(boosting4, newdata = combined[1:32402, -44], n.trees = 150, type = "response")
##                > (0.001 * i), 1, 0)
##   error4[i] <- mean(pr != combined[1:32402, 44])
## }
## 
## 
## 
## error5 <- c() #misclassfication for the fifth model
## for(i in 1:length(thresholds)){
##   pr <- c()
##   pr <- ifelse(predict(boosting5, newdata = combined[1:32402, -44], n.trees = 800, type = "response")
##                > (0.001 * i), 1, 0)
##   error5[i] <- mean(pr != combined[1:32402, 44])
## }
## 
## 
## 
## error6 <- c() #misclassfication for the trained model
## for(i in 1:length(thresholds)){
##   pr <- c()
##   pr <- ifelse(predict(boostingtrain, newdata = combined[1:32402, -44], n.trees = 150, type = "prob")[, 2]
##                > (0.001 * i), 1, 0)
##   error6[i] <- mean(pr != combined[1:32402, 44])
## }
## 
## 
## 
## par(mfrow=c(1, 3))
## plot(thresholds, error)
## plot(thresholds, error2)
## plot(thresholds, error3)
## plot(thresholds, error4)
## plot(thresholds, error5)
## plot(thresholds, error6)

## ------------------------------------------------------------------------
set.seed(100)
thresh <- 0.5



a <- predict(boosting1, newdata = combined[32403:48598, -44], n.trees = 5000, type = "response")
a1 <- (a > thresh)
a2 <- mean(a1 == combined[32403:48598, 44])



b <- predict(boosting2, newdata = combined[32403:48598, -44], n.trees = 2000, type = "response")
b1 <- (b > 0.3)
b2 <- mean(b1 == combined[32403:48598, 44])



c <- predict(boosting3, newdata = combined[32403:48598, -44], n.trees = 5000, type = "response")
c1 <- (c > thresh)
c2 <- mean(c1 == combined[32403:48598, 44])



d <- predict(boosting4, newdata = combined[32403:48598, -44], n.trees = 200, type = "response")
d1 <- (d > thresh)
d2 <- mean(d1 == combined[32403:48598, 44])



e <- predict(boosting5, newdata = combined[32403:48598, -44], n.trees = 800, type = "response")
e1 <- (e > thresh)
e2 <- mean(e1 == combined[32403:48598, 44])



f <- predict(boostingtrain, newdata = combined[32403:48598, -44], n.trees = 150, type = "raw")
f1 <- as.numeric(f) - 1
f2 <- mean(f1 == combined[32403:48598, 44])



a2
b2
c2
d2
e2
f2

## ------------------------------------------------------------------------
final.auc4 <- boosting5

final.thres4 <- boosting4

