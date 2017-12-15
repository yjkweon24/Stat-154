## ----setup, include=FALSE------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)
library(rpart) #decision tree
library(mlr, warn.conflicts = T) #missing values imputation
library(missForest, warn.conflicts = T) #missing values imputation
library(mi, warn.conflicts = T) #missing values imputation
library(mice, warn.conflicts = T) #missing values imputation
library(VIM) #finding patterns of missing values
library(Hmisc) #missing values imputation
library(lattice)
library(arules) #discretize
library(lme4) #dummfiy
library(dplyr) #Check proportions of data for EDA
options(scipen=999) #get rid of scientific notation 

## ----EDA-----------------------------------------------------------------
train <- read.table("../data/rawdata/adult.data.txt", sep = ",", na.strings = "?",
                    strip.white = T)
test <- read.table("../data/rawdata/adult.test.txt", sep = ",", na.strings = "?",
                   strip.white = T)

dim(train)
dim(test)

colnames(train) <- c("age", "workclass", "fnlwgt", "education", "education-num",
                     "marital-status", "occupation", "relationship", "race", "sex",
                     "capital-gain", "capital-loss", "hours-per-week", "native-country", "income")

colnames(test) <- c("age", "workclass", "fnlwgt", "education", "education-num",
                     "marital-status", "occupation", "relationship", "race", "sex",
                     "capital-gain", "capital-loss", "hours-per-week", "native-country", "income")



#Find missing values and NAs for training set.
for(i in 1:ncol(train)){
  cat("<names of NA rows in", colnames(train)[i], "variable>", "\n")
  cat(rownames(train)[is.na(train[, i])], "\n")
  cat("Number of NA values:  ", length(rownames(train)[is.na(train[, i])]), "\n")
  print("======================================")
  print("======================================")
  
  cat("<names of rows contain missing values in", colnames(train)[i], "variable>", "\n")
  cat(rownames(train[which(train[, i] == ""), ]), "\n")
  cat("Number of Missing values :  ", length(rownames(train[which(train[, i] == ""), ])), "\n")
  print("======================================")
  print("======================================")
  
  cat("<names of rows contain ? values in", colnames(train)[i], "variable>", "\n")
  cat(rownames(train[which(train[, i] == " ?"), ]), "\n")
  cat("Number of ? values :  ", length(rownames(train[which(train[, i] == " ?"), ])), "\n")
  print("======================================")
  print("======================================")
}



#Find missing values and NAs for testing set.
for(i in 1:ncol(test)){
  cat("<names of NA rows in", colnames(test)[i], "variable>", "\n")
  cat(rownames(test)[is.na(test[, i])], "\n")
  cat("Number of NA values:  ", length(rownames(test)[is.na(test[, i])]), "\n")
  print("======================================")
  print("======================================")
  
  cat("<names of rows contain missing values in", colnames(test)[i], "variable>", "\n")
  cat(rownames(test[which(test[, i] == ""), ]), "\n")
  cat("Number of Missing values :  ", length(rownames(test[which(test[, i] == ""), ])), "\n")
  print("======================================")
  print("======================================")
  
  cat("<names of rows contain ? values in", colnames(test)[i], "variable>", "\n")
  cat(rownames(test[which(test[, i] == " ?"), ]), "\n")
  cat("Number of ? values :  ", length(rownames(test[which(test[, i] == " ?"), ])), "\n")
  print("======================================")
  print("======================================")
}



#Get percentage of missing values
apply(train, 2, function(x) sum(is.na(x)) / length(x)) * 100
apply(test, 2, function(x) sum(is.na(x)) / length(x)) * 100



#MICE package to see the pattern 
md.pattern(train)
plot <- aggr(train, col = c('blue', 'yellow'),
                    numbers = TRUE, sortVars = TRUE,
                    labels = names(train), cex.axis = .7,
                    gap = 2, ylab = c("Missing data", "Pattern"))

md.pattern(test)
plot <- aggr(test, col = c('blue', 'yellow'),
                    numbers = TRUE, sortVars = TRUE,
                    labels = names(test), cex.axis = .7,
                    gap = 2, ylab = c("Missing data", "Pattern"))



# Hmisc package to impute missing values
# ww <- aregImpute(~ age + workclass + fnlwgt + education + `education-num` + `marital-status` +
#                    occupation + relationship + race + sex + `capital-gain` + `capital-loss` +
#                    `hours-per-week` + income,
#                  data = train, n.impute = 5, group = "income")



#mlr package to impute missing values
# newworkclass <- impute(train[,2], classes = list(factor = imputeMode(), integer = imputeMean()), dummy.classes = c("integer","factor"), dummy.type = "numeric")
# 
# newoccupation <- impute(train[,7], classes = list(factor = imputeMode(), integer = imputeMean()), dummy.classes = c("integer","factor"), dummy.type = "numeric")
# 
# newcountry <- impute(train[,14], classes = list(factor = imputeMode(), integer = imputeMean()), dummy.classes = c("integer","factor"), dummy.type = "numeric")



#missForest package to impute missing values
# foresting <- missForest(train, maxiter = 5, ntree = 100)
# foresting$OOBerror
# newtrain <- foresting$ximp
# write.csv(newtrain, file = "../data/cleandata/newtrain.csv", col.names = T, row.names = F)
newtrain <- read.csv("../data/cleandata/newtrain.csv", header = T)
dim(newtrain)



# foresting2 <- missForest(test, maxiter = 5, ntree = 100)
# foresting2$OOBerror
# newtest <- foresting2$ximp
# write.csv(newtest, file = "../data/cleandata/newtest.csv", col.names = T, row.names = F)
newtest <- read.csv("../data/cleandata/newtest.csv", header = T)
dim(newtest)



#Check whether the data is messed up while imputing missing values
#They should never show 0, as we are supposed to see only missing value has been changed...
#Compare NA with new number in new data set should show NA, not 0.
t <- matrix(0, 1, ncol(train))
for(i in 1:20){
  a <- sample.int(nrow(newtrain), 1)
  t <- rbind(t, (newtrain[a, ] == train[a, ]))
}
t <- t[-1, ]
t

t2 <- matrix(0, 1, ncol(test))
for(i in 1:20){
  a <- sample.int(nrow(newtest), 1)
  t2 <- rbind(t2, (newtest[a, ] == test[a, ]))
}
t2 <- t2[-1, ]
t2

## ------------------------------------------------------------------------
#See structure and summaries before removing outliers
str(newtest)
summary(newtest)

str(newtrain)
summary(newtrain)



#Deal with outliers for training sets
continuouscol <- c(1, 3, 5, 11, 12, 13) #subset continous variables

par(mfrow = c(2, 3))
for(i in continuouscol){
  boxplot(newtrain[, i], main = paste("boxplot for", colnames(newtrain[i])),
          xlab = colnames(newtrain[i]))
}

for(i in continuouscol){
  den_acc <- density(newtrain[, i], adjust = 1)
  plot(den_acc, main = paste("density plot for", colnames(newtrain[i])))
  polygon(den_acc, col = "red", border = "blue")
}

outlierstrain <- list()
for(i in continuouscol){
  outliers <- boxplot.stats(newtrain[, i])$out
  numbers <- length(outliers)
  outlierstrain[[i]] <- list(outliers, numbers)
}
head(outlierstrain)

fnlwgttrainout <- tail(order(rank(newtrain[, 3])), 15)
fnlout <- c()
for(i in 1:length(fnlwgttrainout)){
  fnlout[i] <- newtrain[fnlwgttrainout[i], 3]
}

#head(order(rank(newtrain[,5])))
table(newtrain[, 11])
gainout <- tail(order(rank(newtrain[, 11])), 159)



#Outliers removing for training sets.
dim(newtrain)
newtrain <- newtrain[-gainout, ]
dim(newtrain)



#Deal with outliers for testing sets
for(i in continuouscol){
  boxplot(newtest[, i], main = paste("boxplot for", colnames(newtest[i])),
          xlab = colnames(newtest[i]))
}

for(i in continuouscol){
  den_acc <- density(newtest[, i], adjust = 1)
  plot(den_acc, main = paste("density plot for", colnames(newtest[i])))
  polygon(den_acc, col = "red", border = "blue")
}

outlierstest <- list()
for(i in continuouscol){
  outliers <- boxplot.stats(newtest[, i])$out
  numbers <- length(outliers)
  outlierstest[[i]] <- list(outliers, numbers)
}
head(outlierstest)

table(newtest[, 11])
gainout <- tail(order(rank(newtest[, 11])), 85)



#Outliers removing for training sets.
dim(newtest)
newtest <- newtest[-gainout, ]
dim(newtest)



#Plots after removing outliers training
for(i in continuouscol){
  boxplot(newtrain[, i], main = paste("boxplot for", colnames(newtrain[i]), "-outliers removed"),
          xlab = colnames(newtrain[i]))
}

for(i in continuouscol){
  den_acc <- density(newtrain[, i], adjust = 1)
  plot(den_acc, main = paste("density plot for", colnames(newtrain[i]), "-outliers removed"))
  polygon(den_acc, col = "red", border = "blue")
}



#Plots after removing outliers testing
for(i in continuouscol){
  boxplot(newtest[, i], main = paste("boxplot for", colnames(newtest[i]), "-outliers removed"),
          xlab = colnames(newtest[i]))
}

for(i in continuouscol){
  den_acc <- density(newtest[, i], adjust = 1)
  plot(den_acc, main = paste("density plot for", colnames(newtest[i]), "-outliers removed"))
  polygon(den_acc, col = "red", border = "blue")
}

## ------------------------------------------------------------------------
#detach("package:plyr", unload=TRUE) #because plyr and dplyr existed together conflicting...

#Check whether categorical variables can be discretized....
plot(newtrain$workclass)
table(newtrain$workclass)
newtrain %>% group_by(workclass) %>%  summarise (n = n()) %>% mutate(freq = n / sum(n))

plot(newtest$workclass)
table(newtest$workclass)
newtest %>% group_by(workclass) %>%  summarise (n = n()) %>% mutate(freq = n / sum(n))



plot(newtrain$education)
table(newtrain$education)
newtrain %>% group_by(education) %>%  summarise (n = n()) %>% mutate(freq = n / sum(n))

plot(newtest$education)
table(newtest$education)
newtest %>% group_by(education) %>%  summarise (n = n()) %>% mutate(freq = n / sum(n))



plot(newtrain$marital.status)
table(newtrain$marital.status)
newtrain %>% group_by(marital.status) %>%  summarise (n = n()) %>% mutate(freq = n / sum(n))

plot(newtest$marital.status)
table(newtest$marital.status)
newtest %>% group_by(marital.status) %>%  summarise (n = n()) %>% mutate(freq = n / sum(n))



plot(newtrain$occupation)
table(newtrain$occupation)
newtrain %>% group_by(occupation) %>%  summarise (n = n()) %>% mutate(freq = n / sum(n))

plot(newtest$occupation)
table(newtest$occupation)
newtest %>% group_by(occupation) %>%  summarise (n = n()) %>% mutate(freq = n / sum(n))



plot(newtrain$relationship)
table(newtrain$relationship)
newtrain %>% group_by(relationship) %>%  summarise (n = n()) %>% mutate(freq = n / sum(n))

plot(newtest$relationship)
table(newtest$relationship)
newtest %>% group_by(relationship) %>%  summarise (n = n()) %>% mutate(freq = n / sum(n))



plot(newtrain$race)
table(newtrain$race)
newtrain %>% group_by(race) %>%  summarise (n = n()) %>% mutate(freq = n / sum(n))

plot(newtest$race)
table(newtest$race)
newtest %>% group_by(race) %>%  summarise (n = n()) %>% mutate(freq = n / sum(n))



plot(newtrain$sex)
table(newtrain$sex)
newtrain %>% group_by(sex) %>%  summarise (n = n()) %>% mutate(freq = n / sum(n))

plot(newtest$sex)
table(newtest$sex)
newtest %>% group_by(sex) %>%  summarise (n = n()) %>% mutate(freq = n / sum(n))



plot(newtrain$native.country)
table(newtrain$native.country)
newtrain %>% group_by(native.country) %>%  summarise (n = n()) %>% mutate(freq = n / sum(n))

plot(newtest$native.country)
table(newtest$native.country)
newtest %>% group_by(native.country) %>%  summarise (n = n()) %>% mutate(freq = n / sum(n))



#Check collinearity issues
newtrain %>% group_by(education) %>% summarise (n = n()) %>% mutate(freq = n / sum(n))
newtrain %>% group_by(education.num) %>% summarise (n = n()) %>% mutate(freq = n / sum(n))

newtrain <- newtrain[, -4]
newtest <- newtest[, -4]

## ------------------------------------------------------------------------
#Find correlations of the data - for collinearity issue checks
cor(newtest[, c(1, 3, 4, 10, 12)])
cor(newtrain[, c(1, 3, 4, 10, 12)])



#remove fnlwght variable.
newtrain <- newtrain[, -3]
newtest <- newtest[, -3]



#See structure and summaries after removing outliers
str(newtest)
summary(newtest)

str(newtrain)
summary(newtrain)



#Analyzing/checking before discretizing
# table(newtrain[,14])
# table(newtest[,14])
# 
# plot(newtrain$education)
# plot(newtrain$occupation)
# plot(newtrain$native.country)
# 
# plot(newtest$education)
# plot(newtest$occupation)
# plot(newtest$native.country)



#Discretize training set
# discretetrainage <- discretize(newtrain$age, method = "interval", categories = 10)
# discretetrainfnlwgt <- discretize(newtrain$fnlwgt, method = "interval", categories = 10)
# discretetrainedunum <- discretize(newtrain$education.num, method = "interval", categories = 10)
# discretetraingain <- discretize(newtrain$capital.gain, method = "interval", categories = 10)
# discretetrainloss <- discretize(newtrain$capital.loss, method = "interval", categories = 10)
# discretetrainhours <- discretize(newtrain$hours.per.week, method = "interval", categories = 10)



#Binning
countrydis <- function(vector){
  len <- length(vector)
  for(i in 1:len){
      if(vector[i] == "United-States"){
        vector[i] <- vector[i]
      }else if(vector[i] == "Mexico"){
        vector[i] <- vector[i]
      }else if(vector[i] == "Philippines"){
        vector[i] <- vector[i]
      }else{
        vector[i] <- "other_countries"
      }
  }
  return(vector)
}

workdis <- function(vector){
  len <- length(vector)
  for(i in 1:len){
    if(vector[i] == "Federal-gov"){
      vector[i] <- vector[i]
    }else if(vector[i] == "Local-gov"){
      vector[i] <- vector[i]
    }else if(vector[i] == "Private"){
      vector[i] <- vector[i]
    }else if(vector[i] == "Self-emp-inc"){
      vector[i] <- vector[i]
    }else if(vector[i] == "Self-emp-not-inc"){
      vector[i] <- vector[i]
    }else if(vector[i] == "State-gov"){
      vector[i] <- vector[i]
    }else{
      vector[i] <- "No-gain"
    }
  }
  return(vector)
}

#discretetraincountry <- as.factor(countrydis(as.character(newtrain$native.country)))



#Discretize testing set
# discretetestage <- discretize(newtest$age, method = "interval", categories = 10)
# discretetestfnlwgt <- discretize(newtest$fnlwgt, method = "interval", categories = 10)
# discretetestedunum <- discretize(newtest$education.num, method = "interval", categories = 10)
# discretetestgain <- discretize(newtest$capital.gain, method = "interval", categories = 10)
# discretetestloss <- discretize(newtest$capital.loss, method = "interval", categories = 10)
# discretetesthours <- discretize(newtest$hours.per.week, method = "interval", categories = 10)
# discretetestcountry <- as.factor(countrydis(as.character(newtest$native.country)))
#Combine training and testing to make the same intervals for discretizing



newtrain$type <- "train"
newtest$type <- "test"
combined <- rbind(newtrain, newtest)



# discreteage <- discretize(combined$age, method = "interval", categories = 10)
# discretefnlwgt <- discretize(combined$fnlwgt, method = "interval", categories = 10)
# discreteedunum <- discretize(combined$education.num, method = "interval", categories = 10)
# discretegain <- discretize(combined$capital.gain, method = "interval", categories = 7) #not enough data
# discreteloss <- discretize(combined$capital.loss, method = "interval", categories = 7) #not enough data
# discretehours <- discretize(combined$hours.per.week, method = "interval", categories = 10)
discretecountry <- as.factor(countrydis(as.character(combined$native.country)))
discreteworkclass <- as.factor(workdis(as.character(combined$workclass)))



# combined$age <- discreteage
# combined$fnlwgt <- discretefnlwgt
# combined$education.num <- discreteedunum
# combined$capital.gain <- discretegain
# combined$capital.loss <- discreteloss
# combined$hours.per.week <- discretehours
combined$native.country <- discretecountry
combined$workclass <- discreteworkclass



dim(combined)
newtrain2 <- combined[1:sum(combined$type == "train"), -14]
newtest2 <- combined[(sum(combined$type == "train") + 1):nrow(combined), -14]
dim(newtrain2)
dim(newtest2)



#plots
par(mfrow = c(2, 2)) #set how many plots on the palete.

for(i in 1:12){
  plot(newtrain2[, i], newtrain2[, 13])
}

for(i in 1:12){
  plot(newtest2[, i], newtest2[, 13])
}



#Assignining discretized variables
# newtrain2 <- newtrain
# newtest2 <- newtest
# dim(newtrain2)
# dim(newtest2)
# 
# newtrain2$age <- discretetrainage
# newtrain2$fnlwgt <- discretetrainfnlwgt
# newtrain2$education.num <- discretetrainedunum
# newtrain2$capital.gain <- discretetraingain
# newtrain2$capital.loss <- discretetrainloss
# newtrain2$hours.per.week <- discretetrainhours
# newtrain2$native.country <- discretetraincountry
# 
# newtest2$age <- discretetestage
# newtest2$fnlwgt <- discretetestfnlwgt
# newtest2$education.num <- discretetestedunum
# newtest2$capital.gain <- discretetestgain
# newtest2$capital.loss <- discretetestloss
# newtest2$hours.per.week <- discretetesthours
# newtest2$native.country <- discretetestcountry



#Dummify training set
dumtrainwork <- dummy(newtrain2$workclass)
dumtrainmarry <- dummy(newtrain2$marital.status)
dumtrainoccu <- dummy(newtrain2$occupation)
dumtrainrelation <- dummy(newtrain2$relationship)
dumtrainrace <- dummy(newtrain2$race)
dumtrainsex <- dummy(newtrain2$sex)
dumtraincountry <- dummy(newtrain2$native.country)



#Dummify testing set
dumtestwork <- dummy(newtest2$workclass)
dumtestmarry <- dummy(newtest2$marital.status)
dumtestoccu <- dummy(newtest2$occupation)
dumtestrelation <- dummy(newtest2$relationship)
dumtestrace <- dummy(newtest2$race)
dumtestsex <- dummy(newtest2$sex)
dumtestcountry <- dummy(newtest2$native.country)



#Take out columns
newtrain2 <- newtrain2[, -c(2, 4, 5, 6, 7, 8, 12)]
newtest2 <- newtest2[, -c(2, 4, 5, 6, 7, 8, 12)]



#Assigning dummified variables
newtrain2 <- cbind(newtrain2, dumtrainwork, dumtrainmarry, dumtrainoccu,
                   dumtrainrelation, dumtrainrace, dumtrainsex, dumtraincountry)
newtrain2[, 45] <- newtrain2$income
newtrain2 <- newtrain2[, -6]
names(newtrain2)[44]<- "income"
dim(newtrain2)

newtest2 <- cbind(newtest2, dumtestwork, dumtestmarry, dumtestoccu,
                   dumtestrelation, dumtestrace, dumtestsex, dumtestcountry)
newtest2[, 45] <- newtest2$income
newtest2 <- newtest2[, -6]
names(newtest2)[44]<- "income"
dim(newtest2)



#fixing...
newtrain2$income <- droplevels(newtrain2$income, c("<=50K.", ">50K."))
newtest2$income <- droplevels(newtest2$income, c("<=50K", ">50K"))

newtest2$income <- as.character(newtest2$income)
newtest2$income <- substr(newtest2$income, 1, nchar(newtest2$income) - 1)
newtest2$income <- as.factor(newtest2$income)



dim(newtrain2)
dim(newtest2)
str(newtrain2)
str(newtest2)

## ---- echo = F, eval = F-------------------------------------------------
## write.csv(newtest2, file = "../data/cleandata/newtest2.csv", col.names = T, row.names = F)
## write.csv(newtrain2, file = "../data/cleandata/newtrain2.csv", col.names = T, row.names = F)

## ---- echo = F, eval = F-------------------------------------------------
## newtrain2 <- read.csv("../data/cleandata/newtrain2.csv", header = T)
## newtest2 <- read.csv("../data/cleandata/newtest2.csv", header = T)
## str(newtrain2)
## str(newtest2)

