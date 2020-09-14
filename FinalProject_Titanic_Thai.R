## Working directory
setwd("...")
## Library used
#install.packages("compare") #comment it out if you had this library
library(compare)
library(tidyverse)
library(rpart)
library(class)
library(e1071)
library(party)
## Import file
train <- read.csv("train.csv")
test <- read.csv("test.csv")
s.test <- read.csv("gender_submission.csv")
## Clean data (missing data will be deleted)
# Merge data
test <- cbind(test,s.test$Survived)
# Rename
colnames(test)[colnames(test)=="s.test$Survived"] <- "Survived"
# Clean/Delete Missing Data (mostly age).
a.train <- na.omit(train)
train.survived <- a.train$Survived
a.test <- na.omit(test)
# This is used for mostly age variable analysis.
# If the Age is valid, Age will be accounted, if not, Age will not be accounted
# Data without Survived
na.train <- within(a.train, rm(Survived))
na.test <- within(a.test, rm(Survived))
nonvars <- c("PassengerId","Name","Ticket","Embarked","Cabin")
a.train <- a.train[,!(names(a.train) %in% nonvars)]
a.test <- a.test[,!(names(a.test) %in% nonvars)]
str(a.train)
str(a.test)
a.train$Sex = as.numeric(a.train$Sex)
a.test$Sex = as.numeric(a.test$Sex)
cor(a.train)
cor(a.test)
#######################################################################
## Logistic Regression
TitanicLog <- glm(Survived ~ . - Parch - Fare, data = a.train, family = binomial)
AgeLog <- glm(Survived ~ Age, data = a.train, family = binomial)
# Always predict die
c.survival <- 0
c.die <- 0
for (i in 1:nrow(a.train)){
  if (a.train[i,1] == 0){
    c.die = c.die + 1
  } else{
    c.survival = c.survival + 1
  }
}
baseline.train = c.die/nrow(a.train)
cat("Baseline prediction is: ", baseline.train, "\n")
predict.lg <- predict(TitanicLog, newdata = a.test, type = "response")
t.lg <- table(a.test$Survived, predict.lg >= 0.5)
print("confusion matrix of logistics Regression")
print(t.lg)
accuracy.lg <- (t.lg[1,1] + t.lg[2,2])/sum(t.lg)
die.lg <- t.lg[1,1]/sum(t.lg[1,])
survival.lg <- t.lg[2,2]/sum(t.lg[2,])
cat("Accuracy of Survival Prediction by logistics regression: ", survival.lg, "\n")
cat("Accuracy of Non-Survival Prediction by logistics regression: ", die.lg, "\n")
cat("Overall Accuracy Prediction of logistics regression: ", accuracy.lg, "\n")
#######################################################################
## Decision Tree
fit_train <- rpart(a.train$Survived~. ,method="class", data=a.train)
name1 = "decisionTree_Titanic"
num = 1
ext = ".pdf"
name2 = paste(name1, num, ext, sep = '')
# plot and save the pdf file
pdf(name2)
plot(fit_train, uniform = T,main = "Decision Tree for Titanic Survival")
text(fit_train, use.n = T, all = T, cex = 0.6)
dev.off()
#predict
pred.dt <- predict(fit_train,a.test[,-7], type = "class")
# table
t.dt <- table(a.test$Survived,pred.dt)
print("confusion matrix of Decision Tree")
print(t.dt)
accuracy.dt <- (t.dt[1,1] + t.dt[2,2])/sum(t.dt)
die.dt <- t.dt[1,1]/sum(t.dt[1,])
survival.dt <- t.dt[2,2]/sum(t.dt[2,])
cat("Accuracy of Survival Prediction by Decision Tree: ", survival.dt, "\n")
cat("Accuracy of Non-Survival Prediction by Decision Tree: ", die.dt, "\n")
cat("Overall Accuracy Prediction of Decision Tree: ", accuracy.dt, "\n")
#######################################################################
## kNN
#accuracy
acc.knn <- numeric()
#error
err.knn <- numeric()
n <- integer()
for (i in 1:100) { #testing from k = 1 to k = 100
  n[i]=i
  knear <- knn(a.train[,-1],a.test[,-7],a.train$Survived,k=i)
  #confusion matrix
  #unlist the winedata_test to make sure it work
  t.knn.check <- table(unlist(a.test[,7]),knear)
  #get diagonal elements of matrix
  td <- diag(t.knn.check)
  #sum all the elements of the matrix t which gives 
  #the examples tested
  sumt <- sum(t.knn.check)
  #sum the diagonal to see how many correct predictions
  sumtd <- sum(td)
  #calculate accuracy
  acc.knn[i] <- sumtd/sumt
  #calculate error
  err.knn[i] <- (sumt-sumtd)/sumt
}
resultsknn <- cbind(n,acc.knn,err.knn)
colnames(resultsknn) <- c("k","Accuracy","Error")
pdf("kNNaccuracy.pdf")
plot(1:100,acc.knn,xlab="k",ylab="Accuracy")
dev.off()
best.knn.index <- which.max(acc.knn)
knear.best <- knn(a.train[,-1],a.test[,-7],a.train$Survived,k=best.knn.index)
## Run KNN with best k
t.knn <- table(unlist(a.test[,7]),knear.best)
print("confusion matrix of kNN")
print(t.knn)
accuracy.knn <- (t.knn[1,1] + t.knn[2,2])/sum(t.knn)
die.knn <- t.knn[1,1]/sum(t.knn[1,])
survival.knn <- t.knn[2,2]/sum(t.knn[2,])
cat("Accuracy of Survival Prediction by kNN: ", survival.knn, "\n")
cat("Accuracy of Non-Survival Prediction by kNN: ", die.knn, "\n")
cat("Overall Accuracy Prediction of kNN: ", accuracy.knn, "\n")
#######################################################################
## Naive Bayes
NB.res <- matrix()
NB.survival <- 0
NB.die <- 0
NB.Model <- naiveBayes(Survived ~., data=a.train)
NB.Pred <- predict(NB.Model, a.test[,-7],type = "raw")
for (i in 1:nrow(NB.Pred)){
  if(NB.Pred[i,1] > NB.Pred[i,2]){
    NB.die = NB.die + 1
    NB.res[i] = 0
  } else {
    NB.survival = NB.survival + 1
    NB.res[i] = 1
  }
}
#Confusion matrix of NB
t.NB <- table(NB.res, a.test$Survived)
print("confusion matrix of NB")
print(t.NB)
accuracy.NB <- (t.NB[1,1] + t.NB[2,2])/sum(t.NB)
die.NB <- t.NB[1,1]/sum(t.NB[1,])
survival.NB <- t.NB[2,2]/sum(t.NB[2,])
cat("Accuracy of Survival Prediction by NB: ", survival.NB, "\n")
cat("Accuracy of Non-Survival Prediction by NB: ", die.NB, "\n")
cat("Overall Accuracy Prediction of NB: ", accuracy.NB, "\n")
#######################################################################
## Random Forest
set.seed(123)
rf.fit <- cforest(as.factor(Survived)~., data = a.train, controls=cforest_unbiased(ntree=100, mtry=3))
rf.pred <- predict(rf.fit, newdata = a.test[,-7], OOB=TRUE, type = "response")
t.rf <- table(rf.pred, a.test$Survived)
print("confusion matrix of Random Forest")
print(t.rf)
accuracy.rf <- (t.rf[1,1] + t.rf[2,2])/sum(t.rf)
die.rf <- t.rf[1,1]/sum(t.rf[1,])
survival.rf <- t.rf[2,2]/sum(t.rf[2,])
cat("Accuracy of Survival Prediction by Random Forest: ", survival.rf, "\n")
cat("Accuracy of Non-Survival Prediction by Random Forest: ", die.rf, "\n")
cat("Overall Accuracy Prediction of Random Forest: ", accuracy.rf, "\n")
#######################################################################
## Compare the Result
# Survival
acc.survival <- cbind(survival.lg, survival.dt, survival.knn, survival.NB, survival.rf)
colnames(acc.survival) <- c("LogR", "DecisionTree", "kNN", "NaiveBayes", "RandomForest")
acc.sur.best <- max(acc.survival)
acc.sur.best.name <- which.max(acc.survival)
if (acc.sur.best.name == 1) {
  cat("Best Accuracy of Survival Prediction is LogR at ", acc.sur.best*100, "% \n")
} else if (acc.sur.best.name == 2) {
  cat("Best Accuracy of Survival Prediction is DecisionTree at ", acc.sur.best*100, "% \n")
} else if (acc.sur.best.name == 3) {
  cat("Best Accuracy of Survival Prediction is kNN at ", acc.sur.best*100, "% \n")
} else if (acc.sur.best.name == 4) {
  cat("Best Accuracy of Survival Prediction is NaiveBayes at ", acc.sur.best*100, "% \n")
} else if (acc.sur.best.name == 5) {
  cat("Best Accuracy of Survival Prediction is RandomForest at ", acc.sur.best*100, "% \n")
}
# Non-Survival
acc.die <- cbind(die.lg, die.dt, die.knn, die.NB, die.rf)
colnames(acc.survival) <- c("LogR", "DecisionTree", "kNN", "NaiveBayes", "RandomForest")
acc.die.best <- max(acc.die)
acc.die.best.name <- which.max(acc.die)
if (acc.die.best.name == 1) {
  cat("Best Accuracy of Non-Survival Prediction is LogR at ", acc.die.best*100, "% \n")
} else if (acc.die.best.name == 2) {
  cat("Best Accuracy of Non-Survival Prediction is DecisionTree at ", acc.die.best*100, "% \n")
} else if (acc.die.best.name == 3) {
  cat("Best Accuracy of Non-Survival Prediction is kNN at ", acc.die.best*100, "% \n")
} else if (acc.die.best.name == 4) {
  cat("Best Accuracy of Non-Survival Prediction is NaiveBayes at ", acc.die.best*100, "% \n")
} else if (acc.die.best.name == 5) {
  cat("Best Accuracy of Non-Survival Prediction is RandomForest at ", acc.die.best*100, "% \n")
}
# Overall
acc.over <- cbind(accuracy.lg, accuracy.dt, accuracy.dt, accuracy.NB, accuracy.rf)
colnames(acc.survival) <- c("LogR", "DecisionTree", "kNN", "NaiveBayes", "RandomForest")
acc.over.best <- max(acc.over)
acc.over.best.name <- which.max(acc.over)
if (acc.over.best.name == 1) {
  cat("Best Accuracy of Overall Prediction is LogR at ", acc.over.best*100, "% \n")
} else if (acc.over.best.name == 2) {
  cat("Best Accuracy of Overall Prediction is DecisionTree at ", acc.over.best*100, "% \n")
} else if (acc.over.best.name == 3) {
  cat("Best Accuracy of Overall Prediction is kNN at ", acc.over.best*100, "% \n")
} else if (acc.over.best.name == 4) {
  cat("Best Accuracy of Overall Prediction is NaiveBayes at ", acc.over.best*100, "% \n")
} else if (acc.over.best.name == 5) {
  cat("Best Accuracy of Overall Prediction is RandomForest at ", acc.over.best*100, "% \n")
}
#######################################################################
## Graph
logr <- rbind(survival.lg, die.lg, accuracy.lg)
dt <- rbind(survival.dt, die.dt, accuracy.dt)
knn.r <- rbind(survival.knn, die.knn, accuracy.knn)
nb <- rbind(survival.NB, die.NB, accuracy.NB)
rf <- rbind(survival.rf, die.rf, accuracy.rf)
# Grouped Bar Plot
titanic.pred <- cbind(logr,dt,knn.r,nb,rf)
titanic.pred <- as.table(titanic.pred)
colnames(titanic.pred) <- c("LogR", "DT", "kNN", "NaiveBayes", "RF")
rownames(titanic.pred) <- c("Survival", "Non-Survival", "Overall")
pdf("Algorithm Comparison of Titanic Survival Prediction.pdf")
barplot(titanic.pred, main="Algorithms Comparison for \n Titanic Survival Model",
        xlab="Algorithms", ylab = "Accuracy", col=c("darkblue","gold1","darkgreen"),
        beside=TRUE)
legend(8.5, 0.9,
       legend = rownames(titanic.pred),
       col=c("darkblue","gold1","darkgreen"),
       cex = 0.8,
       pch= 15)
dev.off()

