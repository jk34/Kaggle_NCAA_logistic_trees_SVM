#Installing package
library('stringr')


#to use training data in blog_utility.r:
source('blog_utility.R')
trainData <- data.frame()
for(i in 2008:2012) {
  x<-train_frame_model(i)
  trainData <- rbind(trainData, x)
}

trainData$Win<-factor(trainData$Win)

#if errors for 2014, its because WICHITA ST WAS UNDEFEATED
#test data
testData <- data.frame()
for (i in 2013:2013) {
  testData <- rbind(testData, test_frame_model(i))
}
#this works, since blog_utility.r is updated. 
#If it doesn't, need to re-click "Source"



#train the training set
library('rpart')
train_rpart <- rpart(Win ~ A_WST6 + A_SEED + B_WST6 + B_SEED, data = trainData, 
                     method = "class")



predictions_rpart <- predict(train_rpart, newdata = testData, type = "prob")
predictions <- predictions_rpart[, 1]
subfile <- data.frame(id = testData$Matchup, pred = predictions)
write.csv(subfile, file = "tree_model.csv", row.names = FALSE)

#to see how well model matched actual tournament results
tourneyRes<-read.csv("tourney_compact_results.csv") 
season_matches <- tourneyRes[which(tourneyRes$season == "2013"), ]
team <- vector()
result <- vector()
for(i in c(1:nrow(season_matches))) {
  row <- season_matches[i, ]
  if(row$wteam < row$lteam) {
    vector <- paste("2013","_",row$wteam,"_", row$lteam, sep ="")
    team <- c(team, vector)
    result <- c(result, 1)
  } else {
    oth <- paste("2013", "_", row$lteam, "_", row$wteam, sep ="")
    team <- c(team, oth)
    result <- c(result, 0)
  }
}
actual_data_frame <- data.frame("Matchup" = team, "Win" = result)
head(actual_data_frame)

#check how accurate model was
final_rpart<-merge(subfile,actual_data_frame,by.x="id",by.y="Matchup")
#remove NA
final_rpart<-final_rpart[complete.cases(final_rpart),]
#get results of accuracy
sum<-0
for(i in c(1:nrow(final_rpart))) {
  row <- final_rpart[i, ]
  if(row$Win==1)
    #sum <- row$pred*row$Win + sum
    sum <- log(row$pred)*row$Win + sum
  else
    sum <- log(1-row$pred) + sum
  #sum <- 1 - row$pred + sum
}
logloss<- (-sum)/nrow(final_rpart)
#1.008 for rpart (goal it to minimize log-loss)
#1.0598 if using 2010-2012 as training and 2013 as test 
#(not 2014 because of undefeated Wichita St)



#logistic regression
glm.fit = glm(Win ~ A_WST6 + A_SEED + A_TWPCT + A_BPI + B_WST6 + B_SEED + B_TWPCT + B_BPI, data = trainData, family=binomial)
#TWPCT had high-pvalue so its insignificant
glm.fit = glm(Win ~ A_WST6 + A_SEED + A_BPI + B_WST6 + B_SEED + B_BPI, data = trainData, family=binomial)
predictions_logit <- predict(glm.fit, testData, type = "response")
#type="response" tells R to output probs of the form P(Y=1|X)
#DON'T need
#predictions <- predictions_logit[, 1]
#because predictions_rpart gives 2 columns
#but predictions_logit is only ONE column
predictions <- predictions_logit
subfile <- data.frame(id = testData$Matchup, pred = predictions)
write.csv(subfile, file = "logistic_model.csv", row.names = FALSE)

#to see how well model matched actual tournament results
tourneyRes<-read.csv("tourney_compact_results.csv") 
season_matches <- tourneyRes[which(tourneyRes$season == "2013"), ]
team <- vector()
result <- vector()
for(i in c(1:nrow(season_matches))) {
  row <- season_matches[i, ]
  if(row$wteam < row$lteam) {
    vector <- paste("2013","_",row$wteam,"_", row$lteam, sep ="")
    team <- c(team, vector)
    result <- c(result, 1)
  } else {
    oth <- paste("2013", "_", row$lteam, "_", row$wteam, sep ="")
    team <- c(team, oth)
    result <- c(result, 0)
  }
}
actual_data_frame <- data.frame("Matchup" = team, "Win" = result)
head(actual_data_frame)

#check how accurate model was
final_logit<-merge(subfile,actual_data_frame,by.x="id",by.y="Matchup")
#remove NA
final_logit<-final_logit[complete.cases(final_logit),]
#get results of accuracy
sum<-0
for(i in c(1:nrow(final_logit))) {
  row <- final_logit[i, ]
  if(row$Win==1)
    #sum <- row$pred*row$Win + sum
    sum <- log(row$pred)*row$Win + sum
  else
    sum <- log(1-row$pred) + sum
    #sum <- 1 - row$pred + sum
}
logloss_logit<- (-sum)/nrow(final_logit)
#.5935 for logistic regression (it was 1.008 for rpart, so logistic regression is better here)
# for 2014 as test set?

#.692 if using 2010-2012 as training and 2013 as test (1.06 for rpart)
#.684 if using BPI, SEED and A_wST6, .682 if just BPI, A_WST6 (no SEED since it has much higher p-value than A_WST6 and A_BPI)
#.64 if ONLY BPI

#.688 if using 2008-2012 as training and 2013 as test and BPI, SEED and A_wST6, 
#.688 ALSO if just BPI, A_WST6 (no SEED)
#.656 if ONLY BPI


#SVM
library('e1071')
#default SVM assumes regression
#but outcome of win/loss is classification
#so SVM in R requires converting data frame to "factor" for classification
#see p.359 of "Introduction to Statistical Learning" 

trainDataSVM <- trainData
trainDataSVM$Win <- factor(trainDataSVM$Win)
#svmfit =svm (Win ~ A_WST6 + A_SEED + B_WST6 + B_SEED , data = trainDataSVM , kernel ="linear", cost =10,scale =FALSE, probability=TRUE )
svmfit =svm (Win ~ A_WST6 + A_SEED + B_WST6 + B_SEED , data = trainDataSVM , kernel ="polynomial", degree=3,cost =10,scale =FALSE, probability=TRUE )
#use CV with a range of cost values
set.seed (1)
tune.out = tune(svm, Win ~ A_WST6 + A_SEED + B_WST6 + B_SEED, data=trainDataSVM , kernel ="polynomial",ranges =list (cost=c(0.1, 1 ,5 ,10 ,50), degree=c(1,2,3,4,5)))
summary(tune.out)
#cost of .1 and .01 give lowest CV error for LINEAR KERNEL
#cost 1,5,10 and degree 3 give lowest CV error (about .29) for POLYNOMIAL KERNEL

#access the best model from tune()
bestmod =tune.out$best.model
summary(bestmod)

#doesn'twork: 
testDataSVM<-testData
testDataSVM$Win <-factor(testDataSVM$Win)
predictions_SVM <- predict(bestmod, testDataSVM, type = "response",probability=TRUE)

#type="response" tells R to output probs of the form P(Y=1|X)

#possibly getting only "NA" values in predictions because you need to convert
#outputs of SVM to probabilities. 
predictions <- predictions_SVM#[, 1]
subfile <- data.frame(id = testData$Matchup, pred = predictions)
write.csv(subfile, file = "SVM_model.csv", row.names = FALSE)

#to see how well model matched actual tournament results
tourneyRes<-read.csv("tourney_compact_results.csv") 
season_matches <- tourneyRes[which(tourneyRes$season == "2013"), ]
team <- vector()
result <- vector()
for(i in c(1:nrow(season_matches))) {
  row <- season_matches[i, ]
  if(row$wteam < row$lteam) {
    vector <- paste("2013","_",row$wteam,"_", row$lteam, sep ="")
    team <- c(team, vector)
    result <- c(result, 1)
  } else {
    oth <- paste("2013", "_", row$lteam, "_", row$wteam, sep ="")
    team <- c(team, oth)
    result <- c(result, 0)
  }
}
actual_data_frame <- data.frame("Matchup" = team, "Win" = result)
head(actual_data_frame)

#check how accurate model was
final_logit<-merge(subfile,actual_data_frame,by.x="id",by.y="Matchup")
#remove NA
final_logit<-final_logit[complete.cases(final_logit),]
#get results of accuracy
sum<-0
for(i in c(1:nrow(final_logit))) {
  row <- final_logit[i, ]
  if(row$Win==1)
    #sum <- row$pred*row$Win + sum
    sum <- log(row$pred)*row$Win + sum
  else
    sum <- log(1-row$pred) + sum
  #sum <- 1 - row$pred + sum
}
logloss_SVM<- (-sum)/nrow(final_logit)
#nothing yet for SVM


#Random Forest
library('randomForest')
binaryTrain=ifelse(trainData$Win==1,"Win","Lose")
binaryTest=ifelse(testData$Win==1,"Win","Lose")
binTrain=data.frame(trainData,binaryTrain)
binTest=data.frame(testData,binaryTest)
formula = as.formula(binaryTrain ~ A_WST6 + A_SEED  + A_BPI + B_WST6 + B_SEED  + B_BPI)
rf = randomForest(formula, data=binTrain, mtry=2, ntree=5000, importance=TRUE)
predictions_RF <- predict(rf, binTest, type = "prob")
predictions <- data.frame(predictions_RF)


subfile <- data.frame(id = testData$Matchup, pred = predictions)
subfile<-subfile[,names(subfile)[-c(2)]] 
#only need prob of win, while the 2nd column was prob of loss
write.csv(subfile, file = "RF_model.csv", row.names = FALSE)

#to see how well model matched actual tournament results
tourneyRes<-read.csv("tourney_compact_results.csv") 
season_matches <- tourneyRes[which(tourneyRes$season == "2013"), ]
team <- vector()
result <- vector()
for(i in c(1:nrow(season_matches))) {
  row <- season_matches[i, ]
  if(row$wteam < row$lteam) {
    vector <- paste("2013","_",row$wteam,"_", row$lteam, sep ="")
    team <- c(team, vector)
    result <- c(result, 1)
  } else {
    oth <- paste("2013", "_", row$lteam, "_", row$wteam, sep ="")
    team <- c(team, oth)
    result <- c(result, 0)
  }
}
actual_data_frame <- data.frame("Matchup" = team, "Win" = result)
head(actual_data_frame)

#check how accurate model was
final_logit<-merge(subfile,actual_data_frame,by.x="id",by.y="Matchup")
#remove NA
final_logit<-final_logit[complete.cases(final_logit),]
#get results of accuracy
sum<-0
for(i in c(1:nrow(final_logit))) {
  row <- final_logit[i, ]
  if(row$Win==1)
    #sum <- row$pred*row$Win + sum
    sum <- log(row$pred)*row$Win + sum
  else
    sum <- log(1-row$pred) + sum
  #sum <- 1 - row$pred + sum
}
logloss_RF<- (-sum)/nrow(final_logit)
#.651 if using 2008-2012 as training and 2013 as test and BPI, SEED and A_wST6,mtry=2, ntree=5000
#.650 if using ntree=1000, .657 for ntree=500, .648 if ntree=10000

#.688 if using 2008-2012 as training and 2013 as test and BPI, SEED and A_wST6, 
#.688 ALSO if just BPI, A_WST6 (no SEED)
#.656 if ONLY BPI


#KNN
binaryTrain=ifelse(trainData$Win==1,"Win","Lose")
binaryTest=ifelse(testData$Win==1,"Win","Lose")
binTrain=data.frame(trainData,binaryTrain)
binTest=data.frame(testData,binaryTest)
c1=as.factor(binTrain$binaryTrain)
binTrain1<-binTrain[,names(binTrain)[-c(1,2,3,4,13)]] 
binTest1<-binTest[,names(binTest)[-c(1,2,3,4,13)]] 
set.seed(1)
output <- knn(binTrain1, binTest1, c1, k = 320, prob=TRUE)

#predictions_KNN <- predict(output, binTest, type = "prob")
#predictions <- data.frame(output)
#predictions<-predictions[,names(predictions)[-c(2)]]

subfile <- data.frame(id = testData$Matchup, pred = attributes(output)$prob) #predictions)
write.csv(subfile, file = "KNN_model.csv", row.names = FALSE)

#to see how well model matched actual tournament results
tourneyRes<-read.csv("tourney_compact_results.csv") 
season_matches <- tourneyRes[which(tourneyRes$season == "2013"), ]
team <- vector()
result <- vector()
for(i in c(1:nrow(season_matches))) {
  row <- season_matches[i, ]
  if(row$wteam < row$lteam) {
    vector <- paste("2013","_",row$wteam,"_", row$lteam, sep ="")
    team <- c(team, vector)
    result <- c(result, 1)
  } else {
    oth <- paste("2013", "_", row$lteam, "_", row$wteam, sep ="")
    team <- c(team, oth)
    result <- c(result, 0)
  }
}
actual_data_frame <- data.frame("Matchup" = team, "Win" = result)
head(actual_data_frame)

#check how accurate model was
final_logit<-merge(subfile,actual_data_frame,by.x="id",by.y="Matchup")
#remove NA
final_logit<-final_logit[complete.cases(final_logit),]
#get results of accuracy
sum<-0
for(i in c(1:nrow(final_logit))) {
  row <- final_logit[i, ]
  if(row$Win==1)
    #sum <- row$pred*row$Win + sum
    sum <- log(row$pred)*row$Win + sum
  else
    sum <- log(1-row$pred) + sum
  #sum <- 1 - row$pred + sum
}
logloss_KNN<- (-sum)/nrow(final_logit)
#best value was .6947, for k=320 if using 2008-2012 as training and 2013 as test

#.651 if using 2008-2012 as training and 2013 as test and BPI, SEED and A_wST6,mtry=2, ntree=5000
#.650 if using ntree=1000, .657 for ntree=500, .648 if ntree=10000

#.688 if using 2008-2012 as training and 2013 as test and BPI, SEED and A_wST6, 
#.688 ALSO if just BPI, A_WST6 (no SEED)
#.656 if ONLY BPI



#LDA
ldafit <- lda(Win ~ A_WST6 + A_SEED + A_TWPCT + A_BPI + B_WST6 + B_SEED + B_TWPCT + B_BPI, data = trainData)
ldafit$means
#  A_WST6   A_SEED   A_TWPCT    A_BPI   B_WST6   B_SEED   B_TWPCT    B_BPI
#0 4.410714 9.321429 0.7172202 60.11310 4.488095 5.267857 0.7794048 26.13095
#1 4.531646 5.025316 0.7911582 18.20886 4.411392 8.278481 0.7422722 49.58861

ldafit$scaling
#only shows LD1, that is, only 1 linear discriminant

predLDA <- predict(ldafit, testDF, type = "prob")
predLDA <- data.frame(predLDA)
subfile<-data.frame(id=testData$Matchup, pred=predLDA$posterior.1)

tourneyRes<-read.csv("tourney_compact_results.csv") 
season_matches <- tourneyRes[which(tourneyRes$season == "2013"), ]
team <- vector()
result <- vector()
for(i in c(1:nrow(season_matches))) {
  row <- season_matches[i, ]
  if(row$wteam < row$lteam) {
    vector <- paste("2013","_",row$wteam,"_", row$lteam, sep ="")
    team <- c(team, vector)
    result <- c(result, 1)
  } else {
    oth <- paste("2013", "_", row$lteam, "_", row$wteam, sep ="")
    team <- c(team, oth)
    result <- c(result, 0)
  }
}
actual_data_frame <- data.frame("Matchup" = team, "Win" = result)
head(actual_data_frame)

#check how accurate model was
final_logit<-merge(subfile,actual_data_frame,by.x="id",by.y="Matchup")
#remove NA
final_logit<-final_logit[complete.cases(final_logit),]
#get results of accuracy
sum<-0
for(i in c(1:nrow(final_logit))) {
  row <- final_logit[i, ]
  if(row$Win==1)
    #sum <- row$pred*row$Win + sum
    sum <- log(row$pred)*row$Win + sum
  else
    sum <- log(1-row$pred) + sum
  #sum <- 1 - row$pred + sum
}
logloss_LDA<- (-sum)/nrow(final_logit)
#.665 for LDA using all features and 2008-12 training 2013 test
#.6308 when using BPI, SEED, TWPCT

#for RF
#.651 if using 2008-2012 as training and 2013 as test and BPI, SEED and A_wST6,mtry=2, ntree=5000
#.650 if using ntree=1000, .657 for ntree=500, .648 if ntree=10000

#not as good as Logistic Regression logloss of .5935


#plot LDA
#this is not working
ldafit <- lda(Win ~ A_WST6 + A_SEED + A_TWPCT + A_BPI + B_WST6 + B_SEED + B_TWPCT + B_BPI, data = trainData)
prop.ldafit = ldafit$svd^2/sum(ldafit$svd^2)
pLDA <- predict(ldafit, trainData)
dataset=data.frame(win=trainData[,"Win"], lda=pLDA$x)
p1 <- ggplot(dataset) + geom_point(aes(ldafit$scaling.LD1, colour = win, shape = win), size = 2.5) +  labs(x = paste("LD1 (", percent(prop.ldafit[1]), ")", sep=""))
p1

