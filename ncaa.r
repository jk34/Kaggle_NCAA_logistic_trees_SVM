#Installing package
install.packages('stringr')
library('stringr')

#RESPONSE VARIABLES
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
 train_data_frame <- data.frame("Matchup" = team, "Win" = result)
 head(train_data_frame)
 
 
 
 #PREDICTORS

#Selecting and sorting the playoff teamIDs least to greatest for season A
tourneySeeds<-read.csv("tourney_seeds.csv") 
regSeason<-read.csv("regular_season_compact_results.csv")
playoff_teams <- sort(tourneySeeds$team[which(tourneySeeds$season == "2013")])
 
#Selecting the seeds for season A
playoff_seeds <- tourneySeeds[which(tourneySeeds$season == "2013"), ]
 
#Selecting the regular season statistics for season A
season <- regSeason[which(regSeason$season == "2013"), ]
 
#Wins by team
win_freq_table <- as.data.frame(table(season$wteam))
wins_by_team <- win_freq_table[win_freq_table$Var1 %in% playoff_teams, ]
# see http://www.r-bloggers.com/how-to-get-the-frequency-table-of-a-categorical-variable-as-a-data-frame-in-r/
#for how freq is number of occurrences for each var1 in the win_freq_table
 
 
#Losses by team
loss_freq_table <- as.data.frame(table(season$lteam))
loss_by_team <- loss_freq_table[loss_freq_table$Var1 %in% playoff_teams, ]

#to add Wichita State, since they were undefeated in reg. season
#below WORKS for 2014
#loss_by_team$Var1 <- as.character(loss_by_team$Var1)
#loss_by_team<-rbind(loss_by_team, data.frame(Var1=1455, Freq=0))
#loss_by_team <- loss_by_team[order(loss_by_team$Var1),]
#Var1 Freq
#681 1455    0
 
#Total Win Percentage
gamesplayed <- as.vector(wins_by_team$Freq + loss_by_team$Freq)
total_winpct <- round(wins_by_team$Freq / gamesplayed, digits = 3)
total_winpct_by_team <- as.data.frame(cbind(as.vector(loss_by_team$Var1), total_winpct))
#total_winpct_by_team[which(total_winpct_by_team$V1=="1455"), ]
#V1 total_winpct
#68 1455            1  - since Wichita St was undefeated

colnames(total_winpct_by_team) <- c("Var1", "Freq")
 
#Num of wins in last 6 games
wins_last_six_games_by_team <- data.frame()
for(i in playoff_teams) {
  games <- season[which(season$wteam == i | season$lteam == i), ]
  numwins <- sum(tail(games$wteam) == i)
  put <- c(i, numwins)
  wins_last_six_games_by_team <- rbind(wins_last_six_games_by_team, put)
}
colnames(wins_last_six_games_by_team) <- c("Var1", "Freq")
 
#Seed in tournament
pattern <- "[A-Z]([0-9][0-9])"
team_seeds <- as.data.frame(str_match(playoff_seeds$seed, pattern))
seeds <- as.numeric(team_seeds$V2)
playoff_seeds$seed  <- seeds
seed_col <- vector()
BPI_col<-vector()
for(i in playoff_teams) {
  val <- match(i, playoff_seeds$team)
  seed_col <- c(seed_col, playoff_seeds$seed[val])
  BPI_col <- c(BPI_col, playoff_seeds$BPI[val])
}
#team_seed <- data.frame("Var1" = playoff_teams, "Freq" =seed_col)
team_seed <- data.frame(playoff_teams, seed_col, BPI_col)
#team_seed<-data.frame()
#team_seed <- cbind(playoff_teams, seed_col,BPI_col) 

#Combining columns together
#team_metrics <- data.frame()
team_metrics <- data.frame(total_winpct_by_team, wins_last_six_games_by_team$Freq, team_seed$seed_col, team_seed$BPI_col)#team_seed$Freq)
#team_metrics <- cbind(total_winpct_by_team, wins_last_six_games_by_team$Freq, team_seed$seed_col, team_seed$BPI_col)#team_seed$Freq)
colnames(team_metrics) <- c("TEAMID", "A_TWPCT","A_WST6", "A_SEED","A_BPI")

#to use training data in blog_utility.r:
source('blog_utility.R')
trainData <- data.frame()
for(i in 2008:2012) {
  x<-train_frame_model(i)
  trainData <- rbind(trainData, x)
}


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





#Neural Networks

#m <- model.matrix( 
#  ~Win + A_WST6 + A_SEED + A_TWPCT + A_BPI + B_WST6 + B_SEED + B_TWPCT + B_BPI, data = trainData
#)
#don't use model.matrix since these are not categorical variables
ncaanet <- neuralnet(Win ~ A_WST6 + A_SEED + A_TWPCT + A_BPI + B_WST6 + B_SEED + B_TWPCT + B_BPI, data=trainData, hidden = 4, lifesign = "minimal", 
                      linear.output = FALSE, threshold = 0.1)
ncaanet.results <- compute(ncaanet, testData[,5:12])
#columns 5-12 of testData are A_WST6,A_SEED,A_TWPCT,A_BPI and similarly for B

subfile <- data.frame(id = testData$Matchup, pred = ncaanet.results$net.result)
write.csv(subfile, file = "neuralnet_model.csv", row.names = FALSE)

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
#.671 for neural net if using 2008-2012 as training and 2013 as test and TWPCT, BPI, SEED and A_wST6, 
#hidden=4 and threshold=.1

#2.179 if using TWPCT, BPI, SEED, testDataN<-testData[,-c(1:4,6,10)]
#.6706 if using BPI, SEED

#.688 for logistic regression if using 2008-2012 as training and 2013 as test and BPI, SEED and A_wST6, 
#.688 ALSO if just BPI, A_WST6 (no SEED)
#.656 if ONLY BPI