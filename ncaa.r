#RESPONSE VARIABLES
tourneyRes<-read.csv("tourney_compact_results.csv") 
season_matches <- tourneyRes[which(tourneyRes$season == "2014"), ]
team <- vector()
result <- vector()
for(i in c(1:nrow(season_matches))) {
     row <- season_matches[i, ]
     if(row$wteam < row$lteam) {
         vector <- paste("2014","_",row$wteam,"_", row$lteam, sep ="")
         team <- c(team, vector)
         result <- c(result, 1)
     } else {
         oth <- paste("2014", "_", row$lteam, "_", row$wteam, sep ="")
         team <- c(team, oth)
         result <- c(result, 0)
     }
 }
 train_data_frame <- data.frame("Matchup" = team, "Win" = result)
 head(train_data_frame)
 
 
 
 #PREDICTORS
 #Installing package
install.packages('stringr')
library('stringr')
#Selecting and sorting the playoff teamIDs least to greatest for season A
tourneySeeds<-read.csv("tourney_seeds.csv") 
regSeason<-read.csv("regular_season_compact_results.csv")
playoff_teams <- sort(tourneySeeds$team[which(tourneySeeds$season == "2014")])
 
#Selecting the seeds for season A
playoff_seeds <- tourneySeeds[which(tourneySeeds$season == "2014"), ]
 
#Selecting the regular season statistics for season A
season <- regSeason[which(regSeason$season == "2014"), ]
 
#Wins by team
win_freq_table <- as.data.frame(table(season$wteam))
wins_by_team <- win_freq_table[win_freq_table$Var1 %in% playoff_teams, ]
# see http://www.r-bloggers.com/how-to-get-the-frequency-table-of-a-categorical-variable-as-a-data-frame-in-r/
#for how freq is number of occurrences for each var1 in the win_freq_table
 
 
#Losses by team
loss_freq_table <- as.data.frame(table(season$lteam))
loss_by_team <- loss_freq_table[loss_freq_table$Var1 %in% playoff_teams, ]

#to add Wichita State, since they were undefeated in reg. season
loss_by_team$Var1 <- as.character(loss_by_team$Var1)
loss_by_team<-rbind(loss_by_team, data.frame(Var1=1455, Freq=0))
loss_by_team <- loss_by_team[order(loss_by_team$Var1),]
#loss_by_team[which(loss_by_team$Var1=="1455"), ]
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
for(i in playoff_teams) {
  val <- match(i, playoff_seeds$team)
  seed_col <- c(seed_col, playoff_seeds$seed[val])
}
team_seed <- data.frame("Var1" = playoff_teams, "Freq" =seed_col)
 
#Combining columns together
team_metrics <- data.frame()
team_metrics <- cbind(total_winpct_by_team, wins_last_six_games_by_team$Freq, team_seed$Freq)
colnames(team_metrics) <- c("TEAMID", "A_TWPCT","A_WST6", "A_SEED")

#to use training data in blog_utility.r:
source('blog_utility.R')
trainData <- data.frame()
for(i in 2010:2013) {
  x<-train_frame_model(i)
  trainData <- rbind(trainData, x)
}

#test data
testData <- data.frame()
for (i in 2014:2014) {
  testData <- rbind(testData, test_frame_model(i))
}
#this works, since blog_utility.r is updated. 
#If it doesn't, need to re-click "Source"


#train the training set
train_rpart <- rpart(Win ~ A_WST6 + A_SEED + B_WST6 + B_SEED, data = trainData, 
                     method = "class")



predictions_rpart <- predict(train_rpart, newdata = testData, type = "prob")
predictions <- predictions_rpart[, 1]
subfile <- data.frame(id = testData$Matchup, pred = predictions)
write.csv(subfile, file = "tree_model.csv", row.names = FALSE)

#to see how well model matched actual tournament results
tourneyRes<-read.csv("tourney_compact_results.csv") 
season_matches <- tourneyRes[which(tourneyRes$season == "2014"), ]
team <- vector()
result <- vector()
for(i in c(1:nrow(season_matches))) {
  row <- season_matches[i, ]
  if(row$wteam < row$lteam) {
    vector <- paste("2014","_",row$wteam,"_", row$lteam, sep ="")
    team <- c(team, vector)
    result <- c(result, 1)
  } else {
    oth <- paste("2014", "_", row$lteam, "_", row$wteam, sep ="")
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




#logistic regression
glm.fit = glm(Win ~ A_WST6 + A_SEED + A_TWPCT + B_WST6 + B_SEED + B_TWPCT, data = trainData, family=binomial)
#TWPCT had high-pvalue so its insignificant
glm.fit = glm(Win ~ A_WST6 + A_SEED + B_WST6 + B_SEED , data = trainData, family=binomial)
predictions_logit <- predict(glm.fit, testData, type = "response")
#type="response" tells R to output probs of the form P(Y=1|X)
predictions <- predictions_logit[, 1]
subfile <- data.frame(id = testData$Matchup, pred = predictions)
write.csv(subfile, file = "tree_model.csv", row.names = FALSE)

#to see how well model matched actual tournament results
tourneyRes<-read.csv("tourney_compact_results.csv") 
season_matches <- tourneyRes[which(tourneyRes$season == "2014"), ]
team <- vector()
result <- vector()
for(i in c(1:nrow(season_matches))) {
  row <- season_matches[i, ]
  if(row$wteam < row$lteam) {
    vector <- paste("2014","_",row$wteam,"_", row$lteam, sep ="")
    team <- c(team, vector)
    result <- c(result, 1)
  } else {
    oth <- paste("2014", "_", row$lteam, "_", row$wteam, sep ="")
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
logloss<- (-sum)/nrow(final_logit)
#.5935 for logistic regression (it was 1.008 for rpart, so logistic regression is better here)



#SVM
library('e1071')
svmfit =svm (Win ~ A_WST6 + A_SEED + B_WST6 + B_SEED , data = trainData , kernel ="linear", cost =10,scale =FALSE )
#use CV with a range of cost values
set.seed (1)
tune.out = tune(svm, Win ~ A_WST6 + A_SEED + B_WST6 + B_SEED, data=trainData , kernel ="linear",ranges =list (cost=c(0.001 , 0.01 , 0.1, 1 ,5 ,10 ,100) ))
summary(tune.out)
#cost of .1 and .01 give lowest CV error

#access the best model from tune()
bestmod =tune.out$best.model
summary(bestmod)

#doesn'twork: 
predictions_SVM <- predict(bestmod, testData, type = "response")

#type="response" tells R to output probs of the form P(Y=1|X)

#possibly getting only "NA" values in predictions because you need to convert
#outputs of SVM to probabilities. 
predictions <- predictions_SVM#[, 1]
subfile <- data.frame(id = testData$Matchup, pred = predictions)
write.csv(subfile, file = "SVM_model.csv", row.names = FALSE)

#to see how well model matched actual tournament results
tourneyRes<-read.csv("tourney_compact_results.csv") 
season_matches <- tourneyRes[which(tourneyRes$season == "2014"), ]
team <- vector()
result <- vector()
for(i in c(1:nrow(season_matches))) {
  row <- season_matches[i, ]
  if(row$wteam < row$lteam) {
    vector <- paste("2014","_",row$wteam,"_", row$lteam, sep ="")
    team <- c(team, vector)
    result <- c(result, 1)
  } else {
    oth <- paste("2014", "_", row$lteam, "_", row$wteam, sep ="")
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
logloss<- (-sum)/nrow(final_logit)
#.5935 for logistic regression (it was 1.008 for rpart