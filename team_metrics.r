install.packages('stringr')
library('stringr')
#Selecting and sorting the playoff teamIDs least to greatest for season A
playoff_teams <- sort(tourneySeeds$team[which(tourneySeeds$season == "A")])
 
#Selecting the seeds for season A
playoff_seeds <- tourneySeeds[which(tourneySeeds$season == "A"), ]
 
#Selecting the regular season statistics for season A
season <- regSeason[which(regSeason$season == "A"), ]
 
#Wins by team
win_freq_table <- as.data.frame(table(season$wteam))
wins_by_team <- win_freq_table[win_freq_table$Var1 %in% playoff_teams, ]
 
#Losses by team
loss_freq_table <- as.data.frame(table(season$lteam))
loss_by_team <- loss_freq_table[loss_freq_table$Var1 %in% playoff_teams, ]
 
#Total Win Percentage
gamesplayed <- as.vector(wins_by_team$Freq + loss_by_team$Freq)
total_winpct <- round(wins_by_team$Freq / gamesplayed, digits = 3)
total_winpct_by_team <- as.data.frame(cbind(as.vector(loss_by_team$Var1), total_winpct))
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