library(data.table)
library(rvest)
library(stringr)
library(tidyr)
library(rJava)
library(XLConnect)
library(dplyr)
library(reshape)
library(glmnet)

setwd("/home/jerry/Desktop/Kaggle_NCAA_logistic_trees_SVM")

#use web scraping to get Win% for each team from ESPN
all<-data.frame()
for (i in 1:15) {
  pageurl <- paste0("http://www.espn.com/mens-college-basketball/bpi/_/page/",i,"/view/bpi")
  webpage <- read_html(pageurl)
  standings02 <- html_nodes(webpage, 'table')
  season02b <- html_table(standings02)[[2]]
  all<-rbind(season02b, all)
}
all<-all[ order(all[,1]), ]

BPI2017<-all[,c("RK","TEAM")]
colnames(BPI2017) <- c("BPI", "Team")
BPI2017$Team<-sub("(?s)^(.*)(?i:\\1)$|[A-Z-]{1,4}$", "\\1", BPI2017$Team, perl=TRUE)
write.csv(BPI2017, file = "BPI2017.csv", row.names = FALSE)