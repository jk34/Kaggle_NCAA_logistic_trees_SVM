# Predicting the winners of the 2017 NCAA basketball tournament and previous tournaments using R and Python

See "ncaa17.ipynb" for the IPython notebook and "ncaa17.py" at https://github.com/jk34/Kaggle_NCAA_logistic_trees_SVM for the Python script I used to generate the predictions for the upcoming 2017 tournament.

## Summary

-I used Seed, BPI and the regular season winning percentages of each team to predict the winner of each potential matchup. In the code, I used the following variables as predictors: SEED,WST_6, TWPCT (details at https://statsguys.wordpress.com/2014/03/15/data-analytics-for-beginners-march-machine-learning-mania-part-ii/).

-According to http://www.espn.com/mens-college-basketball/bpi, BPI is:

The College Basketball Power Index (BPI) is a measure of team strength that is meant to be the best predictor of performance going forward. BPI represents how many points above or below average a team is."

-Full description of BPI is explained here: http://www.espn.com/mens-college-basketball/story/_/id/7561413/bpi-college-basketball-power-index-explained

-For the 2017 tournament, the generated predictions are in "predictions.csv". Some of the predictions make intuitive sense as they match historical results. For example, the code below predicts 2-seeded Duke to beat 15-seeded Troy with a 96% probability and 2-seeded Kentucky with a 98.8% probability to beat 15-seeded N Kentucky. This makes sense because ever since 1985, the 2-seeded teams have beaten the 15-seeded teams 93.75% out of all matchups


## Background

I wanted to predict the winners of the 2017 NCAA basketball tournament because I have enjoyed watching the tournament in previous years and I also enjoy using Python programming for manipulating data, performing data analysis, and generating predictions on data sets

I obtained my data from Kaggle: https://www.kaggle.com/c/march-machine-learning-mania-2017/data

"RegularSeasonCompactResults.csv" contains data for the winners and losers of each game in all the regular seasons from 1985 to 2017

"Teams.csv" contains each team in NCAA basketball and their team ID value

"TourneyCompactResults.csv" contains the results of each tournament game for the 1985 to 2016 tournaments

"TourneySeeds.csv" contains the seeds for each team in the tournaments from 1985 to 2016

I generated the predictions for the 2017 tournament using the code in "ncaa17.py"

I also generated predictions for previous seasons (not the 2017 tournament using code in R. I utilized the blog_utility.r from https://statsguys.wordpress.com/2014/03/15/data-analytics-for-beginners-march-machine-learning-mania-part-ii/

The analysis I have performed is in ncaa.r. I have tried to use trees, logistic regression, and support vector machines to generate predictive models on previous seasons and tournament results and use the 2014 or 2013 NCAA tournament results as the test data

I also included BPI as a predictor, which is a rough estimate for how good teams really are. For 2011-12 seasons and afterwards, I used the BPI rankings from http://espn.go.com/mens-college-basketball/bpi/_/season/2012 because I think these BPI rankings do not consider tournament results into the rankings. That is because that link has “NCAA tournament information” which predicts the seeds and which teams will make tournament or not. For seasons before 2011-12, I used the Pomeroy rankings instead http://kenpom.com/index.php?y=2014, which also tries to determine how good teams really are

## Data Exploration

The CSV files provided by Kaggle are assumed to be already cleaned. I could perform further data exploration, but due to time constraints, I chose not to. I could have generated histograms and boxplots to look for noticeable typos and outliers

## Results

I used the log-loss values of each model to determine which one predicts results that more accurately match actual results. 

Logistic regression was clearly a better model than rpart because it had a lower log-loss value of .692 if using 2010-2012 as training and 2013 as test (1.06 for rpart). The log-loss for logistic regression was further reduced to .684 if only using BPI, SEED and WST6 as predictors. It was further reduced slightly to .682 if I just used BPI and WST6 (not SEED since it has much higher p-value than A_WST6 and A_BPI). I got a more noticeable improvement to .64 if only using BPI.

If I used 2008-2012 as the training set instead, the log-loss was .688 if using BPI, SEED and A_wST6 as predictors. The log-loss remained the same if I just used BPI and A_WST6 (no SEED) as predictors. The log-loss noticeably decreased to .656 if only using BPI

I then tried to determine the log-loss for support vector machine, but I could not get the code for it to work.

I then tried random forest. It computed the log-loss value as .651 if using 2008-2012 as the training and 2013 as the test set using BPI, SEED and WST6 as the predictors, mtry=2 (using only 2 of the 3 predictors in each tree split), ntree=5000 (using 5000 different trees). The log-loss values did not change much if I varied the number of trees as it was .650 if using ntree=1000, .657 for ntree=500, and .648 if ntree=10000. These log-loss values are very similar to the values from logistic regression

Finally, I used K-nearest neighbors. The best log-loss value was .6947, for k=320 if using 2008-2012 as the training and 2013 as test set. This is slightly larger than the log-loss values from random forest and logistic regression. So the predictions from random forest and logistic regression are more slightly more accurate than K-nearest neighbors (using 2008-12 as the training set and 2013 as the test set)

## Conclusion

The predictions should not taken to seriously. I probably should have used GridSearchCV and 10-fold CV in Python or the caret package in R to find a better value for the parameters in the models I used. 

For the 2017 predictions, I also could have compared the best score from Logistic Regression with other models, such as Gradient Boosting, K-Nearest Neighbors, SVM, etc But I just wanted to generate predictions quickly since the tournament is coming up soon
