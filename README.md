# Kaggle_NCAA_logistic_trees_SVM
My work in using machine learning algorithms on NCAA basketball tournament data. I obtained the data from https://www.kaggle.com/c/march-machine-learning-mania-2015

I also utilize the blog_utility.r from https://statsguys.wordpress.com/2014/03/15/data-analytics-for-beginners-march-machine-learning-mania-part-ii/

The analysis I have performed is in ncaa.r. I have tried to use trees, logistic regression, and support vector machines to generate predictive models on previous seasons and tournament results and use the 2014 or 2013 NCAA tournament results as the test data

I used the log-loss values of each model to determine which one predicts results that more accurately match actual results. I used the following variables as predictors: SEED,WST_6, TWPCT (details at https://statsguys.wordpress.com/2014/03/15/data-analytics-for-beginners-march-machine-learning-mania-part-ii/). I also included BPI as a predictor, which is a rough estimate for how good teams really are. For 2011-12 seasons and afterwards, I used the BPI rankings from http://espn.go.com/mens-college-basketball/bpi/_/season/2012 because I think these BPI rankings do not consider tournament results into the rankings. That is because that link has “NCAA tournament information” which predicts the seeds and which teams will make tournament or not. For seasons before 2011-12, I used the Pomeroy rankings instead http://kenpom.com/index.php?y=2014, which also tries to determine how good teams really are

Logistic regression was clearly a better model than rpart because it had a lower log-loss value of .692 if using 2010-2012 as training and 2013 as test (1.06 for rpart). The log-loss for logistic regression was further reduced to .684 if only using BPI, SEED and WST6 as predictors. It was further reduced slightly to .682 if I just used BPI and WST6 (not SEED since it has much higher p-value than A_WST6 and A_BPI). I got a more noticeable improvement to .64 if only using BPI.

If I used 2008-2012 as the training set instead, the log-loss was .688 if using BPI, SEED and A_wST6 as predictors. The log-loss remained the same if I just used BPI and A_WST6 (no SEED) as predictors. The log-loss noticeably decreased to .656 if only using BPI

I then tried to determine the log-loss for support vector machine, but I could not get the code for it to work.

I then tried random forest. It computed the log-loss value as .651 if using 2008-2012 as the training and 2013 as the test set using BPI, SEED and WST6 as the predictors, mtry=2 (using only 2 of the 3 predictors in each tree split), ntree=5000 (using 5000 different trees). The log-loss values did not change much if I varied the number of trees as it was .650 if using ntree=1000, .657 for ntree=500, and .648 if ntree=10000. These log-loss values are very similar to the values from logistic regression

Finally, I used K-nearest neighbors. The best log-loss value was .6947, for k=320 if using 2008-2012 as the training and 2013 as test set. This is slightly larger than the log-loss values from random forest and logistic regression. So the predictions from random forest and logistic regression are more slightly more accurate than K-nearest neighbors (using 2008-12 as the training set and 2013 as the test set)
