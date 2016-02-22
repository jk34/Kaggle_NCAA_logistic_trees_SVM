import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
import pybrain
from pybrain.datasets import SupervisedDataSet
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import SoftmaxLayer

def team_metrics_by_season(seasonletter):
    tourneySeeds= pd.read_csv("tourney_seeds.csv", sep=',') 
    regSeason= pd.read_csv("regular_season_compact_results.csv", sep=',')
    tourneyRes= pd.read_csv("tourney_compact_results.csv", sep=',') 

    #Selecting the seeds for season A
    season_seeds = tourneySeeds[tourneySeeds.season == seasonletter] 
    playoff_teams = season_seeds.sort_values(['team'], ascending=[1])
    playoff_seeds = season_seeds

    #Selecting the regular season statistics for season A
    season = regSeason[regSeason.season == seasonletter] 

    #Wins by team
    win_freq_table = season.wteam
    wins_count = win_freq_table.value_counts()
    wins_by_team = pd.DataFrame(data= {'team': wins_count.index.values, 'wins': wins_count.values} )
    #wins_by_team.columns=['team', 'wins']
    #Losses by team
    loss_freq_table = season.lteam
    loss_count = loss_freq_table.value_counts()
    loss_by_team = pd.DataFrame(data= {'team': loss_count.index.values, 'loss': loss_count.values} )

    #Total Win Percentage
    games_df = wins_by_team.merge(loss_by_team, on='team')
    #MERGE is the same as JOIN in SQL
    #CONCAT simply adds two series/dataframes together
    #MERGE adds them together when they share the COMMON ID (the ID to JOIN them on)
    
    games_df['games'] = games_df['wins'] + games_df['loss']
    games_df['winpct'] = games_df['wins']/games_df['games']
    total_winpct_by_team = games_df.loc[:,['team','winpct']]

    #get seeds
    team_seeds = playoff_seeds['seed'].str.extract('(\d+)').astype(int)
    playoff_seeds.seed = team_seeds
    playoff_teams['seed'] = playoff_seeds.seed
 
    #combining columns together
    team_metrics = total_winpct_by_team.merge(playoff_teams, on='team')

    #only keep columns teamID, TW_PCT, A_SEED, A_BPI
    team_metrics = team_metrics.loc[:,['team','winpct','seed','BPI']]
    team_metrics.columns = ['TEAMID', 'A_TWPCT', 'A_SEED','A_BPI']
    return team_metrics



def train_frame_model(seasonletter):
    #PREDICTORS
    tourneySeeds= pd.read_csv("tourney_seeds.csv", sep=',') 
    regSeason= pd.read_csv("regular_season_compact_results.csv", sep=',')
    tourneyRes= pd.read_csv("tourney_compact_results.csv", sep=',') 


    teamMetrics = team_metrics_by_season(seasonletter)
    
    season_matches = tourneyRes[tourneyRes.season == seasonletter]
    #print "seasonmatches", season_matches.head()
    #season_matches = season_matches.sort_values(['team'], ascending=[1])

    team = pd.Series()
    '''for index, row in season_matches.iterrows():
        print row['wteam'], row['lteam']
        if (row['wteam'] < row['lteam']):
            matchup = "2013_" + row['wteam'].astype(str) + '_' + row['lteam'].astype(str)
            team.append(pd.Series([matchup]))
            result.append(pd.Series([1]))
        else:
            matchup = "2013_" + row['lteam'].astype(str) + '_' + row['wteam'].astype(str)
            team.append(pd.Series([matchup]))
            result.append(pd.Series([0]))'''
    str_seasonletter = str(seasonletter)
    team = str_seasonletter + "_" + season_matches.wteam.astype(str) + '_' + season_matches.lteam.astype(str)
    #team.append(pd.Series([matchup]))
    #loc returns the indices. so if ixs = season_matches['wteam'] > season_matches['lteam'], then
    #season_matches[ixs] returns the indices where season_matches['wteam'] > season_matches['lteam']
    ixs = season_matches['wteam'] > season_matches['lteam']
    team[ixs] = str_seasonletter + "_" + season_matches.loc[ixs,'lteam'].astype(str) + '_' + season_matches.loc[ixs,'wteam'].astype(str)
    result = pd.Series(np.ones(ixs.shape))
    result= 1-ixs
    #result is initialized to all 1's
    #when ixs is satisfied, result is set to 0 because 1-1=0, where ixs=1 because boolean=True
    #else, result is 1-0=1

    list_series = [team,result]
    labels = range(len(list_series))
    model_data_frame  = pd.concat(list_series, levels=labels,axis=1)
    model_data_frame.columns = ['Matchup', 'Win']

    teamMetrics_away = teamMetrics
    #team_metrics_away = team_metrics_away.loc[:,['TEAMID','A_TWPCT','A_SEED','A_BPI']]
    teamMetrics_away.columns = ['TEAMID', 'B_TWPCT', 'B_SEED','B_BPI']

    df2 = model_data_frame['Matchup'].str.split('_', expand=True)
    #convert to numbers
    df2 = df2.astype(int)
    df2.columns = ['Season','HomeID', 'AwayID']
    df2 = df2.loc[:,['HomeID', 'AwayID']]
    model_data_frame = model_data_frame.join(df2)

    model_data_frame = pd.merge(model_data_frame, teamMetrics, how='left', left_on='HomeID', right_on='TEAMID')
    model_data_frame = model_data_frame.drop('TEAMID', 1)
    model_data_frame.columns = ['Matchup','Win','HomeID', 'AwayID','A_TWPCT', 'A_SEED','A_BPI']

    model_data_frame = pd.merge(model_data_frame, teamMetrics_away, how='left', left_on='AwayID', right_on='TEAMID')
    model_data_frame = model_data_frame.drop('TEAMID', 1)

    return model_data_frame


def test_frame_model(seasonletter):
    model_data_frame = submissionFile(seasonletter)

    teamMetrics = team_metrics_by_season(seasonletter) 
    teamMetrics_away = teamMetrics
    teamMetrics_away.columns = ['TEAMID', 'B_TWPCT', 'B_SEED','B_BPI']

    df2 = model_data_frame['Matchup'].str.split('_', expand=True)
    #convert to numbers
    df2 = df2.astype(int)
    df2.columns = ['Season','HomeID', 'AwayID']
    df2 = df2.loc[:,['HomeID', 'AwayID']]
    model_data_frame = model_data_frame.join(df2)

    model_data_frame = pd.merge(model_data_frame, teamMetrics, how='left', left_on='HomeID', right_on='TEAMID')
    model_data_frame = model_data_frame.drop('TEAMID', 1)
    model_data_frame.columns = ['Matchup','Win','HomeID', 'AwayID','A_TWPCT', 'A_SEED','A_BPI']

    model_data_frame = pd.merge(model_data_frame, teamMetrics_away, how='left', left_on='AwayID', right_on='TEAMID')
    model_data_frame = model_data_frame.drop('TEAMID', 1)

    return model_data_frame


def submissionFile(seasonletter):
    #PREDICTORS
    #Selecting and sorting the playoff teamIDs least to greatest for season A
    tourneySeeds= pd.read_csv("tourney_seeds.csv", sep=',')
    season_seeds = tourneySeeds[tourneySeeds.season == seasonletter] 
    playoffTeams = season_seeds['team']
    playoffTeams = playoffTeams.sort_values(ascending=[1])
    print "playoffteams", playoffTeams.head()
    numTeams = len(playoffTeams.index)
    str_seasonletter = str(seasonletter)
    idcol = pd.Series(str_seasonletter+ "_" + "_".join([str(a),str(b)]) for a,b in combinations(playoffTeams,2))
    form = idcol.to_frame()
    form.columns=['Matchup']
    form['result'] = np.NaN

    return form



def main():


    testData = pd.DataFrame()
    for i in range(2013,2014):
        y = test_frame_model(i)
        testData  = pd.concat([testData, y], axis=0)

    #all steps above were for TEST SET
    #NOW, We need to get TRAINING SET (for example, using seasons 2008-2012)
    trainData = pd.DataFrame()
    for i in range(2008,2013): #from 2008 to 2013-1, so from 2008 to 2012
        x = train_frame_model(i)
        trainData  = pd.concat([trainData, x], axis=0)
        #axis=0 means we concat x BELOW trainData. Axis=1, means we append similar to cbind


    #logistic
    model = LogisticRegression(C=.01)
    features=['A_TWPCT','A_SEED','A_BPI','B_TWPCT','B_SEED','B_BPI']
    model.fit(trainData[features], trainData['Win'])
    
    predicted = np.array(model.predict_proba(testData[features]))
    predicted = pd.DataFrame(predicted)
    predicted.columns=['Win','Loss']
    
    #print "predicte", predicted[1:5, 1]
    subfile  = pd.concat([testData.Matchup, predicted.Loss], axis=1)
    subfile.columns=['id', 'pred']


    tourneyResults= pd.read_csv("tourney_compact_results.csv", sep=',') 
    season_matches = tourneyResults[tourneyResults.season == 2013]
    team = pd.Series()
    str_seasonletter = str(2013)
    team = str_seasonletter + "_" + season_matches.wteam.astype(str) + '_' + season_matches.lteam.astype(str)
    ixs = season_matches['wteam'] > season_matches['lteam']
    team[ixs] = str_seasonletter + "_" + season_matches.loc[ixs,'lteam'].astype(str) + '_' + season_matches.loc[ixs,'wteam'].astype(str)
    result = pd.Series(np.ones(ixs.shape))
    result= 1-ixs
    list_series = [team,result]
    labels = range(len(list_series))
    actual_data_frame  = pd.concat(list_series, levels=labels,axis=1)
    actual_data_frame.columns = ['Matchup', 'Win']

    final_logit = pd.merge(subfile, actual_data_frame, left_on='id', right_on='Matchup')
    #final_logit = final_logit.drop('Win', 1)
    final_logit = final_logit.drop('Matchup', 1)
    print "final_logit", final_logit.head()

    print "logistic", log_loss(final_logit.Win, np.array(final_logit.pred)) 



    #neural network
    #ds = SupervisedDataSet(6,1)
    ds = ClassificationDataSet(6,nb_classes=2, class_labels=['1','0'])
    ds.setField( 'input', trainData[features] )
    '''ytrain = trainData['Win']
    ds.setField( 'target', ytrain)
    y_train = y_train.reshape( -1, 1 )
    hidden_size=2
    net = buildNetwork( input_size, hidden_size, target_size, bias = True )
    trainer = BackpropTrainer( net, ds )'''
    fnn = buildNetwork( 6, 5, 6, outclass=SoftmaxLayer )
    trainer = BackpropTrainer( fnn, dataset=trainData, momentum=0.1, verbose=True, weightdecay=0.01)


    trainer.trainUntilConvergence( verbose = True, validationProportion = 0.15, maxEpochs = 1000, continueEpochs = 10 )
    #nn.fit(trainData[features], trainData['Win'])
    predicted = net.activateOnDataset( ds )
    print "predicte", predicted.head()
    '''predicted = np.array(model.predict_proba(testData[features]))
    predicted = pd.DataFrame(predicted)
    predicted.columns=['Win','Loss']'''
    
    #print "predicte", predicted[1:5, 1]
    subfile  = pd.concat([testData.Matchup, predicted.Loss], axis=1)
    subfile.columns=['id', 'pred']


    tourneyResults= pd.read_csv("tourney_compact_results.csv", sep=',') 
    season_matches = tourneyResults[tourneyResults.season == 2013]
    team = pd.Series()
    str_seasonletter = str(2013)
    team = str_seasonletter + "_" + season_matches.wteam.astype(str) + '_' + season_matches.lteam.astype(str)
    ixs = season_matches['wteam'] > season_matches['lteam']
    team[ixs] = str_seasonletter + "_" + season_matches.loc[ixs,'lteam'].astype(str) + '_' + season_matches.loc[ixs,'wteam'].astype(str)
    result = pd.Series(np.ones(ixs.shape))
    result= 1-ixs
    list_series = [team,result]
    labels = range(len(list_series))
    actual_data_frame  = pd.concat(list_series, levels=labels,axis=1)
    actual_data_frame.columns = ['Matchup', 'Win']

    final_logit = pd.merge(subfile, actual_data_frame, left_on='id', right_on='Matchup')
    #final_logit = final_logit.drop('Win', 1)
    final_logit = final_logit.drop('Matchup', 1)
    print "neural networks", log_loss(final_logit.Win, np.array(final_logit.pred)) 

if __name__ == '__main__':
    main()

