import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def team_metrics_by_season(seasonletter):
    #get seeds for all teams in previous tournaments
    tourneySeeds= pd.read_csv("tourney_seeds.csv", sep=',') 
 
    #get seeds for all teams in 2017 tournament
    tourneySeeds1= pd.read_csv("TourneySeeds2017BPI.csv", sep=',') 

    #read in BPI2017.csv, which was generated from getBPI.r
    #it contains the BPI rankings for each team in the 2017 tournament
    BPI17= pd.read_csv("BPI2017.csv", sep=',')

    #modify names of certain teams to match their names in teams.csv
    #for example, "Wichita State" from BPI2017.csv needs to be changed to "Wichita St" to
    #match "Wichita St" in teams.csv
    BPI17['Team'] = BPI17.Team.str.replace(r"\bState\b", "St")
    BPI17.set_value(11, 'Team', "St Mary's CA")
    BPI17.set_value(27, 'Team', "Miami FL")
    BPI17.set_value(38, 'Team', "VA Commonwealth")
    BPI17.set_value(44, 'Team', "MTSU")
    BPI17.set_value(72, 'Team', "ETSU")
    BPI17.set_value(74, 'Team', "Monmouth NJ")
    BPI17.set_value(77, 'Team', "Mississippi")
    BPI17.set_value(79, 'Team', "FL Gulf Coast")
    BPI17.set_value(82, 'Team', "NC State")
    BPI17.set_value(91, 'Team', "Col Charleston")
    BPI17.set_value(203, 'Team', "Mt St Mary's")
    BPI17.set_value(185, 'Team', "S Dakota St")
    BPI17.set_value(150, 'Team', "NC Central")
    BPI17.set_value(136, 'Team', "Kent")
    BPI17.set_value(175, 'Team', "N Kentucky")
    BPI17.set_value(177, 'Team', "TX Southern")

    teamNames= pd.read_csv("teams.csv", sep=',') 

    #merge teamNames, which contains the names and team_id of each team, with BPI17, which contains the BPI of each team
    BPI2017= pd.merge(BPI17,teamNames, left_on='Team',right_on='team_name')

    #get just the teams and their seeds for the 2017 tournament
    tourneySeeds17 = tourneySeeds1[tourneySeeds1["season"]==2017]

    #merge to get the team name, team_id, seed, and BPI of each team in the 2017 tournament
    tourneySeeds17F= pd.merge(tourneySeeds17,BPI2017, left_on='team',right_on='team_id')

    #only need columns season, seed, team, BPI_y
    tourneySeeds17F.drop(tourneySeeds17F.columns[[3,5,6,7]], axis=1, inplace=True)

    #change "BPI_y" to "BPI"
    tourneySeeds17F=tourneySeeds17F.rename(columns = {'BPI_y':'BPI'})

    #append 2017 data to data containing seasons 1985-2014 (could also include seasons 2015-2016, but I figured the training set is already large enough)
    tourneySeeds=tourneySeeds.append(pd.DataFrame(data=tourneySeeds17F))

    #save this into csv file because it will later be loaded into submissionFile in test_frame_model
    tourneySeeds.to_csv("tourney_seeds17.csv", sep=',') 

    #load in the regular season data for seasons 1985-2014,2017
    regSeason= pd.read_csv("regular_season_compact_results2017.csv", sep=',')

    #convert all column names to lower-case
    regSeason.columns = map(str.lower, regSeason.columns)

    #Selecting the season, seed, team, and BPI for a given season.
    #Ex: To get the data for the 2010 season, where seasonletter=2010
    season_seeds = tourneySeeds[tourneySeeds.season == seasonletter] 

    #sort values by the team id in the "team" column
    playoff_teams = season_seeds.sort_values(['team'], ascending=[1])


    playoff_seeds = season_seeds

    #Selecting the regular season statistics for a given season
    season = regSeason[regSeason.season == seasonletter] 

    #Count the number of wins for each team in a given regular season
    win_freq_table = season.wteam
    wins_count = win_freq_table.value_counts()
    wins_by_team = pd.DataFrame(data= {'team': wins_count.index.values, 'wins': wins_count.values} )

    #Losses by team
    loss_freq_table = season.lteam
    loss_count = loss_freq_table.value_counts()
    loss_by_team = pd.DataFrame(data= {'team': loss_count.index.values, 'loss': loss_count.values} )

    #Total Win Percentage for each team in a given regular season
    games_df = wins_by_team.merge(loss_by_team, on='team')
    games_df['games'] = games_df['wins'] + games_df['loss']
    games_df['winpct'] = games_df['wins']/games_df['games']
    total_winpct_by_team = games_df.loc[:,['team','winpct']]

    #extract numerical value of seeds for each playoff team
    #extract numerical value of seed. For example, for seed "W01", get just "1"
    team_seeds = playoff_seeds['seed'].str.extract('(\d+)').astype(int)
    playoff_seeds.seed = team_seeds
    playoff_teams['seed'] = playoff_seeds.seed
 
    #combining columns together
    #to get season, seed, team, BPI, and winpct for each team in the tournament
    team_metrics = total_winpct_by_team.merge(playoff_teams, on='team')

    #only keep columns teamID, TW_PCT, A_SEED, A_BPI
    team_metrics = team_metrics.loc[:,['team','winpct','seed','BPI']]
    team_metrics.columns = ['TEAMID', 'A_TWPCT', 'A_SEED','A_BPI']
    return team_metrics



def train_frame_model(seasonletter):
    #get results of each game in previous tournaments
    #the season, winning team's id (wteam), and losing team's id (lteam)
    tourneyRes= pd.read_csv("tourney_compact_results.csv", sep=',') 

    #get results for a given tournament season
    season_matches = tourneyRes[tourneyRes.season == seasonletter]

    #get the id, winning percentage, seed, and BPI for each team in a given season
    teamMetrics = team_metrics_by_season(seasonletter)

    #each entry in "team" looks something like "2010_1115_1457"
    #In that example, seasonletter=2010, wteam=1115, lteam=1457
    #this format is necessary because Kaggle requires a similar format when accepting the predictions in a csv file
    team = pd.Series()
    str_seasonletter = str(seasonletter)
    team = str_seasonletter + "_" + season_matches.wteam.astype(str) + '_' + season_matches.lteam.astype(str)
    
    #loc returns the indices.
    #if ixs = season_matches['wteam'] > season_matches['lteam'], then
    #season_matches[ixs] returns the indices where season_matches['wteam'] > season_matches['lteam']
    ixs = season_matches['wteam'] > season_matches['lteam']
    team[ixs] = str_seasonletter + "_" + season_matches.loc[ixs,'lteam'].astype(str) + '_' + season_matches.loc[ixs,'wteam'].astype(str)
    result = pd.Series(np.ones(ixs.shape))
    result= 1-ixs
    #result is initialized to all 1's
    #when ixs is satisfied, result is set to 0 because 1-1=0, where ixs=1 because boolean=True
    #else, result is 1-0=1

    #want to generate a dataframe containing the matchups and the win column
    #result will look like:
    #Matchup           Win
    #2010_1124_1181    0
    #2010_1277_1397    1
    #The team with the lower id value is placed before the team with the higher id
    #Win=0 means the 1st team lost. So team 1124 lost to team 1181 in the 2010 season
    #Win=1 means the 1st team won. So team 1277 beat team 1397 in the 2010 season
    list_series = [team,result]
    labels = range(len(list_series))
    model_data_frame  = pd.concat(list_series, levels=labels,axis=1)
    model_data_frame.columns = ['Matchup', 'Win']


    #will create a dataframe containing Matchup, Win (1 or 0), HomeID, AwayID, and the Winpct/Seed/BPI for the HomeID and AwayID
    #for simplicity, HomeID refers to the team with the lower team_id value
    #it is not related to the seeds of the teams in each matchup
    
    #For example, for the entry
    #2010_1115_1457  1  1115  1457 0.531250 16 220.0 0.566667 16 223.0
    #In the 2010 season, team with id=1115 is the "home" team and team with id="1457" is the "away" team
    #the home team won because Win=1, it had a .531 Winpct in the regular season, it was a 16 seed, and its BPI was 222. The away team had a .566 Winpct, 16 seed, and BPI=223
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


def test_frame_model(seasonletter):
    #similar to train_frame_model, but it generates a dataframe for the test set
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
    #print "MODEL DATA FRAME", model_data_frame
    return model_data_frame


def submissionFile(seasonletter):
    #Selecting and sorting the playoff teamIDs least to greatest for season A
    tourneySeeds= pd.read_csv("tourney_seeds17.csv", sep=',')
    season_seeds = tourneySeeds[tourneySeeds.season == seasonletter] 
    playoffTeams = season_seeds['team']
    playoffTeams = playoffTeams.sort_values(ascending=[1])
    numTeams = len(playoffTeams.index)
    str_seasonletter = str(seasonletter)
    idcol = pd.Series(str_seasonletter+ "_" + "_".join([str(a),str(b)]) for a,b in combinations(playoffTeams,2))
    form = idcol.to_frame()
    form.columns=['Matchup']
    form['result'] = np.NaN

    return form



def main():

    #Train model on training set (for example, using seasons 2008-2012)
    trainData = pd.DataFrame()
    for i in range(2008,2013): #from 2008 to 2013-1, so from 2008 to 2012
        x = train_frame_model(i)
        trainData  = pd.concat([trainData, x], axis=0)
        #axis=0 means we concat x BELOW trainData. Axis=1, means we append similar to cbind

    #use upcoming tournament (season 2017) as test set
    testData = pd.DataFrame()
    for i in range(2017,2018):
        y = test_frame_model(i)
        testData  = pd.concat([testData, y], axis=0)


    '''Use Logistic Regression
    I just used C=.01, but I probably should use GridSearchCV and 10-fold CV to find a better value for C
    I also could have compared the best score from Logistic Regression with other models, such as Gradient Boosting, K-Nearest Neighbors, SVM, etc
    But I just wanted to generate predictions quickly since the tournament is coming up soon'''

    #predictions will be generated in predictions.csv
    #each row in predictions.csv contains the probability of each team winning in a potential matchup
    #For example, for the row
    #1112 1116 .636
    #That means teamid=1112 has a .636 chance to beat teamid=1116

    model = LogisticRegression(C=.01)
    features=['A_TWPCT','A_SEED','A_BPI','B_TWPCT','B_SEED','B_BPI']
    model.fit(trainData[features], trainData['Win'])
    
    predicted = np.array(model.predict_proba(testData[features]))
    predicted = pd.DataFrame(predicted)
    predicted.columns=['Win','Loss']

    subfile1  = pd.concat([testData.HomeID, testData.AwayID, predicted.Loss], axis=1)
    subfile1.to_csv("predictions.csv", sep=',')   


    #generate bar plots showing win probabilities of a few matchups using Seaborn
    teamNames= pd.read_csv("teams.csv", sep=',')
    outNames= pd.merge(subfile1, teamNames,left_on='HomeID',right_on='team_id')
    outNames= pd.merge(outNames, teamNames,left_on='AwayID',right_on='team_id')
    outNames.drop(outNames.columns[[0,1,3,5]], axis=1, inplace=True)

    #change "BPI_y" to "BPI"
    outNames = outNames.rename(columns = {'team_name_x':'TeamA', 'team_name_y':'TeamB', 'Loss':'WinProb'})
    #cols = outNames.columns.tolist()
    #cols = cols[-1] + cols[:-1]
    #outNames = outNames[cols]
    outNames = outNames[['TeamA', 'TeamB', 'WinProb']]
    outNames.to_csv("predictions1.csv", sep=',')   

    #generate plots
    teamlst=[]
    winlst=[]

    for index, row in outNames.iterrows():
        if (row[0]=="Butler" and row[1]=="Winthrop") or (row[1]=="Butler" and row[0]=="Winthrop"):
            teamlst.append(row[0])
            teamlst.append(row[1])
            winlst.append(round(100*row[2],1))
            winlst.append(round(100*(1-row[2]),1))
        elif (row[0]=="Maryland" and row[1]=="Xavier") or (row[1]=="Maryland" and row[0]=="Xavier"):
            teamlst.append(row[0])
            teamlst.append(row[1])
            winlst.append(round(100*row[2],1))
            winlst.append(round(100*(1-row[2]),1))
        elif (row[0]=="Kent" and row[1]=="UCLA") or (row[1]=="Kent" and row[0]=="UCLA"):
            teamlst.append(row[0])
            teamlst.append(row[1])
            winlst.append(round(100*row[2],1))
            winlst.append(round(100*(1-row[2]),1))
        elif (row[0]=="Creighton" and row[1]=="Rhode Island") or (row[1]=="Creighton" and row[0]=="Rhode Island"):
            teamlst.append(row[0])
            teamlst.append(row[1])
            winlst.append(round(100*row[2],1))
            winlst.append(round(100*(1-row[2]),1))

    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as plot

    with PdfPages('winprobs.pdf') as pdf_pages:
        fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        list_of_axes = (ax1,ax2,ax3,ax4)
        for i,j in zip(range(0,len(teamlst), 2),list_of_axes):
            sns.barplot(teamlst[i:i+2], winlst[i:i+2],ax=j)
            j.axes.set_title('Which team will win?', fontsize=14,color="b",alpha=0.3)
            j.set_ylabel("Win Probability (%)",size = 12,color="r",alpha=0.5)

            for p in j.patches:
                j.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    	pdf_pages.savefig()
    	plt.close()
    

if __name__ == '__main__':
    main()
