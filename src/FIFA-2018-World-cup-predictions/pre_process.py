#!/usr/bin/env python3.6
# coding: utf-8
import pandas as pd
import numpy as np


def get_results(path):
    """
        获取 results.csv 数据，并进行预处理
    """
    #  读取数据
    # world_cup = pd.read_csv('datasets/WorldCup2018Dataset.csv')
    results = pd.read_csv(path)

    # world_cup.head()
    print(results.head())

    #  Adding goal difference and establishing who is the winner
    winner = []
    for i in range(len(results['home_team'])):
        if results['home_score'][i] > results['away_score'][i]:
            winner.append(results['home_team'][i])
        elif results['home_score'][i] < results['away_score'][i]:
            winner.append(results['away_team'][i])
        else:
            winner.append('Draw')

    results['winning_team'] = winner

    # adding goal difference column
    results['goal_difference'] = np.absolute(
        results['home_score'] - results['away_score'])

    print(results.head())
    return results


def get_1930_data(results):
    """
        获取从1930年开始的世界杯数据，并进行预处理
    """
    pass


def get_wordcup_data(worldcup_teams, data):
    """
        获取世界杯32强的数据
    """
    df_teams_home = data[data['home_team'].isin(worldcup_teams)]

    df_teams_away = data[data['away_team'].isin(worldcup_teams)]
    df_teams = pd.concat((df_teams_home, df_teams_away))
    df_teams.drop_duplicates()

    df_teams.count()
    df_teams.head()

    return df_teams


def process_df_teams_data(df_teams):
    """
        预处理世界杯32强数据，选出我想要的数据
    """
    # create an year column to drop games before 1930
    year = []
    for row in df_teams['date']:
        year.append(int(row[:4]))

    df_teams['match_year'] = year
    df_teams_1930 = df_teams[df_teams.match_year >= 1930]
    df_teams_1930.head()

    # dropping columns that wll not affect matchoutcomes
    df_teams_1930 = df_teams.drop(
        [
            'date', 'home_score', 'away_score', 'tournament', 'city',
            'country', 'goal_difference', 'match_year'
        ],
        axis=1)
    df_teams_1930.head()

    # Building the model
    # the prediction label: The winning_team column will show "2" if the home team has won, "1" if it was a tie, and "0" if the away team has won.

    df_teams_1930 = df_teams_1930.reset_index(drop=True)
    df_teams_1930.loc[df_teams_1930.winning_team == df_teams_1930.home_team,
                      'winning_team'] = 2
    df_teams_1930.loc[df_teams_1930.winning_team == 'Draw', 'winning_team'] = 1
    df_teams_1930.loc[df_teams_1930.winning_team == df_teams_1930.away_team,
                      'winning_team'] = 0

    df_teams_1930.head()

    # convert home team and away team from categorical variables to continous inputs
    #  Get dummy variables
    final = pd.get_dummies(
        df_teams_1930,
        prefix=['home_team', 'away_team'],
        columns=['home_team', 'away_team'])

    return final


def get_game_schedule(csv_path, ranking):
    """
        获取世界杯赛程中前48场
    """
    game_schedule = pd.read_csv(csv_path)

    # Create new columns with ranking position of each team
    game_schedule.insert(1, 'first_position', game_schedule['Home Team'].map(
        ranking.set_index('Team')['Position']))
    game_schedule.insert(2, 'second_position', game_schedule['Away Team'].map(
        ranking.set_index('Team')['Position']))

    #  第一轮一共 48 场，每个小组 6 场
    game_schedule = game_schedule.iloc[:48, :]
    print(game_schedule.tail())

    return game_schedule


def clean_and_predict(matches, ranking, final, logreg):

    #  Initialization of auxiliary list for data cleaning
    positions = []

    #  Loop to retrieve each team's position according to FIFA ranking
    for match in matches:
        positions.append(
            ranking.loc[ranking['Team'] == match[0], 'Position'].iloc[0])
        positions.append(
            ranking.loc[ranking['Team'] == match[1], 'Position'].iloc[0])

    #  Creating the DataFrame for prediction
    pred_set = []

    #  Initializing iterators for while loop
    i = 0
    j = 0

    #  'i' will be the iterator for the 'positions' list, and 'j' for the list of matches (list of tuples)
    while i < len(positions):
        dict1 = {}

        #  If position of first team is better, he will be the 'home' team, and vice-versa
        if positions[i] < positions[i + 1]:
            dict1.update({
                'home_team': matches[j][0],
                'away_team': matches[j][1]
            })
        else:
            dict1.update({
                'home_team': matches[j][1],
                'away_team': matches[j][0]
            })

        #  Append updated dictionary to the list, that will later be converted into a DataFrame
        pred_set.append(dict1)
        i += 2
        j += 1

    #  Convert list into DataFrame
    pred_set = pd.DataFrame(pred_set)
    backup_pred_set = pred_set

    #  Get dummy variables and drop winning_team column
    pred_set = pd.get_dummies(
        pred_set,
        prefix=['home_team', 'away_team'],
        columns=['home_team', 'away_team'])

    #  Add missing columns compared to the model's training dataset
    missing_cols2 = set(final.columns) - set(pred_set.columns)
    for c in missing_cols2:
        pred_set[c] = 0
    pred_set = pred_set[final.columns]

    #  Remove winning team column
    pred_set = pred_set.drop(['winning_team'], axis=1)

    #  Predict!
    predictions = logreg.predict(pred_set)
    for i in range(len(pred_set)):
        print(
            backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
        if predictions[i] == 2:
            print("Winner: " + backup_pred_set.iloc[i, 1])
        elif predictions[i] == 1:
            print("Draw")
        elif predictions[i] == 0:
            print("Winner: " + backup_pred_set.iloc[i, 0])
        print('Probability of ' + backup_pred_set.iloc[i, 1] + ' winning: ',
              '%.3f' % (logreg.predict_proba(pred_set)[i][2]))
        print('Probability of Draw: ',
              '%.3f' % (logreg.predict_proba(pred_set)[i][1]))
        print('Probability of ' + backup_pred_set.iloc[i, 0] + ' winning: ',
              '%.3f' % (logreg.predict_proba(pred_set)[i][0]))
        print("")
