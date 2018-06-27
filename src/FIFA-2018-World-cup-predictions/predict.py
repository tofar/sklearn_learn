#!/usr/bin/env python3.6
# coding: utf-8

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from pre_process import get_results, get_wordcup_data, process_df_teams_data, get_game_schedule, clean_and_predict

#  读取数据
results = get_results('./datasets/results.csv')

#  wins is a good metric to analyze and predict outcomes of matches in the tournament
# tournament and venue won't add much to our predictions
# historical match records will be used

# narrowing to team patcipating in the world cup
worldcup_teams = [
    'Australia', ' Iran', 'Japan', 'Korea Republic', 'Saudi Arabia', 'Egypt',
    'Morocco', 'Nigeria', 'Senegal', 'Tunisia', 'Costa Rica', 'Mexico',
    'Panama', 'Argentina', 'Brazil', 'Colombia', 'Peru', 'Uruguay', 'Belgium',
    'Croatia', 'Denmark', 'England', 'France', 'Germany', 'Iceland', 'Poland',
    'Portugal', 'Russia', 'Serbia', 'Spain', 'Sweden', 'Switzerland'
]

df_teams = get_wordcup_data(worldcup_teams, results)

final = process_df_teams_data(df_teams)

# adding Fifa rankings
# the team which is positioned higher on the FIFA Ranking will be considered "favourite" for the match
# and therefore, will be positioned under the "home_teams" column
# since there are no "home" or "away" teams in World Cup games.

#  Loading new datasets
ranking = pd.read_csv('./datasets/FIFA_Rankings.csv')
game_schedule = get_game_schedule('./datasets/Schedule.csv', ranking)

#  List for storing the group stage games
pred_set = []

#  Loop to add teams to new prediction dataset based on the ranking position of each team
for index, row in game_schedule.iterrows():
    if row['first_position'] < row['second_position']:
        pred_set.append({
            'home_team': row['Home Team'],
            'away_team': row['Away Team'],
            'winning_team': None
        })
    else:
        pred_set.append({
            'home_team': row['Away Team'],
            'away_team': row['Home Team'],
            'winning_team': None
        })

pred_set = pd.DataFrame(pred_set)
backup_pred_set = pred_set

#  Get dummy variables and drop winning_team column
pred_set = pd.get_dummies(
    pred_set,
    prefix=['home_team', 'away_team'],
    columns=['home_team', 'away_team'])

#  Add missing columns compared to the model's training dataset
missing_cols = set(final.columns) - set(pred_set.columns)
for c in missing_cols:
    pred_set[c] = 0
pred_set = pred_set[final.columns]

#  Remove winning team column
pred_set = pred_set.drop(['winning_team'], axis=1)

# 逻辑回归
logreg = LogisticRegression(C=1000.0, random_state=100)
#  Separate X and Y sets
X = final.drop(['winning_team'], axis=1)
Y = final["winning_team"]
Y = Y.astype('int')

#  Separate train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.30, random_state=42)

logreg.fit(X_train, Y_train)

# group matches
predictions = logreg.predict(pred_set)
for i in range(game_schedule.shape[0]):
    print(backup_pred_set.iloc[i, 1] + " and " + backup_pred_set.iloc[i, 0])
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

# 16强
group_16 = [('Uruguay', 'Portugal'), ('France', 'Croatia'),
            ('Brazil', 'Mexico'), ('England', 'Colombia'), ('Spain', 'Russia'),
            ('Argentina', 'Peru'), ('Germany', 'Switzerland'), ('Poland',
                                                                'Belgium')]
clean_and_predict(group_16, ranking, final, logreg)

# 八强
quarters = [('Portugal', 'France'), ('Spain', 'Argentina'),
            ('Brazil', 'England'), ('Germany', 'Belgium')]
clean_and_predict(quarters, ranking, final, logreg)

# 四强
semi = [('Portugal', 'Brazil'), ('Argentina', 'Germany')]
clean_and_predict(semi, ranking, final, logreg)

#  决赛
finals = [('Brazil', 'Germany')]
clean_and_predict(finals, ranking, final, logreg)
