#!/usr/bin/env python3.6
# coding: utf-8

import pandas as pd
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from pre_process import get_results, get_wordcup_data, process_teams_data, get_game_schedule, clean_and_predict

#  读取数据
results = get_results('./datasets/results.csv')

# 世界杯32强
worldcup_teams = [
    'Australia', ' Iran', 'Japan', 'Korea Republic', 'Saudi Arabia', 'Egypt',
    'Morocco', 'Nigeria', 'Senegal', 'Tunisia', 'Costa Rica', 'Mexico',
    'Panama', 'Argentina', 'Brazil', 'Colombia', 'Peru', 'Uruguay', 'Belgium',
    'Croatia', 'Denmark', 'England', 'France', 'Germany', 'Iceland', 'Poland',
    'Portugal', 'Russia', 'Serbia', 'Spain', 'Sweden', 'Switzerland'
]

teams_data = get_wordcup_data(worldcup_teams, results)

final = process_teams_data(teams_data)

# 读取历史排名数据
ranking = pd.read_csv('./datasets/FIFA_Rankings.csv')
# 获取2018世界杯赛程
game_schedule = get_game_schedule('./datasets/Schedule.csv', ranking)

#  预测数据集
pred_set = []

#  如果历史排名高的话则置于 home_team
for index, row in game_schedule.iterrows():
    if row['first_position'] < row['second_position']:
        pred_set.append({
            'home_team': row['Home Team'],
            'away_team': row['Away Team'],
        })
    else:
        pred_set.append({
            'home_team': row['Away Team'],
            'away_team': row['Home Team'],
        })

pred_set = pd.DataFrame(pred_set)
backup_pred_set = pred_set

#  将数据横铺, 并取出冗余
pred_set = pd.get_dummies(
    pred_set,
    prefix=['home_team', 'away_team'],
    columns=['home_team', 'away_team'])

#  合并
missing_cols = set(final.columns) - set(pred_set.columns)
for c in missing_cols:
    pred_set[c] = 0

pred_set = pred_set[final.columns]
pred_set = pred_set.drop(['winning_team'], axis=1)
# 线性逻辑回归
logreg = LogisticRegression(C=1000.0, random_state=100)

X = final.drop(['winning_team'], axis=1)
Y = final["winning_team"]
Y = Y.astype('int')

# 按照 1:9 分割测试和训练数据
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.10, random_state=100)

logreg.fit(X_train, Y_train)
predictions = logreg.predict(pred_set)
print("\n开始预测32强的对战情况\n")
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
print("\n开始预测16强的对战情况\n")
group_16 = [('Uruguay', 'Portugal'), ('France', 'Croatia'),
            ('Brazil', 'Mexico'), ('England', 'Colombia'), ('Spain', 'Russia'),
            ('Argentina', 'Peru'), ('Germany', 'Switzerland'), ('Poland',
                                                                'Belgium')]
clean_and_predict(group_16, ranking, final, logreg)

# 八强
print("\n开始预测8强的对战情况\n")
quarters = [('Portugal', 'France'), ('Spain', 'Argentina'),
            ('Brazil', 'England'), ('Germany', 'Belgium')]
clean_and_predict(quarters, ranking, final, logreg)

# 四强
print("\n开始预测4强的对战情况\n")
semi = [('Portugal', 'Brazil'), ('Argentina', 'Germany')]
clean_and_predict(semi, ranking, final, logreg)

#  决赛
print("\n开始预测决赛的对战情况\n")
finals = [('Brazil', 'Germany')]
clean_and_predict(finals, ranking, final, logreg)
