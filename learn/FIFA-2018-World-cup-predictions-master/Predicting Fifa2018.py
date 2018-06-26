#  coding: utf-8

#  In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#  import matplotlib.ticker as ticker
#  import matplotlib.ticker as plticker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#  In[2]:

#  load data
world_cup = pd.read_csv('datasets/World Cup 2018 Dataset.csv')
results = pd.read_csv('datasets/results.csv')

#  In[3]:

world_cup.head()

#  In[4]:

results.head()

#  In[5]:

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

results.head()

#  In[6]:

# lets work with a subset of the data one that includes games played by Nigeria in a Nigeria dataframe
df = results[(results['home_team'] == 'Nigeria')
             | (results['away_team'] == 'Nigeria')]
nigeria = df.iloc[:]
nigeria.head()

#  In[7]:

# creating a column for year and the first world cup was held in 1930
year = []
for row in nigeria['date']:
    year.append(int(row[:4]))

nigeria['match_year'] = year
nigeria_1930 = nigeria[nigeria.match_year >= 1930]
nigeria_1930.count()

#  In[8]:

# what is the common game outcome for nigeria visualisation
wins = []
for row in nigeria_1930['winning_team']:
    if row != 'Nigeria' and row != 'Draw':
        wins.append('Loss')
    else:
        wins.append(row)

winsdf = pd.DataFrame(wins, columns=['Nigeria_Results'])

# plotting
fig, ax = plt.subplots(1)
fig.set_size_inches(10.7, 6.27)
sns.set(style='darkgrid')
sns.countplot(x='Nigeria_Results', data=winsdf)

#  In[9]:

#  wins is a good metric to analyze and predict outcomes of matches in the tournament
# tournament and venue won't add much to our predictions
# historical match records will be used

#  In[10]:

# narrowing to team patcipating in the world cup
worldcup_teams = [
    'Australia', ' Iran', 'Japan', 'Korea Republic', 'Saudi Arabia', 'Egypt',
    'Morocco', 'Nigeria', 'Senegal', 'Tunisia', 'Costa Rica', 'Mexico',
    'Panama', 'Argentina', 'Brazil', 'Colombia', 'Peru', 'Uruguay', 'Belgium',
    'Croatia', 'Denmark', 'England', 'France', 'Germany', 'Iceland', 'Poland',
    'Portugal', 'Russia', 'Serbia', 'Spain', 'Sweden', 'Switzerland'
]
df_teams_home = results[results['home_team'].isin(worldcup_teams)]
df_teams_away = results[results['away_team'].isin(worldcup_teams)]
df_teams = pd.concat((df_teams_home, df_teams_away))
df_teams.drop_duplicates()
df_teams.count()

#  In[11]:

df_teams.head()

#  In[12]:

# create an year column to drop games before 1930
year = []
for row in df_teams['date']:
    year.append(int(row[:4]))
df_teams['match_year'] = year
df_teams_1930 = df_teams[df_teams.match_year >= 1930]
df_teams_1930.head()

#  In[13]:

# dropping columns that wll not affect matchoutcomes
df_teams_1930 = df_teams.drop(
    [
        'date', 'home_score', 'away_score', 'tournament', 'city', 'country',
        'goal_difference', 'match_year'
    ],
    axis=1)
df_teams_1930.head()

#  In[14]:

# Building the model
# the prediction label: The winning_team column will show "2" if the home team has won, "1" if it was a tie, and "0" if the away team has won.

df_teams_1930 = df_teams_1930.reset_index(drop=True)
df_teams_1930.loc[df_teams_1930.winning_team == df_teams_1930.home_team,
                  'winning_team'] = 2
df_teams_1930.loc[df_teams_1930.winning_team == 'Draw', 'winning_team'] = 1
df_teams_1930.loc[df_teams_1930.winning_team == df_teams_1930.away_team,
                  'winning_team'] = 0

df_teams_1930.head()

#  In[32]:

# convert home team and away team from categorical variables to continous inputs
#  Get dummy variables
final = pd.get_dummies(
    df_teams_1930,
    prefix=['home_team', 'away_team'],
    columns=['home_team', 'away_team'])

#  Separate X and y sets
X = final.drop(['winning_team'], axis=1)
y = final["winning_team"]
y = y.astype('int')

#  Separate train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

#  In[33]:

final.head()

#  In[34]:

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
score = logreg.score(X_train, y_train)
score2 = logreg.score(X_test, y_test)

print("Training set accuracy: ", '%.3f' % (score))
print("Test set accuracy: ", '%.3f' % (score2))

#  In[35]:

# adding Fifa rankings
# the team which is positioned higher on the FIFA Ranking will be considered "favourite" for the match
# and therefore, will be positioned under the "home_teams" column
# since there are no "home" or "away" teams in World Cup games.

#  Loading new datasets
ranking = pd.read_csv('datasets/fifa_rankings.csv')
fixtures = pd.read_csv('datasets/fixtures.csv')

#  List for storing the group stage games
pred_set = []

#  In[19]:

#  Create new columns with ranking position of each team
fixtures.insert(1, 'first_position', fixtures['Home Team'].map(
    ranking.set_index('Team')['Position']))
fixtures.insert(2, 'second_position', fixtures['Away Team'].map(
    ranking.set_index('Team')['Position']))

#  We only need the group stage games, so we have to slice the dataset
fixtures = fixtures.iloc[:48, :]
fixtures.tail()

#  In[20]:

#  Loop to add teams to new prediction dataset based on the ranking position of each team
for index, row in fixtures.iterrows():
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

pred_set.head()

#  In[21]:

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

pred_set.head()

#  In[22]:

# group matches
predictions = logreg.predict(pred_set)
for i in range(fixtures.shape[0]):
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

#  In[63]:

#  List of tuples before
group_16 = [('Uruguay', 'Portugal'), ('France', 'Croatia'),
            ('Brazil', 'Mexico'), ('England', 'Colombia'), ('Spain', 'Russia'),
            ('Argentina', 'Peru'), ('Germany', 'Switzerland'), ('Poland',
                                                                'Belgium')]

#  In[64]:


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


#  In[65]:

clean_and_predict(group_16, ranking, final, logreg)

#  In[72]:

#  List of matches
quarters = [('Portugal', 'France'), ('Spain', 'Argentina'),
            ('Brazil', 'England'), ('Germany', 'Belgium')]

#  In[73]:

clean_and_predict(quarters, ranking, final, logreg)

#  In[74]:

#  List of matches
semi = [('Portugal', 'Brazil'), ('Argentina', 'Germany')]

#  In[75]:

clean_and_predict(semi, ranking, final, logreg)

#  In[77]:

#  Finals
finals = [('Brazil', 'Germany')]

#  In[78]:

clean_and_predict(finals, ranking, final, logreg)
