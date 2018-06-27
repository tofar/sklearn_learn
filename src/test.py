import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from pre_process import get_results, get_wordcup_data, process_df_teams_data


def test_one_team(team_name, data):
    # lets work with a subset of the data one that includes games played by Brazil in a Brazil dataframe
    df = data[(data['home_team'] == team_name)
              | (data['away_team'] == team_name)]
    nigeria = df.iloc[:]
    print(nigeria.head())

    # creating a column for year and the first world cup was held in 1930
    year = []
    for row in nigeria['date']:
        year.append(int(row[:4]))

    nigeria['match_year'] = year
    nigeria_1930 = nigeria[nigeria.match_year >= 1930]
    nigeria_1930.count()

    # what is the common game outcome for nigeria visualisation
    wins = []
    for row in nigeria_1930['winning_team']:
        if row != team_name and row != 'Draw':
            wins.append('Loss')
        else:
            wins.append(row)

    winsdf = pd.DataFrame(wins, columns=[team_name])

    # plotting
    fig, ax = plt.subplots(1)
    fig.set_size_inches(10.7, 6.27)
    sns.set(style='darkgrid')
    sns.countplot(x=team_name, data=winsdf)
    plt.show()


def test_acurancy(data, method):
    #  Separate X and Y sets
    X = data.drop(['winning_team'], axis=1)
    Y = data["winning_team"]
    Y = Y.astype('int')
    # 按照 1:9 分割测试和训练数据

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.10, random_state=100)

    # sc = StandardScaler()
    # sc.fit(X_train)
    # X_train_std = sc.transform(X_train)
    # X_test_std = sc.transform(X_test)

    # X_combined_std = np.vstack((X_train_std, X_test_std))
    # Y_combined = np.hstack((Y_train, Y_test))

    # logreg.fit(X_train_std, Y_train)
    method.fit(X_train, Y_train)

    train_score = method.score(X_train, Y_train)
    test_train_score = method.score(X_test, Y_test)

    print("Training set accuracy: ", '%.3f' % (train_score))
    print("Test set accuracy: ", '%.3f' % (test_train_score))


if __name__ == "__main__":
    #  读取数据
    results = get_results('./datasets/results.csv')

    #  wins is a good metric to analyze and predict outcomes of matches in the tournament
    # tournament and venue won't add much to our predictions
    # historical match records will be used

    # narrowing to team patcipating in the world cup
    worldcup_teams = [
        'Australia', ' Iran', 'Japan', 'Korea Republic', 'Saudi Arabia',
        'Egypt', 'Morocco', 'Nigeria', 'Senegal', 'Tunisia', 'Costa Rica',
        'Mexico', 'Panama', 'Argentina', 'Brazil', 'Colombia', 'Peru',
        'Uruguay', 'Belgium', 'Croatia', 'Denmark', 'England', 'France',
        'Germany', 'Iceland', 'Poland', 'Portugal', 'Russia', 'Serbia',
        'Spain', 'Sweden', 'Switzerland'
    ]
    # 测试一支队伍
    test_one_team('Brazil', results)

    df_teams = get_wordcup_data(worldcup_teams, results)
    final = process_df_teams_data(df_teams)
    logreg = LogisticRegression(C=1000.0, random_state=100)

    # 测试准确率
    test_acurancy(final, logreg)
