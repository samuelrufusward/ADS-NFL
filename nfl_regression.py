import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE


feature_columns = ['Age', 'Height', 'Weight', 'Overall', 'Speed',
 'Acceleration', 'Agility', 'Change of Dir', 'Strength', 'Jumping',
 'Awareness', 'Carrying', 'Break Tackle', 'Juke Move', 'Spin Move',
 'Trucking', 'Stiff Arm', 'BC Vision', 'Catching', 'Catch In Traffic',
 'Spec Catch', 'Release', 'Short RR', 'Medium RR', 'Deep RR',
 'Throw Power', 'Throw Acc Short', 'Throw Acc Mid', 'Throw Acc Deep',
 'Throw Under Pressure', 'Throw On The Run', 'Play Action', 'Break Sack',
 'Run Block', 'Run Block Power', 'Run Block Finesse', 'Pass Block',
 'Pass Block Power', 'Pass Block Finesse', 'Impact Blocking',
 'Lead Blocking', 'Tackle', 'Hit Power', 'Pursuit', 'Man Coverage',
 'Zone Coverage', 'Press', 'Play Recognition', 'Power Moves',
 'Finesse Moves', 'Block Shedding', 'Kick Power', 'Kick Accuracy',
 'Kick Return', 'Stamina', 'Injury', 'Toughness', 'Years Pro']

team_mapping = {"PHI": "Eagles", "BUF": "Bills", "ATL": "Falcons", "PIT": "Steelers",
                "CLE": "Browns", "CIN": "Bengals", "IND": "Colts", "TEN": "Titans", 
                "MIA": "Dolphins", "BAL": "Ravens", "NE": "Patriots", "HOU": "Texans",
                "JAX": "Jaguars", "NYG": "Giants", "NO": "Saints", "TB": "Buccaneers",
                "WAS": "Commanders", "ARI": "Cardinals", "CAR": "Panthers", "DAL": "Cowboys", 
                "GB": "Packers", "CHI": "Bears", "NYJ": "Jets", "DET": "Lions", 
                "OAK": "Raiders", "LA": "Rams", "MIN": "Vikings", "LAC": "Chargers", 
                "KC": "Chiefs", "SF": "49ers", "DEN": "Broncos", "SEA": "Seahawks", 
                "LV": "Raiders"}

def add_abbreviated(player_stats):
    player_stats["abrev_name"] = player_stats.displayName.apply(lambda x: str(x)[0]+ "."+"".join(str(x).split(" ")[1:]))
    player_stats.abrev_name = player_stats.abrev_name.apply(lambda x: str(x).lower())

def map_teams(play_data):
    play_data['Team'] = play_data.possessionTeam.apply(lambda x: team_mapping[str(x)])

def process_plays(play_data):
    playDescText = play_data.playDescription.apply(lambda x: " ".join(str(x).split()[1:]))
    playDescText = playDescText.apply(lambda x: "".join(x.split(")")[1:]) if str(x).startswith("(") else x)
    receivers = playDescText.map(lambda x: str(x).lower().split(" ")[str(x).lower().split(" ").index("to")+1] if ("pass" in str(x).lower().split(" ") and "to" in str(x).lower().split(" ")) & ("intercept" not in str(x).lower()) else np.nan)
    receivers.loc[receivers.map(lambda x: str(x)[-1]) == "."] = receivers.loc[receivers.map(lambda x: str(x)[-1]) == "."].apply(lambda x: "".join(str(x)[:-1]))
    receivers.loc[receivers=="no"] = np.nan
    return receivers

def merge_plays_with_stats(play_data, player_stats, receivers):
    play_data['abrev_name'] = receivers
    receiver_stats = play_data.merge(player_stats, how='left', left_on=['Team','abrev_name'], right_on=['Team','abrev_name'])
    receiver_stats = receiver_stats.rename(columns={'abrev_name': "receiver"})
    receiver_stats.dropna(axis='rows', subset='nflId', inplace=True)
    receiver_stats.reset_index(drop=True, inplace=True)
    return receiver_stats

def get_yards_by_stats(receiver_stats, player_stats, weeks, year):
    if year == '2021': yards_moniker = 'offensePlayResult'
    elif year == '2023': yards_moniker = 'prePenaltyPlayResult'
    avg_weekly_yards_by_stats = pd.DataFrame({'nflId': receiver_stats['nflId'].astype('int32').unique()})
    avg_weekly_yards_by_stats['weekly_yards'] = avg_weekly_yards_by_stats['nflId'].map(lambda x: receiver_stats[receiver_stats['nflId'] == x][yards_moniker].sum() / weeks)
    avg_weekly_yards_by_stats = avg_weekly_yards_by_stats.merge(player_stats[feature_columns + ['nflId']], how='inner', on='nflId')
    return avg_weekly_yards_by_stats[feature_columns], avg_weekly_yards_by_stats['weekly_yards']

def preprocess_year_data(year):
    weeks = pd.read_csv(f'BigDataBowl{year}/data{year}/games.csv')['week'].max()
    player_stats = pd.read_csv(f'BigDataBowl{year}/merged_df.csv').drop('Unnamed: 0', axis='columns')
    play_data = pd.read_csv(f'BigDataBowl{year}/data{year}/plays.csv')
    add_abbreviated(player_stats)
    map_teams(play_data)
    receivers = process_plays(play_data)
    receiver_stats = merge_plays_with_stats(play_data, player_stats, receivers)
    return receiver_stats, player_stats, weeks

def find_top_select_k_best(reg, train, test):
    X_train, Y_train = train
    X_test, Y_test = test
    train_scores = []
    test_scores = []
    for i in range(1, X_train.shape[1] + 1):
        k_best = SelectKBest(f_regression, k=i).fit(X_train, Y_train)
        Xk_train = k_best.transform(X_train)
        Xk_test = k_best.transform(X_test)
        reg.fit(Xk_train, Y_train)
        train_scores.append(reg.score(Xk_train, Y_train))
        test_scores.append(reg.score(Xk_test, Y_test))
    k = np.argmax(test_scores) + 1 
    return (k, train_scores[k-1], test_scores[k - 1])

def find_top_rfe(reg, train, test):
    X_train, Y_train = train
    X_test, Y_test = test
    train_scores = []
    test_scores = []
    for i in range(1, X_train.shape[1] + 1):
        rfe = RFE(reg, n_features_to_select=i).fit(X_train, Y_train)
        Xk_train = rfe.transform(X_train)
        Xk_test = rfe.transform(X_test)
        reg.fit(Xk_train, Y_train)
        train_scores.append(reg.score(Xk_train, Y_train))
        test_scores.append(reg.score(Xk_test, Y_test))
    k = np.argmax(test_scores) + 1 
    return (k, train_scores[k - 1], test_scores[k - 1])
