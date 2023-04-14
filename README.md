# ADS-NFL

Looking into optimising pass plays in NFL matches by using Madden player statistics.

## Installation

Clone ADS-NFL

```
git clone https://github.com/samuelrufusward/ADS-NFL.git
```

Prepare anaconda environment, `mamba` also works

```
conda env create -n ads -f environment.yml
```

## Dataset

Download train data from [NFL Big Data Bowl 2021](https://www.kaggle.com/c/nfl-big-data-bowl-2021)

Download test data from [NFL Big Data Bowl 2023](https://www.kaggle.com/c/nfl-big-data-bowl-2023)

## Regression

Import functions needed for regression data

```
import nfl_regression
```

Preprocess train/test splits of Madden stats

```
train_receiver_stats, train_player_stats, train_weeks = nfl_regression.preprocess_year_data('2021')
X_train, Y_train = nfl_regression.get_yards_by_stats(train_receiver_stats, train_player_stats, train_weeks, '2021')

test_receiver_stats, test_player_stats, test_weeks = nfl_regression.preprocess_year_data('2023')
X_test, Y_test = nfl_regression.get_yards_by_stats(test_receiver_stats, test_player_stats, test_weeks, '2023')
```

We only consider these features

```
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
```

Train regression on average weekly yards gained 

```
from sklearn.svm import SVR
reg = SVR(kernel='poly')
reg = reg.fit(X_train, Y_train)
```

Score against test split

```
reg.score(X_test, Y_test)
```

Predict average weekly yards gained by using only a player's statistics!

```
reg.predict(test_player_stats[test_player_stats['abrev_name'] == 't.brady'][feature_columns])
```
