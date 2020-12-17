# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:46:29 2020

@author: mtdic
"""

#%%
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder


### Stuff created by me
sys.path.append(r"C:\Users\mtdic\Documents\GitHub\big_data_bowl_2021")
import viz_util as viz
import data_util as du

os.chdir(r"E:/NFL/big_data_bowl/2021")

#%%
#### Non-Tracking Data Import and Cleaning

##reading in 
#includes schedule info for games
df_games = pd.read_csv("data/provided/games.csv")

#includes play-by-play info on specific plays
df_plays = pd.read_csv("data/provided/plays.csv")

#includes background info for players
df_players = pd.read_csv("data/provided/players.csv")
df_players['height'] = du.standardize_heights(df_players['height'])
df_players['birthDate'], df_players['age'] = du.convert_birthdates_to_ages(df_players['birthDate'])


### BONUS Data: 
df_targets = pd.read_csv("data/provided/targetedReceiver.csv")

#%%

### Making a training set for the incompletions over expected model
df_train = df_plays.merge(df_targets)

### Creating penalty fields to use in target variable creation
df_train['penalty_code_list'] = df_train['penaltyCodes'].apply(lambda x: x.split(';') if type(x) == str else [])
df_train['defense_cov_penalty_accepted'] = df_train.apply(lambda x: True if (x['playResult'] > x['offensePlayResult']) and
                                                                            (('ICT' in x['penalty_code_list']) or
                                                                             ('DPI' in x['penalty_code_list']) or
                                                                             ('DH' in x['penalty_code_list']))
                                                                    else False, axis = 1)
df_train['offense_opi_penalty_accepted'] = df_train.apply(lambda x: True if (x['playResult'] < x['offensePlayResult']) and
                                                                            ('OPI' in x['penalty_code_list'])
                                                                    else False, axis = 1)

## Make the target variable (defensive win/defense lose by penalty/defense lose by completion)
df_train['target_descr'] = df_train.apply(lambda x: 'D-Win' if ( x['offense_opi_penalty_accepted'] or ((x['passResult'] in ['I', 'IN']) and 
                                                                                                        not x['defense_cov_penalty_accepted'] ))
                                         else 'D-Lose' if (x['defense_cov_penalty_accepted'] or (x['passResult'] == 'C'))
                                         else None, axis = 1)
df_train = df_train[df_train['target_descr'].isin(['D-Lose', 'D-Win'])]
label_encoder = LabelEncoder()
df_train['target'] = label_encoder.fit_transform(df_train['target_descr'])

#%%

## Data Cleaning.. Removing any plays where the ball is not in play

## Read in tracking data iteratively
WEEKS = list(range(1,18))
tracking_dfs = []
for w in WEEKS:
    df_tracking = pd.read_csv(f"data/provided/week{w}.csv")
    df_tracking = df_tracking[(df_tracking['displayName'] == 'Football')]
    tracking_dfs.append(df_tracking)
ball_tracking_df = pd.concat(tracking_dfs)
del df_tracking, tracking_dfs, w

## Make a column indicating if the ball is within the bounds of the field
ball_tracking_df['football_in_play'] = ball_tracking_df.apply(lambda x:
                                                 1 if ((x['displayName'] == "Football")
                                                   and ((x['y'] < 53.3) and (x['y'] > 0) and
                                                        (x['x'] < 120) and (x['x'] > 0)))
                                                 else 0, axis=1)    


## Creating a DF to be merged in containing x,y position of football throughout each play
football_in_play_df = (ball_tracking_df.groupby(["gameId", "playId"])
                                        .agg({'football_in_play': np.sum})
                                        .reset_index())
football_in_play_df['ball_in_play_ind'] = football_in_play_df['football_in_play'].apply(lambda x: True if x > 0
                                                                                         else False)

df_train = df_train.merge(football_in_play_df[['gameId', 'playId', 'ball_in_play_ind']], how = 'inner')

## Only keep those where the ball is in play.
df_train = df_train[df_train['ball_in_play_ind']]

#%%

## Pre-feature engineering
### Load player tracking data for all plays in training set.  Include ball, defense, and receiver 
## Read in tracking data iteratively
WEEKS = list(range(1,18))
tracking_dfs = []
for w in WEEKS:
    df_tracking = pd.read_csv(f"data/provided/week{w}.csv")
    df_tracking = df_tracking.merge(df_targets)
    df_tracking = df_tracking.merge(df_games[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr', 'week']])
    df_tracking = df_tracking.merge(df_plays[['gameId', 'playId', 'possessionTeam', 'yardlineNumber', 'absoluteYardlineNumber']])
    df_tracking['isDefense'] = df_tracking.apply(lambda x:
                                                 True if (((x['team'] == "home") &
                                                               (x['possessionTeam'] == x['visitorTeamAbbr']))
                                                           |  ((x['team'] == "away") &
                                                               (x['possessionTeam'] == x['homeTeamAbbr'])))
                                                 else False, axis=1)
    df_tracking['isTargetedReceiver'] = df_tracking.apply(lambda x: True if x['nflId'] == x['targetNflId'] else False, axis=1)
    df_tracking = df_tracking[(df_tracking['displayName'] == 'Football') | 
                              (df_tracking['isTargetedReceiver']) |
                              (df_tracking['isDefense'])]
    tracking_dfs.append(df_tracking)
train_tracking_df = pd.concat(tracking_dfs)
del df_tracking, tracking_dfs, w


## Orient coordinates so offense is always driving to the right
train_tracking_df = du.orient_coords(train_tracking_df)

## Find closest defender/receiver to ball at time of arrival and distances between of the 3 pairs.
distances_df = du.find_distances_at_arrival(train_tracking_df)

### Merge those features in
df_train = df_train.merge(distances_df[['gameId', 'playId', 'defender_distance_to_football',
                                        'receiver_corner_dist_between', 'receiver_distance_to_football',
                                        'defenderNflId']])


#%%

### Find number of yards past LOS the ball was thrown 
## (may want to weed out easy screen passes by looking at >5 yards downfield...
#    plot it out to see what makes sense)
yards_past_los_df = du.find_distance_thrown_downfield(train_tracking_df)
df_train = df_train.merge(yards_past_los_df)

#%%

### Feature engineering
####  Heights of corner/receiver
df_train = df_train.merge(df_players[['nflId', 'displayName', 'height']], left_on = 'targetNflId', right_on = 'nflId').drop(columns = 'nflId')
df_train = df_train.rename(columns = {'height': 'receiver_height',
                                      'displayName': 'receiver_name'})

df_train = df_train.merge(df_players[['nflId', 'displayName', 'height']], left_on = 'defenderNflId', right_on = 'nflId').drop(columns = 'nflId')
df_train = df_train.rename(columns = {'height': 'defender_height',
                                      'displayName': 'defender_name'})

df_train['receiver_minus_defender_height'] = df_train['receiver_height'] - df_train['defender_height']

#%%

## Modeling Individual Features and Plotting
#### See R script
df_train[['gameId', 'playId', 'targetNflId', 'receiver_name', 'defenderNflId', 'defender_name',
          'receiver_height', 'defender_height', 'receiver_minus_defender_height', 'target', 'target_descr',
          'yds_thrown_past_los', 'defender_distance_to_football', 'receiver_corner_dist_between',
          'receiver_distance_to_football']].to_csv("data/created/model_coverage_wins.csv", index = False)


#%%

## Read in tracking data iteratively
WEEKS = list(range(1,18))
tracking_dfs = []
for w in WEEKS:
    df_tracking = pd.read_csv(f"data/provided/week{w}.csv")
    df_tracking = df_tracking[(df_tracking['position'].isin(['WR', 'CB', 'QB'])) | 
                              (df_tracking['displayName'] == 'Football')]
    df_tracking = df_tracking.merge(df_games[['gameId', 'homeTeamAbbr','visitorTeamAbbr']], on = 'gameId')
    tracking_dfs.append(df_tracking)
tracking_df = pd.concat(tracking_dfs)
del df_tracking, tracking_dfs, w

#%%
### Goal-line Fade Analysis: Data Cleaning

## Define goal line plays as plays inside the offense's 10 yard line
MAX_YD_LINE = 10
goal_line_plays = df_plays[(df_plays['yardlineNumber'] <= MAX_YD_LINE) &
                           (df_plays['yardlineSide'] != df_plays['possessionTeam'])].copy()

## Only keep tracking data for these plays for this analysis
goal_line_tracking_df = df_tracking.merge(goal_line_plays, on = ['gameId', 'playId'])
del tracking_df

# Find the closest defender to all of the goal line passes
closest_defender_df = du.find_closest_defender_to_ball(goal_line_tracking_df)

## Merge closest defender and targets
goal_line_tracking_df = goal_line_tracking_df.merge(closest_defender_df, how='left', on=['gameId', 'playId'])
goal_line_tracking_df = goal_line_tracking_df.merge(df_targets, how='left', on=['gameId', 'playId'])

## Limit to certain routes (goal line fades)
ROUTES = ['GO', 'POST']
goal_line_tracking_df = goal_line_tracking_df[((goal_line_tracking_df['route'].isin(ROUTES)) &
                                              ((goal_line_tracking_df['nflId'] == goal_line_tracking_df['targetNflId']) |
                                               (goal_line_tracking_df['nflId'] == goal_line_tracking_df['defenderNflId'])))]

goal_line_target_defenders_df = (goal_line_tracking_df.groupby(
                                ['gameId', 'playId', 'targetNflId', 'defenderNflId',
                                 'distance_to_football']).size().reset_index().drop(columns=0))
goal_line_plays_df = df_plays.merge(goal_line_target_defenders_df, on = ['gameId', 'playId'])

## Merge in Receiver/defender info
df_players.columns = [c+'_receiver' if c!='nflId' else c for c in df_players.columns]
goal_line_plays_df = goal_line_plays_df.merge(df_players, how='left', left_on = 'targetNflId',
                                              right_on = 'nflId', suffixes=('', '_receiver'))
df_players.columns = [c.replace('_receiver', '_defender') if c!='nflId' else c for c in df_players.columns]
goal_line_plays_df = goal_line_plays_df.merge(df_players, how='left', left_on = 'defenderNflId',
                                              right_on = 'nflId', suffixes=('', '_defender'))
df_players.columns = [c.replace('_defender', '') for c in df_players.columns]

#%%
#### Goal-line Fade Analysis: Visualizing Results

## Visualize Receivers
receiver_cnts = (goal_line_plays_df.groupby('displayName_receiver')
                 .size().reset_index().rename(columns={0:'Goal Line Fades',
                                              'displayName_receiver': 'Receiver'})
                 .sort_values('Goal Line Fades', ascending=True))

receiver_cnts = receiver_cnts[receiver_cnts['Goal Line Fades'] >= 3]
#receiver_plt = sns.barplot(x="Receiver", y="Goal Line Fades", color = "#3182bd", data=receiver_cnts, saturation=0.5)
plt.style.use('fivethirtyeight')
plt.barh(receiver_cnts['Receiver'], receiver_cnts['Goal Line Fades'], color = "#3182bd")
plt.title('2018-19 Goal Line Fades by Receiver')
#plt.xticks(rotation=70, horizontalalignment='right')
plt.tight_layout()
plt.show()


## Visualize Defenders
defender_cnts = (goal_line_plays_df.groupby(['displayName_defender'])
                 .size().reset_index().rename(columns={0:'Fades Defended',
                                              'displayName_defender': 'Defender'})
                 .sort_values('Fades Defended', ascending=False))
defender_cnts[defender_cnts['Fades Defended'] >= 2]

### Unfortunately, it seems that the sample size is too low for this analysis

#%%
#### Goal-line Fade Analysis: Calculating Metrics (for fun, despite low sample size)

## Completion %
goal_line_plays_df['completion'] = goal_line_plays_df.apply(lambda x:
                                     1 if (x['passResult'] == 'C') 
                                     else 0 if x['passResult'] in ['I', 'IN']
                                     else None, axis=1)

goal_line_plays_df['pass_interference'] =  goal_line_plays_df['isDefensivePI'].apply(lambda x: 1 if x else 0) 


defender_pcts_df = (goal_line_plays_df
                    .groupby('displayName_defender')
                    .agg({'completion': np.mean,
                          'pass_interference': np.mean,
                          'playId': len})
                    .rename(columns={'completion':'completion_pct',
                                     'pass_interference': 'pass_interference_pct',
                                     'playId': 'targeted_cnt'}))
defender_pcts_df[defender_pcts_df['targeted_cnt'] >= 3].sort_values('completion_pct', ascending = True)

#%%
#### Receiver and Corner Clustering
#####  Goal: Find "groups" of similar receivers/corners based on physical traits (height/weight/age)
#####          TODO: - Normalize heights/weights
#####                - Add Combine metrics

## Perform clustering for corners
corner_X = df_players[df_players['position'] == 'CB'][['height', 'weight']]
corner_X['height_std'] = corner_X['height'].apply(lambda x: (x-np.mean(corner_X['height']))/np.std(corner_X['height']))
corner_X['weight_std'] = corner_X['weight'].apply(lambda x: (x-np.mean(corner_X['weight']))/np.std(corner_X['weight']))
kmeans_corner = KMeans(n_clusters=3, random_state=0).fit(corner_X[['height_std', 'weight_std']])
corner_X['corner_cluster'] = kmeans_corner.labels_
## Dedup
corner_X = corner_X.groupby(['height', 'weight', 'corner_cluster']).size().reset_index().drop(columns=0)
df_players_corner_cluster = df_players[df_players['position'] == 'CB'].merge(corner_X, on = ['height', 'weight'])
#df_players_corner_cluster['corner_cluster'] = df_players_corner_cluster['corner_cluster'].apply(lambda x:
#                                                     "Big" if x == 1 else "Small" if x==0 else "Medium")


plt.scatter(df_players['height'], df_players['weight'])
plt.title('All Players Height vs. Weight')
plt.show()

sns.scatterplot(x='height', y='weight', hue = 'corner_cluster', palette="deep",
            data = df_players_corner_cluster)
plt.title('Cornerbacks Height vs. Weight')
plt.show()

## Perform clustering for receivers
receiver_X = df_players[df_players['position'] == 'WR'][['height', 'weight']]
receiver_X['height_std'] = receiver_X['height'].apply(lambda x: (x-np.mean(receiver_X['height']))/np.std(receiver_X['height']))
receiver_X['weight_std'] = receiver_X['weight'].apply(lambda x: (x-np.mean(receiver_X['weight']))/np.std(receiver_X['weight']))
kmeans_receiver = KMeans(n_clusters=3, random_state=0).fit(receiver_X[['height_std', 'weight_std']])
receiver_X['receiver_cluster'] = kmeans_receiver.labels_
## Dedup
receiver_X = receiver_X.groupby(['height', 'weight', 'receiver_cluster']).size().reset_index().drop(columns=0)
df_players_receiver_cluster = df_players[df_players['position'] == 'WR'].merge(receiver_X, on = ['height', 'weight'])
df_players_receiver_cluster['receiver_cluster'] = df_players_receiver_cluster['receiver_cluster'].apply(lambda x:
                                                     "Big" if x == 1 else "Small" if x==2 else "Medium")

sns.scatterplot(x='height', y='weight', hue = 'receiver_cluster', palette="deep",
            data = df_players_receiver_cluster)
plt.title('Receivers Height/Weight Clusters')
plt.show()

df_wr_corner_clusters = pd.concat([df_players_receiver_cluster, df_players_corner_cluster])
df_players_w_clusters = df_players.merge(df_wr_corner_clusters[['nflId', 'receiver_cluster', 'corner_cluster']], on = 'nflId', how = 'left')

#%%
#### Receiver and corner height groupings based on Ptiles
receiver_low = np.percentile(receiver_X['height'], (1.00/3.00)*100)
receiver_hgh = np.percentile(receiver_X['height'], (2.00/3.00)*100)
corner_low = np.percentile(corner_X['height'], (1.00/3.00)*100)
corner_hgh = np.percentile(corner_X['height'], (2.00/3.00)*100)

df_players_w_clusters['WR_height_group'] = df_players_w_clusters.apply(lambda x:
                                        None if x['position'] != 'WR' else
                                        'Big' if x['height'] > receiver_hgh else
                                        'Small' if x['height'] <= receiver_low else
                                        'Medium', axis=1)

wr_clusters_df = df_players_w_clusters[df_players_w_clusters['position'] == 'WR']
viz.create_height_dotplot(wr_clusters_df, "WR")

    
df_players_w_clusters['CB_height_group'] = df_players_w_clusters.apply(lambda x:
                                        None if x['position'] != 'CB' else
                                        'Big' if x['height'] >= corner_hgh else
                                        'Small' if x['height'] < corner_low else
                                        'Medium', axis=1)
    
    
cb_clusters_df = df_players_w_clusters[df_players_w_clusters['position'] == 'CB']
viz.create_height_dotplot(cb_clusters_df, "CB")

#%%
#### Find route/receiver type/corner type combinations that lead to high/low EPA
##### Possible viz: Heatmap split by ReceiverType/Route on X axis and Corner Type on Y Axis, 
##                  colored by EPA, transparency is sample size

# Find the closest defender to all of the goal line passes
wr_cb_tracking_df = tracking_df.merge(df_plays, on = ['gameId', 'playId'])
del tracking_df
closest_defender_df = du.find_closest_defender_to_ball(wr_cb_tracking_df)

## Merge closest defender and targets
wr_cb_tracking_df = wr_cb_tracking_df.merge(closest_defender_df, how='left', on=['gameId', 'playId'])
wr_cb_tracking_df = wr_cb_tracking_df.merge(df_targets, how='left', on=['gameId', 'playId'])

## Limit to players of interest to save space
wr_cb_tracking_df = wr_cb_tracking_df[((wr_cb_tracking_df['nflId'] == wr_cb_tracking_df['targetNflId']) |
                                     (wr_cb_tracking_df['nflId'] == wr_cb_tracking_df['defenderNflId']))]

## Compile info down to the play level
wr_cb_routes = (wr_cb_tracking_df.groupby(['gameId', 'playId', 'targetNflId', 'defenderNflId', 'distance_to_football',
                                          'route']).size().reset_index().rename(columns={0:'count'}))
del wr_cb_tracking_df

## Merge extra info back into play-level DF
wr_cb_plays_df = df_plays.merge(wr_cb_routes, on = ['gameId', 'playId'])

## Completion %
wr_cb_plays_df['completion'] = wr_cb_plays_df.apply(lambda x:
                                     1 if (x['passResult'] == 'C') 
                                     else 0 if x['passResult'] in ['I', 'IN']
                                     else None, axis=1)

## Pass interference %
wr_cb_plays_df['pass_interference'] =  wr_cb_plays_df['isDefensivePI'].apply(lambda x: 1 if x else 0) 

## Merge in Receiver/defender info
df_players_w_clusters.columns = [c+'_receiver' if c not in ['nflId', 'receiver_cluster', 'corner_cluster']
                      else c for c in df_players_w_clusters.columns]
wr_cb_plays_df = wr_cb_plays_df.merge(df_players_w_clusters[['nflId', 'displayName_receiver', 'receiver_cluster']],
                                              how='left', left_on = 'targetNflId',
                                              right_on = 'nflId')
df_players_w_clusters.columns = [c.replace('_receiver', '_defender') if c not in ['nflId', 'receiver_cluster', 'corner_cluster']
                      else c for c in df_players_w_clusters.columns]
wr_cb_plays_df = wr_cb_plays_df.merge(df_players_w_clusters[['nflId', 'displayName_defender', 'corner_cluster']],
                                              how='left', left_on = 'defenderNflId',
                                              right_on = 'nflId')
df_players_w_clusters.columns = [c.replace('_defender', '') for c in df_players_w_clusters.columns]

## Group by receiver type/corner type/route and calculate EPA and count
route_clusters_epa_df = (wr_cb_plays_df
                        .groupby(['receiver_cluster', 'corner_cluster', 'route'])
                        .agg({'epa': np.mean,
                              'completion': np.mean,
                              'pass_interference': np.mean,
                              'playId': len})
                        .reset_index()
                        .rename(columns={'epa': 'avg_epa',
                                         'completion':'completion_pct',
                                         'pass_interference': 'pass_interference_pct',
                                         'playId': 'targeted_cnt'})
                        .sort_values('avg_epa'))

#%%
#### Visualizations of EPA by Route/Receiver Type/Corner Type

##### All combinations with > 50 targets
decent_n_routes = route_clusters_epa_df[route_clusters_epa_df['targeted_cnt'] > 50]

decent_n_pivot = pd.pivot_table(data=decent_n_routes,
                    index='corner_cluster',
                    values='avg_epa',
                    columns=['receiver_cluster', 'route'])

g = sns.heatmap(decent_n_pivot, cmap = 'Blues')
g.set_facecolor('gray')
plt.title('EPA By Route and Receiver/Corner Size\nwith >50 Targets')
plt.show()

##### Top 5 route types (and >50 targets)
top_5_routes = list(wr_cb_plays_df.route.value_counts().index)[0:5]
top_5_routes_df = decent_n_routes[decent_n_routes['route'].isin(top_5_routes)]

top_5_rts_pivot = pd.pivot_table(data=top_5_routes_df,
                    index='corner_cluster',
                    values='avg_epa',
                    columns=['receiver_cluster', 'route'])

g = sns.heatmap(top_5_rts_pivot, cmap = 'Blues')
g.set_facecolor('gray')
plt.title('EPA By Route and Receiver/Corner Size\nwith >50 Targets, for 5 most common routes')
plt.show()

### Removing Receiver Cluster from top 5 routes heatmap
top_5_rts_no_wr_pivot = pd.pivot_table(data=top_5_routes_df,
                    index='corner_cluster',
                    values='avg_epa',
                    columns='route')

g = sns.heatmap(top_5_rts_no_wr_pivot, cmap = 'Blues')
g.set_facecolor('gray')
plt.title('EPA By Route and Corner Size\nwith >50 Targets, for 5 most common routes')
plt.show()

#%%

wr_cb_plays_ht_df = wr_cb_plays_df.merge(df_players_w_clusters[['nflId', 'WR_height_group']],
                                              how='left', left_on = 'targetNflId',
                                              right_on = 'nflId')
wr_cb_plays_ht_df = wr_cb_plays_ht_df.merge(df_players_w_clusters[['nflId', 'CB_height_group']],
                                              how='left', left_on = 'defenderNflId',
                                              right_on = 'nflId')

## Group by receiver type/corner type/route and calculate EPA and count
route_ht_grps_epa_df = (wr_cb_plays_ht_df
                        .groupby(['WR_height_group', 'CB_height_group', 'route'])
                        .agg({'epa': np.mean,
                              'completion': np.mean,
                              'pass_interference': np.mean,
                              'playId': len})
                        .reset_index()
                        .rename(columns={'epa': 'avg_epa',
                                         'completion':'completion_pct',
                                         'pass_interference': 'pass_interference_pct',
                                         'playId': 'targeted_cnt'})
                        .sort_values('avg_epa'))

##### All combinations with > 30 targets
decent_n_routes = route_ht_grps_epa_df[route_ht_grps_epa_df['targeted_cnt'] > 30]

decent_n_pivot = pd.pivot_table(data=decent_n_routes,
                    index='CB_height_group',
                    values='avg_epa',
                    columns=['WR_height_group', 'route'])

g = sns.heatmap(decent_n_pivot, cmap = 'Greens')
g.set_facecolor('gray')
plt.title('EPA By Route and Receiver/Corner Size\nwith >50 Targets')
plt.show()

##### Top 5 route types (and >50 targets)
top_5_routes = list(wr_cb_plays_ht_df.route.value_counts().index)[0:5]
top_5_routes_df = decent_n_routes[decent_n_routes['route'].isin(top_5_routes)]

top_5_rts_pivot = pd.pivot_table(data=top_5_routes_df,
                    index='CB_height_group',
                    values='avg_epa',
                    columns=['WR_height_group', 'route'])

g = sns.heatmap(top_5_rts_pivot, cmap = 'Greens')
g.set_facecolor('gray')
plt.title('EPA By Route and Receiver/Corner Size\nwith >50 Targets, for 5 most common routes')
plt.show()

### Removing Receiver Cluster from top 5 routes heatmap
top_5_rts_no_wr_pivot = pd.pivot_table(data=top_5_routes_df,
                    index='CB_height_group',
                    values='avg_epa',
                    columns='route')

g = sns.heatmap(top_5_rts_no_wr_pivot, cmap = 'Greens')
g.set_facecolor('gray')
plt.title('EPA By Route and Corner Size\nwith >50 Targets, for 5 most common routes')
plt.show()

#%%
decent_n_pivot = pd.pivot_table(data=decent_n_routes,
                    index='CB_height_group',
                    values='completion_pct',
                    columns=['WR_height_group', 'route'])

g = sns.heatmap(decent_n_pivot, cmap = 'Greens')
g.set_facecolor('gray')
plt.title('Completion % By Route and Receiver/Corner Size\nwith >50 Targets')
plt.show()

##### Top 5 route types (and >50 targets)
top_5_routes = list(wr_cb_plays_ht_df.route.value_counts().index)[0:5]
top_5_routes_df = decent_n_routes[decent_n_routes['route'].isin(top_5_routes)]

top_5_rts_pivot = pd.pivot_table(data=top_5_routes_df,
                    index='CB_height_group',
                    values='completion_pct',
                    columns=['WR_height_group', 'route'])

g = sns.heatmap(top_5_rts_pivot, cmap = 'Greens')
g.set_facecolor('gray')
plt.title('Completion % By Route and Receiver/Corner Size\nwith >50 Targets, for 5 most common routes')
plt.show()

##### Deep route types (and >50 targets)
deep_routes = ['GO', 'POST']
deep_routes_df = decent_n_routes[decent_n_routes['route'].isin(deep_routes)]

deep_rts_pivot = pd.pivot_table(data=deep_routes_df,
                    index='CB_height_group',
                    values='completion_pct',
                    columns=['WR_height_group', 'route'])

g = sns.heatmap(deep_rts_pivot, cmap = 'Greens')
g.set_facecolor('gray')
plt.title('Completion % By Route and Receiver/Corner Size\nwith >50 Targets, for GO/POST routes')
plt.show()


### Removing Receiver Cluster from top 5 routes heatmap
top_5_rts_no_wr_pivot = pd.pivot_table(data=top_5_routes_df,
                    index='CB_height_group',
                    values='completion_pct',
                    columns='route')

g = sns.heatmap(top_5_rts_no_wr_pivot, cmap = 'Greens')
g.set_facecolor('gray')
plt.title('Completion % By Route and Corner Size\nwith >50 Targets, for 5 most common routes')
plt.show()



#%%
"""
Other Ideas:
    1. Which players are the best at closely tracking receivers as they try to get open?
      - Co-travel analytics
      - In other words: which corners are best at minimizing "separation" throughout a receiver's route?
		 ○ Note: may need to isolate to man coverage plays here
      - Could we use "clusters" above to identify which physical traits allow DBs to track certain receivers better?
      - IMPACT: Inform DC/DB coaches when game planning matchups between their DBs and opponents receivers. 
    
"""

#%%

## Determine which side of the ball each receiver and CB are on, from the offense's perspective
##  At the start of the play

tracking_df = tracking_df[tracking_df['event']=='ball_snap'].copy()
tracking_df = du.orient_coords(tracking_df)

## Determine the side of the ball that each receiver/corner is on
### Get a dataframe with the position of the ball.. marking any time that the ball location is wonky
ball_position_df = (tracking_df[tracking_df['displayName'] == 'Football']
                  .groupby(['gameId', 'playId', 'x', 'y']).size().reset_index().drop(columns=0)
                  .rename(columns = {'x': 'football_x',
                           'y': 'football_y'}))
ball_position_df['football_x'] = ball_position_df.apply(lambda x: 'Invalid' if ((x['football_y'] < 0) or 
                                                                      (x['football_y'] > 53.3))
                                                         else x['football_x'], axis=1)
ball_position_df['football_y'] = (ball_position_df['football_y']
                                  .apply(lambda y: 'Invalid' if ((y < 0) or (y > 53.3))
                                         else y))

### Get a dataframe with the position of the QB (in case ball location sucks).. marking any time that the QB location is wonky
qb_position_df = (tracking_df[tracking_df['position'] == 'QB']
                  .groupby(['gameId', 'playId', 'x', 'y']).size().reset_index().drop(columns=0)
                  .rename(columns = {'x': 'qb_x',
                           'y': 'qb_y'}))
qb_position_df['qb_x'] = qb_position_df.apply(lambda x: 'Invalid' if ((x['qb_y'] < 0) or 
                                                                      (x['qb_y'] > 53.3))
                                                         else x['qb_x'], axis=1)
qb_position_df['qb_y'] = (qb_position_df['qb_y']
                           .apply(lambda y: 'Invalid' if ((y < 0) or (y > 53.3))
                                            else y))

## Make a data frame with the furthest left and right locations of CBs and WRs
## To determine whether a player is out wide or in the "slot" somewhere between out wide and the ball
wr_cb_max_min_ys = (tracking_df[tracking_df['position'].isin(['CB', 'WR'])]
                    .groupby(['gameId', 'playId', 'position'])
                    .aggregate({'y': [max, min]}))
wr_cb_max_min_ys.columns = ['_'.join(col) for col in wr_cb_max_min_ys.columns]
wr_cb_max_min_ys = wr_cb_max_min_ys.reset_index()


### Noticing a few plays where the ball is outside the field throughout the duration of the play:
     ### game 2018123001, play 435
     ### game 2018093006, play 623
### Calling these "invalid" and using the QB position to determine side of the ball for WRs/CBs
#### Only plays this won't work with plays with bad ball sensor locations that are also Wildcat plays with a QB out wide
tracking_df = tracking_df.merge(ball_position_df)
tracking_df = tracking_df.merge(qb_position_df)
tracking_df = tracking_df.merge(wr_cb_max_min_ys)
tracking_df['side_of_ball'] = tracking_df.apply(lambda x: None if (x['displayName'] == 'Football' 
                                                                   or x['football_y'] == 'Invalid')
                                                else 'Left' if x['y'] > x['football_y']
                                                else 'Right' if x['y'] < x['football_y']
                                                else 'On-ball', axis=1)
tracking_df['side_of_ball'] = tracking_df.apply(lambda x: None if (x['football_y'] == 'Invalid'
                                                                   and x['qb_y'] == 'Invalid')
                                                else 'Left' if (x['football_y'] == 'Invalid'
                                                                and x['y'] > x['qb_y'])
                                                else 'Right' if (x['football_y'] == 'Invalid'
                                                                and x['y'] < x['qb_y'])
                                                else 'On-ball' if (x['football_y'] == 'Invalid'
                                                                and x['y'] == x['qb_y'])
                                                else x['side_of_ball'], axis=1)

tracking_df['out_wide_ind'] = tracking_df.apply(lambda x: 1 if (x['y'] == x['y_max'] or
                                                                x['y'] == x['y_min'])
                                                     else 0, axis=1)

tracking_df['right_side_ind'] = (tracking_df['side_of_ball']
                                 .apply(lambda x: 1 if x == 'Right' else 0) )
tracking_df['left_side_ind'] = (tracking_df['side_of_ball']
                                 .apply(lambda x: 1 if x == 'Left' else 0) )

wr_cb_right_left_side_pcts = (tracking_df[tracking_df['position'].isin(['CB', 'WR'])]
                              .groupby('nflId')
                              .aggregate({'right_side_ind': [np.mean],
                                          'left_side_ind': [np.mean],
                                          'out_wide_ind': [np.mean, len]}).reset_index())
wr_cb_right_left_side_pcts.columns = wr_cb_right_left_side_pcts.columns.droplevel(1)
wr_cb_right_left_side_pcts.columns = ['nflId', 'pct_right', 'pct_left', 'pct_out_wide', 'n_plays']

wr_cb_right_left_side_pcts = wr_cb_right_left_side_pcts.merge(df_players[['nflId', 'displayName', 'position']])

#%%
"""

This cell seeks to measure which corners are best at keeping minimal space between themselves and receivers
throughout the course of a play.

Methodology:
    - Find the closest offensive player (their "matchup") to each corner at the start of a play
    - Measure distance between corner and their "matchup" for each frame throughout the play
    - Calculate summary distance metrics
        - Average distance throughout the play
        - "Closeout" distance (difference between distance at time of pass arrival and time of pass release)
    - Measure correlation between distance metrics above and performance metrics (EPA, completion % allowed, etc.)
    - Compare corners

Drawbacks:
    - Closest offensive player is not always the person that the corner is responsible for guarding throughout the play
      - Ex. zone coverage and offensive formations with no receivers out wide
    - 

"""

### Source tracking data
tracking_df = pd.read_csv(r"C:\Users\mtdic\Documents\GitHub\big_data_bowl_2021\data\week15_kc_lac_tracking.csv") ## subset

## Read in full data iteratively
#os.chdir(r"E:\NFL\big_data_bowl\2021")
#WEEKS = list(range(1,18))
#tracking_dfs = []
#for w in WEEKS:
#    df_tracking = pd.read_csv(f"data/provided/week{w}.csv")
#    df_tracking = df_tracking[(df_tracking['position'].isin(['WR', 'CB', 'QB'])) | 
#                              (df_tracking['displayName'] == 'Football')]
#    df_tracking = df_tracking.merge(df_games[['gameId', 'homeTeamAbbr','visitorTeamAbbr']], on = 'gameId')
#    tracking_dfs.append(df_tracking)
#tracking_df = pd.concat(tracking_dfs)
#del df_tracking, tracking_dfs, w


tracking_df = tracking_df.merge(df_plays[['gameId', 'playId', 'possessionTeam', 'penaltyCodes',
                                          'epa', 'passResult', 'playResult', 'offensePlayResult',
                                          'isDefensivePI']])
tracking_df = tracking_df.merge(df_games[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']])


## Get the presnap opponents
presnap_opponent_df = du.find_closest_presnap_opponents(tracking_df, position = 'CB')

### Compare the trajectories for the CB and person they were lined up closest to
full_traj_dfs = []
traj_summary_dfs = []
for i, row in presnap_opponent_df.iterrows():
    full_traj_df, traj_summary_df = (du.compare_trajectories(tracking_df,
                                                          row['gameId'],
                                                          row['playId'],
                                                          row['nflId'],
                                                          row['nflId_opponent']))
    full_traj_dfs.append(full_traj_df)
    traj_summary_dfs.append(traj_summary_df)
traj_summary_full_df = pd.concat(traj_summary_dfs).reset_index().drop(columns='index')
all_full_traj_df = pd.concat(full_traj_dfs).reset_index().drop(columns='index')

## Merge trajectory summaries with targets
traj_summary_full_df = traj_summary_full_df.merge(df_targets,
                                                  left_on=['gameId', 'playId', 'nflId_off'],
                                                  right_on = ['gameId', 'playId', 'targetNflId'],
                                                  how = 'left')

## Sometimes the target dataframe lists the QB as the targeted receiver when a sack happens.. let's remove those.
traj_summary_full_df['targetNflId'] = traj_summary_full_df.apply(lambda x: np.nan if x['passResult'] == 'S'
                                                                 else x['targetNflId'], axis = 1)
traj_summary_full_df['isTargeted'] = traj_summary_full_df['targetNflId'].apply(lambda x: ~np.isnan(x))

## Useful column to calcuate completion % while disregarding penalty plays
traj_summary_full_df['completion_ind'] = traj_summary_full_df.apply(lambda x: 1 if x['passResult'] == 'C'
                                                                    else np.nan if ((x['passResult'] == 'S') or
                                                                                 ('DPI' in x['penaltyCodes'].split(';')) or
                                                                                 ('ICT' in x['penaltyCodes'].split(';')) or
                                                                                 ('DH' in x['penaltyCodes'].split(';')) or 
                                                                                 ('DOF') in x['penaltyCodes'].split(';') or
                                                                                 ( (x['offensePlayResult'] == 0) and (x['playResult'] > 0)))
                                                                    else 0 if x['passResult'] in ['IN', 'I']
                                                                    else np.nan, axis = 1)

## Useful column to calculate penalties on defensive coverage (TODO: figure out if penalty is called on the player being evaluated)
traj_summary_full_df['cov_penalty_ind'] = traj_summary_full_df.apply(lambda x: 1 if (('DPI' in x['penaltyCodes'].split(';')) or
                                                                                     ('ICT' in x['penaltyCodes'].split(';')) or
                                                                                     ('DH' in x['penaltyCodes'].split(';')))
                                                                                    else 0, axis = 1)


## Merge defensive player name
df_defensive_player = df_players[['nflId', 'displayName']].rename(columns={'displayName': 'displayNameDefense'})
traj_summary_full_df = traj_summary_full_df.merge(df_defensive_player, left_on = 'nflId_def', right_on = 'nflId').drop(columns='nflId')

## Merge offensive player name
df_offensive_player = df_players[['nflId', 'displayName']].rename(columns={'displayName': 'displayNameOffense'})
traj_summary_full_df = traj_summary_full_df.merge(df_offensive_player, left_on = 'nflId_off', right_on = 'nflId').drop(columns='nflId')

coverage_summary = (traj_summary_full_df[traj_summary_full_df['isTargeted']].groupby(['displayNameDefense'])
                                                        .agg({'epa': np.mean,
                                                              'playId': len,
                                                              'completion_ind': np.nanmean,
                                                              'cov_penalty_ind': np.mean}).sort_values('epa')
                                                        .rename(columns = {'playId': 'n_targets',
                                                                           'epa': 'avg_epa',
                                                                           'completion_ind': 'completion_pct',
                                                                           'cov_penalty_ind': 'cov_penalty_pct'}))

traj_summary_full_df.to_csv(r"C:\Users\mtdic\Documents\GitHub\big_data_bowl_2021\data\kc_lac_traj_summary_12_2018.csv", index = False)