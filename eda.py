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

from sklearn.cluster import KMeans

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
std_birthdates, ages = du.convert_birthdates_to_ages(df_players['birthDate'])
df_players['birthDate'] = std_birthdates
df_players['age'] = ages

### BONUS Data: 
df_targets = pd.read_csv("data/provided/targetedReceiver.csv")

#%%


## Read in tracking data iteratively, filtering to goal line plays
WEEKS = list(range(1,18))
tracking_dfs = []
for w in WEEKS:
    df_tracking = pd.read_csv(f"data/provided/week{w}.csv")
    df_tracking = df_tracking[(df_tracking['position'].isin(['WR', 'CB'])) | 
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

## Only keep tracking data for this analysis
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
      - Could we use "clusters" above to identify which physical traits allow DBs to track certain receivers better?
      - IMPACT: Inform DC/DB coaches when game planning matchups between their DBs and opponents receivers. 
    
"""