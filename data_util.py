# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:16:43 2020

@author: mtdic
"""

import os
import re
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.cluster import KMeans, DBSCAN
import seaborn as sns

#defining events that designate pass arrival
PASS_ARRIVAL_EVENTS = ['pass_outcome_caught',
                      'pass_arrived',
                      'pass_outcome_incomplete',
                      'pass_outcome_interception',
                      'pass_outcome_touchdown']

def orient_coords(df):
    """
    Standardizing tracking data so its always in direction of offense instead of 
     raw on-field coordinates.
    
    Parameters
    ----------
    df : DataFrame
        Player tracking dataframe.

    Returns
    -------
    df : DataFrame
        Player tracking dataframe with standardized coordinate x/y fields
         (original field retained as x_orig/y_orig).

    """
    
    ## Retain old fields
    df['x_orig'] = df['x']
    df['y_orig'] = df['y']
    
    ## Flip the field if they're going to the "left"
    df['x'] = df.apply(lambda x: 120-x['x'] if x['playDirection'] == "left" else x['x'], axis=1)
    df['y'] = df.apply(lambda x: 160/3-x['y'] if x['playDirection'] == "left" else x['y'], axis=1)
    
    return df


def find_closest_defender_to_ball(df_plays_tracking_merge):
    """
    Find the closest defender to the ball at the time the ball arrives for a 
     DataFrame of plays/tracking data.
    
    Parameters
    ----------
    df_plays_tracking_merge : DataFrame
        DataFrame with play and tracking info.
    
    Returns
    -------
    df_defender : DataFrame
        DataFrame with closest defender to the ball at time of ball arrival.
    
    """
    
    ## Adding several columns to be used in calculations
    df_plays_tracking_merge['side_of_ball'] = df_plays_tracking_merge.apply(lambda x:
                                                 "offense" if (((x['team'] == "home") &
                                                               (x['possessionTeam'] == x['homeTeamAbbr']))
                                                           |  ((x['team'] == "away") &
                                                               (x['possessionTeam'] == x['visitorTeamAbbr'])))
                                                 else "football" if x['team'] == 'football' 
                                                 else "defense", axis=1)
    
    df_plays_tracking_merge['defensive_team'] = df_plays_tracking_merge.apply(lambda x:
                                                 x['visitorTeamAbbr'] if x['possessionTeam'] == x['homeTeamAbbr']
                                                 else x['homeTeamAbbr'], axis=1)
    
    df_plays_tracking_merge['football_in_play'] = df_plays_tracking_merge.apply(lambda x:
                                                 1 if x['displayName'] == "Football"
                                                 else 0, axis=1)    
    PASS_ARRIVAL_EVENTS = ['pass_outcome_caught',
                          'pass_arrived',
                          'pass_outcome_incomplete',
                          'pass_outcome_interception',
                          'pass_outcome_touchdown']
    
    df_plays_tracking_merge['pass_arrived'] = df_plays_tracking_merge['event'].apply(lambda x: 1 if x 
                                                                            in PASS_ARRIVAL_EVENTS else 0)
    
    
    
    ## Creating a DF to be merged in containing x,y position of football throughout each play
    football_in_play_df = (df_plays_tracking_merge.groupby(["gameId", "playId", "frameId"])
                                   .agg({'football_in_play': np.sum})
                                   .reset_index())
    football_in_play_df = football_in_play_df[football_in_play_df['football_in_play'] > 0]
    
    football_position_df = (df_plays_tracking_merge[df_plays_tracking_merge['displayName'] == 'Football']
                               [['gameId','playId','frameId','x','y']]
                               .merge(football_in_play_df, on=["gameId", "playId", "frameId"])
                               .rename(columns = {'x': 'football_x',
                                                  'y': 'football_y'}))
    
    pass_arrival_frame_df = (df_plays_tracking_merge[df_plays_tracking_merge['pass_arrived'] == 1]
                               .groupby(["gameId", "playId"]).agg({'frameId':np.min}).reset_index())
    
    ### Create a DF with distances to football at that event
    distance_to_football_df = df_plays_tracking_merge.merge(pass_arrival_frame_df, on=["gameId", "playId", "frameId"])
    distance_to_football_df = distance_to_football_df.merge(football_position_df, on=["gameId", "playId", "frameId"])
    distance_to_football_df['distance_to_football'] = distance_to_football_df.apply(lambda x: 
                                                                        np.sqrt( (x['x'] - x['football_x'])**2 +
                                                                         (x['y'] - x['football_y'])**2), axis=1)
    
    ## Select the records with the minimum distance and get the player info
    min_distance_df = (distance_to_football_df[(distance_to_football_df['side_of_ball'] == "defense")]
                       .groupby(["gameId", "playId"]).agg({'distance_to_football': np.min}).reset_index())
    distance_to_football_df = (distance_to_football_df[(distance_to_football_df['side_of_ball'] == "defense")]
                                .merge(min_distance_df, on=["gameId", "playId", "distance_to_football"]))
    distance_to_football_df = (distance_to_football_df[["gameId", "playId", "nflId", "football_x", "football_y",
                                                       "distance_to_football"]].rename(columns={'nflId':'defenderNflId'}))
    
    return distance_to_football_df


def convert_birthdates_to_ages(birthdates):
    """
    Standardize a list of dates in various formats and create a field for 

    Parameters
    ----------
    birthdates : list of strings
        Mixed dates formats as strings (birthDate field from the "players.csv" provided file.)

    Returns
    -------
    std_birthdates, ages : list of datetime, list of float
        A list of standardized birthdates and a list of ages

    """
    end_of_2018_season = datetime.strptime("2018-12-31", '%Y-%m-%d')
    
    regex_yyyy_mm_dd = re.compile(r"\d{4}-\d{2}-\d{2}")
    regex_mm_dd_yyyy = re.compile(r"\d{2}\/\d{2}\/\d{4}")
    
    std_birthdates = []
    ages = []
    for b in birthdates:
        if re.match(regex_yyyy_mm_dd, b):
            std_bday = datetime.strptime(b, "%Y-%m-%d")
            age = abs((end_of_2018_season - std_bday).days)/365.00
        elif re.match(regex_mm_dd_yyyy, b):
            std_bday = datetime.strptime(b, "%m/%d/%Y")
            age = abs((end_of_2018_season - std_bday).days)/365.00
        else:
            age = None
            std_bday = None
        std_birthdates.append(std_bday)
        ages.append(age)
        
    
    return std_birthdates, ages


def standardize_heights(heights):
    """
    Standardize a list of dates in various formats and create a field for 

    Parameters
    ----------
    heights : list of strings
        Mixed height formats as strings (e.g. 6'4"/6-4 or 76 inches)

    Returns
    -------
    heights_std : list of ints
        A list of standardized heights (in inches) as an integer

    """
    
    heights_std = []
    for h in heights:
        if ('-' in h) or ("'" in h):
            if '-' in h:
                char = '-'
            elif "'" in h:
                char = "'"
            feet_std = int(h.split(char)[0])
            inches_std = int(h.split('-')[1])
            height_std = feet_std*12+inches_std
            heights_std.append(height_std)
        else:
            heights_std.append(int(h))
    
    return heights_std


def read_tracking_for_player(nflId):
    """
    Read in player tracking data for a particular player.

    Parameters
    ----------
    nflId : int
        ID number for a player assigned by the NFL.

    Returns
    -------
    tracking_df : DataFrame
        Player tracking data for a specific player.
    """
    
    os.chdir(r"E:/NFL/big_data_bowl/2021")
    
    ## Read in tracking data iteratively, subset to player and ball
    WEEKS = list(range(1,18))
    tracking_dfs = []
    for w in WEEKS:
        df_tracking = pd.read_csv(f"data/provided/week{w}.csv")
        df_tracking = df_tracking[(df_tracking['nflId'] == nflId) | 
                                  (df_tracking['position'] == 'QB') | ## Getting QB in case ball position is wrong
                                  (df_tracking['displayName'] == 'Football')]
        tracking_dfs.append(df_tracking)
    tracking_df = pd.concat(tracking_dfs)
    
    return tracking_df


def get_start_position_df(nflId, tracking_df):
    """
    Get limit the tracking data to the time of the ball snap and determine side of the ball

    Parameters
    ----------
    nflId : int
        ID number for a player assigned by the NFL.

    Returns
    -------
    tracking_df : DataFrame
        Player tracking data for a specific player at the time of ball snap.

    """
    tracking_df = tracking_df[tracking_df['event']=='ball_snap'].copy()
    tracking_df = orient_coords(tracking_df)
    
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
    
    ### Noticing a few plays where the ball is outside the field throughout the duration of the play:
         ### game 2018123001, play 435
         ### game 2018093006, play 623
    ### Calling these "invalid" and using the QB position to determine side of the ball for WRs/CBs
    #### Only plays this won't work with plays with bad ball sensor locations that are also Wildcat plays with a QB out wide
    tracking_df = tracking_df.merge(ball_position_df)
    tracking_df = tracking_df.merge(qb_position_df)
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
        
    tracking_df['right_side_ind'] = (tracking_df['side_of_ball']
                                     .apply(lambda x: 1 if x == 'Right' else 0) )
    tracking_df['left_side_ind'] = (tracking_df['side_of_ball']
                                     .apply(lambda x: 1 if x == 'Left' else 0) )
    
    tracking_df['x_minus_ball_x'] = tracking_df.apply(lambda x: x['x']-x['football_x'] if x['football_x'] != 'Invalid'
                                                      else None, axis=1)
    
    tracking_df = tracking_df[tracking_df['nflId']==nflId].copy()
    
    return tracking_df


def get_top_speeds(position = 'DB'):
    """

    Parameters
    ----------
    position : str, optional
        Player position to gather top speeds for. The default is 'DB', which gets all CBs and safeties.

    Returns
    -------
    top_speeds_df : TYPE
        DESCRIPTION.

    """
    os.chdir(r"E:/NFL/big_data_bowl/2021")
    
    if position == 'DB':
        positions = ['CB', 'DB', 'FS', 'SS', 'S']
    else:
        positions = [position]
    
    ## Read in tracking data iteratively, subset to player and ball
    WEEKS = list(range(1,18))
    top_speeds_dfs = []
    for w in WEEKS:
        df_tracking = pd.read_csv(f"data/provided/week{w}.csv")
        df_tracking = df_tracking[(df_tracking['position'].isin(positions)) &
                                  (df_tracking['s'] < 20) &
                                  ((df_tracking['gameId'] != 2018102101) &
                                   (df_tracking['nflId'] != 2495775.0))]  ## remove impossible speeds and anomalous Eric Weddle game
        top_speeds_df = (df_tracking.query("s < 13")
                                    .groupby('nflId').agg({'s': max,
                                                           'a': max,
                                                           'playId': 'nunique'})
                                    .reset_index()
                                    .rename(columns = {'s': 'max_speed',
                                                       'a': 'max_accel',
                                                       'playId': 'n_plays'}))
        top_speeds_dfs.append(top_speeds_df)
    top_speeds_concat = pd.concat(top_speeds_dfs)
    top_speeds_df = (top_speeds_concat.groupby('nflId').agg({'max_speed': max,
                                                            'max_accel': max,
                                                            'n_plays': sum})
                                    .reset_index())
    
    return top_speeds_df


class DBIsland:
    def __init__(self, nflId):
        self.nflId = nflId
        self.player_tracking_df = read_tracking_for_player(nflId)
        self.start_position_df = get_start_position_df(nflId, self.player_tracking_df)
        self.start_position_clusters = None
    
    def cluster_start_positions(self, eps = 1, min_samples=5):
        db_clusters = (DBSCAN(eps=eps, min_samples=min_samples)
                       .fit(self.start_position_df[['x_minus_ball_x', 'y']]))
        
        self.start_position_df['cluster'] = db_clusters.labels_
        
        self.start_position_clusters = (self.start_position_df[self.start_position_df['cluster'] != -1]
                                         .groupby(['cluster', 'side_of_ball'])
                                         .agg({'x_minus_ball_x': np.mean,
                                               'y': np.mean,
                                               'playId': len})
                                         .reset_index()
                                         .rename(columns = {'x_minus_ball_x': 'cluster_center_x',
                                                            'y': 'cluster_center_y',
                                                            'playId': 'n_plays'}))
    
    def plot_start_position_clusters(self):
        
        if self.start_position_clusters is None:
            self.cluster_start_positions()
        
        self.start_position_df['negative_y'] = self.start_position_df['y']*-1
        
        ### Make the scatterplot
        sns.scatterplot(data=self.start_position_df,
                        y='x_minus_ball_x', x='negative_y',
                        hue = 'cluster', palette="deep")
        