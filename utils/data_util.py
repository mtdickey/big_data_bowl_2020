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


def find_distances_at_arrival(df):
    """
    Find the closest defender to the ball at the time the ball arrives for a
     DataFrame of plays/tracking data.  Measure distances between defender and ball, 
     targeted receiver and ball, and defender and targeted receiver.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with tracking, play, game, and targets info (see training dataframe creation cells).
    
    Returns
    -------
    df_defender : DataFrame
        DataFrame with closest defender and receiver to the ball at time of ball arrival.
    
    """
    
    ## Adding column to be used in calculations
    df['pass_arrived'] = df['event'].apply(lambda x: 1 if x in PASS_ARRIVAL_EVENTS else 0)
    
    ## Creating a DF to be merged in containing x,y position of football throughout each play    
    football_position_df = (df[df['displayName'] == 'Football']
                            [['gameId','playId','frameId','x','y']]
                            .rename(columns = {'x': 'football_x', 'y': 'football_y'}))
    
    pass_arrival_frame_df = (df[df['pass_arrived'] == 1]
                               .groupby(["gameId", "playId"])
                               .agg({'frameId':np.min}).reset_index())
    
    ### Create a DF with distances to football at that event
    distance_to_football_df = df.merge(pass_arrival_frame_df, on=["gameId", "playId", "frameId"])
    distance_to_football_df = distance_to_football_df.merge(football_position_df, on=["gameId", "playId", "frameId"])
    distance_to_football_df['distance_to_football'] = distance_to_football_df.apply(lambda x: np.sqrt( (x['x'] - x['football_x'])**2 +
                                                                (x['y'] - x['football_y'])**2), axis=1)
    
    ## Select the defense records with the minimum distance and get the player info
    min_d_distance_df = (distance_to_football_df[(distance_to_football_df['isDefense'])]
                       .groupby(["gameId", "playId"]).agg({'distance_to_football': np.min}).reset_index())
    d_distance_to_football_df = (distance_to_football_df[(distance_to_football_df['isDefense'])]
                                .merge(min_d_distance_df, on=["gameId", "playId", "distance_to_football"]))
    d_distance_to_football_df = (d_distance_to_football_df[["gameId", "playId", "nflId", "football_x", "football_y",
                                                        "x", "y", "distance_to_football"]]
                               .rename(columns={'x': 'defender_x',
                                                'y': 'defender_y',
                                                'nflId':'defenderNflId',
                                                'distance_to_football': 'defender_distance_to_football'}))
    
    ## Do the same for the targeted receiver
    t_receiver_distance_df = (distance_to_football_df[(distance_to_football_df['isTargetedReceiver'])]
                              [['gameId', 'playId', 'nflId', 'distance_to_football', 'x', 'y']]
                              .rename(columns = {'distance_to_football': 'receiver_distance_to_football',
                                       'x': 'receiver_x', 'y': 'receiver_y', 'nflId': 'receiverNflId'}))
    
    ## Merge receiver and defender distance DFs and calculate distance between defender and receiver
    distances_df = t_receiver_distance_df.merge(d_distance_to_football_df, on = ['gameId', 'playId'])
    distances_df['receiver_corner_dist_between'] = distances_df.apply(lambda x: 
                                np.sqrt( (x['receiver_x'] - x['defender_x'])**2 +
                                         (x['receiver_y'] - x['defender_y'])**2), axis=1)
    
    return distances_df


def find_distance_thrown_downfield(df):
    """
    Calculate the number of yards downfield (past the line of scrimmage) that passes were thrown on each play.

    Parameters
    ----------
    df : DataFrame
        DataFrame with tracking, play, game, and targets info (see training dataframe creation cells)..

    Returns
    -------
    downfield_dist_df : DataFrame
        DataFrame with number of yards downfield that passes were thrown.

    """
    
    ## Find football at time of "arrival"
    ## Adding column to be used in calculations
    df['pass_arrived'] = df['event'].apply(lambda x: 1 if x in PASS_ARRIVAL_EVENTS else 0)
    
    
    ## Creating a DF to be merged in containing x,y position of football throughout each play    
    football_position_df = (df[df['displayName'] == 'Football']
                            [['gameId','playId','frameId','x','y', 'absoluteYardlineNumber', 'event']])
    
    ## Get the position of the ball at the time of snap for relative LOS proxy
    ball_snap_df = (football_position_df[football_position_df['event'] == 'ball_snap']
                    .drop(columns = ['event', 'frameId']).rename(columns={'x': 'snap_football_x'}))
    
    ## Dataframe with exact/first frame of pass arrival 
    pass_arrival_frame_df = (df[df['pass_arrived'] == 1]
                               .groupby(["gameId", "playId"])
                               .agg({'frameId':np.min}).reset_index())
        
    ## Get the position of the ball at the time pass arrival
    ball_pass_arrived_df = (pass_arrival_frame_df.merge(football_position_df)
                            .rename(columns = {'x': 'arrival_football_x'}))
    
    ## Merge the two together
    downfield_dist_df = ball_pass_arrived_df.merge(ball_snap_df, on = ['gameId', 'playId'])
    downfield_dist_df['yds_thrown_past_los'] = downfield_dist_df['arrival_football_x'] - downfield_dist_df['snap_football_x']
    downfield_dist_df = downfield_dist_df[['gameId', 'playId', 'yds_thrown_past_los']]
    
    return downfield_dist_df


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


def find_closest_presnap_opponents(tracking_df, position = None, side = None): 
    """
    Find the closest offensive player lined up across from CBs on each play
    
    Parameters
    ----------
    tracking_df : DataFrame
        Player tracking dataframe with play data merged.
    position : str
        Optional parameter that limits the returned data to specific positions.
         (i.e. closest to 'CB' position).  Can specify "DB" to get safeties and CBs.
    side : str
        Optional parameter that limits the returned data to "offense" or "defense" players.
        
    Returns
    -------
    tracking_df : DataFrame
        Data with offensive and defensive players that were closest to each other 
        at the time that the ball was snapped.
    
    """
    
    ## Limit to the frame at time of ball-snap and orient coordinates
    tracking_df = tracking_df[tracking_df['event']=='ball_snap'].copy()
    tracking_df = orient_coords(tracking_df)
    
    ## Identify the opposite team in a column to merge on
    tracking_df['opposing_team'] = tracking_df.apply(lambda x: 'home' if x['team'] == 'away'
                                                     else 'away', axis = 1)
    
    tracking_df['defensive_team'] = tracking_df.apply(lambda x:
                                            'away' if x['possessionTeam'] == x['homeTeamAbbr']
                                            else 'home', axis=1)
    
    tracking_df['off_or_def'] = tracking_df.apply(lambda x:
                                            'defense' if x['team'] == x['defensive_team']
                                            else 'offense', axis=1)
    
    ## Merge the pre-snap positions of opposing team
    opposing_player_lineup_merge_df = tracking_df.merge(tracking_df[['gameId', 'playId', 'x', 'y', 'nflId', 'team']],
                                        left_on = ['gameId', 'playId', 'opposing_team'],
                                        right_on = ['gameId', 'playId', 'team'],
                                        suffixes = ('', '_opponent'))
    
    ## Calculate distance to other players
    opposing_player_lineup_merge_df['distance_to_player'] = (opposing_player_lineup_merge_df
                                                .apply(lambda x: np.sqrt((x['x'] - x['x_opponent'])**2 + 
                                                                         (x['y'] - x['y_opponent'])**2),
                                                       axis=1))
    
    ## Get the minimum distance for each player, play, and game and keep only that value
    closest_player_distance_df = (opposing_player_lineup_merge_df.groupby(['gameId', 'nflId', 'playId'])
                                  .agg({'distance_to_player': min}).reset_index())
    
    opposing_player_lineup_merge_df = opposing_player_lineup_merge_df.merge(closest_player_distance_df)

    if position is not None:
        if position == 'DB':
            positions = ['CB', 'FS', 'SS', 'DB']
        else:
            positions = [position]
        
        opposing_player_lineup_merge_df = opposing_player_lineup_merge_df[
            opposing_player_lineup_merge_df['position'].isin(positions)]
    
    if side is not None:
        opposing_player_lineup_merge_df = opposing_player_lineup_merge_df[
            opposing_player_lineup_merge_df['off_or_def'] == side]
    
    closest_presnap_opponent = opposing_player_lineup_merge_df[['gameId', 'playId', 'nflId',
                                                                 'nflId_opponent', 'distance_to_player']]
    
    return closest_presnap_opponent


def compare_trajectories(tracking_df, gameId, playId, nflId_def, nflId_off):
    """
    Compare the series of coordinates of two players throughout a specific play.

    Parameters
    ----------
    gameId : int
        ID number for the game.
    playId : int
        ID number for the play within the game.
    nflId_def : int
        ID number for the defensive player assigned by the NFL..
    nflId_off : int
        ID number for the offensive player assigned by the NFL.

    Returns
    -------
    full_traj_df : DataFrame
        DataFrame with all frames, events, and distances between the 2 players.
    traj_summary_df : TYPE
        DataFrame with 1 record per play, summarizing distance between the 2 players.

    """
    
    ## Offensive trajectory
    off_player_traj_df = tracking_df[(tracking_df['nflId'] == nflId_off) &
                                     (tracking_df['gameId'] == gameId) &
                                     (tracking_df['playId'] == playId)].copy()
    ## Defensive trajectory
    def_player_traj_df = tracking_df[(tracking_df['nflId'] == nflId_def) &
                                     (tracking_df['gameId'] == gameId) &
                                     (tracking_df['playId'] == playId)].copy()
    
    ## Dropping some duplicate columns that we don't need in each
    def_player_traj_df = def_player_traj_df.drop(columns = ['epa', 'passResult', 'offensePlayResult',
                                                            'penaltyCodes', 'playResult', 'isDefensivePI'])
    
    ## Merge into one
    full_traj_df = off_player_traj_df.merge(def_player_traj_df, on = ['gameId', 'playId', 'frameId'],
                                              suffixes = ('_off', '_def'))
    
    ## Calculate distance between players
    full_traj_df['distance_between'] = (full_traj_df.apply(lambda x: np.sqrt(
                                            (x['x_def'] - x['x_off'])**2 + 
                                            (x['y_def'] - x['y_off'])**2), axis=1))
    
    ## Limit fields
    full_traj_df = full_traj_df[['gameId', 'playId', 'frameId', 'nflId_off', 'nflId_def',
                                 'event_off', 'x_off','y_off', 'x_def', 'y_def', 's_def',
                                 'a_def', 'distance_between', 'route_off','offensePlayResult',
                                 'epa', 'passResult', 'playResult', 'isDefensivePI', 'penaltyCodes']]
    
    ## Get the route
    route_df = full_traj_df[~pd.isna(full_traj_df['route_off'])]
    if len(route_df) > 0:
        route = list(route_df['route_off'])[0]
    else:
        route = None
    
    ## Get frame and distance at time of pass release
    pass_release_df = full_traj_df[full_traj_df['event_off'] == 'pass_forward']
    if len(pass_release_df) > 0:
        dist_at_release = list(pass_release_df.sort_values('frameId')['distance_between'])[0]
    else:
        dist_at_release = np.nan
    
    ## Get frame and distance at time of pass arrival
    pass_arrival_df = full_traj_df[full_traj_df['event_off'].isin(PASS_ARRIVAL_EVENTS)]
    if len(pass_arrival_df) > 0:
        dist_at_arrival = list(pass_arrival_df.sort_values('frameId')['distance_between'])[0]
    else:
        dist_at_arrival = np.nan
    
    ## Summarize to 1 record per trajectory
    traj_summary_df = (full_traj_df.fillna('-1')
                                  .groupby(['gameId', 'playId', 'nflId_off', 'nflId_def',
                                             'epa', 'passResult', 'playResult', 'isDefensivePI',
                                             'offensePlayResult', 'penaltyCodes'])
                                   .agg({'distance_between':np.mean})
                                   .rename(columns = {'distance_between': 'avg_distance'})
                                   .reset_index())
    
    traj_summary_df['dist_at_release'] = dist_at_release
    traj_summary_df['dist_at_arrival'] = dist_at_arrival
    traj_summary_df['closure'] = dist_at_release - dist_at_arrival
    traj_summary_df['route'] = route
    
    full_traj_df = full_traj_df.drop(columns=['epa', 'passResult', 'playResult', 'isDefensivePI'])
    
    return full_traj_df, traj_summary_df


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
        