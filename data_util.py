# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:16:43 2020

@author: mtdic
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime

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

#defining events that designate pass arrival
PASS_ARRIVAL_EVENTS = ['pass_outcome_caught',
                      'pass_arrived',
                      'pass_outcome_incomplete',
                      'pass_outcome_interception',
                      'pass_outcome_touchdown']

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