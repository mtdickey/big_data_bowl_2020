# Big Data Bowl 2021

Analysis for NFL's 3rd Big Data Bowl, running from October 2020 to January 5th, 2021 on [Kaggle](https://www.kaggle.com/c/nfl-big-data-bowl-2021/overview).

#### Organization

##### Analysis
  - Notebooks:
    - [*Fifty-Fifty.ipynb*](https://github.com/mtdickey/big_data_bowl_2021/tree/main/analysis/notebooks/Fifty-Fifty.ipynb): Analyzing how defenses can make "50-50" balls more in their favor.
    - [*Corner_Receiver_Clusters.ipynb*](https://github.com/mtdickey/big_data_bowl_2021/tree/main/analysis/notebooks/Corner_Receiver_Clusters.ipynb): Early idea to analyze which routes are most/least effective for each receiver/corner matchup.
  - Scripts:
    - *regressions_viz.py*: Modeling coverage success using tracking data and player attributes
    - *eda.py*: Various exploratory analyses (less organized than the notebooks)
  
##### Utility Functions (utils)

  - *Visualization*
     - *viz_util.py*: Useful plotting functions in Python
     - *plot_plays.R*: Script used to animate plays in GIF form  (Credit to Tom Bliss' work [here](https://www.kaggle.com/tombliss/tutorial/notebook#AnimatePlays))
  - *Data Wrangling*
     - *data_util.py*: Various functions to work with player tracking data
