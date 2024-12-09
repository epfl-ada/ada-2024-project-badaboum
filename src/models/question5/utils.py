from datetime import datetime
from src.models.question5.datasets_loading import get_data
from src.models.question5.utils import *

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

def split_compound_score(type_="ceremony"):

    # Initial data
    df = get_data()

    # Create lists to store all movies sentiment scores 
    before_scores = []
    after_scores = []

    # Group reviews by movie
    df_grouped = df.groupby('imdb_id')

    for movie_id, group in df_grouped:
        
        # Get the ceremony date for the current movie
        date = group['ceremony_date'].iloc[0]  
        
        if(type_ == "nomination"):
            date = group['nomination_date'].iloc[0]  

        # Split the reviews according to their publication date (before/after the oscar ceremony)
        before_ceremony = group.loc[group['date'] < date, ['text_compound', 'imdb_id', 'winner']]
        after_ceremony = group.loc[group['date'] >= date, ['text_compound', 'imdb_id', 'winner']]
    
        before_scores.append(before_ceremony)
        after_scores.append(after_ceremony)

    return before_scores, after_scores


def prepare_data(df, imdb_id, nomination_date, ceremony_date, type_="both"):

    first_date = nomination_date
    seconde_date = ceremony_date

    if(type_ == "nomination"):
        seconde_date = nomination_date
    elif(type_ == "ceremony"):
        first_date = ceremony_date
    
    # Filter only around the oscar bumpy
    filtered_reviews = df.loc[
    (df['imdb_id'] == imdb_id)
    & (df['date'] >= first_date - pd.DateOffset(months=2))
    & (df['date'] <= seconde_date + pd.DateOffset(months=2)) ]

    # Compute the mean and count of the reviews for each day
    grouped_reviews_mean = filtered_reviews.groupby(pd.Grouper(key='date', freq='D'))["text_compound"].mean()
    grouped_reviews_count = filtered_reviews.groupby(pd.Grouper(key='date', freq='D'))["text_compound"].count()

    # Apply a rolling window to smooth the mean and the count
    grouped_reviews_mean_smoothed = grouped_reviews_mean.rolling(window=10, min_periods=1).mean()
    grouped_reviews_count_smoothed = grouped_reviews_count.rolling(window=3, min_periods=1).mean()

    return grouped_reviews_mean_smoothed, grouped_reviews_count_smoothed,


def prepare_data_for_all_movies(df):
    results = {}
    # To keep track of processed movie IDs
    seen_movies = set() 

    for _, row in df.iterrows():
        imdb_id = row['imdb_id']
        ceremony_date = pd.to_datetime(row['ceremony_date'])
        nomination_date = pd.to_datetime(row['nomination_date'])

        if imdb_id in seen_movies:
            continue

        # Mark this movie as processed
        seen_movies.add(imdb_id)

        # Call the prepare_data function for each movie
        grouped_mean, grouped_count = prepare_data(df, imdb_id, nomination_date, ceremony_date, type_="ceremony")

        # Adjust the dates by subtracting the ceremony date (making ceremony day 0)
        grouped_mean.index = (grouped_mean.index - ceremony_date).days
        grouped_count.index = (grouped_count.index - ceremony_date).days

        # Store the results in a dictionary
        results[imdb_id] = {
            'mean': grouped_mean,
            'count': grouped_count
        }

    return results