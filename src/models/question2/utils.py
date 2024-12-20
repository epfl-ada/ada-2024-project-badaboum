import pandas as pd
from src.models.question2.datasets_loading import load_reviews_augmented

def find_ceremony_date(df, imdb_id):
    """ 
    Find the ceremony date of a specific movie
    """
    return df[df["imdb_id"] == imdb_id].iloc[0].ceremony_date

    
def compute_timeline_unique(df, imdb_id):
    """
    Compute the average compound score and number of reviews over time for a specific movie

    Parameters:
        df (pd.DataFrame): DataFrame containing the reviews, each with 'imdb_id', 
                           'date', 'ceremony_date', and 'text_compound' columns.
                           
        imdb_id (str): ID of the movie we want to analyze
    
    Returns:
        pd.DataFrame, pd.DataFrame: Two DataFrames containing the positive and 
                                    negative reviews, with dates centered around 
                                    each movie's ceremony date.
    """
    # Find the ceremony
    ceremony_date = find_ceremony_date(df, imdb_id)
    
    # Filter only around the oscar bumpy
    filtered_reviews = df.loc[
    (df['imdb_id'] == imdb_id)
    & (df['date'] >= ceremony_date)]

    # Compute the mean and count of the reviews for each day
    grouped_reviews_mean = filtered_reviews.groupby(pd.Grouper(key='date', freq='D'))["text_compound"].mean()
    grouped_reviews_count = filtered_reviews.groupby(pd.Grouper(key='date', freq='D'))["text_compound"].count()

    # Apply a rolling window to smooth the mean and the count
    grouped_reviews_mean_smoothed = grouped_reviews_mean.rolling(window=30, min_periods=1).mean()
    grouped_reviews_count_smoothed = grouped_reviews_count.rolling(window=30, min_periods=1).mean()

    return grouped_reviews_mean_smoothed, grouped_reviews_count_smoothed


def split_compound_scores_individual(df, imdb_id):
    """
    Splits reviews into positive and negative reviews for one specific movie.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the reviews, each with 'imdb_id', 
                           'date', 'ceremony_date', and 'text_compound' columns.

        imdb_id (str): ID of the movie we want to analyze

    Returns:
        pd.DataFrame, pd.DataFrame: Two DataFrames containing the positive and 
                                    negative reviews.
    """
    
    ceremony_date = find_ceremony_date(df, imdb_id)
    
    # Filter only around the oscar bumpy
    filtered_reviews = df.loc[
    (df['imdb_id'] == imdb_id)
    & (df['date'] >= ceremony_date)]

    # Separate the positive and negative reviews
    positive_reviews = filtered_reviews[filtered_reviews["text_compound"] >= 0]
    negative_reviews = filtered_reviews[filtered_reviews["text_compound"] < 0]

    return positive_reviews, negative_reviews


def split_compound_scores_global(df, type_="all"):
    """
    Splits reviews into positive and negative reviews globally across all movies, 
    centered around each movie's ceremony date.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the reviews, each with 'imdb_id', 
                           'date', 'ceremony_date', and 'text_compound' columns.

        type_ ("all", "winners", "loosers"): select if we want all the reviews or only
                                             the one of winning/loosing movies
    
    Returns:
        pd.DataFrame, pd.DataFrame: Two DataFrames containing the positive and 
                                    negative reviews, with dates centered around 
                                    each movie's ceremony date.
    """
    # Calculate the relative date for each review
    df['ceremony_date'] = pd.to_datetime(df['ceremony_date'], format='%Y-%m-%d')

    # Center the reviews around the ceremony date of the corresponding movie
    df["relative_date"] = df["date"] - df["ceremony_date"]
    df["relative_date"] = df["relative_date"].dt.days
    
    # Split the DataFrame into positive and negative reviews
    positive_reviews = df[(df["text_compound"] >= 0) & (df["relative_date"] >= 0)]
    negative_reviews = df[(df["text_compound"] < 0) & (df["relative_date"] >= 0)]

    # Filter the reviews if we want only the winning/loosing movies
    if(type_ == "winners"):
        positive_reviews = df[(df["text_compound"] >= 0) & (df["relative_date"] >= 0) & (df["winner"] == True)]
        negative_reviews = df[(df["text_compound"] < 0) & (df["relative_date"] >= 0) & (df["winner"] == True)]
    elif(type_ == "loosers"):
        positive_reviews = df[(df["text_compound"] >= 0) & (df["relative_date"] >= 0) & (df["winner"] == False)]
        negative_reviews = df[(df["text_compound"] < 0) & (df["relative_date"] >= 0) & (df["winner"] == False)]
    
    return positive_reviews, negative_reviews


def select_visualization_groups(type_1, type_2):

    df = load_reviews_augmented()
    
    positive_reviews_global, negative_reviews_global = split_compound_scores_global(df)
    positive_reviews_winners, negative_reviews_winners = split_compound_scores_global(df, type_="winners")
    positive_reviews_loosers, negative_reviews_loosers = split_compound_scores_global(df, type_="loosers")

    df_1 = 0
    df_2 = 0 

    if(type_1 == "pos_glob"):
        df_1 = positive_reviews_global
    elif(type_1 == "pos_win"):
        df_1 = positive_reviews_winners
    elif(type_1 == "pos_loos"):
        df_1 = positive_reviews_loosers
    elif(type_1 == "neg_win"):
        df_1 = negative_reviews_winners
        
    if(type_2 == "neg_glob"):
        df_2 = negative_reviews_global
    elif(type_2 == "neg_win"):
        df_2 = negative_reviews_winners
    elif(type_2 == "neg_loos"):
        df_2 = negative_reviews_loosers
    elif(type_2 == "pos_loos"):
        df_2 = positive_reviews_loosers

    return df_1, df_2