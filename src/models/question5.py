from datetime import datetime
from scipy.stats import wilcoxon
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_data():

    # Get the reviews
    PATH = 'data/'
    reviews = pd.read_csv(PATH +'imdb_reviews/imdb_reviews_with_compound.csv')

    # Ensure the 'date' column is in datetime format
    reviews['date'] = pd.to_datetime(reviews['date'])
    
    # Get the oscar movies table
    oscar_movies = pd.read_csv(PATH +'oscar_movies.csv')

    # Only keep the imdb movie id the ceremony date and the winner flag
    oscar_ceremonies = oscar_movies[['tconst','ceremony_date', 'winner']]

    # Rename the movie id column to be consistent with the review dataframe
    oscar_ceremonies =  oscar_ceremonies.rename(columns={"tconst": "imdb_id"})

    # Add the ceremony date of the movie to the review
    reviews = reviews.join(oscar_ceremonies.set_index('imdb_id'), on= 'imdb_id')

    return reviews

def split_compound_score_around_ceremony():

    # Initial data
    df = get_data()

    # Create lists to store all movies sentiment scores 
    before_scores = []
    after_scores = []

    # Group reviews by movie
    df_grouped = df.groupby('imdb_id')

    for movie_id, group in df_grouped:
        
        # Get the ceremony date for the current movie
        ceremony_date = group['ceremony_date'].iloc[0]  

        # Split the reviews according to their publication date (before/after the oscar ceremony)
        before_ceremony = group.loc[group['date'] < ceremony_date, ['text_compound', 'imdb_id', 'winner']]
        after_ceremony = group.loc[group['date'] >= ceremony_date, ['text_compound', 'imdb_id', 'winner']]
    
        before_scores.append(before_ceremony)
        after_scores.append(after_ceremony)

    return before_scores, after_scores

def perform_statistical_tests():

    # Get the reviews already splitted
    before_nomination, after_nomination = split_compound_score_around_ceremony()
    
    # Flatten the lists for the test
    before_nomination_flat = [item['text_compound'] for sublist in before_nomination for item in sublist.to_dict(orient='records')]
    after_nomination_flat = [item['text_compound'] for sublist in after_nomination for item in sublist.to_dict(orient='records')]


    # Plot the distribution of the sentiment scores in both cases
    plt.hist(before_nomination_flat, bins=200)
    plt.show()

    plt.hist(after_nomination_flat, bins=200)
    plt.show()

    # List for the pairwise results
    results = []

    # Number of tested movies
    count = 0
    
    for i in range(0,len(before_nomination)):

        # Get the reviews for one specific movie
        before = before_nomination[i]['text_compound'].tolist()
        after = after_nomination[i]['text_compound'].tolist()

        if len(before) == 0 or len(after) == 0:
            continue  # Skip this movie

        movie_id = before_nomination[i]['imdb_id'].tolist()[0]
        winner = before_nomination[i]['winner'].tolist()[0]
        
        # Truncate both lists to the length of the shorter one
        min_length = min(len(before), len(after))
        before = before[:min_length]
        after = after[:min_length]
        
        # Perform the Wilcoxon test (with reviews specific to one movie)
        stat, p = wilcoxon(before, after)
        results.append({'Movie ID': movie_id, 'Winner': winner , 'p-value': p})

    results_df = pd.DataFrame(results)

    # Movies where the hypothesis was rejected (p value of 0.05)
    count_reject = (results_df['p-value'] < 0.05).sum()

    # Percentage of movie where the hypothesis was rejected
    percentage_rejected = count_reject / results_df.shape[0] * 100

    print(f'There are {count_reject} rejected movies (p-value < 0.05). That represents only {percentage_rejected} percent of the movies')

    return results_df

    
def prepare_data(df, imdb_id, nomination_date, ceremony_date):
    
    # Filter only around the oscar bumpy
    filtered_reviews = df.loc[
    (df['imdb_id'] == imdb_id)
    & (df['date'] >= nomination_date - pd.DateOffset(months=2))
    & (df['date'] <= ceremony_date + pd.DateOffset(months=2)) ]

    # Compute the mean and count of the reviews for each day
    grouped_reviews_mean = filtered_reviews.groupby(pd.Grouper(key='date', freq='D'))["text_compound"].mean()
    grouped_reviews_count = filtered_reviews.groupby(pd.Grouper(key='date', freq='D'))["text_compound"].count()

    # Apply a rolling window to smooth the mean and the count
    grouped_reviews_mean_smoothed = grouped_reviews_mean.rolling(window=10, min_periods=1).mean()
    grouped_reviews_count_smoothed = grouped_reviews_count.rolling(window=3, min_periods=1).mean()

    return grouped_reviews_mean_smoothed, grouped_reviews_count_smoothed


def plot_oscar_bump(type_):

    # Custom case for Milestone 2
    imdb_id = 'tt0405159'
    nomination_date = pd.to_datetime('2005-01-25')
    ceremony_date = pd.to_datetime('2005-02-27')

    # Get the data
    df_init = get_data()

    # Prepare the data
    grouped_reviews_mean, grouped_reviews_count = prepare_data(df_init, imdb_id, nomination_date, ceremony_date)

    # Set the theme of the plot
    sns.set_theme(style="darkgrid", rc={'figure.figsize':(11.7,8.27)})

    # Set the plot titles
    title = ""
    xlabel = ""
    ylabel = ""
    df = df_init

    if(type_ == "compound"):
        title = "Compound score over time for the movie: "
        xlabel = "Date"
        ylabel = "Mean compound score"
        df = grouped_reviews_mean
    elif(type_ == "count"):
        title = "Review count over time for the movie: "
        xlabel = "Date"
        ylabel = "Mean review count"
        df = grouped_reviews_count


    # Plot the timeline
    sns.lineplot(x=df.index, y=df, zorder=1).set(xlabel=xlabel, ylabel=ylabel, title=title+imdb_id)

    # Add the points to the plot
    nomination_value = df.loc[nomination_date]
    ceremony_value = df.loc[ceremony_date]
    
    sns.scatterplot(x=[nomination_date], y=[nomination_value], color='red', s=100, marker='o',
                    label='Oscar Nomination', zorder=2)

    sns.scatterplot(x=[ceremony_date], y=[ceremony_value], color='blue', s=100, marker='o',
                    label='Ceremony Date', zorder=2)