from datetime import datetime
import pandas as pd
import seaborn as sns

def get_data():

    # Get the reviews
    PATH = 'data/reviews/'
    reviews = pd.read_csv(PATH +'reviews_with_compound.csv')

    # Ensure the 'date' column is in datetime format
    reviews['date'] = pd.to_datetime(reviews['date'])

    return reviews


def prepare_data(df, imdb_id, ceremony_date):
    
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
    

def plot_oscar_timeline(type_):

    # Custom case for Milestone 2
    imdb_id = 'tt0405159'
    ceremony_date = pd.to_datetime('2005-02-27')

    # Get the data
    df_init = get_data()

    # Prepare the data
    grouped_reviews_mean, grouped_reviews_count = prepare_data(df_init, imdb_id, ceremony_date)

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