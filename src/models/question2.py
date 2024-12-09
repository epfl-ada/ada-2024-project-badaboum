from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def get_data():

    # Get the reviews
    PATH = 'data/'
    reviews = pd.read_csv(PATH +'imdb_reviews/imdb_reviews_with_compound.csv')

    # Ensure the 'date' column is in datetime format
    reviews['date'] = pd.to_datetime(reviews['date'])

    # Get the oscar movies table
    oscar_movies = pd.read_csv(PATH +'oscar_movies.csv')

    # Only keep the imdb movie id the ceremony date and the winner flag
    oscar_ceremonies = oscar_movies[['tconst','ceremony_date', 'oscar_year', 'winner']]

    # Rename the movie id column to be consistent with the review dataframe
    oscar_ceremonies =  oscar_ceremonies.rename(columns={"tconst": "imdb_id"})

    # Add the ceremony date of the movie to the review
    reviews = reviews.join(oscar_ceremonies.set_index('imdb_id'), on='imdb_id')

    return reviews


def prepare_data(df, imdb_id):

    ceremony_date = df[df["imdb_id"] == imdb_id].iloc[0].ceremony_date
    
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
    
    ceremony_date = df[df["imdb_id"] == imdb_id].iloc[0].ceremony_date
    
    # Filter only around the oscar bumpy
    filtered_reviews = df.loc[
    (df['imdb_id'] == imdb_id)
    & (df['date'] >= ceremony_date)]

    positive_reviews = filtered_reviews[filtered_reviews["text_compound"] >= 0]
    negative_reviews = filtered_reviews[filtered_reviews["text_compound"] < 0]

    return positive_reviews, negative_reviews


def plot_compound_scores_individual(imdb_id):
    
    # Get the data
    df_init = get_data()

    # Prepare the data
    positive_reviews, negative_reviews = split_compound_scores_individual(df_init, imdb_id)

    sns.scatterplot(data=positive_reviews, x="date", y="text_compound")
    
    sns.scatterplot(data=negative_reviews, x="date", y="text_compound") 

    plt.show()

    sns.kdeplot(data=positive_reviews[["date", "text_compound"]], x="date")
    sns.kdeplot(data=negative_reviews[["date", "text_compound"]], x="date")

def split_compound_scores_global(df, type_="all"):
    """
    Splits reviews into positive and negative reviews globally across all movies, 
    centered around each movie's ceremony date.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the reviews, each with 'imdb_id', 
                           'date', 'ceremony_date', and 'text_compound' columns.
    
    Returns:
        pd.DataFrame, pd.DataFrame: Two DataFrames containing the positive and 
                                    negative reviews, with dates centered around 
                                    each movie's ceremony date.
    """
    # Calculate the relative date for each review
    df['ceremony_date'] = pd.to_datetime(df['ceremony_date'], format='%Y-%m-%d')

    df["relative_date"] = df["date"] - df["ceremony_date"]
    df["relative_date"] = df["relative_date"].dt.days
    
    # Split the DataFrame into positive and negative reviews
    positive_reviews = df[(df["text_compound"] >= 0) & (df["relative_date"] >= 0)]
    negative_reviews = df[(df["text_compound"] < 0) & (df["relative_date"] >= 0)]

    if(type_ == "winners"):
        positive_reviews = df[(df["text_compound"] >= 0) & (df["relative_date"] >= 0) & (df["winner"] == True)]
        negative_reviews = df[(df["text_compound"] < 0) & (df["relative_date"] >= 0) & (df["winner"] == True)]
    elif(type_ == "loosers"):
        positive_reviews = df[(df["text_compound"] >= 0) & (df["relative_date"] >= 0) & (df["winner"] == False)]
        negative_reviews = df[(df["text_compound"] < 0) & (df["relative_date"] >= 0) & (df["winner"] == False)]
    
    return positive_reviews, negative_reviews

def plot_compound_scores_global(type_1, type_2):
    
    # Get the data
    df_init = get_data()

    # Prepare the data
    positive_reviews_global, negative_reviews_global = split_compound_scores_global(df_init)
    positive_reviews_winners, negative_reviews_winners = split_compound_scores_global(df_init, type_="winners")
    positive_reviews_loosers, negative_reviews_loosers = split_compound_scores_global(df_init, type_="loosers")

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

    ax = sns.scatterplot(data=df_1, x="relative_date", y="text_compound", label = type_1)
    sns.scatterplot(data=df_2, x="relative_date", y="text_compound", label = type_2)

    # Add a title and axis labels
    ax.set_title("Scatterplot of Relative Date vs. Sentiment Score", fontsize=16)
    ax.set_xlabel("Relative Date (Days)", fontsize=14)
    ax.set_ylabel("Sentiment Score (Compound)", fontsize=14)

    plt.legend()

    # Show the plot
    plt.show()


    ax = sns.kdeplot(data=df_1[["relative_date", "text_compound"]], x="relative_date", label = type_1)
    sns.kdeplot(data=df_2[["relative_date", "text_compound"]], x="relative_date", label = type_2)

    # Add a title and axis labels
    ax.set_title("Density Distributions of Reviews over Time", fontsize=16)
    ax.set_xlabel("Relative Date (Days)", fontsize=14)
    ax.set_ylabel("Distribution Density", fontsize=14)

    plt.legend()

    # Show the plot
    plt.show()
    
    # Extract the 'relative_date' values
    dates_1 = df_1['relative_date']
    dates_2 = df_2['relative_date']
    
    # Perform the KS test
    ks_stat, p_value = ks_2samp(dates_1, dates_2)
    
    print(f"KS Statistic: {ks_stat}, p-value: {p_value}")
    
def plot_oscar_timeline(imdb_id,type_):

    # Get the data
    df_init = get_data()

    # Prepare the data
    grouped_reviews_mean, grouped_reviews_count = prepare_data(df_init, imdb_id)

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