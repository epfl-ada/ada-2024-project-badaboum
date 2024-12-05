from datetime import datetime
from scipy.stats import wilcoxon
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import statsmodels.api as sm

def get_data():

    # Get the reviews
    PATH = 'data/'
    reviews = pd.read_csv(PATH +'imdb_reviews/imdb_reviews_with_compound.csv')

    # Ensure the 'date' column is in datetime format
    reviews['date'] = pd.to_datetime(reviews['date'])
    
    # Get the oscar movies table
    oscar_movies = pd.read_csv(PATH +'oscar_movies.csv')

    # Get the oscar nomination dates table
    oscar_nomination_dates = pd.read_csv(PATH + 'oscar_nomination_dates.csv')

    # Rename the movie id column to be consistent with the review dataframe
    oscar_nomination_dates =  oscar_nomination_dates.rename(columns={"year": "oscar_year"})

    # Only keep the imdb movie id the ceremony date and the winner flag
    oscar_ceremonies = oscar_movies[['tconst','ceremony_date', 'oscar_year', 'winner']]

    # Rename the movie id column to be consistent with the review dataframe
    oscar_ceremonies =  oscar_ceremonies.rename(columns={"tconst": "imdb_id"})

    # Add the ceremony date of the movie to the review
    reviews = reviews.join(oscar_ceremonies.set_index('imdb_id'), on='imdb_id')

    # Add the nomination date of the movie to the review
    reviews = reviews.join(oscar_nomination_dates.set_index('oscar_year'), on='oscar_year')

    return reviews


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

def plot_proportions():
    
    before_scores, after_scores = split_compound_score(type_="ceremony")
    before_scores_nomination, after_scores_nomination = split_compound_score(type_="nomination")

    # Flatten the lists 
    before_flat = [item['text_compound'] for sublist in before_scores for item in sublist.to_dict(orient='records')]
    after_flat = [item['text_compound'] for sublist in after_scores for item in sublist.to_dict(orient='records')]

    before_nomination_flat = [item['text_compound'] for sublist in before_scores_nomination for item in sublist.to_dict(orient='records')]
    after_nomination_flat = [item['text_compound'] for sublist in after_scores_nomination for item in sublist.to_dict(orient='records')]

    after_flat_winner = [item['text_compound'] for sublist in after_scores for item in sublist.to_dict(orient='records') if item["winner"]==True]
    after_flat_looser = [item['text_compound'] for sublist in after_scores for item in sublist.to_dict(orient='records') if item["winner"]==False]

    # Define the bins and labels
    bins = [-1, -0.8, -0.2, 0.2, 0.8, 1]
    labels = ["Really Negative", "Negative", "Neutral", "Positive", "Really Positive"]
    
    # Bin the scores
    before_categories = pd.cut(before_flat, bins=bins, labels=labels)
    after_categories = pd.cut(after_flat, bins=bins, labels=labels)

    before_categories_nomination = pd.cut(before_nomination_flat, bins=bins, labels=labels)
    after_categories_nomination = pd.cut(after_nomination_flat, bins=bins, labels=labels)

    after_winner_categories = pd.cut(after_flat_winner, bins=bins, labels=labels)
    after_looser_categories = pd.cut(after_flat_looser, bins=bins, labels=labels)

    # Count the occurrences in each category
    before_counts = before_categories.value_counts() / len(before_flat)
    after_counts = after_categories.value_counts() / len(after_flat)

    before_counts_nomination = before_categories_nomination.value_counts() / len(before_nomination_flat)
    after_counts_nomination = after_categories_nomination.value_counts() / len(after_nomination_flat)

    after_winner_counts = after_winner_categories.value_counts() / len(after_flat_winner)
    after_looser_counts = after_looser_categories.value_counts() / len(after_flat_looser)
    
    # Plot the distributions
    x = range(len(labels))
    
    plt.bar(x, before_counts, width=0.4, label="Before the ceremony", align='center', alpha=0.7)
    plt.bar(x, after_counts, width=0.4, label="After the ceremony", align='edge', alpha=0.7)
    
    # Add labels and legend
    plt.xticks(x, labels, rotation=45)
    plt.xlabel("Sentiment Categories")
    plt.ylabel("Proportion")
    plt.title("Distribution of Compound Scores Before and After the ceremony")
    plt.legend()
    plt.tight_layout()
    
    # Show the plot
    plt.show()

    plt.bar(x, before_counts_nomination, width=0.4, label="Before the nomination", align='center', alpha=0.7)
    plt.bar(x, after_counts_nomination, width=0.4, label="After the nomination", align='edge', alpha=0.7)
    
    # Add labels and legend
    plt.xticks(x, labels, rotation=45)
    plt.xlabel("Sentiment Categories")
    plt.ylabel("Proportion")
    plt.title("Distribution of Compound Scores Before and After the nomination")
    plt.legend()
    plt.tight_layout()
    
    # Show the plot
    plt.show()

    plt.bar(x, after_winner_counts, width=0.4, label="After Nomination and won", align='center', alpha=0.7)
    plt.bar(x, after_looser_counts, width=0.4, label="After Nomination and didn't win", align='edge', alpha=0.7)
    
    # Add labels and legend
    plt.xticks(x, labels, rotation=45)
    plt.xlabel("Sentiment Categories")
    plt.ylabel("Proportion")
    plt.title("Distribution of Compound Scores After the ceremony for winning and non-winning movies")
    plt.legend()
    plt.tight_layout()
    
    # Show the plot
    plt.show()

    
def perform_statistical_test_compound(type_="ceremony"):
    
    # Get the reviews already splitted
    before_nomination, after_nomination = split_compound_score(type_="ceremony")

    if(type_== "nomination"):
        before_nomination, after_nomination = split_compound_score(type_="nomination")
    
    # Flatten the lists for the test
    before_nomination_flat = [item['text_compound'] for sublist in before_nomination for item in sublist.to_dict(orient='records')]
    after_nomination_flat = [item['text_compound'] for sublist in after_nomination for item in sublist.to_dict(orient='records')]


    # Plot the distribution of the sentiment scores in both cases
    plt.hist(before_nomination_flat, bins=200)
    plt.yscale('log')
    plt.show()

    plt.hist(after_nomination_flat, bins=200)
    plt.yscale('log')
    plt.show()

    # List for the pairwise results
    results = []

    # Number of tested movies
    count = 0
    
    for i in range(0,len(before_nomination)):

        # Get the reviews for one specific movie
        before = before_nomination[i]['text_compound'].tolist()
        after = after_nomination[i]['text_compound'].tolist()

        # Skip this movie if reviews are missing
        if len(before) == 0 or len(after) == 0:
            continue  

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


def perform_regression_compound(type_score="all", type_date="ceremony"):
    
    # Get the reviews already splitted
    before_nomination, after_nomination = split_compound_score(type_="ceremony")

    if(type_date == "nomination"):
        
        before_nomination, after_nomination = split_compound_score(type_="nomination")
        
    before_nomination_flat = [item['text_compound'] for sublist in before_nomination for item in sublist.to_dict(orient='records')]
    after_nomination_flat = [item['text_compound'] for sublist in after_nomination for item in sublist.to_dict(orient='records')]

    if(type_score=="positive"):
        
        before_nomination_flat = [item['text_compound'] for sublist in before_nomination for item in sublist.to_dict(orient='records') if 
                              item['text_compound'] >= 0]
        after_nomination_flat = [item['text_compound'] for sublist in after_nomination for item in sublist.to_dict(orient='records') if
                                 item['text_compound'] >= 0]
    elif(type_score=="negative"):

        before_nomination_flat = [item['text_compound'] for sublist in before_nomination for item in sublist.to_dict(orient='records') if 
                              item['text_compound'] <= 0]
        after_nomination_flat = [item['text_compound'] for sublist in after_nomination for item in sublist.to_dict(orient='records') if
                                 item['text_compound'] <= 0]

    # Find the target length (length of the shorter list)
    target_length = min(len(before_nomination_flat), len(after_nomination_flat))

    # Randomly sample the longer list and keep the shorter list as is
    before_nomination_flat = random.sample(before_nomination_flat, target_length) if len(before_nomination_flat) > target_length else before_nomination_flat
    after_nomination_flat = random.sample(after_nomination_flat, target_length) if len(after_nomination_flat) > target_length else after_nomination_flat

    before_nomination_final = pd.DataFrame(before_nomination_flat)
    after_nomination_final = pd.DataFrame(after_nomination_flat)

    before_nomination_final["time"] = 0
    after_nomination_final["time"] = 1

    final_df = pd.concat([before_nomination_final, after_nomination_final])

    final_df = final_df.rename(columns={0: "compound"})


    sns.stripplot(x='time', y='compound', data=final_df, jitter=0.2, alpha=0.5)

    # Independent variable (time) and dependent variable (score)
    X = final_df['time']
    y = final_df['compound']

    # Add a constant for the intercept
    X = sm.add_constant(X)  # Adds a column of ones

    # Fit the model
    model = sm.OLS(y, X).fit()

    # Print the summary
    print(model.summary())

    return final_df

#def perform_distribution_analysis():

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


def plot_oscar_bump_unique_movie(imdb_id,type_):
    
    # Get the data
    df_init = get_data()

    nomination_date = pd.to_datetime(df_init[df_init.imdb_id==imdb_id].nomination_date.values[0])
    ceremony_date = pd.to_datetime(df_init[df_init.imdb_id==imdb_id].ceremony_date.values[0])

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


def plot_oscar_bump_all_movies():
    
    # Get the data
    df_init = get_data()

    # Prepare the data
    results = prepare_data_for_all_movies(df_init)

    all_compounds = []
    all_counts = []

    # Collect all the mean values from the results
    for imdb_id, data in results.items():
        all_compounds.append(data['mean'])
        all_counts.append(data['count'])

    # Concatenate all the mean values
    combined_compounds = pd.concat(all_compounds, axis=0)
    combined_counts = pd.concat(all_counts, axis=0)

    # Group by the relative date (index) and calculate the mean for each day
    daily_mean_coumpounds = combined_compounds.groupby(combined_compounds.index).mean()
    daily_mean_counts = combined_counts.groupby(combined_counts.index).mean()

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    
    # Example daily_mean_compounds data
    # daily_mean_coumpounds is assumed to be a pandas Series with index as the time variable
    x = daily_mean_coumpounds.index  # Time (relative to ceremony date)
    y = daily_mean_coumpounds.values  # Mean sentiment scores
    
    # Fit a linear regression model
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # Generate the regression line
    regression_line = intercept + slope * x

    # Plot the overall mean for the compound score and the number of reviews
    plt.figure(figsize=(10, 5))
    plt.plot(daily_mean_coumpounds, label='Combined Mean Sentiment')

    # Plot the regression line
    plt.plot(x, regression_line, color='red', linestyle='--')
    
    plt.title('Mean of All Movie Sentiment Scores')
    plt.xlabel('Time')
    plt.ylabel('Mean Sentiment Score')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(daily_mean_counts, label='Combined Mean Number Of Reviews')
    plt.title('Count of All Movie Sentiment Reviews')
    plt.xlabel('Time')
    plt.ylabel('Mean Number Of Reviews')
    plt.legend()
    plt.show()