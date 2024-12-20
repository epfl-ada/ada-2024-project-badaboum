from src.models.oscar_bump.datasets_loading import get_data
from src.models.oscar_bump.utils import *
from scipy.stats import linregress

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_compound_distribution(type_="ceremony"):
    """
    Plot the distribution of the compound score before and after the ceremony
    
    Parameters:
        type_ (str): The type of date to consider (ceremony or nomination)
    """
    # Get the reviews already splitted
    before, after = split_compound_score(type_="ceremony")

    # If the type is nomination, split the compound score for the nomination date
    if(type_== "nomination"):
        before, after = split_compound_score(type_="nomination")
    
    # Flatten the lists for the test
    before_flat = [item['text_compound'] for sublist in before for item in sublist.to_dict(orient='records')]
    after_flat = [item['text_compound'] for sublist in after for item in sublist.to_dict(orient='records')]

    # Plot the distribution of the sentiment scores in both cases
    sns.kdeplot(before_flat, color='red', linewidth=2, log_scale=(False, True), bw_adjust=0.5, clip=(-1, 1), label = 'Before')
    sns.kdeplot(after_flat, color='green', linewidth=2, log_scale=(False, True), bw_adjust=0.5, clip=(-1, 1), label = 'After')
    plt.legend()
    plt.yscale('log')
    plt.xlabel("Compound score")
    plt.ylabel("Number of reviews (log scale)")
    plt.title(f"Compound Score Distribution before and after the {type_}" )
    plt.show()

    # Perform the Kolmogorov-Smirnov test and print the p-value
    print(f"Kolmogorov-Smirnov test p-value: {stats.kstest(before_flat, after_flat).pvalue}")


def plot_oscar_bump_unique_movie(imdb_id,type_):
    """
    Plot the compound score over time for a specific movie
    
    Parameters:
        imdb_id (str): The IMDB ID of the movie
        type_ (str): The type of score to consider (compound or count)
    """
    # Get the data
    df_init = get_data()

    # Get the nomination and ceremony date
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

    # Set the labels according to the type
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
    
    # Add the points to the plot
    sns.scatterplot(x=[nomination_date], y=[nomination_value], color='red', s=100, marker='o',
                    label='Oscar Nomination', zorder=2)

    # Add the points to the plot
    sns.scatterplot(x=[ceremony_date], y=[ceremony_value], color='blue', s=100, marker='o',
                    label='Ceremony Date', zorder=2)


def plot_oscar_bump_all_movies():
    """
    Plot the compound score over time for all movies
    """   
    # Get the data
    df_init = get_data()

    # Prepare the data
    results = prepare_data_for_all_movies(df_init)

    all_compounds = []
    all_counts = []

    # Collect all the mean values from the results
    for _, data in results.items():
        all_compounds.append(data['mean'])
        all_counts.append(data['count'])

    # Concatenate all the mean values
    combined_compounds = pd.concat(all_compounds, axis=0)
    combined_counts = pd.concat(all_counts, axis=0)

    # Group by the relative date (index) and calculate the mean for each day
    daily_mean_coumpounds = combined_compounds.groupby(combined_compounds.index).mean()
    daily_mean_counts = combined_counts.groupby(combined_counts.index).mean()

    # Time (relative to ceremony date)
    x = daily_mean_coumpounds.index 
    
    # Mean sentiment scores
    y = daily_mean_coumpounds.values  
    
    # Fit a linear regression model
    slope, intercept, _, _, _ = linregress(x, y)
    
    # Generate the regression line
    regression_line = intercept + slope * x

    # Plot the overall mean for the compound score and the number of reviews
    plt.figure(figsize=(10, 5))
    plt.plot(daily_mean_coumpounds, label='Combined Mean Sentiment')

    # Plot the regression line
    plt.plot(x, regression_line, color='red', linestyle='--')
    
    # Add labels and legend
    plt.title('Mean of All Movie Sentiment Scores around the Ceremony Date')
    plt.xlabel('Time (Days)')
    plt.ylabel('Mean Sentiment Score')
    plt.legend()
    plt.show()

    # Plot the overall mean for the number of reviews   
    plt.figure(figsize=(10, 5))
    plt.plot(daily_mean_counts, label='Combined Mean Number Of Reviews')
    plt.title('Count of All Movie Reviews around the Ceremony Date')
    plt.xlabel('Time (Days)')
    plt.ylabel('Mean Number Of Reviews')
    plt.legend()
    plt.show()


def plot_proportions():
    """
    Plot the distribution of the compound score for different situations
    """
    
    # Split the compound score for the ceremony and nomination date
    before_scores, after_scores = split_compound_score(type_="ceremony")
    before_scores_nomination, after_scores_nomination = split_compound_score(type_="nomination")

    # Flatten the lists for the plot
    before_flat = [item['text_compound'] for sublist in before_scores for item in sublist.to_dict(orient='records')]
    after_flat = [item['text_compound'] for sublist in after_scores for item in sublist.to_dict(orient='records')]

    before_nomination_flat = [item['text_compound'] for sublist in before_scores_nomination for item in sublist.to_dict(orient='records')]
    after_nomination_flat = [item['text_compound'] for sublist in after_scores_nomination for item in sublist.to_dict(orient='records')]

    after_flat_winner = [item['text_compound'] for sublist in after_scores for item in sublist.to_dict(orient='records') if item["winner"]==True]
    after_flat_looser = [item['text_compound'] for sublist in after_scores for item in sublist.to_dict(orient='records') if item["winner"]==False]

    # Define the bins and labels
    bins = [-1, -0.8, -0.2, 0.2, 0.8, 1]
    labels = ["Really Negative", "Negative", "Neutral", "Positive", "Really Positive"]
    
    # Categorize the compound scores
    before_categories = pd.cut(before_flat, bins=bins, labels=labels)
    after_categories = pd.cut(after_flat, bins=bins, labels=labels)

    before_categories_nomination = pd.cut(before_nomination_flat, bins=bins, labels=labels)
    after_categories_nomination = pd.cut(after_nomination_flat, bins=bins, labels=labels)

    after_winner_categories = pd.cut(after_flat_winner, bins=bins, labels=labels)
    after_looser_categories = pd.cut(after_flat_looser, bins=bins, labels=labels)

    # Calculate the proportions
    before_counts = before_categories.value_counts() / len(before_flat)
    after_counts = after_categories.value_counts() / len(after_flat)

    before_counts_nomination = before_categories_nomination.value_counts() / len(before_nomination_flat)
    after_counts_nomination = after_categories_nomination.value_counts() / len(after_nomination_flat)

    after_winner_counts = after_winner_categories.value_counts() / len(after_flat_winner)
    after_looser_counts = after_looser_categories.value_counts() / len(after_flat_looser)
    
    # Plot the proportions
    x = range(len(labels))
    
    plt.bar(x, before_counts, width=0.4, label="Before the ceremony", align='center', alpha=0.7)
    plt.bar(x, after_counts, width=0.4, label="After the ceremony", align='edge', alpha=0.7)
    
    plt.xticks(x, labels, rotation=45)
    plt.xlabel("Sentiment Categories")
    plt.ylabel("Proportion")
    plt.title("Distribution of Compound Scores Before and After the ceremony")
    plt.legend()
    plt.tight_layout()
    
    plt.show()

    plt.bar(x, before_counts_nomination, width=0.4, label="Before the nomination", align='center', alpha=0.7)
    plt.bar(x, after_counts_nomination, width=0.4, label="After the nomination", align='edge', alpha=0.7)
    
    plt.xticks(x, labels, rotation=45)
    plt.xlabel("Sentiment Categories")
    plt.ylabel("Proportion")
    plt.title("Distribution of Compound Scores Before and After the nomination")
    plt.legend()
    plt.tight_layout()
    
    plt.show()

    plt.bar(x, after_winner_counts, width=0.4, label="After the ceremony and won", align='center', alpha=0.7)
    plt.bar(x, after_looser_counts, width=0.4, label="After the ceremony and didn't win", align='edge', alpha=0.7)
    
    plt.xticks(x, labels, rotation=45)
    plt.xlabel("Sentiment Categories")
    plt.ylabel("Proportion")
    plt.title("Distribution of Compound Scores After the ceremony for winning and non-winning movies")
    plt.legend()
    plt.tight_layout()
    
    plt.show()


def plot_proportions_change():
    """
    Plot the change in the distribution of the compound score for different situations
    """
    # Split the compound score for the ceremony and nomination date
    before_scores, after_scores = split_compound_score(type_="ceremony")
    before_scores_nomination, after_scores_nomination = split_compound_score(type_="nomination")

    # Flatten the lists for the plot
    before_flat = [item['text_compound'] for sublist in before_scores for item in sublist.to_dict(orient='records')]
    after_flat = [item['text_compound'] for sublist in after_scores for item in sublist.to_dict(orient='records')]

    before_nomination_flat = [item['text_compound'] for sublist in before_scores_nomination for item in sublist.to_dict(orient='records')]
    after_nomination_flat = [item['text_compound'] for sublist in after_scores_nomination for item in sublist.to_dict(orient='records')]

    after_flat_winner = [item['text_compound'] for sublist in after_scores for item in sublist.to_dict(orient='records') if item["winner"]==True]
    after_flat_looser = [item['text_compound'] for sublist in after_scores for item in sublist.to_dict(orient='records') if item["winner"]==False]

    # Define the bins and labels
    bins = [-1, -0.8, -0.2, 0.2, 0.8, 1]
    labels = ["Really Negative", "Negative", "Neutral", "Positive", "Really Positive"]
    
    # Categorize the compound scores
    before_categories = pd.cut(before_flat, bins=bins, labels=labels)
    after_categories = pd.cut(after_flat, bins=bins, labels=labels)

    before_categories_nomination = pd.cut(before_nomination_flat, bins=bins, labels=labels)
    after_categories_nomination = pd.cut(after_nomination_flat, bins=bins, labels=labels)

    after_winner_categories = pd.cut(after_flat_winner, bins=bins, labels=labels)
    after_looser_categories = pd.cut(after_flat_looser, bins=bins, labels=labels)

    # Calculate the proportions
    before_counts = before_categories.value_counts() / len(before_flat)
    after_counts = after_categories.value_counts() / len(after_flat)

    before_counts_nomination = before_categories_nomination.value_counts() / len(before_nomination_flat)
    after_counts_nomination = after_categories_nomination.value_counts() / len(after_nomination_flat)

    after_winner_counts = after_winner_categories.value_counts() / len(after_flat_winner)
    after_looser_counts = after_looser_categories.value_counts() / len(after_flat_looser)
    
    # Calculate the change in the distribution
    change_ceremony = after_counts - before_counts
    change_nomination = after_counts_nomination - before_counts_nomination
    change_winner = after_winner_counts - after_looser_counts

    # Plot the change in the distribution
    x = range(len(labels))
    
    plt.bar(x, change_ceremony, width=0.4, label="Change After Ceremony", align='center', alpha=0.7, color="b")
    plt.axhline(0, color='black',linewidth=1)  
    
    plt.xticks(x, labels, rotation=45)
    plt.xlabel("Sentiment Categories")
    plt.ylabel("Change in Proportion")
    plt.title("Change in Compound Scores Distribution After the Ceremony")
    plt.tight_layout()
    
    plt.show()

    plt.bar(x, change_nomination, width=0.4, label="Change After Nomination", align='center', alpha=0.7, color="g")
    plt.axhline(0, color='black',linewidth=1) 
    
    plt.xticks(x, labels, rotation=45)
    plt.xlabel("Sentiment Categories")
    plt.ylabel("Change in Proportion")
    plt.title("Change in Compound Scores Distribution After the Nomination")
    plt.tight_layout()
    
    plt.show()
    
    plt.bar(x, change_winner, width=0.4, label="Change for a Winner compared to a Looser", align='center', alpha=0.7, color="r")
    plt.axhline(0, color='black',linewidth=1)
    plt.xticks(x, labels, rotation=45)
    plt.xlabel("Sentiment Categories")
    plt.ylabel("Change in Proportion")
    plt.title("Change in Compound Scores Distribution After Ceremony: Winners vs. Losers")
    plt.tight_layout()
    
    plt.show()

