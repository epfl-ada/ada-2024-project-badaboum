import seaborn as sns
import matplotlib.pyplot as plt

from src.models.reviews_over_time.datasets_loading import load_reviews_augmented
from src.models.reviews_over_time.utils import *

def plot_compound_scores_individual(imdb_id):
    """
    Plot the compound scores of the reviews over time for a specific movie.
    
    Parameters:
        imdb_id (str): The IMDb ID of the movie.
    """
    # Get the data
    df_init = load_reviews_augmented()

    # Prepare the data
    positive_reviews, negative_reviews = split_compound_scores_individual(df_init, imdb_id)

    # Plot the compound scores
    sns.scatterplot(data=positive_reviews, x="date", y="text_compound")
    
    sns.scatterplot(data=negative_reviews, x="date", y="text_compound") 

    plt.show()

    sns.kdeplot(data=positive_reviews[["date", "text_compound"]], x="date")
    sns.kdeplot(data=negative_reviews[["date", "text_compound"]], x="date")


def plot_compound_scores_global(type_1, type_2):
    """
    Plot the compound scores of the reviews over time for two different types of movies.
    
    Parameters:
        type_1 (str): The first type of movie.
        type_2 (str): The second type of movie.
    """
    # Select the data
    df_1, df_2 = select_visualization_groups(type_1, type_2)

    ax = sns.kdeplot(data=df_1[["relative_date", "text_compound"]], x="relative_date", label = type_1)
    sns.kdeplot(data=df_2[["relative_date", "text_compound"]], x="relative_date", label = type_2)

    # Add a title and axis labels
    ax.set_title("Density Distributions of Reviews over Time", fontsize=16)
    ax.set_xlabel("Relative Date (Days)", fontsize=14)
    ax.set_ylabel("Distribution Density", fontsize=14)
    
    # Add a legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Review Types")
    
    # Adjust the layout to make space for the legend
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_oscar_timeline_unique(imdb_id,type_):
    """
    Plot the compound scores or the review count over time in a line for a specific movie.
    
    Parameters:
        imdb_id (str): The IMDb ID of the movie.
        type_ (str): The type of the plot we want to generate. It can be "compound" or "count"
    """
    # Get the data
    df_init = load_reviews_augmented()

    # Prepare the data
    grouped_reviews_mean, grouped_reviews_count = compute_timeline_unique(df_init, imdb_id)

    # Set the theme of the plot
    sns.set_theme(style="darkgrid", rc={'figure.figsize':(11.7,8.27)})

    # Set the plot titles
    title = ""
    xlabel = ""
    ylabel = ""
    df = df_init

    # Select the data to plot
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