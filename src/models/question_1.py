import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

DATA_PATH = "data/"


def load_oscar_movies() -> pd.DataFrame:
    return pd.read_csv(f"{DATA_PATH}oscar_movies.csv")


def load_oscar_winners_nominees() -> pd.DataFrame:
    data = load_oscar_movies()

    oscar_winners = data[data["winner"] == True]
    oscar_nominees = data[data["winner"] == False]

    return oscar_winners, oscar_nominees


# Load the other movies data (not oscar nominated)
def load_other_movies() -> pd.DataFrame:
    return pd.read_csv(f"{DATA_PATH}all_other_movies.csv")


def print_ttset_winner_vs_nomiees_ratings():
    oscar_winners, oscar_nominees = load_oscar_winners_nominees()

    # Perform a t-test to determine if the average ratings of winners and nominees are significantly different
    t_stat, p_val = ttest_ind(
        oscar_winners["averageRating"], oscar_nominees["averageRating"]
    )
    print("T-statistic: ", t_stat)
    print("P-value: ", p_val)


def plot_winner_vs_nominees_vs_box_office_hit_ratings():
    oscar_winners, oscar_nominees = load_oscar_winners_nominees()
    other_movies = load_other_movies()

    # Remove movies with missing revenue data
    other_movies = other_movies.dropna(subset=["revenue"])
    # Drop 0 revenue movies
    other_movies = other_movies[other_movies["revenue"] > 0]

    # Get the top box office hit of each year
    number_top_movies_per_year = 1

    top_movies = (
        other_movies.sort_values("revenue", ascending=False)
        .groupby("release")
        .head(number_top_movies_per_year)
    )
    top_movies = top_movies.reset_index(drop=True)

    # Keep only where we have data for oscars
    top_movies = top_movies[
        top_movies["release"].isin(
            list(oscar_winners["release"].values)
            + list(oscar_nominees["release"].values)
        )
    ]

    # Add to the box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=0, y="averageRating", data=top_movies)
    sns.boxplot(x=1, y="averageRating", data=oscar_nominees)
    sns.boxplot(x=2, y="averageRating", data=oscar_winners)
    plt.title("Average rating of Box office hits and Oscar winners and nominees")
    plt.xlabel("")
    plt.ylabel("Average Rating")
    plt.legend(["Box office hit", "Nominee", "Winner"])
    plt.xticks([0, 1, 2], ["Box office hit", "Nominee", "Winner"])
    plt.show()
