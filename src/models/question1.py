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


def get_top_box_office_movies(
    other_movies: pd.DataFrame,
    oscar_winners: pd.DataFrame,
    oscar_nominees: pd.DataFrame,
    number_top_movies_per_year: int = 1,
) -> pd.DataFrame:
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

    return top_movies


def plot_winner_vs_nominees_vs_box_office_hit_ratings():
    oscar_winners, oscar_nominees = load_oscar_winners_nominees()
    other_movies = load_other_movies()

    top_movies = get_top_box_office_movies(other_movies, oscar_winners, oscar_nominees)

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


def get_audience_oscar_concordance():
    oscar_winners, oscar_nominees = load_oscar_winners_nominees()
    other_movies = load_other_movies()

    # Combine the data
    combined_data = pd.concat([oscar_winners, oscar_nominees, other_movies])

    # For each year, get the rating of the oscars winner, and the highest raiting movie
    audience_winners = []
    top_ratings = []

    # Get the top rated movie(s) of each year
    for year in combined_data["release"].unique():

        # Get the top rated movies of the year
        movies_of_year = combined_data[combined_data["release"] == year]
        top_rated_movies = movies_of_year.sort_values("averageRating", ascending=False)

        # Keep the top rated movies with equal top rating
        top_rated_movies = top_rated_movies[
            top_rated_movies["averageRating"]
            == top_rated_movies.iloc[0]["averageRating"]
        ]

        audience_winners.append((int(year), top_rated_movies["primaryTitle"].values))
        top_ratings.append((int(year), top_rated_movies["averageRating"].values[0]))

    top_rated_movies_df = pd.DataFrame(top_ratings, columns=["release", "topRating"])

    # Check if the top rated movie of the year was the Oscar winner
    audience_oscar_same = []
    for year, movies in audience_winners:
        # Get the oscars winner of the year
        oscars_winner = combined_data[
            (combined_data["release"] == year) & (combined_data["winner"] == True)
        ]
        oscars_winner = oscars_winner["primaryTitle"].values

        if len(oscars_winner) == 0:
            continue

        # Check if the oscars winner is in the top rated movies
        if oscars_winner[0] in movies:
            audience_oscar_same.append((int(year), True))
        else:
            audience_oscar_same.append((int(year), False))

    # Create a new dataframe
    audience_oscar_same_df = pd.DataFrame(
        audience_oscar_same, columns=["release", "same"]
    )

    return audience_oscar_same_df, top_rated_movies_df


def plot_oscar_winners_vs_audience_concordance():
    audience_oscar_same_df = get_audience_oscar_concordance()

    # Plot the years where the top rated movie was the Oscar winner
    plt.figure(figsize=(10, 2))
    sns.barplot(x="release", y="same", data=audience_oscar_same_df)

    plt.yticks([0, 1], ["No", "Yes"])
    plt.ylabel("Top rated movie was the Oscar winner")

    plt.xlabel("Year")
    plt.xticks(rotation=90)
    plt.title("Was the top rated movie the Oscar winner?")

    plt.show()


def plot_oscar_winners_vs_audience_ratings_gap():
    _, top_rated_movies_df = get_audience_oscar_concordance()
    oscar_winners, _ = load_oscar_winners_nominees()

    # Merge the data
    top_rated_vs_oscar = pd.merge(
        top_rated_movies_df, oscar_winners, on="release", how="inner"
    )

    # Get the gap between the top rated movie and the Oscar winner
    top_rated_vs_oscar["gap_percent"] = (
        (top_rated_vs_oscar["topRating"] - top_rated_vs_oscar["averageRating"])
        / top_rated_vs_oscar["averageRating"]
        * 100
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(x="release", y="gap_percent", data=top_rated_vs_oscar)
    plt.xticks(rotation=90)

    plt.ylabel("Difference in ratings (%)")
    plt.xlabel("Year")

    plt.title(
        "Difference in ratings between the top rated movie and the Oscar winner (%)"
    )
    plt.show()
