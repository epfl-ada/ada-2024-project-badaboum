import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from ..utils.data_parsing import parse_str_to_list


def load_data():

    PATH = "data/"
    winners = pd.read_csv(f"{PATH}oscar_movies.csv")
    non_winners = pd.read_csv(f"{PATH}all_other_movies.csv")

    # parse genres
    winners["IMDB_genres"] = parse_str_to_list(winners["IMDB_genres"])
    non_winners["IMDB_genres"] = parse_str_to_list(non_winners["IMDB_genres"])
    return winners, non_winners


def plot_genre_distribution():

    # Load the data
    winners, non_winners = load_data()

    # Calculate genre frequencies for winners and non-winners
    def calculate_genre_distribution(genres_list):
        all_genres = [genre for genres in genres_list for genre in genres]
        genre_counts = Counter(all_genres)
        total = sum(genre_counts.values())
        return {genre: count / total for genre, count in genre_counts.items()}

    winner_genre_distribution = calculate_genre_distribution(winners["IMDB_genres"])
    non_winner_genre_distribution = calculate_genre_distribution(
        non_winners["IMDB_genres"]
    )

    # Align genres between winner and non-winner distributions
    all_genres = set(winner_genre_distribution.keys()).union(
        set(non_winner_genre_distribution.keys())
    )

    # Create aligned genre distributions with 0 for missing genres
    aligned_winner_genre_distribution = {
        genre: winner_genre_distribution.get(genre, 0) for genre in all_genres
    }
    aligned_non_winner_genre_distribution = {
        genre: non_winner_genre_distribution.get(genre, 0) for genre in all_genres
    }

    # Plotting side-by-side bars for genre distribution
    x = np.arange(len(all_genres))  # the label locations
    width = 0.35  # the width of the bars

    plt.figure(figsize=(10, 6))
    plt.bar(
        x - width / 2,
        aligned_winner_genre_distribution.values(),
        width,
        label="Winners",
    )
    plt.bar(
        x + width / 2,
        aligned_non_winner_genre_distribution.values(),
        width,
        label="Non-Winners",
    )
    plt.xlabel("Genres")
    plt.ylabel("Proportion")
    plt.title("Genre Distribution: Winners vs. Non-Winners")
    plt.legend()
    plt.xticks(x, all_genres, rotation=90)
    plt.show()
