import matplotlib.pyplot as plt
from src.models.oscar_ratings.kmean_categories import (
    get_embedded_categories,
    perform_kmeans,
    reduce_dim,
)
from src.models.oscar_ratings.datasets_loading import (
    load_oscar_winners_nominees_all_categories,
    load_oscar_winners_nominees_best_pict,
    load_oscar_movies_all_categories,
    load_other_movies,
)
from src.models.oscar_ratings.box_office_hit import get_top_box_office_movies
from src.models.oscar_ratings.networks import (
    get_causal_effect_for_new_cat,
    get_causal_effect_for_base_cat,
)
import seaborn as sns
import pandas as pd
import numpy as np


def plot_winner_vs_nominees_vs_box_office_hit_ratings_best_pict():
    oscar_winners, oscar_nominees = load_oscar_winners_nominees_best_pict()
    other_movies = load_other_movies()
    top_movies = get_top_box_office_movies(other_movies, oscar_winners, oscar_nominees)

    # Add to the box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=0, y="averageRating", data=top_movies)
    sns.boxplot(x=1, y="averageRating", data=oscar_nominees)
    sns.boxplot(x=2, y="averageRating", data=oscar_winners)
    plt.title(
        "Average rating distribution of Box office hits and Oscar winners and nominees"
    )
    plt.xlabel("")
    plt.ylabel("Average rating distribution")
    plt.legend(["Box office hit", "Nominee", "Winner"])
    plt.xticks([0, 1, 2], ["Box office hit", "Nominee", "Winner"])
    plt.show()


def plot_winner_vs_nominees_ratings_all_cat():
    oscar_winners, oscar_nominees = load_oscar_winners_nominees_all_categories()

    # Add to the box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=1, y="averageRating", data=oscar_nominees)
    sns.boxplot(x=2, y="averageRating", data=oscar_winners)
    plt.title("Average rating distribution of Oscar winners and nominees")
    plt.xlabel("")
    plt.ylabel("Average rating distribution")
    plt.legend(["Nominee", "Winner"])
    plt.xticks([0, 1], ["Nominee", "Winner"])
    plt.show()


def plot_ratings_over_release_year():
    oscar_winner_nominee_df = pd.concat(load_oscar_winners_nominees_all_categories())

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=oscar_winner_nominee_df, x="release", y="averageRating", hue="any_win"
    )
    sns.regplot(
        data=oscar_winner_nominee_df,
        x="release",
        y="averageRating",
        scatter=False,
        color="red",
        line_kws={"linestyle": "-", "linewidth": 3},
    )

    plt.title("Average Rating of Oscar Winners and Nominees Over Time")
    plt.xlabel("Release Year")
    plt.ylabel("Average Rating")
    plt.legend(title="Oscar Win", labels=["Yes", "No"])

    plt.show()


def plot_numvotes_vs_ratings():
    oscar_winner_nominee_df = pd.concat(load_oscar_winners_nominees_all_categories())

    plt.figure(figsize=(10, 6))

    scatter = sns.scatterplot(
        data=oscar_winner_nominee_df,
        x="numVotes",
        y="averageRating",
        hue="any_win",
    )

    plt.title("Number of Votes vs Average Rating")
    plt.xlabel("Number of Votes")
    plt.ylabel("Average Rating")
    scatter.legend_.set_title("Oscar Win")

    plt.show()


def plot_log_numvotes_vs_ratings():
    oscar_winner_nominee_df = pd.concat(load_oscar_winners_nominees_all_categories())

    oscar_winner_nominee_df["log_numVotes"] = oscar_winner_nominee_df["numVotes"].apply(
        lambda x: 0 if x == 0 else np.log(x)
    )

    oscar_winner_nominee_df = oscar_winner_nominee_df.sort_values(by="any_win")

    plt.figure(figsize=(10, 6))

    scatter = sns.scatterplot(
        data=oscar_winner_nominee_df,
        x="log_numVotes",
        y="averageRating",
        hue="any_win",
        alpha=0.6,
        edgecolor=None,
    )

    plt.title("Number of Votes vs Average Rating")
    plt.xlabel("log(Number of Votes)")
    plt.ylabel("Average Rating")
    scatter.legend_.set_title("Oscar Win")

    plt.show()


def plot_causal_effect_of_each_cat(cat_effects, cat_standard_errors):
    sorted_cat_effects = sorted(cat_effects.items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    sns.barplot(
        x=[x[1] for x in sorted_cat_effects],
        y=[x[0] for x in sorted_cat_effects],
        ax=ax,
    )

    # Add error bars
    for i, category in enumerate(sorted_cat_effects):
        ax.errorbar(
            x=category[1],
            y=i,
            xerr=cat_standard_errors[category[0]],
            fmt="o",
            color="black",
        )

    plt.title("Causal effect of winning an oscar on the rating of a movie")
    plt.xlabel("Causal effect (in rating points)")
    plt.ylabel("Category")


def plot_causal_effect_of_new_cat():
    cat_effects, cat_standard_errors = get_causal_effect_for_new_cat()
    plot_causal_effect_of_each_cat(cat_effects, cat_standard_errors)


def plot_causal_effect_of_base_cat():
    cat_effects, cat_standard_errors = get_causal_effect_for_base_cat()
    plot_causal_effect_of_each_cat(cat_effects, cat_standard_errors)


def plot_ratings_vs_nb_oscars():
    oscar_movies_df = load_oscar_movies_all_categories()
    nb_oscars = (
        oscar_movies_df[["tconst", "averageRating", "winner"]]
        .groupby(["tconst", "averageRating"])
        .sum()["winner"]
        .reset_index()
    )
    nb_oscars["winner"].value_counts()

    sns.boxplot(x="winner", y="averageRating", data=nb_oscars)
    plt.title("Average rating of movies vs number of oscars won")
    plt.xlabel("Number of oscars won")
    plt.ylabel("Average rating")
    plt.show()


# Clustering of categories
def plot_elbow_curve(sses, optimal_k, k_start: int = 2, k_end: int = 30):
    k_list = range(k_start, k_end)

    # Plot the sses
    plt.plot(k_list, sses)
    plt.xlabel("K")
    plt.ylabel("Sum of Squared Errors")
    plt.grid(linestyle="--")

    # Add a point for the optimal k
    plt.scatter(optimal_k, sses[optimal_k - k_start], c="red")
    plt.legend(["SSE", "Optimal K"])

    plt.show()


def plot_clusters_2d(k: int, min_samples: int = 10):
    embedded_categories, categories = get_embedded_categories(min_samples)
    labels = perform_kmeans(embedded_categories, k)

    reduced_dim = reduce_dim(embedded_categories, 2)
    sns.set_theme(style="darkgrid", rc={"figure.figsize": (11.7, 8.27)})
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=reduced_dim[:, 0], y=reduced_dim[:, 1], hue=labels, palette="tab10"
    )
    plt.title("Clusters of categories (PCA 2D)")

    for i, text in enumerate(categories):
        # Dont saturate the image:
        if i % 3 == 0:
            plt.annotate(
                text, (reduced_dim[i, 0], reduced_dim[i, 1]), fontsize=8, alpha=0.8
            )

    plt.show()


def plot_clusters_3d(k: int, min_samples: int = 10):
    embedded_categories, _ = get_embedded_categories(min_samples)
    labels = perform_kmeans(embedded_categories, k)

    reduced_dim = reduce_dim(embedded_categories, 3)
    sns.set_theme(style="darkgrid", rc={"figure.figsize": (11.7, 8.27)})
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        reduced_dim[:, 0], reduced_dim[:, 1], reduced_dim[:, 2], c=labels, cmap="tab10"
    )
    plt.title("Clusters of categories (PCA 3D)")
    plt.show()
