import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from ..utils.data_parsing import parse_str_to_list
import statsmodels.api as sm


def load_data():

    PATH = "data/"
    nominees = pd.read_csv(f"{PATH}oscar_movies.csv")
    non_nominees = pd.read_csv(f"{PATH}all_other_movies.csv")

    # parse genres
    nominees["IMDB_genres"] = parse_str_to_list(nominees["IMDB_genres"])
    non_nominees["IMDB_genres"] = parse_str_to_list(non_nominees["IMDB_genres"])

    # drop \N genres
    nominees["IMDB_genres"] = nominees["IMDB_genres"].apply(
        lambda x: [genre for genre in x if genre != r"\N"]
    )
    non_nominees["IMDB_genres"] = non_nominees["IMDB_genres"].apply(
        lambda x: [genre for genre in x if genre != r"\N"]
    )

    # parse countries
    nominees["countries"] = parse_str_to_list(nominees["countries"])
    non_nominees["countries"] = parse_str_to_list(non_nominees["countries"])

    return nominees, non_nominees


# Calculate feature frequencies for nominees and non-nominees
def calculate_distribution(feature_list):
    all_items = [item for items in feature_list for item in items]
    item_counts = Counter(all_items)
    total = sum(item_counts.values())
    return {item: count / total for item, count in item_counts.items()}


def plot_distribution(feature: str, top_n: int = None):
    """
    Generic function to plot the distribution of a specified feature.

    Args:
        feature (str): Column name to analyze (e.g., 'IMDB_genres', 'countries').
        top_n (int, optional): Number of top categories to include. If None, include all.
    """
    # Load the data
    nominees, non_nominees = load_data()

    nominee_distribution = calculate_distribution(nominees[feature])
    non_nominee_distribution = calculate_distribution(non_nominees[feature])

    # Align categories by creating a union of all keys
    all_items = set(nominee_distribution.keys()).union(
        set(non_nominee_distribution.keys())
    )
    aligned_nominee_distribution = {
        item: nominee_distribution.get(item, 0) for item in all_items
    }
    aligned_non_nominee_distribution = {
        item: non_nominee_distribution.get(item, 0) for item in all_items
    }

    # Optionally limit to top `n` categories
    if top_n:
        combined_distribution = {
            item: aligned_nominee_distribution[item]
            + aligned_non_nominee_distribution[item]
            for item in all_items
        }
        top_items = sorted(
            combined_distribution.keys(),
            key=lambda x: combined_distribution[x],
            reverse=True,
        )[:top_n]
        aligned_nominee_distribution = {
            item: aligned_nominee_distribution[item] for item in top_items
        }
        aligned_non_nominee_distribution = {
            item: aligned_non_nominee_distribution[item] for item in top_items
        }
        all_items = top_items

    # Plot side-by-side bars for the feature distribution
    x = np.arange(len(all_items))  # the label locations
    width = 0.35  # the width of the bars

    plt.figure(figsize=(10, 6))
    plt.bar(
        x - width / 2,
        aligned_nominee_distribution.values(),
        width,
        label="nominees",
    )
    plt.bar(
        x + width / 2,
        aligned_non_nominee_distribution.values(),
        width,
        label="Non-nominees",
    )
    plt.xlabel(feature.capitalize())
    plt.ylabel("Proportion")
    plt.title(f"{feature.capitalize()}: nominees vs. Non-nominees")
    plt.legend()
    plt.xticks(x, all_items, rotation=90)
    plt.show()


def plot_bias(feature: str, top_n: int = None):
    """
    Generic function to calculate and plot bias factors for a specified feature.

    Args:
        feature (str): Column name to analyze (e.g., 'IMDB_genres', 'countries').
        top_n (int, optional): Number of top categories to include. If None, include all.
    """
    # Load the data
    nominees, non_nominees = load_data()

    nominee_distribution = calculate_distribution(nominees[feature])
    non_nominee_distribution = calculate_distribution(non_nominees[feature])

    # Align categories between nominee and non-nominee distributions
    all_items = set(nominee_distribution.keys()).union(
        set(non_nominee_distribution.keys())
    )

    # Create aligned distributions with 0 for missing categories
    aligned_nominee_distribution = {
        item: nominee_distribution.get(item, 0) for item in all_items
    }
    aligned_non_nominee_distribution = {
        item: non_nominee_distribution.get(item, 0) for item in all_items
    }

    # Calculate the bias factor for each category
    bias_factors = {
        item: (
            aligned_nominee_distribution[item] / aligned_non_nominee_distribution[item]
            if aligned_non_nominee_distribution[item] > 0
            else np.nan  # Avoid division by zero
        )
        for item in all_items
    }

    # Sort categories by bias factor
    sorted_items = sorted(
        bias_factors.keys(), key=lambda x: bias_factors[x], reverse=True
    )

    # Optionally limit to top `n` categories
    if top_n:
        sorted_items = sorted_items[:top_n]

    # Plot bias factors
    plt.figure(figsize=(10, 6))
    plt.bar(
        sorted_items,
        [bias_factors[item] for item in sorted_items],
        color="skyblue",
    )
    plt.axhline(1, color="red", linestyle="--", label="No Bias")
    plt.xlabel(feature.capitalize())
    plt.ylabel("Bias Factor (nominees/Non-nominees)")
    plt.title(f"{feature.capitalize()} Bias Factors: nominees vs. Non-nominees")
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

    # Print sorted bias factors for detailed inspection
    for item in sorted_items:
        print(f"{item}: {bias_factors[item]:.2f}")


def ols(feature: str = "IMDB_genres"):
    # target is nominated
    # genres are the predictors
    # Load the data
    nominees, non_nominees = load_data()

    # Add "nominated" column to oscar_movies and drop "winner" column
    nominees["nominated"] = True
    nominees = nominees.drop(columns=["winner"])

    # Add "nominated" column to all_other_movies with False
    non_nominees["nominated"] = False

    # Combine both dataframes
    data = pd.concat([nominees, non_nominees], ignore_index=True)

    # One hot encode genres
    X_genres = data[feature].explode()
    X_genres = pd.get_dummies(X_genres).groupby(level=0).sum()
    y_genres = data["nominated"]  # Target

    # Add a constant for intercept
    X_genres = sm.add_constant(X_genres)

    # Fit the OLS model
    ols_model_genres = sm.OLS(y_genres, X_genres).fit()

    # Summary of the genre-focused OLS model
    ols_summary_genres = ols_model_genres.summary()
    return ols_summary_genres


def plot_runtime_distribution():

    # Load the data
    nominees, non_nominees = load_data()

    # Extract runtimes
    nominee_runtimes = nominees["runtime"].dropna()
    non_nominee_runtimes = non_nominees["runtime"].dropna()

    # Plot box plot comparison without showing outliers
    plt.figure(figsize=(6, 4))
    plt.boxplot(
        [nominee_runtimes, non_nominee_runtimes],
        labels=["nominees", "Non-nominees"],
        showfliers=False,
    )
    plt.xlabel("Category")
    plt.ylabel("Runtime (minutes)")
    plt.title("Runtime Distribution (Without Outliers): nominees vs. Non-nominees")
    plt.show()


def plot_top_country_distribution(top_n=10):
    # Load data
    nominees, non_nominees = load_data()

    nominee_country_distribution = calculate_distribution(nominees["countries"])
    non_nominee_country_distribution = calculate_distribution(non_nominees["countries"])

    # Combine distributions and find top `n` countries
    all_countries = set(nominee_country_distribution.keys()).union(
        set(non_nominee_country_distribution.keys())
    )
    combined_distribution = {
        country: nominee_country_distribution.get(country, 0)
        + non_nominee_country_distribution.get(country, 0)
        for country in all_countries
    }

    top_countries = sorted(
        combined_distribution.keys(),
        key=lambda x: combined_distribution[x],
        reverse=True,
    )[:top_n]

    # Aggregate remaining countries into "Others"
    def aggregate_top_countries(distribution, top_countries):
        aggregated_distribution = {
            country: distribution.get(country, 0) for country in top_countries
        }
        others = sum(
            value
            for country, value in distribution.items()
            if country not in top_countries
        )
        aggregated_distribution["Others"] = others
        return aggregated_distribution

    nominee_top_distribution = aggregate_top_countries(
        nominee_country_distribution, top_countries
    )
    non_nominee_top_distribution = aggregate_top_countries(
        non_nominee_country_distribution, top_countries
    )

    # Plot side-by-side bars for the top countries
    all_countries_top = list(nominee_top_distribution.keys())
    x = np.arange(len(all_countries_top))  # the label locations
    width = 0.35  # the width of the bars

    plt.figure(figsize=(10, 6))
    plt.bar(
        x - width / 2,
        nominee_top_distribution.values(),
        width,
        label="nominees",
    )
    plt.bar(
        x + width / 2,
        non_nominee_top_distribution.values(),
        width,
        label="Non-nominees",
    )
    plt.xlabel("Countries")
    plt.ylabel("Proportion")
    plt.title(f"Top {top_n} Country Distribution: nominees vs. Non-nominees")
    plt.legend()
    plt.xticks(x, all_countries_top, rotation=90)
    plt.show()
