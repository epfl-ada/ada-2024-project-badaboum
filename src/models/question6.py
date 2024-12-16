import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from ..utils.data_parsing import parse_str_to_list
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


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


def correlation(feature: str = "IMDB_genres"):
    """
    Calculate the correlation between the nominated category and the specified feature.
    """
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

    # Calculate the correlation matrix
    correlation_matrix = X_genres.corrwith(y_genres)

    # Visualize the correlation matrix
    plt.figure(figsize=(10, 6))
    correlation_matrix.plot(kind="bar", color="skyblue")

    plt.title("Correlation between Genres and Nomination")
    plt.ylabel("Correlation")
    plt.xlabel("Genre")
    plt.xticks(rotation=90)
    plt.show()

    return correlation_matrix


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

    # Visualize coefficients and p-values
    plt.figure(figsize=(10, 6))
    # Sort the coefficients by value
    ols_model_genres.params[1:] = ols_model_genres.params[1:].sort_values()
    ols_model_genres.params[1:].plot(kind="bar", color="skyblue")

    plt.title("OLS Model Coefficients: Genres")
    plt.ylabel("Coefficient")
    plt.xlabel("Genre")
    plt.xticks(rotation=90)
    plt.show()

    return ols_summary_genres


def vif(feature: str = "IMDB_genres"):
    """
    Calculate the Variance Inflation Factor (VIF) for the feature.
    """
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

    # Calculate the Variance Inflation Factor (VIF)
    vif_genres = pd.DataFrame()
    vif_genres["feature"] = X_genres.columns
    vif_genres["VIF"] = [
        variance_inflation_factor(X_genres.values, i) for i in range(X_genres.shape[1])
    ]

    return vif_genres


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
