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
        label="Nominees",
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


def ols(feature: str, top_n: int = 20):
    """
    Perform OLS regression using the specified feature and visualize significant predictors.

    Args:
        feature (str): The feature to analyze (e.g., 'IMDB_genres').
        top_n (int): Number of top predictors to visualize if significant. Default is 20.
    """
    # Load the data
    nominees, non_nominees = load_data()

    # Add "nominated" column to nominees and drop "winner" column
    nominees["nominated"] = True
    nominees = nominees.drop(columns=["winner"])

    # Add "nominated" column to non_nominees with False
    non_nominees["nominated"] = False

    # Combine both dataframes
    data = pd.concat([nominees, non_nominees], ignore_index=True)

    # One hot encode genres
    X = data[feature].explode()
    X = pd.get_dummies(X).groupby(level=0).sum()
    y = data["nominated"]  # Target

    # Add a constant for intercept
    X = sm.add_constant(X)

    # Fit the OLS model
    ols_model = sm.OLS(y, X).fit()

    # Summary of the genre-focused OLS model
    ols_summary = ols_model.summary()

    # Display the OLS model summary
    print(ols_summary)

    # Get coefficients and p-values excluding the intercept
    params = ols_model.params[1:]  # Exclude intercept
    p_values = ols_model.pvalues[1:]  # Exclude intercept

    # Filter significant predictors (p < 0.05)
    significant_params = params[p_values < 0.05]

    # Select top_n by absolute value
    if top_n:
        top_params = (
            significant_params.abs().nlargest(top_n).index
        )  # Get top_n absolute values
        sorted_params = significant_params.loc[top_params].sort_values(
            ascending=True
        )  # Sort by actual values
    else:
        sorted_params = significant_params.sort_values(
            ascending=True
        )  # Sort all by actual values

    # Adjust dynamic height for visualization
    height_per_elem = 0.5  # Height per bar
    dynamic_height = max(8, len(sorted_params) * height_per_elem)  # Minimum height

    # Visualize coefficients and p-values
    plt.figure(figsize=(10, dynamic_height))
    sorted_params.plot(kind="barh")  # Horizontal bar chart

    plt.title(f"OLS Model Coefficients: Top {top_n} Significant Predictors")
    plt.xlabel("Coefficient")
    plt.ylabel(feature)
    plt.axvline(0, color="red", linestyle="--")  # Reference line at 0
    plt.tight_layout()
    plt.show()


def ols_runtime():
    """
    Fit an OLS model using runtime as the predictor for nomination.
    """
    # Load the data
    nominees, non_nominees = load_data()

    # Add "nominated" column and drop "winner" column
    nominees["nominated"] = True
    nominees = nominees.drop(columns=["winner"])

    non_nominees["nominated"] = False

    # Combine both datasets
    data = pd.concat([nominees, non_nominees], ignore_index=True)

    # Drop rows with missing runtime
    data = data.dropna(subset=["runtime"])

    # Define the predictor (X) and target (y)
    X_runtime = data[["runtime"]]
    y_nominated = data["nominated"]

    # Add a constant for intercept
    X_runtime = sm.add_constant(X_runtime)

    # Fit the OLS model
    ols_model_runtime = sm.OLS(y_nominated, X_runtime).fit()

    # Summary of the runtime-focused OLS model
    ols_summary_runtime = ols_model_runtime.summary()

    return ols_summary_runtime


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
