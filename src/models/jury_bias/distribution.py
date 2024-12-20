import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


# Calculate feature frequencies for nominees and non-nominees
def calculate_distribution(feature_list):
    all_items = [item for items in feature_list for item in items]
    item_counts = Counter(all_items)
    total = sum(item_counts.values())
    return {item: count / total for item, count in item_counts.items()}


def plot_categorical_distribution(data: pd.DataFrame, feature: str, top_n: int = None):
    """
    Generic function to plot the distribution of a specified feature.

    Args:
        feature (str): Column name to analyze (e.g., 'IMDB_genres', 'countries').
        top_n (int, optional): Number of top categories to include. If None, include all.
    """
    # Split the data into nominees and non-nominees
    nominees = data[data["nominated"]]
    non_nominees = data[~data["nominated"]]

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


def plot_runtime_distribution(
    data: pd.DataFrame,
):
    """
    Plot the distribution of runtime for nominees and non-nominees.

    Args:
        data (pd.DataFrame): The dataset.
    """
    # Split the data into nominees and non-nominees
    nominees = data[data["nominated"]]
    non_nominees = data[~data["nominated"]]

    # Extract runtimes
    nominee_runtimes = nominees["runtime"].dropna()
    non_nominee_runtimes = non_nominees["runtime"].dropna()

    # Plot box plot comparison without showing outliers
    plt.figure(figsize=(6, 4))
    plt.boxplot(
        [nominee_runtimes, non_nominee_runtimes],
        labels=["Nominees", "Non-nominees"],
        showfliers=False,
    )
    plt.xlabel("Category")
    plt.ylabel("Runtime (minutes)")
    plt.title("Runtime Distribution: nominees vs. Non-nominees")
    plt.show()
