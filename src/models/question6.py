import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from ..utils.data_parsing import parse_str_to_list
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils import resample


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

    # Add "nominated" column to nominees and drop "winner" column
    nominees["nominated"] = True
    nominees = nominees.drop(columns=["winner"])

    # Add "nominated" column to non_nominees with False
    non_nominees["nominated"] = False

    # Combine both dataframes
    data = pd.concat([nominees, non_nominees], ignore_index=True)

    return data


# Calculate feature frequencies for nominees and non-nominees
def calculate_distribution(feature_list):
    all_items = [item for items in feature_list for item in items]
    item_counts = Counter(all_items)
    total = sum(item_counts.values())
    return {item: count / total for item, count in item_counts.items()}


def plot_distribution(data: pd.DataFrame, feature: str, top_n: int = None):
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


def calculate_vif(X: pd.DataFrame):
    """
    Calculate Variance Inflation Factor (VIF) for each feature.
    Args:
        X (pd.DataFrame): DataFrame of predictors.

    Returns:
        pd.DataFrame: DataFrame with features and their corresponding VIF scores.
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    return vif_data


def filter_high_vif(X: pd.DataFrame, threshold=5.0):
    """
    Remove features with high VIF scores from the dataset.

    Args:
        X (pd.DataFrame): DataFrame of predictors.
        threshold (float): VIF score threshold above which features are removed.

    Returns:
        pd.DataFrame: Filtered DataFrame with low VIF features.
    """
    while True:
        vif = calculate_vif(X)
        max_vif = vif["VIF"].max()
        if max_vif > threshold:
            feature_to_drop = vif.loc[vif["VIF"] == max_vif, "Feature"].values[0]
            print(f"Dropping feature '{feature_to_drop}' with VIF: {max_vif}")
            X = X.drop(columns=[feature_to_drop])
        else:
            break
    return X


def balance_dataset(data, target_column="nominated"):
    """
    Balance the dataset by oversampling the minority class.

    Args:
        data (pd.DataFrame): The dataset to balance.
        target_column (str): The column to balance on.

    Returns:
        pd.DataFrame: Balanced dataset.
    """
    majority = data[data[target_column] == False]
    minority = data[data[target_column] == True]

    # Oversample the minority class
    minority_upsampled = resample(
        minority, replace=True, n_samples=len(majority), random_state=42
    )

    # Combine the balanced dataset
    balanced_data = pd.concat([majority, minority_upsampled]).reset_index(drop=True)

    return balanced_data


def ols_categorical(data: pd.DataFrame, feature: str, top_n: int = 20):
    """
    Perform OLS regression on a balanced dataset and visualize significant predictors.

    Args:
        data (pd.DataFrame): Input dataset.
        feature (str): The feature to analyze (e.g., 'IMDB_genres').
        top_n (int): Number of top predictors to visualize if significant.
    """
    # Balance the dataset
    balanced_data = balance_dataset(data)

    # One hot encode genres
    X = balanced_data[feature].explode()
    X = pd.get_dummies(X).groupby(level=0).sum()
    y = balanced_data["nominated"]

    X = filter_high_vif(X)

    # Add a constant for intercept
    X = sm.add_constant(X)

    # Fit the OLS model
    ols_model = sm.OLS(y, X).fit()

    # Summary of the OLS model
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
        top_params = significant_params.abs().nlargest(top_n).index
        sorted_params = significant_params.loc[top_params].sort_values(ascending=True)
    else:
        sorted_params = significant_params.sort_values(ascending=True)

    # Visualize coefficients
    plt.figure(figsize=(10, 6))
    sorted_params.plot(kind="barh")
    plt.title(f"OLS Model Coefficients: Top {top_n} Significant Predictors")
    plt.xlabel("Coefficient")
    plt.ylabel(feature)
    plt.axvline(0, color="red", linestyle="--")
    plt.tight_layout()
    plt.show()


def ols_continuous(data: pd.DataFrame, feature: str):
    """
    Perform OLS regression on runtime with a balanced dataset.
    """
    # Balance the dataset
    balanced_data = balance_dataset(data)

    # Drop rows with missing runtime
    balanced_data = balanced_data.dropna(subset=["runtime"])

    # Define the predictor (X) and target (y)
    X_runtime = balanced_data[["runtime"]]
    y_nominated = balanced_data["nominated"]

    # Add a constant for intercept
    X_runtime = sm.add_constant(X_runtime)

    # Fit the OLS model
    ols_model_runtime = sm.OLS(y_nominated, X_runtime).fit()

    # Summary of the runtime-focused OLS model
    ols_summary_runtime = ols_model_runtime.summary()

    print(ols_summary_runtime)


def plot_runtime_distribution(
    data: pd.DataFrame,
):
    """
    Plot the distribution of runtime for nominees and non-nominees.
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
        labels=["nominees", "Non-nominees"],
        showfliers=False,
    )
    plt.xlabel("Category")
    plt.ylabel("Runtime (minutes)")
    plt.title("Runtime Distribution (Without Outliers): nominees vs. Non-nominees")
    plt.show()
