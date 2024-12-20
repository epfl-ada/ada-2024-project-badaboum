import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils import resample


def parse_str_to_list(column):
    """
    Parses a column of strings, splitting each string by commas to
    create a list for each entry.

    Parameters:
    column (pd.Series): A pandas Series containing strings separated by commas.

    Returns:
    pd.Series: A Series where each element is a list.
               If an element in the original Series is null, it returns an empty list.
    """
    return column.apply(
        lambda x: [text.strip() for text in x.split(",")] if pd.notnull(x) else []
    )


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


def balance_dataset(data: pd.DataFrame, target: str):
    """
    Balance the dataset by oversampling the minority class.

    Args:
        data (pd.DataFrame): The dataset to balance.
        target (str): The column to balance on.

    Returns:
        pd.DataFrame: Balanced dataset.
    """
    majority = data[data[target] == False]
    minority = data[data[target] == True]

    # Oversample the minority class
    minority_upsampled = resample(
        minority, replace=True, n_samples=len(majority), random_state=42
    )

    # Combine the balanced dataset
    balanced_data = pd.concat([majority, minority_upsampled]).reset_index(drop=True)

    return balanced_data


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
