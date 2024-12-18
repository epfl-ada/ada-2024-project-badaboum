from src.models.question1.datasets_loading import (
    load_oscar_movies_all_categories,
    load_oscar_movies_new_categories,
)
from scipy.stats import ttest_ind
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import numpy as np  # Keep this import for the log transformation
import networkx as nx


def get_causal_effect_for_cat(
    winners_nominees_df: pd.DataFrame, category: str, min_matches: int = 10
):
    winners_nominees_df = winners_nominees_df[
        winners_nominees_df["oscar_category"] == category
    ].reset_index(drop=True)

    control_df: pd.DataFrame = winners_nominees_df[
        winners_nominees_df["winner"] == False
    ]
    treated_df: pd.DataFrame = winners_nominees_df[
        winners_nominees_df["winner"] == True
    ]

    G = nx.Graph()

    for control_id, control_row in control_df.iterrows():
        for treated_id, treated_row in treated_df.iterrows():

            if control_row["release"] == treated_row["release"]:
                diff_nb_votes = abs(control_row["numVotes"] - treated_row["numVotes"])

                G.add_edge(control_id, treated_id, weight=diff_nb_votes)

    # We want to minimize the difference in number of votes between the control and treated group
    matching = nx.min_weight_matching(G)

    if len(matching) < min_matches:
        return np.nan, np.nan

    # Compute the ATE
    def is_treated(node_id):
        return node_id in treated_df.index

    ate = 0
    deltas = []

    for pair in matching:
        control_id = pair[0] if is_treated(pair[1]) else pair[1]
        treated_id = pair[1] if is_treated(pair[1]) else pair[0]

        control_row = winners_nominees_df.loc[control_id]
        treated_row = winners_nominees_df.loc[treated_id]

        delta = treated_row["averageRating"] - control_row["averageRating"]
        ate += delta
        deltas.append(delta)

    ate /= len(matching)

    se_ate = np.std(deltas) / np.sqrt(len(matching))

    return ate, se_ate


def get_causal_effect_for_each_cat(
    oscar_movies_df: pd.DataFrame, min_occurences_for_cat: int = 10
):
    # Get the number of oscar wins for each movie
    nb_wins_df = (
        oscar_movies_df[["tconst", "winner"]][oscar_movies_df["winner"] == True]
        .groupby("tconst")
        .count()
        .reset_index()
    )

    # Remove nominees movies that won in another category
    oscar_movies_df = oscar_movies_df[
        ~(
            # The movie won in another category
            (oscar_movies_df["tconst"].isin(nb_wins_df["tconst"]))
            & (oscar_movies_df["winner"] == False)  # The movie is a nominee
        )
    ]

    single_oscar_movies_df = nb_wins_df[nb_wins_df["winner"] == 1]

    # Keep movies that won in a single category
    oscar_movies_df = oscar_movies_df[
        (oscar_movies_df["tconst"].isin(single_oscar_movies_df["tconst"]))
        | (oscar_movies_df["winner"] == False)
    ]

    categories = oscar_movies_df["oscar_category"].unique()
    ate_per_category = {}
    se_per_category = {}

    for category in categories:
        if (
            oscar_movies_df["oscar_category"].value_counts()[category]
            < min_occurences_for_cat
        ):
            continue

        ate, se_ate = get_causal_effect_for_cat(oscar_movies_df, category)

        # Check if the ATE was computed successfully
        if not np.isnan(ate):
            ate_per_category[category] = ate
            se_per_category[category] = se_ate

    return ate_per_category, se_per_category


def print_causal_effect_for_each_cat(oscar_movies_df: pd.DataFrame):
    ate_per_category, _ = get_causal_effect_for_each_cat(oscar_movies_df)

    # Sort the categories by ATE
    ate_per_category = dict(
        sorted(ate_per_category.items(), key=lambda item: item[1], reverse=True)
    )

    for category, ate in ate_per_category.items():
        print(f"Causal effect for category {category}: {ate}")


def get_causal_effect_for_new_cat(min_occurences_for_cat: int = 10):
    oscar_movies_df = load_oscar_movies_new_categories()

    return get_causal_effect_for_each_cat(oscar_movies_df, min_occurences_for_cat)


def get_causal_effect_for_base_cat(min_occurences_for_cat: int = 10):
    oscar_movies_df = load_oscar_movies_all_categories()

    return get_causal_effect_for_each_cat(oscar_movies_df, min_occurences_for_cat)


def print_causal_effect_for_base_cat():
    oscar_movies_df = load_oscar_movies_all_categories()

    print_causal_effect_for_each_cat(oscar_movies_df)


def print_causal_effect_for_new_cat():
    oscar_movies_df = load_oscar_movies_new_categories()

    print_causal_effect_for_each_cat(oscar_movies_df)
