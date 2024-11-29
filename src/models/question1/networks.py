from src.models.question1.datasets_loading import (
    load_oscar_movies_all_categories,
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
        return np.nan

    # Compute the ATE
    def is_treated(node_id):
        return node_id in treated_df.index

    ATE = 0

    for pair in matching:
        control_id = pair[0] if is_treated(pair[1]) else pair[1]
        treated_id = pair[1] if is_treated(pair[1]) else pair[0]

        control_row = winners_nominees_df.loc[control_id]
        treated_row = winners_nominees_df.loc[treated_id]

        ATE += treated_row["averageRating"] - control_row["averageRating"]

    ATE /= len(matching)

    return ATE


def get_causal_effect_for_each_cat(min_occurences_for_cat: int = 10):
    oscar_movies_df = load_oscar_movies_all_categories()

    categories = oscar_movies_df["oscar_category"].unique()
    ATE_per_category = {}

    for category in categories:
        if (
            oscar_movies_df["oscar_category"].value_counts()[category]
            < min_occurences_for_cat
        ):
            continue

        ATE = get_causal_effect_for_cat(oscar_movies_df, category)

        # Check if the ATE was computed successfully
        if not np.isnan(ATE):
            ATE_per_category[category] = ATE

    return ATE_per_category


def print_causal_effect_for_each_cat():
    ATE_per_category = get_causal_effect_for_each_cat()

    # Sort the categories by ATE
    ATE_per_category = dict(
        sorted(ATE_per_category.items(), key=lambda item: item[1], reverse=True)
    )

    for category, ATE in ATE_per_category.items():
        print(f"Causal effect for category {category}: {ATE}")
