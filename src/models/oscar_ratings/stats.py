from src.models.oscar_ratings.datasets_loading import (
    load_oscar_winners_nominees_all_categories,
    load_oscar_movies_all_categories,
)
from scipy.stats import ttest_ind
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import numpy as np  # Keep this import for the log transformation


def print_ttest_winner_vs_nominees_ratings():
    oscar_winners, oscar_nominees = load_oscar_winners_nominees_all_categories()

    # Perform a t-test to determine if the average ratings of winners and nominees are significantly different
    t_stat, p_val = ttest_ind(
        oscar_winners["averageRating"], oscar_nominees["averageRating"]
    )
    print("T-statistic: ", t_stat)
    print("P-value: ", p_val)


def print_reg_ratings_release_year():
    oscar_winner_nominee_df = pd.concat(load_oscar_winners_nominees_all_categories())

    # Perform an ols regression
    model = sm.OLS.from_formula("averageRating ~ release", data=oscar_winner_nominee_df)
    results = model.fit()
    print(results.summary())


def print_pearson_corr_ratings_numVotes():
    oscar_winner_nominee_df = pd.concat(load_oscar_winners_nominees_all_categories())

    # Compute pearson correlation between number of votes and ratings
    corr = oscar_winner_nominee_df["numVotes"].corr(
        oscar_winner_nominee_df["averageRating"], method="pearson"
    )
    print("Pearson correlation between numVotes and averageRating: ", corr)


def print_spearman_corr_ratings_numVotes():
    oscar_winner_nominee_df = pd.concat(load_oscar_winners_nominees_all_categories())

    # Compute spearman correlation between number of votes and ratings
    corr = oscar_winner_nominee_df["numVotes"].corr(
        oscar_winner_nominee_df["averageRating"], method="spearman"
    )
    print("Spearman correlation between numVotes and averageRating: ", corr)


def print_reg_ratings_log_numVotes():
    oscar_winner_nominee_df = pd.concat(load_oscar_winners_nominees_all_categories())

    # Perform an ols regression
    model = sm.OLS.from_formula(
        "averageRating ~ np.log(numVotes)", data=oscar_winner_nominee_df
    )
    results = model.fit()
    print(results.summary())


def print_reg_ratings_all():
    oscar_winner_nominee_df = pd.concat(load_oscar_winners_nominees_all_categories())

    oscar_winner_nominee_df["log_numVotes"] = np.log(
        oscar_winner_nominee_df["numVotes"]
    )

    scaler = StandardScaler()
    oscar_winner_nominee_df[["release_scaled", "log_numVotes_scaled"]] = (
        scaler.fit_transform(oscar_winner_nominee_df[["release", "log_numVotes"]])
    )

    # Perform an OLS regression
    model = sm.OLS.from_formula(
        "averageRating ~ release_scaled + log_numVotes_scaled + C(any_win)",
        data=oscar_winner_nominee_df,
    )
    results = model.fit(cov_type="HC3")
    print(results.summary())


def print_oscar_categories(min_samples: int = 10):
    oscar_movies = load_oscar_movies_all_categories()

    categories = oscar_movies["oscar_category"].value_counts()
    categories = categories[categories > min_samples]

    for category in categories.index:
        print(f"Category: {category}, Number of samples: {categories[category]}")
