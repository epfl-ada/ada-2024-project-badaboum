import pandas as pd

from src.models.oscar_ratings.category_mapping import map_categories

DATA_PATH = "data/"


def load_oscar_movies_best_pict() -> pd.DataFrame:
    return pd.read_csv(f"{DATA_PATH}oscar_movies.csv")


def load_oscar_winners_nominees_best_pict() -> pd.DataFrame:
    data = load_oscar_movies_best_pict()

    oscar_winners = data[data["winner"] == True]
    oscar_nominees = data[data["winner"] == False]

    return oscar_winners, oscar_nominees


def load_oscar_movies_all_categories() -> pd.DataFrame:
    all_cat_movies = pd.read_csv(f"{DATA_PATH}oscar_movies_all_categories.csv")

    return all_cat_movies


def load_oscar_movies_new_categories() -> pd.DataFrame:
    oscar_movies = load_oscar_movies_all_categories()
    oscar_movies = map_categories(oscar_movies)

    return oscar_movies


# Load the other movies data (not oscar nominated)
def load_other_movies() -> pd.DataFrame:
    return pd.read_csv(f"{DATA_PATH}all_other_movies.csv")


def load_oscar_winners_nominees_all_categories() -> pd.DataFrame:
    data = load_oscar_movies_all_categories()
    winner_nominee_any_cat = (
        data.groupby(["tconst"]).any()[["winner"]].rename(columns={"winner": "any_win"})
    )
    winner_nominee_any_cat = (
        pd.merge(
            winner_nominee_any_cat,
            data.drop(
                columns=[
                    "oscar_category",
                    "oscar_year",
                    "ceremony_date",
                    "winner",
                    "revenue",
                ]
            ),
            on="tconst",
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    winners_any_cat = winner_nominee_any_cat[winner_nominee_any_cat["any_win"] == True]
    nominees_any_cat = winner_nominee_any_cat[
        winner_nominee_any_cat["any_win"] == False
    ]

    return winners_any_cat, nominees_any_cat
