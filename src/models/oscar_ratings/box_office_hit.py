import pandas as pd


def get_top_box_office_movies(
    other_movies: pd.DataFrame,
    oscar_winners: pd.DataFrame,
    oscar_nominees: pd.DataFrame,
    number_top_movies_per_year: int = 1,
) -> pd.DataFrame:
    # Remove movies with missing revenue data
    other_movies = other_movies.dropna(subset=["revenue"])
    # Drop 0 revenue movies
    other_movies = other_movies[other_movies["revenue"] > 0]

    # Get the top box office hit of each year
    number_top_movies_per_year = 1

    top_movies = (
        other_movies.sort_values("revenue", ascending=False)
        .groupby("release")
        .head(number_top_movies_per_year)
    )
    top_movies = top_movies.reset_index(drop=True)

    # Keep only where we have data for oscars
    top_movies = top_movies[
        top_movies["release"].isin(
            list(oscar_winners["release"].values)
            + list(oscar_nominees["release"].values)
        )
    ]

    return top_movies
