import pandas as pd
from argparse import ArgumentParser
import ast

def parse_args():
    parser = ArgumentParser(description="Create final datasets")
    PATH = "../../data/"
    parser.add_argument(
        "--oscars_in",
        default=PATH + "oscar_winners_1929_2016.csv",
        nargs=2,
        type=str,
        help="Input file of the oscar nominee and database winners",
    )

    parser.add_argument(
        "--oscars_out",
        default=PATH + "oscar_movies.csv",
        type=str,
        help="Output file to the processed dataset of the oscar winners and nominees",
    )

    parser.add_argument(
        "--others_out",
        default=PATH + "other_movies.csv",
        type=str,
        help="Output file to the processed dataset of the other movies",
    )

    parser.add_argument(
        "--n_movies",
        default=1000,
        type=int,
        help="Number of movies kept in the other_movies database",
    )

    args = parser.parse_args()

    return args


def extract_countries_cmu(countries_column):
    """
    Extracts country names from a column containing string representations of dictionaries,
    where each dictionary maps keys to country names.

    Parameters:
    countries_column (pd.Series): A pandas Series with string representations of dictionaries,
                                  where each dictionary has country information.

    Returns:
    pd.Series: A Series where each element is a list of country names converted to string.
               If parsing fails for an entry, it returns an empty string.
    """

    def parse_country_dict(country_dict_str):
        try:
            country_dict = ast.literal_eval(
                country_dict_str
            )  # Convert string to dictionary
            # Join the country names into a single string, separated by commas
            return ", ".join(country_dict.values())  # Return only country names
        except (ValueError, SyntaxError):
            return ""  # Return empty list if parsing fails

    return countries_column.apply(parse_country_dict)

def oscars_processing(movies, oscar_winners):
    """
    Merges the movies data and the oscars winners to keep only data for oscar nominees/winners 
    
    Parameters:
    movies (pd.DataFrame): A pandas DataFrame containing the movies' data, from CMU and IMDB
    oscar_winners (pd.DataFrame): A pandas DataFrame containing movies nominated for the oscars
     
    Returns:
    pd.DataFrame: A DataFrame which contains the movies' data for nominated ones at the oscars for Best Motion Picture """

    # Inner join on the oscar and the movies, keeping only the ones nominated and winners
    df = pd.merge(
        movies,
        oscar_winners,
        left_on="primaryTitle",
        right_on="movie_name",
        how="inner",
    )

    # Renaming columns for better understanding
    df = df.rename(columns={"year": "oscar_year"})

    # Dropping unimportant columns
    df = df.drop(columns=["movie_name"])

    # Keeping only the categories of best picture and best motion picture
    oscars_picture = df[
        (df["oscar_category"] == "best motion picture")
        | (df["oscar_category"] == "best picture")
    ]
    oscars_picture = oscars_picture.reset_index(drop=True)

    # Realign with the date, the movies must be released the year before the ceremony
    oscars_picture = oscars_picture[
        oscars_picture["release"] == oscars_picture["oscar_year"] - 1
    ]

    # Remove movie with small number of ratings (there can be duplicates)
    oscars_picture = oscars_picture[oscars_picture["numVotes"] > 10]

    oscars_picture = oscars_picture.reset_index(drop=True)

    return oscars_picture


def others_processing(movies, oscar_winners, n):
    """
    Extracts the top n rated movies that were not nominated to oscars
    
    Parameters:
    movies (pd.DataFrame): A pandas DataFrame containing the movies' data, from CMU and IMDB
    oscar_winners (pd.DataFrame): A pandas DataFrame containing movies nominated for the oscars
     
    Returns:
    pd.DataFrame: A DataFrame which contains the movies' data for NOT nominated ones
    """
    # Left join
    df = pd.merge(
        movies,
        oscar_winners,
        left_on="primaryTitle",
        right_on="movie_name",
        how="left",
        indicator=True,
    )

    # Filter to keep only movies that did not win an Oscar
    df = df[df["_merge"] == "left_only"]

    # Dropping unimportant columns
    df = df.drop(columns=["_merge", "winner", "oscar_category", "year", "movie_name"])

    # Sort by descending over the rating
    df = df.sort_values(by="averageRating", ascending=False)

    # Keep only the top n
    return df.head(n)


def data_processing(oscar_winners, titles, ratings, metadata, n):
    """Creates a DataFrame that combines movies data, ratings and oscar win/nomination
    
    Parameters:
    movies (pd.DataFrame): A pandas DataFrame containing the movies' data, from CMU and IMDB
    oscar_winners (pd.DataFrame): A pandas DataFrame containing movies nominated for the oscars
     
    Returns:
    oscar_picture (pd.DataFrame): A DataFrame which contains the movies' data for nominated ones at the oscars for Best Motion Picture 
    others (pd.DataFrame) A DataFrame which contains the movies' data for NOT nominated ones
    """

    # Creating a df with the titles and their ratings from IMDB
    ratings_title = pd.merge(titles, ratings, on="tconst")

    # Lowercasing the titles
    ratings_title["primaryTitle"] = ratings_title["primaryTitle"].str.lower()
    metadata["name"] = metadata["name"].str.lower()

    movies = pd.merge(ratings_title, metadata, left_on="primaryTitle", right_on="name")

    # Keep only those which are movies
    movies = movies[movies["titleType"] == "movie"]

    # Dropping unimportant columns
    movies = movies.drop(
        columns=[
            "wiki mID",
            "name",
            "freebase mID",
            "languages",
            "isAdult",
            "endYear",
            "originalTitle",
            "titleType",
            "genres_y"
        ]
    )

    # Renaming columns for better understanding
    movies = movies.rename(
        columns={"genres_x": "IMDB_genres"}
    )

    # Converting countries and genres to lists
    movies['countries'] = extract_countries_cmu(movies['countries'])
 
    # Converting the numerical values from string to int
    movies["runtimeMinutes"] = pd.to_numeric(movies["runtimeMinutes"], errors="coerce")
    movies["startYear"] = pd.to_numeric(movies["startYear"], errors="coerce")
    # Only keeping the 4 digit year to have a coherent column
    movies["release"] = movies["release"].str.extract(r"(\d{4})")
    movies["release"] = pd.to_numeric(movies["release"], errors="coerce")

    # Realigning with the correct movies: as several movies may have the same name
    movies = movies[movies["startYear"] == movies["release"]]

    # Realign with the runtime (with a +/- 5 min interval)
    movies = movies[
        (movies["runtimeMinutes"] >= movies["runtime"] - 5)
        & (movies["runtimeMinutes"] <= movies["runtime"] + 5)
    ]
    movies = movies.drop(columns=["startYear", "runtimeMinutes"])

    oscars_picture = oscars_processing(movies, oscar_winners)
    others = others_processing(movies, oscar_winners, n)
    return oscars_picture, others


def main():
    args = parse_args()
    PATH = "../../data/"

    # Importing the datasets
    oscar_winners = pd.read_csv(args.oscars_in)
    titles = pd.read_csv(PATH + "imdb/title.basics.tsv", sep="\t")
    ratings = pd.read_csv(PATH + "imdb/title.ratings.tsv", sep="\t")

    col_names = [
        "wiki mID",
        "freebase mID",
        "name",
        "release",
        "revenue",
        "runtime",
        "languages",
        "countries",
        "genres",
    ]
    metadata = pd.read_csv(
        PATH + "MovieSummaries/movie.metadata.tsv",
        sep="\t",
        names=col_names,
        header=None,
    )

    oscars_picture, others = data_processing(
        oscar_winners, titles, ratings, metadata, args.n_movies
    )

    # Exporting to a csv
    oscars_picture.to_csv(args.oscars_out, index=False)
    others.to_csv(args.others_out, index=False)


if __name__ == "__main__":
    main()
