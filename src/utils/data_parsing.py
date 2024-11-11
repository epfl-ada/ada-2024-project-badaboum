import ast
import pandas as pd


def parse_genres_imdb(column):
    """
    Parses a column of genre strings, splitting each string by commas to
    create a list of genres for each entry.

    Parameters:
    column (pd.Series): A pandas Series containing genre strings,
                        with genres separated by commas.

    Returns:
    pd.Series: A Series where each element is a list of genres.
               If an element in the original Series is null, it returns an empty list.
    """
    return column.apply(lambda x: x.split(",") if pd.notnull(x) else [])


def extract_countries_cmu(countries_column):
    """
    Extracts country names from a column containing string representations of dictionaries,
    where each dictionary maps keys to country names.

    Parameters:
    countries_column (pd.Series): A pandas Series with string representations of dictionaries,
                                  where each dictionary has country information.

    Returns:
    pd.Series: A Series where each element is a list of country names.
               If parsing fails for an entry, it returns an empty list.
    """

    def parse_country_dict(country_dict_str):
        try:
            country_dict = ast.literal_eval(
                country_dict_str
            )  # Convert string to dictionary
            return list(country_dict.values())  # Return only country names
        except (ValueError, SyntaxError):
            return []  # Return empty list if parsing fails

    return countries_column.apply(parse_country_dict)
