import pandas as pd


def parse_str_to_list(column):
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
