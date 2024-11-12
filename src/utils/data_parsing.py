import pandas as pd


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
    return column.apply(lambda x: [text.strip() for text in x.split(",")] if pd.notnull(x) else [])
