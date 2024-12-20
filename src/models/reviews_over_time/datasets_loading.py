import pandas as pd

DATA_PATH = 'data/'

def load_reviews_augmented():
    """
    Load the reviews and combines them with more info about the movie reviewed (ceremony date, year, winner/looser)
    
    Returns:
        reviews (pd.DataFrame): The reviews with the additional info
    """
    # Get the reviews
    reviews = pd.read_csv(DATA_PATH +'imdb_reviews/imdb_reviews_with_compound.csv')

    # Ensure the 'date' column is in datetime format
    reviews['date'] = pd.to_datetime(reviews['date'])

    # Get the oscar movies table
    oscar_movies = pd.read_csv(DATA_PATH +'oscar_movies.csv')

    # Only keep the imdb movie id the ceremony date and the winner flag
    oscar_ceremonies = oscar_movies[['tconst','ceremony_date', 'oscar_year', 'winner']]

    # Rename the movie id column to be consistent with the review dataframe
    oscar_ceremonies =  oscar_ceremonies.rename(columns={"tconst": "imdb_id"})

    # Add the ceremony date of the movie to the review
    reviews = reviews.join(oscar_ceremonies.set_index('imdb_id'), on='imdb_id')

    return reviews