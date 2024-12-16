import pandas as pd

def get_data():

    # Get the reviews
    PATH = 'data/'
    reviews = pd.read_csv(PATH +'imdb_reviews/imdb_reviews_with_compound.csv')

    # Ensure the 'date' column is in datetime format
    reviews['date'] = pd.to_datetime(reviews['date'])
    
    # Get the oscar movies table
    oscar_movies = pd.read_csv(PATH +'oscar_movies.csv')

    # Get the oscar nomination dates table
    oscar_nomination_dates = pd.read_csv(PATH + 'oscar_nomination_dates.csv')

    # Rename the movie id column to be consistent with the review dataframe
    oscar_nomination_dates =  oscar_nomination_dates.rename(columns={"year": "oscar_year"})

    # Only keep the imdb movie id the ceremony date and the winner flag
    oscar_ceremonies = oscar_movies[['tconst','ceremony_date', 'oscar_year', 'winner']]

    # Rename the movie id column to be consistent with the review dataframe
    oscar_ceremonies =  oscar_ceremonies.rename(columns={"tconst": "imdb_id"})

    # Add the ceremony date of the movie to the review
    reviews = reviews.join(oscar_ceremonies.set_index('imdb_id'), on='imdb_id')

    # Add the nomination date of the movie to the review
    reviews = reviews.join(oscar_nomination_dates.set_index('oscar_year'), on='oscar_year')

    # Convert ceremony_date and nomination_date to datetime
    reviews['ceremony_date'] = pd.to_datetime(reviews['ceremony_date'])
    reviews['nomination_date'] = pd.to_datetime(reviews['nomination_date'])

    # Calculate date range
    reviews['lower_bound'] = reviews['nomination_date'] - pd.DateOffset(months=2)
    reviews['upper_bound'] = reviews['ceremony_date'] + pd.DateOffset(months=2)

    # Filter reviews within the date range
    filtered_reviews = reviews[
        (reviews['date'] >= reviews['lower_bound']) & 
        (reviews['date'] <= reviews['upper_bound'])
    ]

    return filtered_reviews