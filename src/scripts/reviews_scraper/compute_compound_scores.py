import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Get the reviews
PATH = '../../../data/imdb_reviews/'
reviews = pd.read_csv(PATH + 'imdb_reviews_best_picture_2years_from_release.csv')

# Analyze the compound score of the reviews
analyzer = SentimentIntensityAnalyzer()
reviews['text_compound'] = reviews['text'].apply(lambda x : analyzer.polarity_scores(x).get('compound'))

# Store the compound scores
reviews.to_csv(PATH + 'imdb_reviews_with_compound.csv')