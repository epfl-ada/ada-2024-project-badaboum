import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from ..utils.data_parsing import parse_str_to_list

def create_dataset():
    """
    Creates the dataset which contains movies' data, with Bafta and Golden Globes stats
    """
    # Importing all the datasets
    PATH = '../data/'
    oscars = pd.read_csv(PATH + 'oscar_movies.csv')

    bafta = pd.read_csv(PATH + 'bafta_films.csv')
    bafta = bafta.drop(columns='workers')
    bafta['nominee'] = bafta['nominee'].str.lower()

    golden_globe = pd.read_csv(PATH + 'golden_globe_awards.csv')
    golden_globe = golden_globe.drop(columns = ['ceremony', 'film'])
    golden_globe['nominee'] = golden_globe['nominee'].str.lower()

    # Cleaning the categories
    bafta['category'] = bafta['category'].str.replace(r' in \d{4}', '', regex=True)
    bafta['category'] = bafta['category'].str.replace(r'^Film \| ', '', regex=True)

    # Keeping the "best film" category and best motion picture
    bafta_best_film = bafta[(bafta['category'] == "Best Film") | 
                            (bafta['category'] == "Film From Any Source")|
                            (bafta['category'] == "Film")]
    bafta_best_film = bafta_best_film.reset_index(drop = True)

    gg_motion_picture = golden_globe[golden_globe['category'].str.startswith("Best Motion Picture -")]
    gg_motion_picture = gg_motion_picture.reset_index(drop = True)

    # Merge Golden globes and Bafta
    merged_df = pd.merge(gg_motion_picture, bafta_best_film, on = 'nominee', how = 'inner')

    merged_df = merged_df.rename(columns = {'year_award':'year_gg', 'category_x': 'category_gg', 'win':'win_gg', 'year':'year_b', 'category_y':'category_b', 'winner':'win_b'})

    all_merged = pd.merge(oscars, merged_df, left_on = 'primaryTitle', right_on = 'nominee', how = 'inner')
    all_merged = all_merged.drop(columns = ['year_film', 'nominee'])
    all_merged = all_merged.rename(columns = {'winner':'win_o'})

    return all_merged

def ratings_by_awards(data):
    df_melted = data.melt(id_vars='averageRating', value_vars=['win_o', 'win_gg', 'win_b'], 
                    var_name='Award', value_name='Won')

    # Plot ratings by awards 
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Award', y='averageRating', hue='Won', data=df_melted)
    plt.title('Movie Ratings by Awards Won')
    plt.xticks(
        ticks=[0, 1, 2],  
        labels=['Oscar', 'Golden Globe', 'BAFTA'], 
    )
    plt.xlabel('Award Type')
    plt.ylabel('Rating')
    plt.legend(title='Won Award')
    plt.show()

    data['total_awards'] = data[['win_o', 'win_gg', 'win_b']].sum(axis=1)

    # Calculate the mean rating for each group based on total awards
    ratings_by_awards = data.groupby('total_awards')['averageRating'].mean().reset_index()

    # Display the average rating by number of awards won
    print("Average rating by number of awards won")
    print(ratings_by_awards)

    # Box plot to see the distribution of ratings by total awards
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='total_awards', y='averageRating', data=data)
    plt.title('Movie Ratings by Total Awards Won')
    plt.xlabel('Total Awards Won')
    plt.ylabel('Rating')
    plt.show()


def awards_by_countries(data):
    data['countries'] = parse_str_to_list(data['countries'])
    

    # Explode the countries so each list of countries becomes separate rows
    exploded = data.explode('countries')

    # Group by the individual countries and sum up the awards
    countries = exploded.groupby('countries')[['win_o', 'win_gg', 'win_b']].sum().reset_index()

    # Set up the positions for the bars
    positions = np.arange(len(countries['countries']))
    bar_width = 0.25

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.bar(positions, countries['win_o'], width=bar_width, label='Oscar')
    plt.bar(positions + bar_width, countries['win_gg'], width=bar_width, label='Golden Globe')
    plt.bar(positions + 2 * bar_width, countries['win_b'], width=bar_width, label='Bafta')

    # Add labels and legend
    plt.xlabel('Country')
    plt.ylabel('Number of Awards Won')
    plt.title('Number of Movies from Each Country That Won Different Awards')
    plt.xticks(positions + bar_width, countries['countries'], rotation=40)
    plt.legend()

    plt.show()


def calculate_genre_distribution(genres_list):
    all_genres = [genre for genres in genres_list for genre in genres]
    genre_counts = Counter(all_genres)
    total = sum(genre_counts.values())
    return {genre: count / total for genre, count in genre_counts.items()}

def awards_by_genre(data):
    data['IMDB_genres'] = parse_str_to_list(data['IMDB_genres'])

    # Have 1 genre per column
    exploded = data.explode('IMDB_genres')

    # Melt the df
    exploded_melted = exploded.melt(
        id_vars=['IMDB_genres'], 
        value_vars=['win_o', 'win_b'],
        var_name='award_type', 
        value_name='is_winner'
    )

    # Filter to keep only the award winners
    award_genre_distribution = exploded_melted[exploded_melted['is_winner']].groupby(['IMDB_genres', 'award_type']).size().unstack(fill_value=0)

    # Renaming columns for clarity
    award_genre_distribution.columns = ['Oscar Winners', 'Bafta Winners']

    award_genre_distribution.plot(kind='bar', figsize=(14, 8), color=['#1f77b4', '#2ca02c'])

    # Add labels and legend
    plt.xlabel('Genre')
    plt.ylabel('Number of Movies')
    plt.title('Genre Distribution Between Oscar and Bafta Winners')
    plt.legend(title='Award Type')
    plt.xticks(rotation=45, ha='right')

    plt.show()