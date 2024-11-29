import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from ..utils.data_parsing import parse_str_to_list
from scipy.stats import ttest_ind

def get_data():
    PATH = "data/other_awards/"
    return pd.read_csv(PATH + 'other_awards.csv')

def ratings_by_awards():
    data = get_data()
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

    # Perform statistical test 
    for award in df_melted['Award'].unique():
        if (award == 'win_o'): name = 'Oscars'
        if (award == 'win_gg'): name = 'Golden globes'
        if (award == 'win_b'): name = 'Bafta'
        print(f"\nAward: {name}")
        award_data = df_melted[df_melted['Award'] == award]
        won_group = award_data[award_data['Won'] == 1]['averageRating']
        not_won_group = award_data[award_data['Won'] == 0]['averageRating']
        t_stat, p_value = ttest_ind(won_group, not_won_group, equal_var=False)
        print("T-statistic:", t_stat)
        print("P-value:", p_value)
        if p_value < 0.05:
            print("Significant difference in ratings.")
        else:
            print("No significant difference in ratings.")
    print('------------------------')
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


def awards_by_countries():
    data = get_data()
    data['countries'] = parse_str_to_list(data['countries'])
    

    # Explode the countries so each list of countries becomes separate rows
    exploded = data.explode('countries')

    # Group by the individual countries and sum up the awards
    countries = exploded.groupby('countries')[['win_o', 'win_gg', 'win_b']].sum().reset_index()
    countries = countries[(countries[['win_o', 'win_gg', 'win_b']].sum(axis=1) > 0)]
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

def awards_by_countries_ratio():
    data = get_data()
    data['countries'] = parse_str_to_list(data['countries'])
    

    # Explode the countries so each list of countries becomes separate rows
    exploded = data.explode('countries')

    total_movies = exploded.groupby('countries').size().reset_index(name='total_movies')
    awards = exploded.groupby('countries')[['win_o', 'win_gg', 'win_b']].sum().reset_index()

    # Merge to calculate the ratio
    countries = pd.merge(awards, total_movies, on='countries')
    countries['ratio_o'] = countries['win_o'] / countries['total_movies']
    countries['ratio_gg'] = countries['win_gg'] / countries['total_movies']
    countries['ratio_b'] = countries['win_b'] / countries['total_movies']

    # Filter countries with at least one award
    countries = countries[(countries[['win_o', 'win_gg', 'win_b']].sum(axis=1) > 0)]

    # Set up the positions for the bars
    positions = np.arange(len(countries['countries']))
    bar_width = 0.25

    # Plotting the ratios
    plt.figure(figsize=(12, 8))
    plt.bar(positions, countries['ratio_o'], width=bar_width, label='Oscar')
    plt.bar(positions + bar_width, countries['ratio_gg'], width=bar_width, label='Golden Globe')
    plt.bar(positions + 2 * bar_width, countries['ratio_b'], width=bar_width, label='Bafta')

    # Add labels and legend
    plt.xlabel('Country')
    plt.ylabel('Awards to Movies Ratio')
    plt.title('Ratio of Awards Won to Total Movies Produced per Country')
    plt.xticks(positions + bar_width, countries['countries'], rotation=40)
    plt.legend()

    plt.tight_layout()
    plt.show()


def calculate_genre_distribution(genres_list):
    all_genres = [genre for genres in genres_list for genre in genres]
    genre_counts = Counter(all_genres)
    total = sum(genre_counts.values())
    return {genre: count / total for genre, count in genre_counts.items()}

def awards_by_genre():
    data = get_data()
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
