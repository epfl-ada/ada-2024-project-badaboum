import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
import numpy as np
import seaborn as sns

sns.set_theme()


def load_data_Q3():
    """
    Load dataframe used for question 3
    """
    df_movies = pd.read_csv('./data/oscar_movies.csv')
    df_director = pd.read_csv('./data/title.crew.tsv', sep='\t')
    df_name = pd.read_csv('./data/name.basics.tsv', sep='\t')



    #fix the warning
    dtype_title = {
    "tconst": "string",
    "titleType": "string",
    "primaryTitle": "string",
    "originalTitle": "string",
    "isAdult": "string",
    "startYear": "string",
    "endYear": "string",
    "runtimeMinutes": "string",
    "genres": "string"
    }
    df_title = pd.read_csv('./data/title.basics.tsv', sep='\t', dtype=dtype_title, na_values="\\N")



    df_not_nominated = pd.read_csv('./data/all_other_movies.csv')
    df_not_nominated['winner'] = False
    df_not_nominated['nominated'] = 0
    df_movies['nominated'] = 1

    

    df_movies = pd.concat([df_movies, df_not_nominated], ignore_index=True)



    #Joins director to each movie
    df_merged = pd.merge(df_movies, df_director, on='tconst', how='inner')



    nconsts = df_name['nconst'].values
    dir_names = df_name['primaryName'].values
    const_dir_mapping = dict(zip(nconsts, dir_names))

    #maps the nconst identifier in IMDB to the director's real name
    def nconst_to_name(const):
        if ',' in const:
            #selects only one director if many
            return[const_dir_mapping[cst] for cst in const.split(',')][0]
        else:
            return const_dir_mapping[const]


    df_merged = df_merged[df_merged['directors'] != '\\N']
    df_merged['directors'] = df_merged['directors'].apply(nconst_to_name)

    df_merged.dropna(subset=['revenue', 'averageRating'], inplace=True)

    df_merged['release'] = df_merged['release'].astype(int)

    for col in df_name.columns:
        df_name = df_name[df_name[col] != '\\N']


    tconsts = df_title['tconst'].values
    movie_names = df_title['primaryTitle'].values
    tconst_dir_mapping = dict(zip(tconsts, movie_names))

    def tconst_to_name(const_lst):
        return[tconst_dir_mapping[cst] for cst in const_lst.split(',') if cst in tconst_dir_mapping] 

    df_name['knownForTitles'] = df_name['knownForTitles'].apply(tconst_to_name)
    df_name['primaryProfession'] = df_name['primaryProfession'].str.split(',')
    df_name = df_name[df_name['primaryProfession'].apply(lambda professions: 'actor' in professions or 'actress' in professions)]





    def compute_profile_score_actors(df, df_actors):
        movies = df['primaryTitle'].values
        revenues = df['revenue'].values

        pairs = dict(zip(movies, revenues))

        def mean_revenue_movie(lst):
            avg = 0
            n_movies = len(lst)
            for movie in lst:
                if movie.lower() in pairs:
                    avg += pairs[movie.lower()] /n_movies
            return avg
        
        df_actors['profile_score'] = df_actors['knownForTitles'].apply(mean_revenue_movie)
        df_actors = df_actors[df_actors['profile_score'] > 1e-6]

        return df_actors

    df_name = compute_profile_score_actors(df_merged, df_name)


    _, _, director_revenue_year = compute_profile_score(df_merged)

    scores = []

    directors = df_merged['directors'].values
    years = df_merged['release'].values

    for i in range(len(directors)):
        scores.append(director_revenue_year[years[i]][directors[i]])

    df_merged['profile_score'] = scores

    #for propensity score matching:
    df_merged["winner"] = df_merged["winner"].astype(int)
    df_merged = df_merged.reset_index(drop=True)

    df_name['knownForTitles'] = df_name['knownForTitles'].apply(lambda x: [l.lower() for l in x])

    movies = df_merged['primaryTitle'].str.lower().values
    titles_actor = df_name['knownForTitles'].values
    actors_score = df_name['profile_score'].values


    scores = []

    for movie in movies:
        score = 0
        for i in range(len(titles_actor)):
            if movie in titles_actor[i]:
                score += actors_score[i]
        scores.append(score)


    df_merged['profile_score_actor'] = scores


    return df_merged, df_name



def compute_profile_score(df):
    """
    Input:
        df: pandas dataframe computed by load_data_Q3

    Output:
        first_year: int, the earliest year in df
        last_year: int, the latest year in df
        director_revenu_year: list of df, director high profile scores for each year
    """
    first_year = df['release'].min()
    last_year = df['release'].max()
    director_revenue_year = {}
    #mean cumulative sum for each year to measure popularity at the given year
    for year in range(int(first_year), int(last_year)+1, 1):
        df_year = df[df['release'] <= year]
        director_revenue = df_year.groupby(['directors'])['revenue'].mean()
        director_revenue_year[year] = director_revenue
    return first_year, last_year, director_revenue_year





def compare_distribution(winner_position_profile, winner_position_rating):
    """
    Perfoms a ttest ind statistical test of winner_posiiton_profile and winner_position_rating and determine if there is a significant difference
    
    Input:
        winner_position_profile: list, position of the oscar winner in the high profile leaderboard of each year 
        winner_position_rating: list, position of the oscar winner in the rating leaderboard of each year
    """

    _, p_value = ttest_ind(winner_position_profile, winner_position_rating)

    print(f'p value = {p_value}')

    alpha = 0.05
    if p_value < alpha:
        print('reject the null hypothesis')
    else:
        print('fail to reject the null hypothesis')

    return p_value




