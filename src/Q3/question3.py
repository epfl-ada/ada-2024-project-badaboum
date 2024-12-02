import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.stats import ttest_ind


sns.set_theme()


def load_data_Q3():
    """
    Load dataframe used for question 3
    """

    df_movies = pd.read_csv('./data/oscar_movies.csv')
    df_director = pd.read_csv('./data/title.crew.tsv', sep='\t')
    df_name = pd.read_csv('./data/name.basics.tsv', sep='\t')

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


    return df_merged


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



def plot_top_directors(director_revenue_year, year, k):
    """
    Plots a bar chart of the top k directors by mean cumulative revenue ('high profile score') for a given year.

    Input:
        director_revenue_year: dict of df, the output of compute_profile_score
        year: int, year, must be between first_year and last_year
        k: int, The number of top directors to display
    """
    
    revenue_year = director_revenue_year[year]
    
    #top k directors
    top_directors = revenue_year.nlargest(k)
    
    top_directors.plot(kind='bar', color='skyblue', alpha=0.8, edgecolor='black')
    plt.title(f"Top {k} Directors in {year}")
    plt.xlabel("Directors")
    plt.ylabel("High profile score")
    plt.xticks(rotation=45, ha='right')
    plt.show()




def plot_winner_position(df):
    """Bar plot of the position of the oscar winner in the ratings and high profile score leaderboard

    Input:
        df: pandas dataframe computed by load_data_Q3

    Output:
        winner_position_profile: list, position of the oscar winner in the high profile leaderboard of each year 
        winner_position_rating: list, position of the oscar winner in the rating leaderboard of each year
    """
    winner_position_profile = []
    winner_position_rating = []


    first_year, last_year, director_revenue_year = compute_profile_score(df)


    for selected_year in range(int(first_year), int(last_year)+1, 1):
        df_year = df[df['release'] == selected_year].copy()
        if len(df_year):
            director_revenue_selected_year = director_revenue_year[selected_year]

            def set_high_profile_score(director):
                if director in director_revenue_selected_year:
                    return director_revenue_selected_year[director]
                else: 
                    return min(director_revenue_selected_year)

            df_year['high profile'] = df_year['directors'].apply(set_high_profile_score)

            scores = df_year['high profile'].values
            ratings = df_year['averageRating'].values
            #0-1 label for logreg
            won = df_year['winner'].astype(int).values

            sort_idx_profile = np.argsort(scores)
            sort_idx_rating = np.argsort(ratings)
            winner_idx = np.argmax(won)

            winner_position_profile.append(sort_idx_profile[winner_idx])
            winner_position_rating.append(sort_idx_rating[winner_idx])
  

    plt.hist(winner_position_profile, label='High profile')
    plt.hist(winner_position_rating, alpha=0.6, label='Ratings')
    plt.title('Position of the winner in the high profile/rating leaderboard')
    plt.xlabel('Position')
    plt.ylabel('Count')
    plt.legend(loc='upper right')
    plt.show()

    return winner_position_profile, winner_position_rating


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
        print('reject the null hypothesis (significant difference)')
    else:
        print('fail to reject the null hypothesis (no significant difference)')






def plot_logreg_diff(df):
    """
    Analyzes the relationship between movie ratings, directors' high-profile scores, 
    and their likelihood of winning an award over different years. The function performs 
    the following steps:
    - Computes correlation between high-profile scores and ratings.
    - Fits logistic regression models to predict awards (binary outcome) using 
      different combinations of features (ratings alone vs. ratings and high-profile scores).
    - Computes the difference in coefficients from the logistic regression models.
    - Plots the coefficient differences and correlations over time.

    Input:
        df: pandas dataframe computed by load_data_Q3
    
    """
    coeff_diff = []
    years_valid = []
    corr = []
    year_corr = []

    first_year, last_year, director_revenue_year = compute_profile_score(df)

    for selected_year in range(int(first_year), int(last_year)+1, 1):

        df_year = df[df['release'] == selected_year].copy()
        if len(df_year):
            ratings = df_year['averageRating'].values
            won = df_year['winner'].astype(int).values
            if np.max(won):
                director_revenue_selected_year = director_revenue_year[selected_year]

                def set_high_profile_score(director):
                    if director in director_revenue_selected_year:
                        return director_revenue_selected_year[director]
                    else: 
                        #set to min instead of zero to have smaller gaps
                        return min(director_revenue_selected_year)

                df_year['high profile'] = df_year['directors'].apply(set_high_profile_score)

                scores = df_year['high profile'].values

                if len(scores) > 2:

                    corr_c = np.corrcoef(scores, ratings)[0,1]
                    corr.append(corr_c)
                    year_corr.append(selected_year)


                    #standardize for logistic regression (scale are very different)
                    scores = (scores - np.mean(scores))/np.std(scores)
                    ratings = (ratings - np.mean(ratings))/np.std(ratings)
                    


                    features = np.vstack((scores, ratings))
                    model_1feature = LogisticRegression()
                    ratings_reshaped = ratings.reshape(-1, 1)
                    if np.sum(np.isnan(ratings)) == 0:
                        model_1feature.fit(ratings_reshaped, won)
                        coef_1 = model_1feature.coef_

                        model_2feature = LogisticRegression()
                        model_2feature.fit(features.T, won)
                        coef_2 = model_2feature.coef_


                        coeff_diff.append(coef_1[0,0]- coef_2[0,0])
                        years_valid.append(selected_year)



    print(f'Mean correlation from {first_year} to {last_year}: {np.mean(corr)}')
    print(f'Mean absolute correlation from {first_year} to {last_year}: {np.mean(np.abs(corr))}')
    plt.plot(years_valid, coeff_diff, label=r'$w_{rating}^1 - w_{rating}^2$')
    plt.plot(year_corr, corr, label=r'$\rho(rating,profile)$')
    plt.xlabel('Years')
    plt.ylabel('Coefficient')
    plt.legend(loc='lower right')
    plt.title('Rating vs High profile: comparison of coefficients')








