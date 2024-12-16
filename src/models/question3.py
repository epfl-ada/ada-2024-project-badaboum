import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.stats import ttest_ind
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf
import networkx as nx


sns.set_theme()


def load_data_Q3():
    """
    Load dataframe used for question 3
    """

    df_movies = pd.read_csv('./data/oscar_movies.csv')
    df_director = pd.read_csv('./data/title.crew.tsv', sep='\t')
    df_name = pd.read_csv('./data/name.basics.tsv', sep='\t')
    df_title = pd.read_csv('./data/title.basics.tsv', sep='\t')

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

    #actor df
    for col in df_name.columns:
        df_name = df_name[df_name[col] != '\\N']

    tconsts = df_title['tconst'].values
    movie_names = df_title['primaryTitle'].values
    tconst_dir_mapping = dict(zip(tconsts, movie_names))

    def tconst_to_name(const_lst):
        return[tconst_dir_mapping[cst] for cst in const_lst.split(',')]

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




def plot_top_actors(df_actor, k):

    top_actors = df_actor.sort_values(by='profile_score', ascending=False).head(k)
    #display name of x-axis
    top_actors.set_index('primaryName', inplace=True)
    top_actors.plot(kind='bar', color='skyblue', alpha=0.8, edgecolor='black')
    plt.title(f"Top {k} Actors")
    plt.xlabel("Actors")
    plt.ylabel("High profile score")
    plt.xticks(rotation=45, ha='right')
    plt.show()




def plot_top_directors_interactive(director_revenue_year, k):
    """
    Creates an interactive bar chart with a slider to display the top k directors by mean cumulative revenue ('high profile score') for any year.

    Input:
        director_revenue_year: dict of pd.Series, the output of compute_profile_score
        k: int, The number of top directors to display
    """
    years = sorted(director_revenue_year.keys())
    frames = []
    for year in years:
        revenue_year = director_revenue_year[year]
        top_directors = revenue_year.nlargest(k)
        
        frames.append(
            go.Bar(
                x=top_directors.index,
                y=top_directors.values,
                name=str(year),
                marker=dict(color='skyblue', line=dict(color='black', width=1))
            )
        )
    
    initial_year = years[0]
    initial_revenue = director_revenue_year[initial_year].nlargest(k)
    fig = go.Figure(
        data=[
            go.Bar(
                x=initial_revenue.index,
                y=initial_revenue.values,
                marker=dict(color='skyblue', line=dict(color='black', width=1))
            )
        ],
        layout=go.Layout(
            title=f"Top {k} Directors in {initial_year}",
            xaxis=dict(title="Directors"),
            yaxis=dict(title="High profile score"),
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            label=str(year),
                            method="update",
                            args=[
                                {"x": [director_revenue_year[year].nlargest(k).index],
                                 "y": [director_revenue_year[year].nlargest(k).values]},
                                {"title": f"Top {k} Directors in {year}"}
                            ],
                        )
                        for year in years
                    ],
                    direction="down",
                    showactive=True,
                )
            ]
        ),
    )
    
    #fig.write_html("director_interactive.html")
    fig.show()








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

    return p_value



def plot_score_distribution(df):
    """
    Plots the distribution of the profile score for winners and losers

    Input:
        df_Q3: pandas dataframe computed by load_data_Q3
    """
    plt.hist(df[df['winner'] == 1]['profile_score'].values, label='Winners')
    plt.hist(df[df['winner'] == 0]['profile_score'].values, alpha=0.6, label='Losers')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.title('Profile score distribution for winners and losers, (y-log scale)')
    plt.xlabel('Profile score')
    plt.ylabel('Count')




def match_and_plot(df):
    mod = smf.logit(formula='winner ~ profile_score', data=df)

    res = mod.fit()

    # Extract the estimated propensity scores
    df['propensity_score'] = res.predict()

    
    def get_similarity(propensity_score1, propensity_score2):
        return 1-np.abs(propensity_score1-propensity_score2)


    treatment_df = df[df['winner'] == 1]
    control_df = df[df['winner'] == 0]


    #inspired from lab 5

# Create an empty undirected graph
    G = nx.Graph()

    # Loop through all the pairs of instances
    for control_id, control_row in control_df.iterrows():
        for treatment_id, treatment_row in treatment_df.iterrows():

            # Calculate the similarity 
            similarity = get_similarity(control_row['propensity_score'],
                                        treatment_row['propensity_score'])

            # Add an edge between the two instances weighted by the similarity between them
            #construit graph biparti fully connected entre treatment et control
            G.add_weighted_edges_from([(control_id, treatment_id, similarity)])

    # Generate and return the maximum weight matching on the generated graph
    matching = nx.max_weight_matching(G)


    matched = []
    diff = []
    ratings = df['averageRating'].values
    for pair in matching:
        matched.append(pair[0])
        matched.append(pair[1])
        diff.append(ratings[pair[0]] - ratings[pair[1]])

    unique_idx = list(set(matched))
    df_balanced = df.iloc[unique_idx]

    won_balanced = df_balanced[df_balanced['winner'] == 1]
    lost_balanced = df_balanced[df_balanced['winner'] == 0]


    plt.hist(won_balanced['averageRating'].values, label='Winners')
    plt.hist(lost_balanced['averageRating'].values, alpha=0.6, label='Losers')
    plt.xlabel('Average Rating')
    plt.ylabel('Count')
    plt.legend(loc='upper right')

    print(f'Average rating of oscar-winning movies: {np.round(won_balanced["averageRating"].mean(),3)}, Other movies: {np.round(lost_balanced["averageRating"].mean(),3)}')
    print(f'Mean difference on each pair: {np.round(np.mean(diff),3)}')

    _, pval = ttest_ind(won_balanced['averageRating'].values, lost_balanced['averageRating'].values)
    plt.title('Average Rating Distribution of Winners and Losers')
    plt.text(6.5, 16, f'p-value: {pval:.3f}', fontsize=12)
    plt.show()




def logreg_ratings(df):
    X = df[['profile score', 'averageRating']]
    winners = df['winner'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logreg = LogisticRegression()
    logreg.fit(X, winners)

    print(f'Profile score coefficient: {logreg.coef_[0,0]}, average rating coefficient: {logreg.coef_[0,1]}')

    x_min, x_max = X_scaled[:,0].min() - 1, X_scaled[:,0].max() + 1
    y_min, y_max = X_scaled[:,1].min() - 1, X_scaled[:,1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))

    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)

    plt.scatter(X_scaled[ winners == 0,0], X_scaled[ winners == 0,1], label='Loser', edgecolor='k', cmap=plt.cm.Paired)
    plt.scatter(X_scaled[ winners == 1,0], X_scaled[ winners == 1,1], label='Winner', edgecolor='k', cmap=plt.cm.Paired)

    plt.xlabel('Profile score (standardized)')
    plt.ylabel('Average Rating (standardized)')
    plt.title('Logistic Regression decision boundary')
    plt.legend(loc= 'upper right')
    plt.show()




def count_won_oscar(df, k):

    directors = df['directors'].unique()
    count_oscar = {}
    for director in directors:
        count_oscar[director] = len(df[(df['directors'] == director) & (df['winner'] == True)])
    


    count_oscar = pd.DataFrame(
        {
            'Directors': list(count_oscar.keys()),
            'Oscar won' : list(count_oscar.values())
        }
    ).sort_values(by='Oscar won', ascending=False).head(k)


    count_oscar.set_index('Directors').plot(kind='bar', color='skyblue', alpha=0.8, edgecolor='black')
    plt.title(f"Top {k} directors with the most oscars")
    plt.xlabel("Directors")
    plt.ylabel("Number of Oscars Won")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(range(0, 3))





def count_won_oscar_actor(df, k):

    df_movies = pd.read_csv('./data/oscar_movies.csv')
    winning_movies = df_movies[df_movies['winner']]['primaryTitle'].values

    actors = df['primaryName'].values
    known_for = df['knownForTitles'].values

    count_oscar = {}
    for i,actor in enumerate(actors):
        count_oscar[actor] = sum([1 for title in known_for[i] if title.lower() in winning_movies])
    


    count_oscar = pd.DataFrame(
        {
            'Actors': list(count_oscar.keys()),
            'Oscar won' : list(count_oscar.values())

        }
    ).sort_values(by='Oscar won', ascending=False).head(k)


    count_oscar.set_index('Actors').plot(kind='bar', color='skyblue', alpha=0.8, edgecolor='black')
    plt.title(f"Top {k} actors with the most oscars")
    plt.xlabel("Actors")
    plt.ylabel("Number of Oscars Won")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(range(0, 4))




def plot_winner_position_actors(df, first_year, last_year):
        winner_position_profile = []
        winner_position_rating = []

        for selected_year in range(int(first_year), int(last_year)+1, 1):
            df_year = df[df['release'] == selected_year].copy()
            if len(df_year):

                

                scores = df_year['profile_score_actor'].values
                ratings = df_year['averageRating'].values
                won = df_year['winner']

                sort_idx_profile = np.argsort(scores)
                sort_idx_rating = np.argsort(ratings)
                winner_idx = np.argmax(won)

                winner_position_profile.append(sort_idx_profile[winner_idx])
                winner_position_rating.append(sort_idx_rating[winner_idx])
    

        plt.hist(winner_position_profile, label='High profile')
        plt.hist(winner_position_rating, alpha=0.6, label='Ratings')
        plt.title('Position of the winner in the high profile/rating leaderboard for actors')
        plt.xlabel('Position')
        plt.ylabel('Count')
        plt.legend(loc='upper right')
        plt.show()

        return winner_position_profile, winner_position_rating




def match_and_plot_actor(df):
    mod = smf.logit(formula='winner ~ profile_score_actor', data=df)

    res = mod.fit()

    # Extract the estimated propensity scores
    df['propensity_score'] = res.predict()

    
    def get_similarity(propensity_score1, propensity_score2):
        return 1-np.abs(propensity_score1-propensity_score2)


    treatment_df = df[df['winner'] == 1]
    control_df = df[df['winner'] == 0]


    #inspired from lab 5

# Create an empty undirected graph
    G = nx.Graph()

    # Loop through all the pairs of instances
    for control_id, control_row in control_df.iterrows():
        for treatment_id, treatment_row in treatment_df.iterrows():

            # Calculate the similarity 
            similarity = get_similarity(control_row['propensity_score'],
                                        treatment_row['propensity_score'])

            # Add an edge between the two instances weighted by the similarity between them
            #construit graph biparti fully connected entre treatment et control
            G.add_weighted_edges_from([(control_id, treatment_id, similarity)])

    # Generate and return the maximum weight matching on the generated graph
    matching = nx.max_weight_matching(G)


    matched = []
    diff = []
    ratings = df['averageRating'].values
    for pair in matching:
        matched.append(pair[0])
        matched.append(pair[1])
        diff.append(ratings[pair[0]] - ratings[pair[1]])

    unique_idx = list(set(matched))
    df_balanced = df.iloc[unique_idx]

    won_balanced = df_balanced[df_balanced['winner'] == 1]
    lost_balanced = df_balanced[df_balanced['winner'] == 0]


    plt.hist(won_balanced['averageRating'].values, label='Winners')
    plt.hist(lost_balanced['averageRating'].values, alpha=0.6, label='Losers')
    plt.xlabel('Average Rating')
    plt.ylabel('Count')
    plt.legend(loc='upper right')

    print(f'Average rating of oscar-winning movies: {np.round(won_balanced["averageRating"].mean(),3)}, Other movies: {np.round(lost_balanced["averageRating"].mean(),3)}')
    print(f'Mean difference on each pair: {np.round(np.mean(diff),3)}')

    _, pval = ttest_ind(won_balanced['averageRating'].values, lost_balanced['averageRating'].values)
    plt.title('Average Rating Distribution of Winners and Losers')
    plt.text(6.5, 14, f'p-value: {pval:.3f}', fontsize=12)
    plt.show()




def plot_score_distribution_actors(df):
    """
    Plots the distribution of the profile score for winners and losers

    Input:
        df_Q3: pandas dataframe computed by load_data_Q3
    """
    plt.hist(df[df['winner'] == 1]['profile_score_actor'].values, label='Winners')
    plt.hist(df[df['winner'] == 0]['profile_score_actor'].values, alpha=0.6, label='Losers')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.title('Actors profile score distribution for winners and losers, (y-log scale)')
    plt.xlabel('Profile score')
    plt.ylabel('Count')
