import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf
import networkx as nx


def plot_top_actors(df_actor, k):

    top_actors = df_actor.sort_values(by='profile_score', ascending=False).head(k)
    #display name of x-axis
    top_actors.set_index('primaryName', inplace=True)
    top_actors.plot(kind='bar', color='skyblue', alpha=0.8, edgecolor='black')
    plt.title(f"Top {k} Actors")
    plt.xlabel("Actors")
    plt.ylabel("High profile score")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()




def count_won_oscar_actor(df):

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
    ).sort_values(by='Oscar won', ascending=False)#.head(k)

    for num in range(count_oscar['Oscar won'].min(), count_oscar['Oscar won'].max()+1):
        count = len(count_oscar[count_oscar['Oscar won'] == num])
        print(f"{count} actors have won {num} Oscars")
        

    count_oscar = count_oscar[count_oscar['Oscar won'] > 2]

    print(f"Actors who have won more than 2 Oscars: {' '.join(count_oscar['Actors'].values)}")




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


    sns.kdeplot(won_balanced['averageRating'].values, label='Winners', fill=True)
    sns.kdeplot(lost_balanced['averageRating'].values, label="Did not win", fill=True, alpha=0.6)

    plt.xlabel('Average Rating')
    plt.ylabel('Density')
    plt.legend(loc='upper right')

    print(f'Average rating of oscar-winning movies: {np.round(won_balanced["averageRating"].mean(),3)}, Other nominees: {np.round(lost_balanced["averageRating"].mean(),3)}')
    print(f'Mean difference on each pair: {np.round(np.mean(diff),3)}')

    _, pval = ttest_ind(won_balanced['averageRating'].values, lost_balanced['averageRating'].values)
    print(f'p-value: {pval}')
    plt.title('Average Rating Distribution of Winners and other nominees (actors)')
    plt.show()