import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import networkx as nx

def plot_top_directors(director_revenue_year, year, k):
    df = director_revenue_year[year].sort_values(ascending=False).head(k)
    df.plot(kind='bar', title=f'Top {k} directors in {year}')
    plt.xlabel('Director')
    plt.ylabel('High profile score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()




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


    #plt.hist(won_balanced['averageRating'].values, label='Winners')
    #plt.hist(lost_balanced['averageRating'].values, alpha=0.6, label='Losers')
    sns.kdeplot(won_balanced['averageRating'].values, label='Won', fill=True)
    sns.kdeplot(lost_balanced['averageRating'].values, label="Did not win", fill=True, alpha=0.6)

    plt.xlabel('Average Rating')
    plt.ylabel('Density')
    plt.legend(loc='upper right')

    print(f'Average rating of oscar-winning movies: {np.round(won_balanced["averageRating"].mean(),3)}, Other nominees: {np.round(lost_balanced["averageRating"].mean(),3)}')
    print(f'Mean difference on each pair: {np.round(np.mean(diff),3)}')

    _, pval = ttest_ind(won_balanced['averageRating'].values, lost_balanced['averageRating'].values)
    print(f'p-value: {pval}')
    plt.title('Average Rating Distribution of Winners and other nominees (directors)')
    plt.show()




def count_won_oscar(df):

    directors = df['directors'].unique()
    count_oscar = {}
    for director in directors:
        count_oscar[director] = len(df[(df['directors'] == director) & (df['winner'] == True)])
    


    count_oscar = pd.DataFrame(
        {
            'Directors': list(count_oscar.keys()),
            'Oscar won' : list(count_oscar.values())
        }
    ).sort_values(by='Oscar won', ascending=False)#.head(k)


    for num in range(count_oscar['Oscar won'].min(), count_oscar['Oscar won'].max()+1):
        count = len(count_oscar[count_oscar['Oscar won'] == num])
        print(f"{count} directors have won {num} Oscars")

    count_oscar = count_oscar[count_oscar['Oscar won'] > 1]

    print(f"Directors who have won more than 1 Oscar: {' '.join(count_oscar['Directors'].values)}")