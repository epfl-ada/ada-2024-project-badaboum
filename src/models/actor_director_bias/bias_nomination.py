import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf
import networkx as nx

def match_and_plot_nominated(df):
    mod = smf.logit(formula='nominated ~ profile_score', data=df)

    res = mod.fit()

    # Extract the estimated propensity scores
    df['propensity_score'] = res.predict()

    
    def get_similarity(propensity_score1, propensity_score2):
        return 1-np.abs(propensity_score1-propensity_score2)


    treatment_df = df[df['nominated'] == 1]
    control_df = df[df['nominated'] == 0]


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

    won_balanced = df_balanced[df_balanced['nominated'] == 1]
    lost_balanced = df_balanced[df_balanced['nominated'] == 0]


    sns.kdeplot(won_balanced['averageRating'].values, label='Nominated', fill=True)
    sns.kdeplot(lost_balanced['averageRating'].values, label='Not nominated', fill=True, alpha=0.6)

    plt.xlabel('Average Rating')
    plt.ylabel('Density')
    plt.legend(loc='upper left')

    print(f'Average rating of nominated movies: {np.round(won_balanced["averageRating"].mean(),3)}, Other movies: {np.round(lost_balanced["averageRating"].mean(),3)}')
    print(f'Mean difference on each pair: {np.round(np.mean(diff),3)}')

    _, pval = ttest_ind(won_balanced['averageRating'].values, lost_balanced['averageRating'].values)
    print(f'p-value: {pval}')
    plt.title('Average Rating Distribution of nominated and not nominated movies (director)')
    plt.show()



def match_and_plot_nominated_actor(df):
    mod = smf.logit(formula='nominated ~ profile_score_actor', data=df)

    res = mod.fit()

    # Extract the estimated propensity scores
    df['propensity_score'] = res.predict()

    
    def get_similarity(propensity_score1, propensity_score2):
        return 1-np.abs(propensity_score1-propensity_score2)


    treatment_df = df[df['nominated'] == 1]
    control_df = df[df['nominated'] == 0]


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

    won_balanced = df_balanced[df_balanced['nominated'] == 1]
    lost_balanced = df_balanced[df_balanced['nominated'] == 0]


    sns.kdeplot(won_balanced['averageRating'].values, label='Nominated', fill=True)
    sns.kdeplot(lost_balanced['averageRating'].values, label='Not nominated', fill=True, alpha=0.6)

    plt.xlabel('Average Rating')
    plt.ylabel('Density')
    plt.legend(loc='upper left')

    print(f'Average rating of nominated movies: {np.round(won_balanced["averageRating"].mean(),3)}, Other movies: {np.round(lost_balanced["averageRating"].mean(),3)}')
    print(f'Mean difference on each pair: {np.round(np.mean(diff),3)}')

    _, pval = ttest_ind(won_balanced['averageRating'].values, lost_balanced['averageRating'].values)
    print(f'p-value: {pval}')
    plt.title('Average Rating Distribution of nominated and not nominated movies (actor)')
    plt.show()