from scipy.stats import wilcoxon
from src.models.question5.datasets_loading import get_data
from src.models.question5.utils import *

import seaborn as sns
import pandas as pd
import statsmodels.api as sm

def perform_regression_compound(type_score="all", type_date="ceremony"):
    
    # Get the reviews already splitted
    before_nomination, after_nomination = split_compound_score(type_="ceremony")

    if(type_date == "nomination"):
        
        before_nomination, after_nomination = split_compound_score(type_="nomination")
        
    before_nomination_flat = [item['text_compound'] for sublist in before_nomination for item in sublist.to_dict(orient='records')]
    after_nomination_flat = [item['text_compound'] for sublist in after_nomination for item in sublist.to_dict(orient='records')]

    if(type_score=="positive"):
        
        before_nomination_flat = [item['text_compound'] for sublist in before_nomination for item in sublist.to_dict(orient='records') if 
                              item['text_compound'] >= 0]
        after_nomination_flat = [item['text_compound'] for sublist in after_nomination for item in sublist.to_dict(orient='records') if
                                 item['text_compound'] >= 0]
    elif(type_score=="negative"):

        before_nomination_flat = [item['text_compound'] for sublist in before_nomination for item in sublist.to_dict(orient='records') if 
                              item['text_compound'] <= 0]
        after_nomination_flat = [item['text_compound'] for sublist in after_nomination for item in sublist.to_dict(orient='records') if
                                 item['text_compound'] <= 0]

    # Find the target length (length of the shorter list)
    target_length = min(len(before_nomination_flat), len(after_nomination_flat))

    # Randomly sample the longer list and keep the shorter list as is
    before_nomination_flat = random.sample(before_nomination_flat, target_length) if len(before_nomination_flat) > target_length else before_nomination_flat
    after_nomination_flat = random.sample(after_nomination_flat, target_length) if len(after_nomination_flat) > target_length else after_nomination_flat

    before_nomination_final = pd.DataFrame(before_nomination_flat)
    after_nomination_final = pd.DataFrame(after_nomination_flat)

    before_nomination_final["time"] = 0
    after_nomination_final["time"] = 1

    final_df = pd.concat([before_nomination_final, after_nomination_final])

    final_df = final_df.rename(columns={0: "compound"})

    sns.stripplot(x='time', y='compound', data=final_df, jitter=0.2, alpha=0.5)

    # Independent variable (time) and dependent variable (score)
    X = final_df['time']
    y = final_df['compound']

    # Add a constant for the intercept
    X = sm.add_constant(X)  # Adds a column of ones

    # Fit the model
    model = sm.OLS(y, X).fit()

    # Print the summary
    print(model.summary())

    return final_df


def perform_statistical_test_compound(type_="ceremony"):
    
    # Get the reviews already splitted
    before_nomination, after_nomination = split_compound_score(type_="ceremony")

    if(type_== "nomination"):
        before_nomination, after_nomination = split_compound_score(type_="nomination")
    
    # Flatten the lists for the test
    before_nomination_flat = [item['text_compound'] for sublist in before_nomination for item in sublist.to_dict(orient='records')]
    after_nomination_flat = [item['text_compound'] for sublist in after_nomination for item in sublist.to_dict(orient='records')]


    # Plot the distribution of the sentiment scores in both cases
    plt.hist(before_nomination_flat, bins=200)
    plt.yscale('log')
    plt.show()

    plt.hist(after_nomination_flat, bins=200)
    plt.yscale('log')
    plt.show()

    # List for the pairwise results
    results = []

    # Number of tested movies
    count = 0
    
    for i in range(0,len(before_nomination)):

        # Get the reviews for one specific movie
        before = before_nomination[i]['text_compound'].tolist()
        after = after_nomination[i]['text_compound'].tolist()

        # Skip this movie if reviews are missing
        if len(before) == 0 or len(after) == 0:
            continue  

        movie_id = before_nomination[i]['imdb_id'].tolist()[0]
        winner = before_nomination[i]['winner'].tolist()[0]
        
        # Truncate both lists to the length of the shorter one
        min_length = min(len(before), len(after))
        before = before[:min_length]
        after = after[:min_length]
        
        # Perform the Wilcoxon test (with reviews specific to one movie)
        stat, p = wilcoxon(before, after)
        results.append({'Movie ID': movie_id, 'Winner': winner , 'p-value': p})

    results_df = pd.DataFrame(results)

    # Movies where the hypothesis was rejected (p value of 0.05)
    count_reject = (results_df['p-value'] < 0.05).sum()

    # Percentage of movie where the hypothesis was rejected
    percentage_rejected = count_reject / results_df.shape[0] * 100

    print(f'There are {count_reject} rejected movies (p-value < 0.05). That represents only {percentage_rejected} percent of the movies')

    return results_df