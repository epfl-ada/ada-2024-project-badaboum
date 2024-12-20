from scipy.stats import wilcoxon
from src.models.oscar_bump.utils import *

import pandas as pd
import statsmodels.api as sm
import random

def perform_regression_compound(type_score="all", type_date="ceremony"):
    """
    Perform a regression analysis on the compound score of the reviews
    
    Parameters:
        type_score (str): The type of score to consider (all, positive or negative)
        type_date (str): The type of date to consider (ceremony or nomination)
        
    Returns:
        final_df (pd.DataFrame): The final dataframe used for the regression analysis
    """
    # Get the reviews already splitted
    before, after = split_compound_score(type_="ceremony")

    # If the type of date is nomination, get the reviews splitted for the nomination date
    if(type_date == "nomination"):
        
        before, after = split_compound_score(type_="nomination")
        
    # Flatten the lists of dataframes
    before_flat = [item['text_compound'] for sublist in before for item in sublist.to_dict(orient='records')]
    after_flat = [item['text_compound'] for sublist in after for item in sublist.to_dict(orient='records')]

    # If the type of score is positive or negative, filter the data accordingly
    if(type_score=="positive"):
        
        before_flat = [item['text_compound'] for sublist in before for item in sublist.to_dict(orient='records') if 
                              item['text_compound'] >= 0]
        after_flat = [item['text_compound'] for sublist in after for item in sublist.to_dict(orient='records') if
                                 item['text_compound'] >= 0]
    elif(type_score=="negative"):

        before_flat = [item['text_compound'] for sublist in before for item in sublist.to_dict(orient='records') if 
                              item['text_compound'] <= 0]
        after_flat = [item['text_compound'] for sublist in after for item in sublist.to_dict(orient='records') if
                                 item['text_compound'] <= 0]

    # Find the target length (length of the shorter list)
    target_length = min(len(before_flat), len(after_flat))

    # Randomly sample the longer list and keep the shorter list as is
    before_flat = random.sample(before_flat, target_length) if len(before_flat) > target_length else before_flat
    after_flat = random.sample(after_flat, target_length) if len(after_flat) > target_length else after_flat

    # Create the final dataframe
    before_final = pd.DataFrame(before_flat)
    after_final = pd.DataFrame(after_flat)

    before_final["time"] = 0
    after_final["time"] = 1

    # Concatenate the dataframes
    final_df = pd.concat([before_final, after_final])

    # Rename the compound score column
    final_df = final_df.rename(columns={0: "compound"})

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
    """
    Perform a Wilcoxon test on the compound score of the reviews
    
    Parameters:
        type_ (str): The type of date to consider (ceremony or nomination)
        
    Returns:
        results_df (pd.DataFrame): The results of the Wilcoxon test for each movie
    """
    # Get the reviews already splitted
    before, after = split_compound_score(type_="ceremony")

    if(type_== "nomination"):
        before, after = split_compound_score(type_="nomination")
    
    # List for the pairwise results
    results = []
    
    for i in range(0,len(before)):

        # Get the reviews for one specific movie
        before_curr = before[i]['text_compound'].tolist()
        after_curr = after[i]['text_compound'].tolist()

        # Skip this movie if reviews are missing
        if len(before_curr) == 0 or len(after_curr) == 0:
            continue  

        # Get the movie id and the winner tag
        movie_id = before[i]['imdb_id'].tolist()[0]
        winner = before[i]['winner'].tolist()[0]
        
        # Truncate both lists to the length of the shorter one
        min_length = min(len(before_curr), len(after_curr))
        before_curr = before_curr[:min_length]
        after_curr = after_curr[:min_length]
        
        # Perform the Wilcoxon test (with reviews specific to one movie)
        _, p = wilcoxon(before_curr, after_curr)
        results.append({'Movie ID': movie_id, 'Winner': winner , 'p-value': p})

    results_df = pd.DataFrame(results)

    # Movies where the hypothesis was rejected (p value of 0.05)
    count_reject = (results_df['p-value'] < 0.05).sum()

    # Percentage of movie where the hypothesis was rejected
    percentage_rejected = count_reject / results_df.shape[0] * 100

    print(f'There are {count_reject} rejected movies (p-value < 0.05). That represents only {percentage_rejected} percent of the movies')

    return results_df