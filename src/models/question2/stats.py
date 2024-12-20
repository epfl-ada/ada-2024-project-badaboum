from scipy.stats import ks_2samp
from src.models.question2.datasets_loading import load_reviews_augmented
from src.models.question2.utils import *

def print_ks_test_review_dates(type_1, type_2):
    """
    Performs the two-sample Kolmogorov-Smirnov test for goodness of fit.

    This test compares the underlying continuous distributions F(x) and G(x) of two independent samples.
    """
    df = load_reviews_augmented()

    # Select the data
    df_1, df_2 = select_visualization_groups(type_1, type_2)
 
    # Extract the 'relative_date' values
    dates_1 = df_1['relative_date']
    dates_2 = df_2['relative_date']
    
    # Perform the KS test
    ks_stat, p_value = ks_2samp(dates_1, dates_2)
    
    print(f"KS Statistic: {ks_stat}, p-value: {p_value}")