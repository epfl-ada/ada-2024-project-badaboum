import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from utils import balance_dataset, filter_high_vif


def ols_categorical(data: pd.DataFrame, feature: str, target="nominated"):
    """
    Perform OLS regression on a balanced dataset and visualize significant predictors.
    Filter out high VIF features.

    Args:
        data (pd.DataFrame): Input dataset.
        feature (str): The categorical feature to analyze (e.g., 'IMDB_genres').
        target (str): The target column for prediction.
    """
    # Balance the dataset
    balanced_data = balance_dataset(data, target)
    # One hot encode genres
    X = balanced_data[feature].explode()
    X = pd.get_dummies(X).groupby(level=0).sum()
    y = balanced_data["nominated"]

    X = filter_high_vif(X)

    # Add a constant for intercept
    X = sm.add_constant(X)

    # Fit the OLS model
    ols_model = sm.OLS(y, X).fit()

    # Summary of the OLS model
    ols_summary = ols_model.summary()

    # Display the OLS model summary
    print(ols_summary)


def ols_continuous(data: pd.DataFrame, feature: str, target="nominated"):
    """
    Perform OLS regression on runtime with a balanced dataset.

    Args:
        data (pd.DataFrame): Input dataset.
        feature (str): The continuous feature to analyze (e.g., 'runtime').
        target (str): The target column for prediction
    """
    # Balance the dataset
    balanced_data = balance_dataset(data, target)

    # Drop rows with missing runtime
    balanced_data = balanced_data.dropna(subset=["runtime"])

    # Define the predictor (X) and target (y)
    X_runtime = balanced_data[feature]
    y_nominated = balanced_data[target]

    # Add a constant for intercept
    X_runtime = sm.add_constant(X_runtime)

    # Fit the OLS model
    ols_model_runtime = sm.OLS(y_nominated, X_runtime).fit()

    # Summary of the runtime-focused OLS model
    ols_summary_runtime = ols_model_runtime.summary()

    print(ols_summary_runtime)


def gls_categorical(data: pd.DataFrame, feature: str, target="nominated"):
    """
    Perform GLS regression on a balanced dataset and visualize significant predictors.
    Filter out high VIF features.

    Args:
        data (pd.DataFrame): Input dataset.
        feature (str): The categorical feature to analyze (e.g., 'IMDB_genres').
        target (str): The target column for prediction.
    """
    # Balance the dataset
    balanced_data = balance_dataset(data, target)

    # One hot encode genres
    X = balanced_data[feature].explode()
    X = pd.get_dummies(X).groupby(level=0).sum()
    y = balanced_data[target]

    X = filter_high_vif(X)

    # Add a constant for intercept
    X = sm.add_constant(X)

    # Fit the GLS model
    gls_model = sm.GLS(y, X).fit()

    # Summary of the GLS model
    gls_summary = gls_model.summary()

    # Display the GLS model summary
    print(gls_summary)

    # Get coefficients and p-values excluding the intercept
    params = gls_model.params[1:]  # Exclude intercept
    p_values = gls_model.pvalues[1:]  # Exclude intercept

    # Filter significant predictors (p < 0.05)
    significant_params = params[p_values < 0.05]

    sorted_params = significant_params.sort_values(ascending=True)

    # Visualize coefficients
    plt.figure(figsize=(10, 8))
    sorted_params.plot(kind="barh")
    plt.title(f"GLS Model Coefficients: Significant Predictors")
    plt.xlabel("Coefficient")
    plt.ylabel(feature)
    plt.axvline(0, color="red", linestyle="--")
    plt.tight_layout()
    plt.show()


def gls_continuous(data: pd.DataFrame, feature: str, target="nominated"):
    """
    Perform GLS regression on runtime with a balanced dataset.

    Args:
        data (pd.DataFrame): Input dataset.
        feature (str): The continuous feature to analyze (e.g., 'runtime').
        target (str): The target column for prediction
    """
    # Balance the dataset
    balanced_data = balance_dataset(data, target)

    # Drop rows with missing runtime
    balanced_data = balanced_data.dropna(subset=["runtime"])

    # Define the predictor (X) and target (y)
    X_runtime = balanced_data[feature]
    y_nominated = balanced_data[target]

    # Add a constant for intercept
    X_runtime = sm.add_constant(X_runtime)

    # Fit the GLS model
    gls_model_runtime = sm.GLS(y_nominated, X_runtime).fit()

    # Summary of the runtime-focused GLS model
    gls_summary_runtime = gls_model_runtime.summary()

    print(gls_summary_runtime)
