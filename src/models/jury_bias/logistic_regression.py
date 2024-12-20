import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def preprocess_data(data, feature_columns, target="nominated"):
    """
    Preprocess the data for training a predictive model.

    Args:
        data (pd.DataFrame): The dataset.
        feature_columns (list): List of feature columns to use.
        target (str): The target column for prediction.

    Returns:
        X (pd.DataFrame): Processed feature matrix.
        y (pd.Series): Target variable.
    """
    # Handle list-like columns
    for col in ["IMDB_genres", "countries"]:
        if data[col].apply(lambda x: isinstance(x, list)).any():
            # Convert lists to strings for one-hot encoding
            data[col] = data[col].apply(
                lambda x: ",".join(map(str, x)) if isinstance(x, list) else x
            )

        # One-hot encode categorical features
        X = pd.get_dummies(data[feature_columns], drop_first=True)
    y = data[target]

    return X, y


def logistic_regression(data, feature_columns, target="nominated"):
    """
    Train a logistic regression model to predict the target variable.

    Args:
        data (pd.DataFrame): The dataset.
        feature_columns (list): List of feature columns to use.
        target (str): The target column for prediction.
    """
    # Check that the target is not part of the feature columns
    if target in feature_columns:
        raise ValueError(
            "Target variable should not be included in the feature columns."
        )

    # Preprocess data
    X, y = preprocess_data(data, feature_columns, target)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Use SMOTE to balance the training set
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # Scale continuous features
    scaler = StandardScaler()
    X_train_sm = scaler.fit_transform(X_train_sm)
    X_test = scaler.transform(X_test)

    # Train logistic regression model with class weight adjustment
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train_sm, y_train_sm)

    # Evaluate the model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba)}")

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        label="Logistic Regression (AUC = {:.2f})".format(
            roc_auc_score(y_test, y_proba)
        ),
    )
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    return model
