import pandas as pd
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def classify():
    # Currently holds 8 data points. Will add at least 92 more.
    # Note: add more data for the model to train on
    data = pd.read_csv('phylogeny.csv')

    # Don't need the first column
    data = data.iloc[:, 1:]

    # Split the data into features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Test: do feature selection with recursive feature elimination
    rfe = RFE(estimator=LogisticRegression(), n_features_to_select=5)
    X = rfe.fit_transform(X, y)

    # Test: use logistic regression
    model = LogisticRegression()

    # Test: train the model
    model.fit(X, y)

    # Will use RFE once again, but modeling with decision trees instead
    # Also user input and cross-validation

classify()