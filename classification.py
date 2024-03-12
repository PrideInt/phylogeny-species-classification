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

    print(data)

    # Split the data into features and target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Test: use logistic regression
    model = LogisticRegression(solver='lbfgs', max_iter=10000)

    # Test: do feature selection with recursive feature elimination
    # Select 5 features
    rfe = RFE(estimator=model, n_features_to_select=5)
    X = rfe.fit_transform(X, y)

    # Test: train the model
    model.fit(X, y)

    # Will use RFE once again, but modeling with decision trees instead
    # Also user input and cross-validation

    ids = parse_legend()

    test_prediction = [['Eukaryota', 'bilateral', 'Tracheophyta', 'composite florets', 'yellow']]

    for i in range(len(test_prediction[0])):
        for j in range(len(ids)):
            if test_prediction[0][i] == ids[j][1]:
                test_prediction[0][i] = int(ids[j][0])

    print(test_prediction)

    # take = input('Enter a prediction: ')

    # print(X[:2, :])
    pred = model.predict(test_prediction)

    for i in range(len(ids)):
        if str(pred[0]) == ids[i][0]:
            pred = ids[i][1]
            break

    print(pred)

def parse_legend():
    tuples = []

    with open('legend.txt', 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split('=')
            if len(line) == 2:
                tuples.append((line[1], line[0]))

    return tuples

classify()