import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from main import load_data, pie_chart, trivial_classifier
from sklearn.linear_model import LogisticRegression
import time
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from main import load_data

# Set the parameter grid for hyperparameter tuning
param_grid = {
    'max_iter': [100, 500, 1000],
    'C': [0.01, 0.1, 1, 10],  # inverse of regularization strength
}


def logistic_regression(data, X, t):
    start_time = time.time()
    # Create a logistic regression model
    log_regression = LogisticRegression()

    # Hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(log_regression, param_grid, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, t)

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Split the data into 80% and 20% for training and test sets
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2, random_state=42)

    # Train the best model on the training set
    best_model.fit(X_train, t_train)

    # Evaluate the model on the test set
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(t_test, predictions)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken:.2f} seconds")
    print(f"Best Hyperparameters: {best_params}")
    return accuracy


def main():
    file_path = 'spam.csv'
    data, X, t = load_data(file_path)  # load data, features, target

    final_accuracy = logistic_regression(data, X, t)
    print(f"Accuracy on the test set: {final_accuracy:.5f}")


if __name__ == "__main__":
    main()

