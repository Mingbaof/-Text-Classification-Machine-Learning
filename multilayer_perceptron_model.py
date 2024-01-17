import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from main import load_data
import time


def multi_layer_perceptron_model(X_train, t_train, X_test, t_test):
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'max_iter': [100, 500, 1000],
        'hidden_layer_sizes': [(100,), (50, 50), (30, 30, 30)],
        'activation': ['logistic', 'relu', 'tanh'],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }

    # Create an MLPClassifier model
    mlp_classifier = MLPClassifier(early_stopping=False, validation_fraction=0.1, n_iter_no_change=10)

    start_time = time.time()
    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(mlp_classifier, param_grid, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, t_train)

    # Get the best model from the grid search and use it to predict on the test set
    best_mlp_classifier = grid_search.best_estimator_
    predictions = best_mlp_classifier.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(t_test, predictions)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken:.2f} seconds")
    return accuracy, grid_search.best_params_


def main():
    file_path = 'spam.csv'
    data, X, t = load_data(file_path)

    # Split the data into 80% and 20% for training and test sets
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2, random_state=42)

    # Train and evaluate the perceptron model with hyperparameter tuning
    accuracy, best_params = multi_layer_perceptron_model(X_train, t_train, X_test, t_test)

    print(f"Accuracy on the test set: {accuracy:.5f}")
    print(f"Best Hyperparameters: {best_params}")


if __name__ == "__main__":
    main()
