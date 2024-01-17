import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(file_path):
    # read the data
    data = pd.read_csv(file_path, encoding='latin-1')
    # drop the useless columns and rename spam: 1 ham: 0
    data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    data.rename(columns={"v1": "label", "v2": "text"}, inplace=True)
    data.replace({'spam': 1, 'ham': 0}, inplace=True)

    X = TfidfVectorizer().fit_transform(data['text']).toarray()  # features
    t = data["label"]  # target

    return data, X, t


def pie_chart(data):
    # Counting the values in the 'label' column
    counts = data['label'].value_counts()

    # Create a pie chart
    plt.pie(counts, labels=['Ham', 'Spam'], autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Ham and Spam Messages in the Dataset')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    #plt.show()

    plt.savefig("Ham_vs_Spam_Pie_Chart" + '.jpg')


def trivial_classifier(data):
    TfidfVectorizer().fit_transform(data['text']).toarray()  # features
    t = data["label"]  # target
    print(f"Trivial classifier: all Legitimate accuracy: {accuracy_score(t, np.zeros_like(t)):.5f}")


def main():
    # load data
    file_path = 'spam.csv'
    data, X, t = load_data(file_path)  # load data, features, target

    pie_chart(data)
    trivial_classifier(data)


if __name__ == "__main__":
    main()
