from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import pickle

CATEGORY = {"positive": 1, "negative": 0}
CATEGORY_INVERSED = {1: "positive", 0: "negative"}

def load_data():
    """Loads review data and returns:
        X_train: reviews for training set
        X_test: reviews for testing set
        y_train: labels for training set
        y_test: labels for testing test
        count_vector: CountVectorizer object that fits training data, 
            it is just a matrix of word (token) counts"""
    # read reviews
    with open("data/reviews.txt") as f:
        reviews = f.readlines()

    reviews = [ review.strip() for review in reviews ]

    # read its corresponding labels 
    with open("data/labels.txt") as f:
        labels = f.readlines()

    labels = [ CATEGORY[label.strip()] for label in labels ]

    # split data to test and train
    # 85% train 15% test
    X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.15)
    # print(X_train[5])

    # to count words of each row ( each review )
    # more at https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer
    count_vector = CountVectorizer()

    X_train = count_vector.fit_transform(X_train)
    X_test = count_vector.transform(X_test)

    print("Saving counting vector...")
    pickle.dump(count_vector, open("results/count_vector.pickle", "wb"))

    return X_train, X_test, y_train, y_test