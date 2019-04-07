import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from preprocess import load_data


if __name__ == "__main__":
     
    # load and preprocess data
    X_train, X_test, y_train, y_test = load_data()
    print("Data loaded")

    # train
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train, y_train)
    print("Model trained")

    # save model
    path = "results/naive_bayes.pickle"
    pickle.dump(naive_bayes, open(path, "wb"))
    print("Model saved to", path)

    # predict
    y_pred = naive_bayes.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"""
Accuracy: {accuracy*100:.2f}%
Precision:{precision*100:.2f}%
Recall:   {recall*100:.2f}%
F1_SCORE: {f1*100:.2f}%
    """)
