from preprocess import CATEGORY_INVERSED
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

import pickle

reviews = [
    "That is a great product",
    "I like this much more than the other bad useless movie", # tricky
    "Let us see the next serie of this",
    "Really nice one",
    "Bad movie",
    "Cannot admit more, this product can't be better then the previous one",
    "Really bad good movie" # makes no sense
]

print("Loading counting vector...")
count_vector = pickle.load(open("results/count_vector.pickle", "rb"))

print("Loading model...")
naive_bayes = pickle.load(open("results/naive_bayes.pickle", "rb"))

x = count_vector.transform(reviews)
result = naive_bayes.predict(x)
result = [ CATEGORY_INVERSED[res] for res in result ]
print(result)