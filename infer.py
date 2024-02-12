import pickle

import numpy as np

from src import common


def test():
    print("Loading dataset...")
    X, y = common.dataset_to_X_y("data/test.csv")
    print("Dataset loaded")
    model = pickle.load(open("models/random_forest", "rb"))
    y_predicted = model.predict(X)
    mistake_rate = np.sum(np.abs(y - y_predicted)) / len(y)
    print(f"Mistake rate: {mistake_rate}")


if __name__ == "__main__":
    test()
