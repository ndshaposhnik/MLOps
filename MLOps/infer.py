import pickle

import common
import numpy as np
import pandas as pd


def infer():
    common.load_data()
    X, y = common.dataset_to_X_y("../data/diabetes/test.csv")
    model = pickle.load(open("../data/random_forest.pth", "rb"))
    y_predicted = model.predict(X)
    mistake_rate = 1 - np.sum(np.abs(y - y_predicted)) / len(y)
    print(f"Accuracy: {mistake_rate.round(2)}")
    df = pd.DataFrame(y_predicted)
    df.to_csv("../data/result.csv", header=False, index=False)


if __name__ == "__main__":
    infer()
