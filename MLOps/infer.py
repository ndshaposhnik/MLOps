import pickle

import numpy as np
import pandas as pd

from MLOps import common


def infer(cfg):
    common.load_data()
    X, y = common.dataset_to_X_y("diabetes/test.csv")
    model = pickle.load(open(cfg["model"]["save_path"], "rb"))
    y_predicted = model.predict(X)
    mistake_rate = 1 - np.sum(np.abs(y - y_predicted)) / len(y)
    print(f"Accuracy: {mistake_rate.round(2)}")
    df = pd.DataFrame(y_predicted)
    df.to_csv(cfg["infer"]["result_file"], header=False, index=False)
