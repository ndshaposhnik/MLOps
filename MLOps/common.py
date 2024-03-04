import os

import pandas as pd
from dvc.api import DVCFileSystem


def normalize_dataframe(df):
    return (df - df.mean()) / df.std()


def dataset_to_X_y(filename, nrows=None, normalize=False):
    data = pd.read_csv(filename, nrows=nrows)
    X = data.loc[:, data.columns[:-1]]
    y = data.loc[:, data.columns[-1]]
    if normalize:
        X = normalize_dataframe(X)
    return X.to_numpy(), y.to_numpy()


def load_data():
    if os.path.isdir("./diabetes"):
        print("The directory with diabetes dataset already exists")
        return
    print("Loading dataset...")
    fs = DVCFileSystem(
        url="https://github.com/ndshaposhnik/MLOps.git",
        rev="task1",
    )
    fs.get("./data", "./diabetes", recursive=True)
    print("Dataset loaded")
