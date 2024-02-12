import pandas as pd


def normalize_dataframe(df):
    return (df - df.mean()) / df.std()


def dataset_to_X_y(filename, nrows=None, normalize=False):
    data = pd.read_csv(filename, nrows=nrows)
    X = data.loc[:, data.columns[:-1]]
    y = data.loc[:, data.columns[-1]]
    if normalize:
        X = normalize_dataframe(X)
    return X.to_numpy(), y.to_numpy()
