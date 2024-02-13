import pickle

import common
from sklearn.ensemble import RandomForestClassifier


def train():
    common.load_data()
    X, y = common.dataset_to_X_y("data/train.csv")
    model = RandomForestClassifier(max_depth=2, random_state=0)
    print("Start training...")
    model.fit(X, y)
    print("Training done")
    pickle.dump(model, open("data/random_forest.pth", "wb"))


if __name__ == "__main__":
    train()
