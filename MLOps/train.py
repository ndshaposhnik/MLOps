import pickle

import common
from sklearn.ensemble import RandomForestClassifier


def train():
    common.load_data()
    return
    X, y = common.dataset_to_X_y("diabetes/train.csv")
    model = RandomForestClassifier(max_depth=2, random_state=0)
    print("Start training...")
    model.fit(X, y)
    print("Training done")
    pickle.dump(model, open("random_forest.pth", "wb"))


if __name__ == "__main__":
    train()
