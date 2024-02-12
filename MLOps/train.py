import pickle

from common import dataset_to_X_y
from sklearn.ensemble import RandomForestClassifier


def train():
    print("Loading dataset...")
    X, y = dataset_to_X_y("../data/train.csv")
    print("Dataset loaded")
    model = RandomForestClassifier(max_depth=2, random_state=0)
    print("Start training...")
    model.fit(X, y)
    print("Training done")
    pickle.dump(model, open("../data/random_forest.pth", "wb"))


if __name__ == "__main__":
    train()
