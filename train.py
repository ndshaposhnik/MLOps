import pickle

from sklearn.ensemble import RandomForestClassifier

from src import common


def train():
    print("Loading dataset...")
    X, y = common.dataset_to_X_y("data/train.csv")
    print("Dataset loaded")
    model = RandomForestClassifier(max_depth=2, random_state=0)
    print("Start training...")
    model.fit(X, y)
    print("Training done")
    pickle.dump(model, open("models/random_forest", "wb"))


if __name__ == "__main__":
    train()
