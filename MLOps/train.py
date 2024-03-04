import pickle

from sklearn.ensemble import RandomForestClassifier

from MLOps import common


def train(cfg) -> None:
    common.load_data()
    X, y = common.dataset_to_X_y("diabetes/train.csv")
    model = RandomForestClassifier(
        max_depth=cfg["train"]["max_depth"],
        random_state=0,
    )
    print("Start training...")
    model.fit(X, y)
    print("Training done")
    pickle.dump(model, open(cfg["model"]["save_path"], "wb"))
