import pickle

import common
import hydra
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    common.load_data()
    X, y = common.dataset_to_X_y("diabetes/train.csv")
    model = RandomForestClassifier(max_depth=cfg["max_depth"], random_state=0)
    print("Start training...")
    model.fit(X, y)
    print("Training done")
    pickle.dump(model, open("random_forest.pth", "wb"))


if __name__ == "__main__":
    train()
