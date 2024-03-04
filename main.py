import fire
import hydra

from MLOps.infer import infer
from MLOps.train import train


def load_config():
    hydra.initialize(config_path="configs", version_base="1.1")
    return hydra.compose(config_name="config.yaml")


def training():
    train(load_config())


def infering():
    infer(load_config())


if __name__ == "__main__":
    fire.Fire(
        {
            "train": training,
            "infer": infering,
        }
    )
