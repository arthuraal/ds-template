import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.datasets import load_iris



@hydra.main(config_path="../config", config_name="main", version_base=None)
def load_data(config: DictConfig):
    """Function to read the data"""
    raw_path = config.data.raw.path
    data = pd.read_csv(raw_path)
    X, y = data.drop("output", axis=1), data.output
    print(f"Saving data into {raw_path}")
    X.to_csv(f"{config.data.raw.input}", index=False)
    y.to_csv(f"{config.data.raw.output}", index=False)


if __name__ == "__main__":
    load_data()