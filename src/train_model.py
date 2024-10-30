import hydra
import pandas as pd
from joblib import dump
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression


@hydra.main(config_path="../config", config_name="main", version_base=None)
def train_model(config: DictConfig):
    """Function to train the model"""
    X_train = pd.read_csv(config.data.processed.train_features)
    y_train = pd.read_csv(config.data.processed.train_labels)

    model = LogisticRegression(solver="liblinear")
    model.fit(X_train, y_train.to_numpy().ravel())

    print("Save model into models/")
    dump(model, "models/log_reg.joblib")


if __name__ == "__main__":
    train_model()
