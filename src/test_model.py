import json
import hydra
import pandas as pd
from joblib import load
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score


@hydra.main(config_path="../config", config_name="main", version_base=None)
def test_model(config: DictConfig):
    """Function to test the model"""
    X_test = pd.read_csv(config.data.processed.test_features)
    y_test = pd.read_csv(config.data.processed.test_labels).to_numpy().ravel()

    model = load('models/log_reg.joblib')
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    # Save metrics in a json file
    metrics = {
        'f1_score': f1,
        'accuracy': accuracy
    }

    with open('metrics/log_reg_metrics.json', 'w') as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    test_model()