import hydra
import pandas as pd
from joblib import dump
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



@hydra.main(config_path="../config", config_name="main", version_base=None)
def prepocess_data(config: DictConfig):
    X = pd.read_csv(config.data.raw.input)
    y = pd.read_csv(config.data.raw.output)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X_train)
    train_features = pd.DataFrame(scaled_features, columns=X.columns.tolist())
    test_features = pd.DataFrame(scaler.transform(X_test), columns=X.columns.tolist())

    train_features.to_csv(config.data.processed.train_features, index=False)
    test_features.to_csv(config.data.processed.test_features, index=False)
    y_train.to_csv(config.data.processed.train_labels, index=False)
    y_test.to_csv(config.data.processed.test_labels, index=False)

    dump(scaler, 'models/scaler.joblib')

if __name__ == "__main__":
    prepocess_data()
