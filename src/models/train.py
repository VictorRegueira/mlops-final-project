import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

class ModelTrainer:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"{self.data_path} not found.")
        self.data = pd.read_csv(self.data_path)
        return self.data

    def train(self):
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        self.model = LinearRegression()
        self.model.fit(X, y)

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def run(self):
        self.load_data()
        self.train()
        self.save_model()

if __name__ == "__main__":
    trainer = ModelTrainer('data\\processed\\processed_dataset.csv', 'models\\model.joblib')
    trainer.run()
