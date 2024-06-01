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
        self.data = self.data.drop('DateTime', axis=1)  # Exclude the DateTime column
        return self.data


    def train(self):
        # Elimina las últimas tres columnas del DataFrame
        X = self.data.iloc[:, :-3]  # Selecciona todas las filas y todas las columnas excepto las últimas tres
        y = self.data['Zone 1 Power Consumption']
        self.model = LinearRegression()
        self.model.fit(X, y)

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def run(self):
        print("Loading data...")
        self.load_data()
        print("Data loaded successfully. Starting training...")
        self.train()
        print("Training completed. Saving model...")
        self.save_model()
        print("Model saved successfully.")

if __name__ == "__main__":
    trainer = ModelTrainer('data\\processed\\processed_dataset.csv', 'models\\model.joblib')
    trainer.run()
