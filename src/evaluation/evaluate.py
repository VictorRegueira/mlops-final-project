import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
import os

class ModelEvaluator:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"{self.data_path} not found.")
        self.data = pd.read_csv(self.data_path)
        # Exclude the 'DateTime' column if present
        if 'DateTime' in self.data.columns:
            self.data = self.data.drop('DateTime', axis=1)
        return self.data


    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"{self.model_path} not found.")
        self.model = joblib.load(self.model_path)
        return self.model

    def evaluate(self):
        X = self.data.iloc[:, :-3]  # Selecciona todas las filas y todas las columnas excepto las últimas tres
        y = self.data['Zone 1 Power Consumption']
        predictions = self.model.predict(X)
        mse = mean_squared_error(y, predictions)
        return mse

    def run(self):
        self.load_data()
        self.load_model()
        mse = self.evaluate()
        print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    evaluator = ModelEvaluator('data\\processed\\processed_dataset.csv', 'models\\model.joblib')
    evaluator.run()
