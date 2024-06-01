import pandas as pd
import os

class DataPreprocessor:
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

    def load_data(self):
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"{self.raw_data_path} not found.")
        self.data = pd.read_csv(self.raw_data_path)
        return self.data

    def preprocess(self):
        processed_data = self.data.copy()
        processed_data.dropna(inplace=True)
        return processed_data

    def save_data(self, data):
        os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
        data.to_csv(self.processed_data_path, index=False)

    def run(self):
        self.load_data()
        processed_data = self.preprocess()
        self.save_data(processed_data)

if __name__ == "__main__":
    preprocessor = DataPreprocessor('data\\raw\\dataset.csv', 'data\\processed\\processed_dataset.csv')
    preprocessor.run()
