import pandas as pd
import argparse
from data.data_preprocessing import preprocess_data
from models.model_training import train_model
from tests.score import score


parser = argparse.ArgumentParser()

parser.add_argument("-p", "--path", type=str, help = "Path to the data")
parser.add_argument("-m", "--model", type=str, choices=['svc'], help = "Choose a model")

args = parser.parse_args()


df = pd.read_csv(args.path)

preprocessed_data = preprocess_data(df)

model = train_model(preprocessed_data['X_train'], preprocessed_data['y_train'], args.model)

print(f"accuracy score = {score(preprocessed_data['X_test'], preprocessed_data['y_test'], model)}")
