import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "heart.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    print(f"Model trained and saved to {MODEL_PATH}")
