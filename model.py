import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

def predict(data):
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    return prediction, probability
