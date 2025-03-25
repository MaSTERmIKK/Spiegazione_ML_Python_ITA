#!/usr/bin/env python3
"""
Carica il modello e il dataset, e effettua una previsione su un singolo campione del test set.
"""

import numpy as np
import pickle

def load_test_data(filename="dataset.npz"):
    data = np.load(filename)
    return data["X_test"]

def load_model(model_filename="model.pkl"):
    with open(model_filename, "rb") as f:
        model = pickle.load(f)
    return model

def predict_sample(model, sample):
    # Assicura che il sample sia in formato bidimensionale
    prediction = model.predict([sample])
    return prediction[0]

if __name__ == '__main__':
    X_test = load_test_data()
    model = load_model()
    sample = X_test[0]  # Usa il primo campione del test set
    pred = predict_sample(model, sample)
    print("Previsione per il campione {}: {}".format(sample, pred))
