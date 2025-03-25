#!/usr/bin/env python3
"""
train_model.py

Carica il dataset di training dal file NPZ, addestra un modello di regressione logistica
e salva il modello in 'model.pkl'.
"""

import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

def load_training_data(filename="dataset.npz"):
    data = np.load(filename)
    return data["X_train"], data["y_train"]

def train_and_save_model(X_train, y_train, model_filename="model.pkl"):
    # Inizializza e addestra il modello
    model = LogisticRegression()
    model.fit(X_train, y_train)
    # Salva il modello addestrato
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print("Modello addestrato e salvato in", model_filename)

if __name__ == '__main__':
    X_train, y_train = load_training_data()
    train_and_save_model(X_train, y_train)
