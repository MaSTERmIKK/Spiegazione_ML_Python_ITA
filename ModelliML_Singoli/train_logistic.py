"""
Carica il dataset di training dal file NPZ, addestra un modello di Logistic Regression
e salva il modello in 'logistic_model.pkl'.
"""

import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

def load_training_data(filename="dataset.npz"):
    data = np.load(filename)
    return data["X_train"], data["y_train"]

def train_and_save_model(X_train, y_train, model_filename="logistic_model.pkl"):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print("Modello Logistic Regression addestrato e salvato in", model_filename)

if __name__ == '__main__':
    X_train, y_train = load_training_data()
    train_and_save_model(X_train, y_train)
