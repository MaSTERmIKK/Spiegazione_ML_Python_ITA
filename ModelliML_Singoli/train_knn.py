"""
Carica il dataset di training dal file NPZ, addestra un modello K-Nearest Neighbors (KNN)
e salva il modello in 'knn_model.pkl'.
"""

import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

def load_training_data(filename="dataset.npz"):
    data = np.load(filename)
    return data["X_train"], data["y_train"]

def train_and_save_model(X_train, y_train, model_filename="knn_model.pkl"):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print("Modello KNN addestrato e salvato in", model_filename)

if __name__ == '__main__':
    X_train, y_train = load_training_data()
    train_and_save_model(X_train, y_train)
