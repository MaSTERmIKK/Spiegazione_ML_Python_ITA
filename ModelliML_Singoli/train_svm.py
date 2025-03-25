"""
Carica il dataset di training dal file NPZ, addestra un modello SVM
e salva il modello in 'svm_model.pkl'.
"""

import numpy as np
import pickle
from sklearn.svm import SVC

def load_training_data(filename="dataset.npz"):
    data = np.load(filename)
    return data["X_train"], data["y_train"]

def train_and_save_model(X_train, y_train, model_filename="svm_model.pkl"):
    model = SVC(probability=True)  # Abilita la stima delle probabilità
    model.fit(X_train, y_train)
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print("Modello SVM addestrato e salvato in", model_filename)

if __name__ == '__main__':
    X_train, y_train = load_training_data()
    train_and_save_model(X_train, y_train)
