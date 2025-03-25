#!/usr/bin/env python3
"""
data_preparation.py

Genera un dataset sintetico per la classificazione, lo divide in training e test,
e lo salva in un file NPZ (dataset.npz).
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def generate_and_save_data(filename="dataset.npz", n_samples=1000, n_features=2, random_state=42):
    # Genera dataset con due classi e due feature informative
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                               n_informative=2, n_redundant=0, random_state=random_state)
    # Suddivide in training e test (70% training, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    # Salva i dati in un unico file NPZ
    np.savez(filename, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print("Dataset salvato in", filename)

if __name__ == '__main__':
    generate_and_save_data()
