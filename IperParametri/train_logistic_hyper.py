"""
train_logistic_hyper.py

Esempio di Logistic Regression con tuning degli iperparametri utilizzando GridSearchCV.
Gli iperparametri esplorati sono:
- C: parametro di regolarizzazione inversa
- solver: algoritmo di ottimizzazione

Il modello ottimizzato viene salvato in 'logistic_model_hyper.pkl'.
"""

import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def load_training_data(filename="dataset.npz"):
    data = np.load(filename)
    return data["X_train"], data["y_train"]

def train_and_save_model(X_train, y_train, model_filename="logistic_model_hyper.pkl"):
    # Modello base
    model = LogisticRegression(max_iter=1000)
    
    # Griglia degli iperparametri
    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    }
    
    # Ottimizzazione tramite Grid Search con validazione incrociata (cv=5)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print("Iperparametri ottimali:", grid_search.best_params_)
    print("Miglior accuratezza ottenuta:", grid_search.best_score_)
    
    best_model = grid_search.best_estimator_
    
    # Salvataggio del modello ottimizzato
    with open(model_filename, "wb") as f:
        pickle.dump(best_model, f)
    print("Modello Logistic Regression ottimizzato salvato in", model_filename)
    
    # Report di classificazione sui dati di training
    y_pred = best_model.predict(X_train)
    print("Report di classificazione (training):\n", classification_report(y_train, y_pred))

if __name__ == '__main__':
    X_train, y_train = load_training_data()
    train_and_save_model(X_train, y_train)
