"""
train_random_forest_hyper.py

Esempio di Random Forest con tuning degli iperparametri utilizzando GridSearchCV.
Gli iperparametri esplorati sono:
- n_estimators: numero di alberi nella foresta
- max_depth: profondità massima degli alberi

Il modello ottimizzato viene salvato in 'random_forest_model_hyper.pkl'.
"""

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def load_training_data(filename="dataset.npz"):
    data = np.load(filename)
    return data["X_train"], data["y_train"]

def train_and_save_model(X_train, y_train, model_filename="random_forest_model_hyper.pkl"):
    model = RandomForestClassifier()
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print("Iperparametri ottimali:", grid_search.best_params_)
    print("Miglior accuratezza ottenuta:", grid_search.best_score_)
    
    best_model = grid_search.best_estimator_
    
    with open(model_filename, "wb") as f:
        pickle.dump(best_model, f)
    print("Modello Random Forest ottimizzato salvato in", model_filename)
    
    y_pred = best_model.predict(X_train)
    print("Report di classificazione (training):\n", classification_report(y_train, y_pred))

if __name__ == '__main__':
    X_train, y_train = load_training_data()
    train_and_save_model(X_train, y_train)
