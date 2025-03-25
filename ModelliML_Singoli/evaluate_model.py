"""
Carica il dataset di test e un modello specificato per valutarne le performance.
Utilizzo:
    python evaluate_model.py <model_filename.pkl>
"""

import sys
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_test_data(filename="dataset.npz"):
    data = np.load(filename)
    return data["X_test"], data["y_test"]

def load_model(model_filename):
    with open(model_filename, "rb") as f:
        model = pickle.load(f)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print("Accuratezza: {:.2f}%".format(acc * 100))
    print("Matrice di confusione:\n", cm)
    print("Report di classificazione:\n", report)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Utilizzo: python evaluate_model.py <model_filename.pkl>")
        sys.exit(1)
    model_filename = sys.argv[1]
    X_test, y_test = load_test_data()
    model = load_model(model_filename)
    evaluate_model(model, X_test, y_test)
