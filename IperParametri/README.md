# Esempi di Hyperparameter Tuning in Machine Learning

Questa repository contiene esempi pratici di come impostare ed ottimizzare gli iperparametri in modelli di Machine Learning utilizzando Python e scikit-learn.

## Cosa Sono Gli Iperparametri?

Gli **iperparametri** sono i parametri che definiscono il comportamento del modello durante l'addestramento e che devono essere impostati prima dell'allenamento. Esempi:
- **Logistic Regression:** `C` (inverso della regolarizzazione), `solver`
- **SVM:** `C`, `kernel`, `gamma`
- **Random Forest:** `n_estimators`, `max_depth`
- **KNN:** `n_neighbors`, `weights`

## Struttura della Repository

- **data_preparation.py:** Genera un dataset sintetico per la classificazione e lo salva in un file NPZ.
- **train_logistic_hyper.py:** Addestra una Logistic Regression ottimizzando gli iperparametri.
- **train_svm_hyper.py:** Addestra una Support Vector Machine ottimizzando gli iperparametri.
- **train_random_forest_hyper.py:** Addestra un modello Random Forest ottimizzando gli iperparametri.
- **train_knn_hyper.py:** Addestra un modello K-Nearest Neighbors ottimizzando gli iperparametri.

## Come Utilizzare

1. **Generare il dataset:**

   ```bash
   python data_preparation.py

