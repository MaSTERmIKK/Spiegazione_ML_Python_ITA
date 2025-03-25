# ML Esempio: Applicazione Pratica di Machine Learning

Questo repository contiene un esempio pratico per l'uso di un modello di Machine Learning in Python. L'obiettivo è fornire un riferimento semplice e modulare per chi desidera imparare ad utilizzare modelli ML per la classificazione.

## Panoramica

Il progetto è realizzato in un singolo file Python (`ml_example.py`) e dimostra i seguenti passaggi fondamentali:

- **Generazione del Dataset:** Creazione di un dataset sintetico per la classificazione.
- **Divisione dei Dati:** Suddivisione del dataset in set di training e test.
- **Addestramento del Modello:** Utilizzo della regressione logistica per l'addestramento.
- **Valutazione del Modello:** Calcolo dell'accuratezza, visualizzazione della matrice di confusione e generazione di un report di classificazione.
- **Visualizzazione dei Dati:** Creazione di un grafico a dispersione (scatter plot) per la visualizzazione dei dati (se a 2 dimensioni).
- **Salvataggio e Caricamento:** Persistenza del modello addestrato tramite `pickle` e successiva ricarica.
- **Previsione:** Esecuzione di una previsione su un campione di dati.

## Prerequisiti

Assicurati di avere installato Python 3.x e le seguenti librerie:
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)

Puoi installare le dipendenze con il seguente comando:

```bash
pip install numpy matplotlib scikit-learn
