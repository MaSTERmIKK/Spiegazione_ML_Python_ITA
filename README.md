# ML Esempio: Applicazione Pratica di Machine Learning

Questo repository contiene vari esempi pratici per l'uso di modelli di ML di Machine Learning in Python. 

L'obiettivo è fornire un riferimento semplice e modulare per chi desidera imparare ad utilizzare modelli ML per la classificazione con una spiegazione allegata in italiano.

## Panoramica

Il progetto contiene un file Python (`Esempio_Generale_ITA.py`) che serve da guida e dimostra i seguenti passaggi fondamentali:

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
