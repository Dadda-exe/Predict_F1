# F1 Race Predictor 🏎️

Questo progetto utilizza **Python**, **scikit-learn** e **pandas** per creare un modello di **regressione logistica** che predice se un pilota di Formula 1 finirà **nei primi 10** di una gara.

## Contenuto
- **Addestramento** del modello usando dati storici (`races.csv`, `drivers.csv`, `results.csv`).
- **Simulazione** di una gara tra 5 piloti casuali.
- **Calcolo delle probabilità** di vittoria per ciascun pilota.

## Come funziona
1. Si caricano e si puliscono i dati.
2. Si addestra un modello di **Logistic Regression**.
3. Si selezionano 5 piloti a caso da un GP random.
4. Si simula una gara stimando chi ha più probabilità di vincere.

## Librerie richieste
- pandas
- scikit-learn
- numpy

Installa le dipendenze con:
```bash
pip install -r requirements.txt
