import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

races_path = r"C:\Users\cdaddario\OneDrive - Deloitte (O365D)\Desktop\F1_PREDICT\races.csv"
results_path = r"C:\Users\cdaddario\OneDrive - Deloitte (O365D)\Desktop\F1_PREDICT\results.csv"
drivers_path = r"C:\Users\cdaddario\OneDrive - Deloitte (O365D)\Desktop\F1_PREDICT\drivers.csv"
races = pd.read_csv(races_path)
results = pd.read_csv(results_path)
drivers = pd.read_csv(drivers_path)

# Sostituzione di valori "\N" e stringhe vuote con NaN
results.replace({"\\N": np.nan, "": np.nan}, inplace=True)
drivers.replace({"\\N": np.nan, "": np.nan}, inplace=True)
races.replace({"\\N": np.nan, "": np.nan}, inplace=True)

# Unione dei dati
data = results.merge(drivers, on='driverId')
data = data.merge(races, on='raceId')

# Conversione delle colonne numeriche
for col in ['grid', 'laps', 'milliseconds', 'positionOrder']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Rimozione delle righe con valori NaN
data.dropna(subset=['grid', 'laps', 'milliseconds', 'positionOrder'], inplace=True)

# Selezione delle caratteristiche per la previsione
features = ['grid', 'laps', 'milliseconds']
target = 'positionOrder'

# Creazione del target binario: 1 se il pilota è tra i primi 10, altrimenti 0
data['target'] = (data[target] <= 10).astype(int)

# Divisione dei dati in input (X) e output (y)
X = data[features]
y = data['target']

# Divisione dei dati in set di addestramento e di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creazione e addestramento del modello di regressione logistica
model = LogisticRegression()
model.fit(X_train, y_train)

# Predizioni e valutazione del modello
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza del modello: {accuracy:.2f}")

# -------------------------
# Simulazione della gara tra 5 piloti random
# -------------------------

# Selezione casuale di un GP dalla colonna "name"
selected_gp = np.random.choice(data['name'].dropna().unique())  # Selezione casuale di un GP

print(f"\nTracciato selezionato per la gara: {selected_gp}")

# Verifica se ci sono risultati per il GP selezionato
valid_results = results[results['raceId'].isin(races[races['name'] == selected_gp]['raceId'])]

if valid_results.empty:
    print("Nessun risultato disponibile per il tracciato selezionato. Impossibile simulare la gara.")
else:
    # Selezioniamo casualmente 5 piloti che hanno risultati nel GP selezionato
    selected_drivers = valid_results.sample(n=5, random_state=42)['driverId'].unique()
    selected_drivers = drivers[drivers['driverId'].isin(selected_drivers)]
    print("\nPiloti selezionati:")
    print(selected_drivers[['forename', 'surname']])

    # Calcoliamo le statistiche di carriera per i piloti selezionati
    career_stats = results[results['driverId'].isin(selected_drivers['driverId'])].copy()  # Copia esplicita

    # Assicuriamoci che le colonne siano numeriche
    career_stats['milliseconds'] = pd.to_numeric(career_stats['milliseconds'], errors='coerce')

    # Filtriamo le righe con valori NaN
    career_stats.dropna(subset=['milliseconds'], inplace=True)

    # Verifica che ci siano risultati validi per la media
    if career_stats.empty:
        print("Nessun dato valido disponibile per calcolare le statistiche di carriera.")
    else:
        # Calcoliamo le statistiche di carriera
        career_stats = career_stats.groupby('driverId').agg({
            'milliseconds': 'mean'  # Tempo medio in millisecondi
        }).reset_index()

        # Creazione di dati di gara casuali per i piloti selezionati
        simulated_data = pd.DataFrame({
            'driverId': selected_drivers['driverId'],
            'grid': np.random.randint(1, 21, size=len(selected_drivers)),  # Posizione in griglia casuale tra 1 e 20
            'laps': np.random.randint(50, 80, size=len(selected_drivers)),  # Numero casuale di giri tra 50 e 80
        })

        # Uniamo i dati delle statistiche di carriera
        simulated_data = simulated_data.merge(career_stats, on='driverId', how='left')

        # Rimuoviamo eventuali righe con NaN nel dataset simulato
        simulated_data.dropna(subset=['grid', 'laps', 'milliseconds'], inplace=True)

        # Verifica che ci siano dati validi prima della previsione
        if simulated_data.empty:
            print("Nessun dato valido disponibile per la previsione. Controlla i dati di input.")
        else:
            # Simuliamo il tempo totale basato sulla posizione in griglia e sulle statistiche di carriera
            simulated_data.loc[:, 'total_time'] = simulated_data['milliseconds'] + (simulated_data['grid'] - 1) * 10  # Aggiungiamo un bonus in base alla posizione in griglia

            # Previsione della probabilità di vincere usando le stesse caratteristiche del modello addestrato
            simulated_data['probability_win'] = model.predict_proba(simulated_data[features])[:, 1]

            # Ordiniamo i piloti in base alla probabilità di vincere
            simulated_data = simulated_data.merge(selected_drivers[['driverId', 'forename', 'surname']], on='driverId')
            simulated_data = simulated_data.sort_values(by='probability_win', ascending=False)

            # Mostra la classifica simulata e il pilota favorito
            print("\nClassifica simulata della gara:")
            print(simulated_data[['forename', 'surname', 'grid', 'laps', 'milliseconds', 'probability_win']].reset_index(drop=True))

            # Pilota con la probabilità più alta di vincere
            winner = simulated_data.iloc[0]
            print(f"\nIl pilota con la probabilità più alta di vincere è {winner['forename']} {winner['surname']} con una probabilità di {winner['probability_win']:.2f}.")
