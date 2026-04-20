import wandb
import pandas as pd

api = wandb.Api()
project_path = "lgand-universit-de-montpellier/rootNav_logs"

print(f"Récupération de la liste des runs pour le projet : {project_path}...")
runs = api.runs(project_path)

all_runs_data = []

for run in runs:
    print(f"Téléchargement des données pour le run : {run.name} (ID: {run.id})...")

    config = run.config
    runtime = run.summary.get("_runtime", "N/A") 
    
    history = run.scan_history()
    
    for row in history:
        row['run_name'] = run.name
        row['runtime'] = runtime
        
        
        all_runs_data.append(row)

print("Conversion des données en tableau (cela peut prendre un instant)...")
df = pd.DataFrame(all_runs_data)

# 3. Sauvegarder en fichier CSV
nom_fichier_csv = "log_rootnav2.csv"
df.to_csv(nom_fichier_csv, index=False)

print(f"Succès ! L'ensemble des données a été sauvegardé et trié dans : {nom_fichier_csv}")