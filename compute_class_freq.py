import os
import torch
import numpy as np
from tqdm import tqdm

def calculate_weights_from_cache(split_folder, num_classes=6):
    class_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0

    # 1. Trouver tous les fichiers caches .pt
    files = [f for f in os.listdir(split_folder) if f.endswith('.pt')]
    
    if len(files) == 0:
        print("❌ Aucun fichier .pt trouvé. Assure-toi d'avoir lancé ton Dataloader au moins une fois pour qu'il génère les caches !")
        return

    print(f"🔍 Comptage des pixels sur {len(files)} fichiers caches .pt...")
    
    # 2. Parcourir les caches et compter
    for filename in tqdm(files):
        filepath = os.path.join(split_folder, filename)
        
        # Charger le dictionnaire mis en cache par _create_cache_file
        cache = torch.load(filepath)
        mask = cache["mask"].numpy() # C'est ton masque de segmentation 0-5
        
        # Compter les occurrences
        counts = np.bincount(mask.flatten(), minlength=num_classes) # Compte les pixels de chaque classe
        class_counts += counts[:num_classes] # limiter à num_classes au cas où il y aurait des valeurs inattendues
        total_pixels += mask.size # nombre total de pixels traités

    # 3. Fréquence de chaque classe
    class_frequencies = class_counts / total_pixels

    # 4. Trouver la médiane de ces fréquences
    median_frequency = np.median(class_frequencies)

    # 5. Calculer les poids finaux
    weights = np.zeros(num_classes, dtype=np.float32)
    for i in range(num_classes):
        if class_frequencies[i] > 0:
            weights[i] = median_frequency / class_frequencies[i]
        else:
            weights[i] = 0.0

    print("\n📊 --- RÉSULTATS ---")
    print(f"Pixels totaux   : {total_pixels}")
    for i in range(num_classes):
        print(f"Classe {i} freq : {class_frequencies[i]:.6f} -> Poids : {weights[i]:.4f}")
    
    # Formatage de la liste pour ton code
    weights_list = [round(float(w), 4) for w in weights]
    print("\n✅ ---> À COPIER DANS run_training.py (ligne 34) :")
    print(f"weights = {weights_list}")

# --- UTILISATION ---
# Mets le chemin vers ton dossier 'train' (là où se trouvent tes images, tes .rsml et tes .pt)
DOSSIER_TRAIN = "/home/loai/Documents/code/RSMLExtraction/RSA_reconstruction/Method/RootNav2/flatten_data" 

calculate_weights_from_cache(DOSSIER_TRAIN, num_classes=6)