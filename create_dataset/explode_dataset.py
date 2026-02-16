import os
import numpy as np
import tifffile
import argparse
from pathlib import Path
from tqdm import tqdm
from openalea.mtg import MTG
from copy import deepcopy
from typing import List, Dict

import os

import openalea.rsml
import tifffile


class LightRSAClass:
    def __init__(self, folder_path: str, load_date_map: bool = False, lazy: bool = True):
        """
        Initialise le DataLoader léger pour un dossier contenant les données.
        
        :param folder_path: Chemin vers le dossier contenant les fichiers.
        :param load_date_map: Si True, le date_map sera chargé (optionnel).
        :param lazy: Si True, le chargement se fera à la demande (lazy loading). 
                     Si False, toutes les données seront chargées immédiatement.
        """
        self.folder_path = folder_path
        self.load_date_map_flag = load_date_map
        self.lazy = lazy

        # Définition des chemins vers les fichiers
        self.image_stack_path = os.path.join(folder_path, "22_registered_stack.tif")
        self.date_map_path = os.path.join(folder_path, "40_date_map.tif")
        self.rsml_expert_file = os.path.join(folder_path, "61_graph_expertized.rsml")
        self.rsml_default_file = os.path.join(folder_path, "61_graph.rsml")

        # Initialisation des attributs (données)
        self._image_stack = None
        self._date_map = None
        self._mtg = None

        # Si lazy=False, charger immédiatement toutes les données
        if not self.lazy:
            self.load_all()

    @property
    def image_stack(self):
        if self._image_stack is None:
            if os.path.exists(self.image_stack_path):
                self._image_stack = tifffile.imread(self.image_stack_path)
            else:
                raise FileNotFoundError(f"Image stack non trouvé : {self.image_stack_path}")
        return self._image_stack

    @property
    def date_map(self):
        if not self.load_date_map_flag:
            return None
        if self._date_map is None:
            if os.path.exists(self.date_map_path):
                self._date_map = tifffile.imread(self.date_map_path)
            else:
                print(f"Avertissement : date_map non trouvé dans {self.date_map_path}")
        return self._date_map

    @property
    def mtg(self):
        if self._mtg is None:
            if os.path.exists(self.rsml_expert_file):
                self._mtg = rsml.rsml2mtg(self.rsml_expert_file)
            elif os.path.exists(self.rsml_default_file):
                self._mtg = rsml.rsml2mtg(self.rsml_default_file)
            else:
                raise FileNotFoundError("Aucun fichier RSML trouvé dans " + self.folder_path)
        return self._mtg

    def load_all(self):
        """
        Charge toutes les données (image stack, MTG et date_map si activé).
        """
        _ = self.image_stack  # Force le chargement de l'image stack
        _ = self.mtg  # Force le chargement du MTG
        if self.load_date_map_flag:
            _ = self.date_map

    def get_data(self):
        """
        Renvoie un dictionnaire contenant l'image stack, le MTG et, éventuellement, le date_map.
        Cette méthode est appelée dans __getitem__ pour retourner les données.
        
        :return: dict avec les clés 'image_stack', 'mtg' et 'date_map'
        """
        return {
            "image_stack": self.image_stack,
            "mtg": self.mtg,
            "date_map": self.date_map
        }

class DirectoryRSAClass:
    def __init__(self, base_dir: str, load_date_map: bool = False, lazy: bool = True):
        """
        Parcourt de manière récursive l'arborescence à partir de base_dir et
        crée une instance LightRSAClass pour chaque dossier contenant les fichiers requis.
        
        :param base_dir: Répertoire racine contenant les sous-dossiers.
        :param load_date_map: Si True, le date_map sera chargé pour chaque dataset.
        :param lazy: Si True, chaque LightRSAClass utilise le chargement paresseux.
        """
        self.base_dir = base_dir
        self.load_date_map_flag = load_date_map
        self.lazy = lazy
        self.loaders = []
        self._scan_directories()

    def _scan_directories(self):
        """
        Parcourt l'arborescence des fichiers et ajoute dans self.loaders tous les dossiers
        contenant le fichier '22_registered_stack.tif'.
        """
        for root, dirs, files in os.walk(self.base_dir):
            # On considère que le dossier est valide s'il contient l'image stack
            if "22_registered_stack.tif" in files:
                loader = LightRSAClass(root, load_date_map=self.load_date_map_flag, lazy=self.lazy)
                self.loaders.append(loader)

    def get_loaders(self):
        """Retourne la liste des LightRSAClass trouvés."""
        return self.loaders

    def __iter__(self):
        return iter(self.loaders)

    def __len__(self):
        return len(self.loaders)

    def __getitem__(self, index):
        """
        Renvoie le dataset correspondant à l'index donné sous forme d'un dictionnaire.
        Si le chargement est lazy, les données seront chargées lors de cet appel.
        """
        loader = self.loaders[index]
        return loader.get_data()

def _truncate_lists(prop: Dict[int, List], idx: int, v: int) -> None:
    val = prop.get(v)
    if isinstance(val, (list, tuple)) and len(val) > idx + 1:
        prop[v] = val[: idx + 1]  # garde 0…idx

def extract_mtg_at_time_t(g: MTG, t: int) -> MTG:
    g_new = deepcopy(g)

    time_prop = g_new.property("time")
    time_h_prop = g_new.property("time_hours")
    diameter_prop = g_new.property("diameter")
    geometry_prop = g_new.property("geometry")
    
    if t == -1: 
        t = max(max(time_prop.values()))

    to_remove = []
    for v, serie in time_prop.items():
        first_t = serie[0]
        if first_t > t:
            to_remove.append(v)
        else:
            idx = max(i for i, tau in enumerate(serie) if tau <= t)

            _truncate_lists(time_prop, idx, v)
            _truncate_lists(time_h_prop, idx, v)
            _truncate_lists(diameter_prop, idx, v)
            _truncate_lists(geometry_prop, idx, v)

            # if list is empty has 1 or less elements, remove vertex
            if len(geometry_prop[v]) <= 1:
                to_remove.append(v)

    for v in to_remove:
        try:
            g_new.remove_tree(v)
        except Exception:
            g_new.remove_vertex(v, reparent_child=False)

    return g_new

def save_rsml(mtg, path):
    try:
        rsml.mtg2rsml(mtg, str(path))
    except AttributeError:
        print(f"Erreur lors de la sauvegarde du RSML à {path}")

def explode_series(base_input_dir, base_output_dir):
    print(f"Scan des données dans : {base_input_dir}")
    dataset_loader = DirectoryRSAClass(base_input_dir, load_date_map=True, lazy=True)
    
    print(f"Trouvé {len(dataset_loader)} séries temporelles.")
    
    output_path = Path(base_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, loader in enumerate(tqdm(dataset_loader.loaders, desc="Traitement des séries")):
        
        try:
            data = loader.get_data()
            image_stack = data['image_stack']
            date_map = data['date_map']
            full_mtg = data['mtg']
        except Exception as e:
            print(f"Erreur lors du chargement de {loader.folder_path}: {e}")
            continue

        series_name = Path(loader.folder_path).name
        series_output_dir = output_path / series_name
        series_output_dir.mkdir(exist_ok=True)

        num_slices = image_stack.shape[0]

        for t in range(num_slices):
            time_folder = series_output_dir / f"t_{t:04d}"
            time_folder.mkdir(exist_ok=True)

            img_slice = image_stack[t]
            tifffile.imwrite(time_folder / "image.tif", img_slice)
            
            if date_map is not None:
                current_time_limit = t + 1
                mask_projected = np.where(
                    (date_map != 0) & (date_map <= current_time_limit), 
                    255, 
                    0
                ).astype(np.uint8)
                tifffile.imwrite(time_folder / "mask_projected.tif", mask_projected)

            if full_mtg is not None:
                try:
                    mtg_t = extract_mtg_at_time_t(full_mtg, t + 1)
                    save_rsml(mtg_t, time_folder / "graph_projected.rsml")
                except Exception as e:
                    pass

    print("\nExtraction terminée avec succès.")

if __name__ == "__main__":
    input = "/home/loai/Documents/code/RSMLExtraction/RSA_deep_working/Data/Val"
    output = "/home/loai/Documents/code/RSMLExtraction/RSA_reconstruction/Method/RootNav2/exploded_data/val/"
    
    explode_series(input, output)