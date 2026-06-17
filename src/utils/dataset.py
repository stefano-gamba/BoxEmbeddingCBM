import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

#-------------------------
# AWA2 DATASET
#-------------------------


def zsl_split_awa2_features(features_path, labels_path, classes_path, 
                                 train_split_path, val_split_path, test_split_path):
    print("Caricamento in corso... (potrebbe richiedere qualche secondo)")
    
    # 1. Carica le feature (37322 x 2048) e le label (37322 x 1)
    # Assumiamo che siano file di testo separati da spazi o tab
    features = np.loadtxt(features_path, dtype=np.float32)
    labels = np.loadtxt(labels_path, dtype=int)
    
    # 2. Carica la mappatura delle classi (ID -> Nome Classe)
    # Il file classes.txt di solito ha due colonne: ID e nome_classe
    classes_df = pd.read_csv(classes_path, sep='\t', header=None, names=['id', 'class_name'])
    
    # Creiamo un dizionario per mappare il nome della classe al suo ID
    class_name_to_id = dict(zip(classes_df['class_name'], classes_df['id']))
    
    # 3. Leggiamo i file di split (che contengono i nomi delle classi)
    with open(train_split_path, 'r') as f:
        train_classes = f.read().splitlines()
        
    with open(val_split_path, 'r') as f:
        val_classes = f.read().splitlines()
        
    with open(test_split_path, 'r') as f:
        test_classes = f.read().splitlines()
        
    # 4. Convertiamo i nomi delle classi negli ID numerici corrispondenti
    train_ids = [class_name_to_id[name] for name in train_classes]
    val_ids = [class_name_to_id[name] for name in val_classes]
    test_ids = [class_name_to_id[name] for name in test_classes]
    
    # 5. Creiamo le maschere booleane per filtrare gli array
    # np.isin controlla se ogni elemento in 'labels' è presente nella lista degli ID
    train_mask = np.isin(labels, train_ids)
    val_mask = np.isin(labels, val_ids)
    test_mask = np.isin(labels, test_ids)
    
    # 6. Splittiamo feature e label
    X_train, y_train = features[train_mask], labels[train_mask]
    X_val, y_val = features[val_mask], labels[val_mask]
    X_test, y_test = features[test_mask], labels[test_mask]
    
    print(f"Dimensioni totali feature: {features.shape}")
    print(f"Training set: {X_train.shape[0]} samples (27 classi)")
    print(f"Validation set: {X_val.shape[0]} samples (13 classi)")
    print(f"Test set: {X_test.shape[0]} samples (10 classi)")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def classical_split_awa2_features(features_path, labels_path, test_size=0.2, val_size=0.1, random_seed=42):
    """
    Carica le feature e le label di AWA2 e crea uno split classico stratificato per tutte le 50 classi.
    
    Di default crea uno split: 70% Train, 10% Validation, 20% Test.
    """
    print("Caricamento dei dati in corso... (potrebbe richiedere qualche secondo)")
    
    # 1. Carica le feature (37322 x 2048) e le label (37322 x 1)
    X = np.loadtxt(features_path, dtype=np.float32)
    y = np.loadtxt(labels_path, dtype=int)
    
    print(f"Dataset caricato correttamente: {X.shape[0]} campioni con {X.shape[1]} feature ciascuno.")
    
    # 2. Primo split: Separiamo il Test Set dal resto (Train + Val)
    # L'argomento stratify=y garantisce che le 50 classi siano bilanciate
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_seed, 
        stratify=y
    )
    
    # 3. Secondo split: Dividiamo il blocco rimanente in Training e Validation
    # Dobbiamo calcolare la proporzione relativa per il validation set.
    # Se test_size=0.2 e val_size=0.1, al blocco temp rimane l'80% dei dati.
    # Il validation deve essere il 10% del totale, quindi il 12.5% dell'80% rimanente (0.1 / 0.8)
    val_relative_size = val_size / (1.0 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_relative_size, 
        random_state=random_seed, 
        stratify=y_temp
    )
    
    # 4. Riepilogo finale
    print("\n--- Risultati dello Split Stratificato (50 Classi) ---")
    print(f"Training set:   {X_train.shape[0]} campioni")
    print(f"Validation set: {X_val.shape[0]} campioni")
    print(f"Test set:       {X_test.shape[0]} campioni")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

class ConceptImplicationDataset(Dataset):
    def __init__(self, ground_truth_matrix):
        """
        ground_truth_matrix: Tensore 2D di shape (num_concepts, num_concepts)
        """
        self.num_concepts = ground_truth_matrix.shape[0]
        
        # Creiamo tutte le combinazioni possibili di indici (i, j)
        i_indices, j_indices = torch.meshgrid(
            torch.arange(self.num_concepts), 
            torch.arange(self.num_concepts), 
            indexing='ij'
        )
        
        # Appiattiamo le matrici in vettori 1D
        self.idx_i = i_indices.flatten()
        self.idx_j = j_indices.flatten()
        self.targets = ground_truth_matrix.flatten()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.idx_i[idx], self.idx_j[idx], self.targets[idx]
    
#-------------------------
# OAI DATASET
#-------------------------

TIMEPOINT_MAP = {
    '00m': '00',
    '48m': '06',
    '72m': '08',
    '96m': '10'
}

VISUAL_CONCEPTS = [
    'xrosfm', # Osteofiti (Femore Mediale)
    'xrscfm', # Sclerosi (Femore Mediale)
    'xrjsm',  # Restringimento Spazio Articolare (Mediale)
    'xrostm', # Osteofiti (Tibia Mediale)
    'xrsctm', # Sclerosi (Tibia Mediale)
    'xrosfl', # Osteofiti (Femore Laterale)
    'xrscfl', # Sclerosi (Femore Laterale)
    'xrjsl',  # Restringimento Spazio Articolare (Laterale)
    'xrostl', # Osteofiti (Tibia Laterale)
    'xrsctl'  # Sclerosi (Tibia Laterale)
]
TARGET_COL = 'xrkl'
BASE_COLS_TO_KEEP = ['id', 'side', 'barcdbu'] + VISUAL_CONCEPTS + [TARGET_COL]

def mappa_immagini_tramite_manifest(root_dir, df_pulito, manifest_path):
    print("1. Lettura del manifest NDA...")
    manifest = pd.read_csv(manifest_path, sep='\t', header=1, low_memory=False)
    manifest.columns = [str(c).strip().lower() for c in manifest.columns]

    # Riferimenti alle colonne esatte che hai stampato
    col_desc = 'image description, i.e. dti, fmri, fast spgr, phantom, eeg, dynamic pet'
    col_subj = "subject id how it's defined in lab/project"

    print("2. Creazione dell'indice di ricerca universale...")
    # Isoliamo solo i record relativi alle ginocchia
    is_knee = manifest[col_desc].str.contains('knee|pa fixed flexion', case=False, na=False)
    manifest_knee = manifest[is_knee].copy()

    # Creiamo un dizionario: ID Paziente -> Mega-stringa con tutti i dati delle sue radiografie del ginocchio
    knee_records_by_subj = {}
    for subj_id, group in manifest_knee.groupby(col_subj):
        # Uniamo TUTTI i valori di TUTTE le colonne in una singola stringa minuscola
        mega_stringa = " | ".join(group.astype(str).values.flatten()).lower()
        knee_records_by_subj[str(subj_id)] = mega_stringa

    print("3. Scansione del disco e verifica incrociata...")
    root_path = Path(root_dir)
    immagini_trovate = {}
    file_jpg_totali_visti = 0

    for folder_name, sas_visit_code in TIMEPOINT_MAP.items():
        time_path = root_path / folder_name
        if not time_path.exists():
            continue

        for img_path in time_path.rglob('*.jpg'):
            file_jpg_totali_visti += 1

            # Dal nome file '00175004_1x1.jpg' ricaviamo le due possibili forme in cui è scritto
            base_id_str = img_path.stem.split('_')[0]                 # '00175004'
            base_id_int_str = str(int(base_id_str)) if base_id_str.isdigit() else base_id_str # '175004'

            # Estraiamo l'ID Paziente (7 cifre)
            match = re.search(r'/([9]\d{6})/', str(img_path))
            if not match:
                continue
            paziente_id = match.group(1)

            # ---> IL CONTROLLO OMNI-SEARCH <---
            if paziente_id in knee_records_by_subj:
                mega_stringa = knee_records_by_subj[paziente_id]

                # Se l'identificativo immagine compare OVUNQUE nei record 'Knee' di questo paziente... è un ginocchio!
                if (base_id_str in mega_stringa) or (base_id_int_str in mega_stringa):
                    logical_key = f"{paziente_id}_{sas_visit_code}"
                    immagini_trovate[logical_key] = str(img_path)

    print(f"\n--- REPORT SCANSIONE ---")
    print(f"File JPG totali visti sul disco: {file_jpg_totali_visti}")
    print(f"Match Riusciti (Ginocchia certificate): {len(immagini_trovate)}")

    print("\n4. Join finale con le annotazioni SAS...")
    df_pulito['logical_key'] = df_pulito['id'] + '_' + df_pulito['timepoint']
    df_pulito['image_path'] = df_pulito['logical_key'].map(immagini_trovate)

    df_finale = df_pulito.dropna(subset=['image_path']).copy()
    df_finale = df_finale.drop(columns=['logical_key'])

    print(f"Dataframe completato: {len(df_finale)} ginocchia singole pronte per l'addestramento.")
    return df_finale

def estrai_roi_dinamica(img_ginocchio):
    """
    Trova dinamicamente lo spazio articolare e restituisce un crop quadrato perfetto.
    L'input deve essere la metà ritagliata (il singolo ginocchio, destro o sinistro).
    """
    # 1. Convertiamo in scala di grigi e array NumPy per l'analisi
    img_np = np.array(img_ginocchio.convert('L'))
    h, w = img_np.shape

    # 2. Isoliamo la fascia centrale verticale (evita i marker R/L ai bordi)
    fascia_centrale = img_np[:, int(w*0.3) : int(w*0.7)]

    # 3. Sommiamo la luminosità riga per riga (Profilo di intensità)
    profilo = np.sum(fascia_centrale, axis=1)

    # 4. Cerchiamo la "valle" scura (il giunto) solo nel blocco centrale dell'immagine
    # per evitare di prendere i bordi neri estremi come falsi positivi
    inizio_ricerca = int(h * 0.35)
    fine_ricerca = int(h * 0.65)

    # L'indice del pixel più scuro in questa zona è la nostra Y del giunto articolare!
    y_giunto = inizio_ricerca + np.argmin(profilo[inizio_ricerca:fine_ricerca])

    # 5. Creiamo un box quadrato centrato sulla Y trovata
    # Usiamo circa l'85% della larghezza per fare un bello zoom e scartare i bordi neri laterali
    lato_box = int(w * 0.85)
    mezzo_lato = lato_box // 2

    top = max(0, y_giunto - mezzo_lato)
    bottom = top + lato_box

    # Se il box sfonda il bordo inferiore, lo spingiamo in su
    if bottom > h:
        bottom = h
        top = h - lato_box

    left = int(w * 0.075) # Centriamo orizzontalmente
    right = left + lato_box

    # Ritagliamo il quadrato perfetto e lo restituiamo
    return img_ginocchio.crop((left, top, right, bottom))

class OAICBMDataset(Dataset):
    def __init__(self, df_finale, c_cols, y_col, transform=None):
        """
        Dataset custom per caricare radiografie singole del ginocchio (ritagliate da img bilaterali),
        i concetti visivi (c) e il target (y).
        """
        # Assicuriamoci che l'indice sia pulito per usare iloc nel __getitem__
        self.df = df_finale.reset_index(drop=True)
        self.c_cols = c_cols
        self.y_col = y_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Caricamento Immagine Bilaterale
        image_path = row['image_path']
        side = int(row['side']) # 1 = Destro, 2 = Sinistro

        try:
            img_bilaterale = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Errore nel caricamento dell'immagine {image_path}: {e}")
            # Fallback (utile in fase di test, in produzione meglio rimuovere i path corrotti a monte)
            img_bilaterale = Image.new('RGB', (1000, 500), (0, 0, 0))

        width, height = img_bilaterale.size

        # 2. Taglio a metà dell'immagine bilaterale
        if side == 1:
            img_meta = img_bilaterale.crop((0, 0, width // 2, height))
        else:
            img_meta = img_bilaterale.crop((width // 2, 0, width, height))

        # 3. EXTRACTION ROI DINAMICA (Il nuovo passaggio!)
        img_quadrata_roi = estrai_roi_dinamica(img_meta)

        # 4. Trasformazioni PyTorch
        if self.transform:
            image = self.transform(img_quadrata_roi)
        else:
            image = img_quadrata_roi

        # 5. Estrazione Concetti Visivi (c)
        # Sostituiamo i NaN nei concetti con 0
        c_feats = row[self.c_cols].astype(float).fillna(0).values.astype(np.float32)

        # Maschera per ignorare i concetti mancanti durante il calcolo della loss nel CBM
        c_mask = (~row[self.c_cols].isna()).values.astype(np.float32)

        # 6. Estrazione Target (y)
        y_feat = np.float32(row[self.y_col])

        return {
            'image': image,
            'c_feats': torch.tensor(c_feats),
            'c_mask': torch.tensor(c_mask),
            'y': torch.tensor(y_feat),
            'image_key': row.get('logical_key', str(idx)) # Utile per debug
        }
    
def mostra_tensore_immagine(tensor, title="Radiografia Ginocchio Singolo"):
    # 1. Definizione degli stessi mean e std usati per normalizzare
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # 2. Convertiamo il tensore in array NumPy e spostiamo i canali (C, H, W -> H, W, C)
    image = tensor.numpy().transpose((1, 2, 0))

    # 3. Denormalizziamo (Moltiplichiamo per std e sommiamo mean)
    image = std * image + mean

    # 4. Tagliamo i valori fuori dal range [0, 1] (per via di approssimazioni in virgola mobile)
    image = np.clip(image, 0, 1)

    # 5. Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()