import numpy as np
import pandas as pd

def load_and_split_awa2_features(features_path, labels_path, classes_path, 
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