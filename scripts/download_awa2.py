import os
import requests
import zipfile
import argparse
from tqdm import tqdm

def scarica_file(url, destinazione):
    """Scarica un file mostrando una barra di avanzamento."""
    response = requests.get(url, stream=True)
    response.raise_for_status() 
    
    dimensione_totale = int(response.headers.get('content-length', 0))
    
    with open(destinazione, 'wb') as file, tqdm(
        desc=destinazione,
        total=dimensione_totale,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=8192):
            dimensione = file.write(data)
            bar.update(dimensione)

def estrai_zip(percorso_zip, cartella_destinazione):
    """Estrae il contenuto di un file zip."""
    print(f"Estrazione di '{percorso_zip}' in '{cartella_destinazione}' in corso...")
    os.makedirs(cartella_destinazione, exist_ok=True)
    
    with zipfile.ZipFile(percorso_zip, 'r') as zip_ref:
        zip_ref.extractall(cartella_destinazione)
    print("Estrazione completata con successo!\n")

if __name__ == "__main__":
    # Configurazione di argparse per leggere i parametri da terminale
    parser = argparse.ArgumentParser(description="Script per scaricare il dataset AwA2 o le sue features.")
    parser.add_argument(
        '--target',
        type=str,
        choices=['dataset', 'features', 'labels', 'all'],
        default='dataset',
        help="Cosa scaricare: 'dataset' (immagini, 13.5GB), 'features' (solo features) o 'all' (entrambi)."
    )
    
    args = parser.parse_args()

    # Dizionario con gli URL e i nomi dei file di destinazione
    downloads = {
        'dataset': {
            'url': "https://cvml.ista.ac.at/AwA2/AwA2-data.zip",
            'file': "AwA2-data.zip",
            'folder': "AwA2_Dataset"
        },
        'features': {
            'url': "https://cvml.ista.ac.at/AwA2/AwA2-features.zip",
            'file': "AwA2-features.zip",
            'folder': "AwA2_Dataset_Features"
        },
        'labels': {
            'url': "https://cvml.ista.ac.at/AwA2/AwA2-base.zip",
            'file': "AwA2-labels.zip",
            'folder': "AwA2_Dataset_Labels"
        }
    }

    # Definiamo la lista degli elementi da scaricare in base alla scelta dell'utente
    targets_to_process = ['dataset', 'features', 'labels'] if args.target == 'all' else [args.target]

    for t in targets_to_process:
        info = downloads[t]
        
        print(f"{'-'*50}")
        print(f"Inizio elaborazione per: {t.upper()}")
        print(f"{'-'*50}")
        
        # Fase di download
        if not os.path.exists(info['file']):
            try:
                scarica_file(info['url'], info['file'])
            except Exception as e:
                print(f"Errore durante il download di {t}: {e}")
                continue # Se fallisce, salta all'elemento successivo (utile se target='all')
        else:
            print(f"Il file '{info['file']}' esiste già. Salto il download.")
            
        # Fase di estrazione
        estrai_zip(info['file'], info['folder'])

    print("Procedura conclusa!")