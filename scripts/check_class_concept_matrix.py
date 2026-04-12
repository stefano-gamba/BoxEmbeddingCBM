import re
from collections import defaultdict

def parse_concepts(filepath):
    """Legge il file dei concetti e restituisce una lista ordinata."""
    concepts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # Rimuove eventuali tag come 
            line = re.sub(r'\\', '', line).strip()
            if not line:
                continue
            parts = line.split()
            # Assumiamo che il primo elemento sia l'ID e il resto il nome del concetto
            if len(parts) >= 2:
                concepts.append(parts[1])
    return concepts

def parse_classes(filepath):
    """Legge il file delle classi e restituisce una lista ordinata."""
    classes = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # Rimuove eventuali tag
            line = re.sub(r'\\', '', line).strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                # Sostituisce il '+' con uno spazio (es. grizzly+bear -> grizzly bear)
                class_name = parts[1].replace('+', ' ')
                classes.append(class_name)
    return classes

def parse_matrix(filepath, num_cols):
    """Legge la matrice e la divide in righe basate sul numero di concetti."""
    matrix = []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        # Estrae solo i caratteri '0' e '1' ignorando spazi, newline e tag
        tokens = re.findall(r'\b[01]\b', content)
        
        # Ricostruisce la matrice riga per riga
        for i in range(0, len(tokens), num_cols):
            row = tokens[i:i+num_cols]
            if len(row) == num_cols:
                matrix.append(row)
    return matrix

def verifica_univocita(classes, matrix):
    """
    Analizza la matrice per trovare firme identiche tra classi diverse.
    """
    # Usiamo un dizionario dove:
    # Key = la stringa della firma (es. "01001...")
    # Value = lista di nomi delle classi che hanno quella firma
    signature_map = defaultdict(list)
    
    for i, cls in enumerate(classes):
        if i < len(matrix):
            # Trasformiamo la riga (lista di '0'/'1') in una stringa unica
            signature_str = "".join(matrix[i])
            signature_map[signature_str].append(cls)
    
    # Filtriamo solo le firme che hanno più di una classe associata
    duplicates = {sig: names for sig, names in signature_map.items() if len(names) > 1}
    
    return signature_map, duplicates

def main():
    # Nomi dei file (assicurati che siano nella stessa cartella dello script)
    concepts_file = '../AwA2_Dataset_Labels/Animals_with_Attributes2/extended_concepts.txt'
    classes_file = '../AwA2_Dataset_Labels/Animals_with_Attributes2/classes.txt'
    matrix_file = '../AwA2_Dataset_Labels/Animals_with_Attributes2/extended_matrix.txt'
    
    # 1. Carica i concetti e le classi
    concepts = parse_concepts(concepts_file)
    classes = parse_classes(classes_file)
    
    # 2. Carica la matrice sapendo che le colonne corrispondono al numero totale dei concetti
    num_concepts = len(concepts)
    matrix = parse_matrix(matrix_file, num_concepts)

    sig_map, duplicates = verifica_univocita(classes, matrix)
    
    # 3. Mappa e stampa i risultati
    print("=== MAPPATURA CLASSI E CONCETTI ===\n")
    for i, cls in enumerate(classes):
        if i < len(matrix):
            active_concepts = []
            for j, val in enumerate(matrix[i]):
                if val == '1':
                    active_concepts.append(concepts[j])
            
            print(f"Classe [{i+1}]: {cls.capitalize()}")
            print(f"Concetti: {', '.join(active_concepts)}")
            print("-" * 50)
        else:
            print(f"Attenzione: Nessuna riga nella matrice trovata per la classe {cls}")
    

    print("=== REPORT UNIVOCITÀ DELLE FIRME ===\n")
    print(f"Totale Classi analizzate: {len(classes)}")
    print(f"Totale Firme Uniche trovate: {len(sig_map)}")
    print("-" * 50)
    
    if not duplicates:
        print("✅ OTTIMO: Tutte le firme sono univoche. Ogni classe ha un set di attributi unico.")
    else:
        print(f"⚠️ ATTENZIONE: Trovate {len(duplicates)} collisioni!")
        print("Le seguenti classi hanno la STESSA firma concettuale e sono indistinguibili per il CBM:")
        for sig, names in duplicates.items():
            # Contiamo quanti attributi '1' ha questa firma duplicata
            num_active = sig.count('1')
            print(f"  • {', '.join(names)} (Attributi attivi: {num_active})")
    
    print("\n" + "="*50 + "\n")

if __name__ == '__main__':
    main()