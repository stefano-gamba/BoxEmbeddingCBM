import numpy as np
from collections import Counter
import nltk
from nltk.corpus import wordnet as wn
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Scarica i dati di WordNet se non sono già presenti nel sistema
nltk.download('wordnet', quiet=True)

def clean_concept_name(name):
    """
    Pulisce i nomi dei predicati di AwA2 per renderli compatibili con WordNet.
    Es. 'polar+bear' -> 'polar_bear', 'has_claws' -> 'has_claws'
    """
    return name.lower().replace("+", "_").replace(" ", "_")

def get_wordnet_descendants(synset):
    """
    Ottiene tutti i discendenti (hyponyms) di un nodo WordNet.
    """
    return set(synset.closure(lambda s: s.hyponyms()))

def compute_combined_conditional_probabilities(predicates_file, matrix_file, labels_file):
    """
    Calcola V_ij combinando le regole di WordNet e le statistiche del dataset come fallback.
    """
    # 1. Caricamento dei file
    with open(predicates_file, 'r') as f:
        predicates = [line.strip().split()[-1] for line in f.readlines()]
        
    M = np.loadtxt(matrix_file) 
    num_classes, num_predicates = M.shape
    labels = np.loadtxt(labels_file, dtype=int)
    
    # 2. Pre-calcolo delle Statistiche del Dataset (Fallback)
    class_counts = np.zeros(num_classes)
    counts = Counter(labels)
    for class_id, count in counts.items():
        idx = class_id - 1 
        if 0 <= idx < num_classes:
            class_counts[idx] = count
            
    D = np.diag(class_counts)
    Co_occ = M.T @ D @ M
    count_j = np.diag(Co_occ)
    count_j_safe = np.maximum(count_j, 1e-10)
    V_data = Co_occ / count_j_safe[np.newaxis, :]
    
    # 3. Inizializzazione matrice finale V e variabili WordNet
    V_final = np.zeros((num_predicates, num_predicates))
    k = len(predicates) # Costante k usata nelle formule del paper per WordNet
    
    # Cerchiamo il synset più comune per ogni concetto
    concept_synsets = {}
    for i, concept in enumerate(predicates):
        cleaned = clean_concept_name(concept)
        synsets = wn.synsets(cleaned)
        # Prendiamo il primo synset se esiste, altrimenti None
        concept_synsets[i] = synsets[0] if synsets else None

    # 4. Calcolo della matrice combinata
    for i in range(num_predicates):
        for j in range(num_predicates):
            syn_i = concept_synsets[i]
            syn_j = concept_synsets[j]
            
            # Condizione 1: Entrambi i concetti sono in WordNet
            if syn_i is not None and syn_j is not None:
                descendants_i = get_wordnet_descendants(syn_i)
                descendants_j = get_wordnet_descendants(syn_j)
                
                # Calcolo P(c_j) tramite WordNet
                p_c_j_wn = len(descendants_j) / k
                
                # Calcolo P(c_i, c_j) trovando i discendenti in comune
                common_descendants = descendants_i.intersection(descendants_j)
                if syn_j in descendants_i:
                    common_descendants.add(syn_j)
                if syn_i in descendants_j:
                    common_descendants.add(syn_i)
                    
                p_c_i_c_j_wn = len(common_descendants) / k 
                
                # Se il concetto j ha validi discendenti nell'ontologia, applichiamo la formula
                if p_c_j_wn > 0:
                    V_final[i, j] = p_c_i_c_j_wn / p_c_j_wn
                else:
                    # Se P(c_j) è 0 (es. nodi foglia o aggettivi senza gerarchia), usiamo le statistiche
                    V_final[i, j] = V_data[i, j]
                    print(f"Attenzione: P({predicates[j]}) è 0 in WordNet. Usando statistiche per V[{i}, {j}].")
            else:
                # Condizione 2: Almeno un attributo non è presente in WordNet, usiamo le statistiche
                V_final[i, j] = V_data[i, j]
                print(f"Attenzione: '{predicates[i]}' o '{predicates[j]}' non trovato in WordNet. Usando statistiche per V[{i}, {j}].")
                
    return V_final, predicates

def plot_and_save_matrix(V_matrix, concept_names, output_pdf):
    """
    Crea una heatmap della matrice delle probabilità condizionate e la salva in PDF.
    """
    # Imposta una dimensione sufficientemente grande per 85 concetti
    plt.figure(figsize=(24, 20))
    
    # Crea la heatmap usando seaborn
    # vmin e vmax fissano la scala dei colori tra 0 (nessuna probabilità) e 1 (certezza)
    sns.heatmap(V_matrix, 
                xticklabels=concept_names, 
                yticklabels=concept_names,
                cmap='viridis', # Puoi usare anche 'Blues' o 'coolwarm'
                vmin=0.0, vmax=1.0)
    
    plt.title('Matrice delle Probabilità Condizionate $V_{ij} = \mathbb{P}(c_i | c_j)$', fontsize=20)
    plt.xlabel('Concetto Condizionante ($c_j$)', fontsize=16)
    plt.ylabel('Concetto Condizionato ($c_i$)', fontsize=16)
    
    # Ruota le etichette per evitare sovrapposizioni
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # Ottimizza gli spazi
    plt.tight_layout()
    
    # Salva il grafico come PDF
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    
    # Chiudi la figura per liberare memoria
    plt.close()
    print(f"Heatmap salvata con successo in '{output_pdf}'")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Utilizzo: python script.py <predicates_file> <matrix_file> <labels_file> <output_file>")
        sys.exit(1)
    
    PREDICATES_FILE = sys.argv[1]
    MATRIX_FILE = sys.argv[2]
    LABELS_FILE = sys.argv[3]
    OUTPUT_FILE = sys.argv[4]
    
    try:
        V_matrix, concept_names = compute_combined_conditional_probabilities(
            PREDICATES_FILE, MATRIX_FILE, LABELS_FILE
        )
        # Salva in formato testuale
        np.savetxt(OUTPUT_FILE, V_matrix, fmt='%.4f')
        print(f"Matrice completata! Salvata in '{OUTPUT_FILE}'.")
        
        # Nuova riga: Genera e salva il grafico PDF
        plot_and_save_matrix(V_matrix, concept_names, 'conditional_probabilities_heatmap.pdf')
        
    except FileNotFoundError as e:
        print(f"Errore: Impossibile trovare uno dei file di input. Dettagli: {e}")
    
    
    np.savetxt(OUTPUT_FILE, V_matrix, fmt='%.4f')
    print(f"Matrice completata! Salvata in '{OUTPUT_FILE}'.")