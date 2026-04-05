import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet as wn
from networkx.drawing.nx_pydot import graphviz_layout
import seaborn as sns
import pandas as pd
import json

import os, sys
# assicurati che la cartella del progetto sia nella ricerca dei moduli
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from utils.dataset import classical_split_awa2_features

NOISY_CONCEPTS = {
    # 1. Sensi, fisicità astratta e cinematica (Non deducibili in modo affidabile da una foto statica)
    'smelly', 'fast', 'slow', 'strong', 'weak', 'muscle', 'agility',
    
    # 2. Comportamento, temperamento e abitudini sociali
    'active', 'inactive', 'nocturnal', 'hibernate', 'fierce', 'timid', 
    'smart', 'group', 'solitary', 'tunnels',
    
    # 3. Dieta e ruoli alimentari (Causerebbero allucinazioni nell'apprendimento)
    'fish', 'meat', 'plankton', 'vegetation', 'insects', 
    'forager', 'grazer', 'hunter', 'scavenger', 'skimmer', 'stalker',
    
    # 4. Habitat, geografia e background (Forzerebbero la rete a ignorare l'animale per guardare lo sfondo)
    'newworld', 'oldworld', 'arctic', 'coastal', 'desert', 'bush', 
    'plains', 'forest', 'fields', 'jungle', 'mountains', 'ocean', 
    'ground', 'water', 'tree', 'cave', 'nestspot',
    
    # 5. Concetti relazionali/umani
    'domestic',

    'swims'
}

def load_awa2_concepts(filepath):
    """Carica i concetti di AwA2 dal file txt (formato: id nome)."""
    concepts = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                # Uniamo in caso di concetti composti, es '1 polar+bear'
                concepts.append(parts[1])
    return concepts

def load_awa2_matrix(filepath):
    """Carica la matrice binaria di incidenza (Classi x Concetti)."""
    return np.loadtxt(filepath, dtype=int)


def build_wordnet_hierarchy(concepts):
    """Costruisce un grafo gerarchico supportando catene custom multi-livello."""
    nltk.download('wordnet', quiet=True)
    G = nx.DiGraph()
    root_node = "Animal"
    G.add_node(root_node)

    new_parents_set = set()
    
    # 1. REGOLE CUSTOM ('a' : 'b' significa 'a' implica 'b', quindi 'a' è figlio di 'b' nell'albero)
    CUSTOM_MAPPING = {
        'paws': 'foot',
        'hooves': 'foot',
        'claws': 'foot',
        'strainteeth': 'tooth',
        'meatteeth': 'tooth',
        'chewteeth': 'tooth',
        'buckteeth': 'tooth',
        'pads': 'foot',
        'tooth': 'mouth', 
        'foot': 'leg',
        'horns': 'head',
        'mouth': 'head',
        'head': 'upper_body_part',
        'hands': 'arm',
        'tail': 'lower_body_part',
        'patches': 'skin_pattern',
        'spots': 'skin_pattern',
        'stripes': 'skin_pattern',
        'longleg': 'leg',
        'longneck': 'upper_body_part',
        'tail': 'lower_body_part',
        'tusks': 'mouth',
        'bipedal': 'foot',
        'quadrapedal': 'foot',
        'arm': 'upper_body_part',
        'flys': 'wings',
        'hops': 'foot',
        'walks': 'foot',
        'leg': 'lower_body_part',
        'flippers': 'swims',
    }

    # 2. Blacklist dei nodi rumorosi di WordNet
    NOISY_PARENTS = {
        'entity', 'abstraction', 'attribute', 'property', 'physical_entity',
        'object', 'whole', 'matter', 'measure', 'fundamental_quantity',
        'physical_property', 'visual_property', 'chromatic_color', 'color',
        'state', 'condition', 'shape', 'psychological_feature', 'event',
        'part', 'group', 'relation', 'substance', 'artifact', 'structure',
        'quality', 'concept', 'category', 'citrus', 'edible_fruit', 'achromatic_color',
        'process', 'shoe', 'footwear', 'animal_foot', 'vertebrate_foot', 'front_tooth',
        'noise_maker', 'device', 'guardianship', 'duty', 'body_part', 'person', 'causal_agent',
        'marking', 'design', 'symptom', 'evidence', 'badge', 'emblem', 'position',
        'animal_material', 'dentine', 'dipterous_insect', 'insect', 'vine', 'vascular_plant',
        'water_sport', 'sport', 'locomotion', 'motion',
    }

    for concept in concepts:
        G.add_node(concept)
        
        # --- FASE 1: CONTROLLO OVERRIDE MANUALE (Con supporto Catene) ---
        if concept in CUSTOM_MAPPING:
            current_node = concept
            visited_custom = set() # Per evitare loop infiniti se sbagli a scrivere regole (es. a->b, b->a)
            
            # Risaliamo la catena del dizionario custom finché ci sono regole
            while current_node in CUSTOM_MAPPING and current_node not in visited_custom:
                visited_custom.add(current_node)
                parent_name = CUSTOM_MAPPING[current_node]
                
                G.add_edge(parent_name, current_node)
                new_parents_set.add(parent_name)
                
                current_node = parent_name
                
            # Finito di risalire le tue regole, attacchiamo la cima della catena alla radice
            G.add_edge(root_node, current_node)
            continue # Passa al prossimo concetto originario

        # --- FASE 2: FALLBACK SU WORDNET ---
        synsets = wn.synsets(concept.replace('+', '_').replace('-', '_'))
        
        if not synsets:
            G.add_edge(root_node, concept)
            continue
            
        syn = synsets[0]
        paths = syn.hypernym_paths()
        
        if not paths or len(paths[0]) == 1:
            G.add_edge(root_node, concept)
        else:
            path = paths[0]
            path_reversed = list(reversed(path))
            previous_node = concept
            MAX_DEPTH = 2 
            current_depth = 0
            
            for i in range(1, len(path_reversed)):
                parent_name = path_reversed[i].name().split('.')[0]
                
                if parent_name in NOISY_PARENTS or current_depth >= MAX_DEPTH:
                    G.add_edge(root_node, previous_node)
                    break
                    
                G.add_edge(parent_name, previous_node)
                new_parents_set.add(parent_name)
                
                previous_node = parent_name
                current_depth += 1
            else:
                G.add_edge(root_node, previous_node)
                
    return G, sorted(list(new_parents_set))


def update_incidence_matrix(original_matrix, original_concepts, new_parents, G):
    """Espande la matrice aggiornando l'appartenenza dei genitori rispetto ai figli."""
    num_classes = original_matrix.shape[0]
    num_original = len(original_concepts)
    num_new = len(new_parents)
    
    # Inizializza la nuova matrice (Classi x (Concetti_Originali + Nuovi_Genitori))
    new_matrix = np.zeros((num_classes, num_original + num_new), dtype=int)
    new_matrix[:, :num_original] = original_matrix
    
    # Mappatura per trovare rapidamente gli indici
    all_concepts = original_concepts + new_parents
    concept_to_idx = {concept: idx for idx, concept in enumerate(all_concepts)}
    
    # Ordine topologico dal basso (foglie) verso l'alto (radice) per propagare i valori
    try:
        nodes_ordered = list(nx.topological_sort(G))
        nodes_reversed = reversed(nodes_ordered) # Partiamo dalle foglie
    except nx.NetworkXUnfeasible:
        print("Attenzione: Il grafo contiene cicli. Eseguo fallback senza propagazione profonda.")
        nodes_reversed = all_concepts

    for node in nodes_reversed:
        if node in concept_to_idx:
            # Trova tutti i figli di questo nodo
            children = list(G.successors(node))
            child_indices = [concept_to_idx[c] for c in children if c in concept_to_idx]
            
            if child_indices:
                # Se una classe ha almeno un figlio (1), allora ha anche il padre (Logical OR)
                parent_idx = concept_to_idx[node]
                # Eseguiamo un OR logico (bit a bit) tra le colonne dei figli
                children_columns = new_matrix[:, child_indices]
                parent_column = np.any(children_columns == 1, axis=1).astype(int)
                
                # Se il nodo aveva già dei valori, li uniamo (OR) con quelli derivati dai figli
                new_matrix[:, parent_idx] = np.logical_or(new_matrix[:, parent_idx], parent_column).astype(int)

    return new_matrix, all_concepts


def compute_conditional_probabilities(G):
    """
    Calcola P(x|y) per ogni coppia di nodi basata sulla topologia dell'albero.
    Implementa le formule euristiche:
    1. P(n) = |descendants(n)| / |nodes|
    2. P(x,y) = |leaves where x,y co-occur| / |leaves|
    3. P(x|y) = P(x,y) / P(y)
    """
    print("Inizio calcolo probabilità condizionate...")
    
    # 1. Identifica Foglie e Nodi
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    
    # Le foglie sono i nodi senza archi in uscita (out_degree == 0)
    leaves = [n for n in nodes if G.out_degree(n) == 0]
    num_leaves = len(leaves)
    
    if num_leaves == 0:
        raise ValueError("Il grafo non ha foglie. Controlla la struttura di G.")

    # 2. Calcolo Probabilità Marginale P(y) per ogni nodo
    # Usiamo il numero di discendenti (più il nodo stesso) diviso il totale dei nodi
    P_marginal = {}
    for n in nodes:
        # nx.descendants trova tutti i nodi raggiungibili "sotto" n
        descendants = nx.descendants(G, n)
        # +1 perché il nodo "n" conta come discendente di se stesso nel calcolo volumetrico
        P_marginal[n] = (len(descendants) + 1) / num_nodes
        
    # 3. Calcolo Probabilità Congiunta P(x,y)
    # Contiamo in quante foglie la coppia (x, y) compare insieme come antenati
    co_occur_counts = {x: {y: 0 for y in nodes} for x in nodes}
    
    for leaf in leaves:
        # Troviamo il percorso genealogico della foglia (antenati + foglia stessa)
        ancestors_set = nx.ancestors(G, leaf).union({leaf})
        
        # Per ogni possibile coppia nel percorso, incrementiamo il contatore
        for x in ancestors_set:
            for y in ancestors_set:
                co_occur_counts[x][y] += 1
                
    # 4. Popoliamo il DataFrame della Matrice di Probabilità Condizionata P(x|y)
    df_cond = pd.DataFrame(0.0, index=nodes, columns=nodes)
    
    for x in nodes:
        for y in nodes:
            # Formula screenshot: aggregato diviso numero di foglie
            p_xy = co_occur_counts[x][y] / num_leaves
            
            # Formula di Bayes: P(x|y) = P(x,y) / P(y)
            # Evitiamo divisioni per zero
            if P_marginal[y] > 0:
                p_x_given_y = p_xy / P_marginal[y]
            else:
                p_x_given_y = 0.0
                
            # Dato che l'autore usa due denominatori diversi (|nodes| e |leaves|), 
            # tronchiamo a 1.0 eventuali anomalie numeriche dell'euristica.
            df_cond.at[x, y] = min(p_x_given_y, 1.0)

    return df_cond

import numpy as np
import pandas as pd

def compute_empirical_matrix(train_incidence, train_labels, concept_names):
    """
    Calcola la matrice delle probabilità condizionate empiriche V_ij dai dati di training.
    
    Args:
        train_incidence: np.array di shape (num_train_classes, num_concepts).
                         Contiene 1 o 0.
        train_labels: np.array o lista di shape (num_train_images,).
                      Contiene l'ID della classe (da 0 a num_train_classes-1) per ogni immagine.
        concept_names: lista di stringhe con i nomi dei concetti nell'ordine corretto.
    """

    # shiftamo train_labels da 1-based a 0-based se necessario
    if np.min(train_labels) == 1:
        train_labels = train_labels - 1

    # 1. Espandiamo la matrice di classe in una matrice a livello di singola immagine
    # Shape risultante: (num_train_images, num_concepts)
    image_concept_matrix = train_incidence[train_labels]
    
    # 2. Calcoliamo le co-occorrenze (Numeratore)
    # Moltiplicando la matrice trasposta per se stessa otteniamo una matrice (num_concepts x num_concepts)
    # dove la cella (i, j) è esattamente il "Numero di immagini con c_i=1 e c_j=1"
    co_occurrences = image_concept_matrix.T @ image_concept_matrix
    
    # 3. Calcoliamo le frequenze marginali (Denominatore)
    # Il "Numero di immagini con c_j=1" corrisponde alla diagonale della matrice di co-occorrenza
    freq_j = np.diag(co_occurrences)
    
    # 4. Calcoliamo V_ij applicando la formula
    # V_ij = co_occurrences[i, j] / freq_j[j]
    # Usiamo np.divide per gestire in sicurezza eventuali divisioni per zero (se un concetto non compare mai)
    V_ij = np.divide(
        co_occurrences, 
        freq_j, 
        out=np.zeros_like(co_occurrences, dtype=float), 
        where=(freq_j != 0)
    )
    
    # Restituiamo il DataFrame etichettato
    return pd.DataFrame(V_ij, index=concept_names, columns=concept_names)

def fuse_probability_matrices(df_graph, df_data):
    """
    Unisce le certezze del grafo (Knowledge-Driven) con le statistiche delle immagini (Data-Driven)
    usando l'operazione di Massimo.
    """
    print("Fusione delle matrici in corso (Max-Pooling semantico)...")
    # np.maximum confronta i DataFrame cella per cella e tiene il valore più alto
    df_fused = np.maximum(df_graph, df_data)
    return df_fused



def visualize_tree(G, original_concepts,output_file="gerarchia_concetti.pdf"):
    """Disegna il grafo e lo salva in PDF."""
    plt.figure(figsize=(60, 32))

    # 1. Creiamo un set per una ricerca più efficiente
    original_set = set(original_concepts)
    
    pos = graphviz_layout(G, prog="dot") 
    
    # Nodi originali (foglie) in blu, nuovi nodi (genitori) in rosso
    out_degrees = dict(G.out_degree())
    node_colors = ['lightgreen' if n in original_set else 'lightcoral' for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600, alpha=0.8)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, edge_color='gray', alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
    
    plt.title("Gerarchia dei Concetti Estesa", fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    print(f"Albero gerarchico salvato in: {output_file}")

import networkx as nx
from itertools import combinations

def build_supervisions_from_graph(G):
    """
    Estrae le regole di supervisione (inclusione e disgiunzione) da un DAG di NetworkX.
    Assunto: arco (u, v) significa che u è genitore di v (v implica u).
    """
    supervisions = []
    
    # --- 1. Regole di INCLUSIONE (Probabilità = 1.0) ---
    # Usiamo nx.descendants per prendere sia i figli diretti che i nipoti.
    # Questo forza la rete a rispettare la gerarchia globale fin da subito.
    for parent in G.nodes():
        descendants = nx.descendants(G, parent)
        for child in descendants:
            # Formato: (Target/Contenitore, Source/Contenuto, Probabilità)
            supervisions.append((parent, child, 1.0))
            
    # --- 2. Regole di DISGIUNZIONE (Probabilità = 0.0) ---
    # Cerchiamo tutti i nodi che hanno lo stesso genitore (fratelli)
    # e imponiamo che i loro box non si sovrappongano.
    for parent in G.nodes():
        children = list(G.successors(parent))
        
        # Se ci sono almeno 2 figli, creiamo le coppie disgiunte simmetriche
        if len(children) > 1:
            for c1, c2 in combinations(children, 2):
                supervisions.append((c1, c2, 0.0))
                supervisions.append((c2, c1, 0.0)) # Regola simmetrica salva-vita!
                
    # Rimuoviamo eventuali duplicati e ordiniamo per pulizia
    supervisions = sorted(list(set(supervisions)))
    
    return supervisions


def numerical_supervision(textual_supervision, all_concepts):
    # Creiamo il dizionario direttamente dalla lista in memoria!
    concept_to_idx = {c.lower(): i for i, c in enumerate(all_concepts)}
    
    supervisioni_numeriche = []
    for target_str, source_str, prob in textual_supervision:
        t_clean = target_str.lower()
        s_clean = source_str.lower()
        
        if t_clean in concept_to_idx and s_clean in concept_to_idx:
            target_idx = concept_to_idx[t_clean]
            source_idx = concept_to_idx[s_clean]
            supervisioni_numeriche.append((target_idx, source_idx, float(prob)))
        else:
            print(f"⚠️ Attenzione: '{target_str}' o '{source_str}' non trovato!")

    import json
    with open("supervisioni_gerarchia_numeriche.json", "w") as f:
        json.dump(supervisioni_numeriche, f, indent=4)
        
    print("Lista numerica salvata in supervisioni_gerarchia_numeriche.json")


def main():
    parser = argparse.ArgumentParser(description="Proietta concetti AwA2 su un Knowledge Graph.")
    parser.add_argument('--concepts', type=str, required=True, help="File txt con i concetti AwA2 (es. predicates.txt)")
    parser.add_argument('--matrix', type=str, required=True, help="File della matrice binaria (es. predicate-matrix-binary.txt)")
    parser.add_argument('--labels', type=str, required=True, help="File con le etichette di training (es. train_labels.txt)")
    args = parser.parse_args()

    # 1. Carica i dati
    print("Caricamento dati...")
    original_concepts = load_awa2_concepts(args.concepts)
    relevant_concepts = [c for c in original_concepts if c not in NOISY_CONCEPTS] # Filtriamo i concetti rumorosi prima di costruire la gerarchia
    original_matrix = load_awa2_matrix(args.matrix)

    # filtriamo la matrice per tenere solo i concetti rilevanti (escludendo quelli rumorosi)
    relevant_concepts_mask = np.isin(original_concepts, relevant_concepts)
    relevant_concepts_matrix = original_matrix[:, relevant_concepts_mask] 

    features_path = 'AwA2_Dataset_Features/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt'
    labels_path = 'AwA2_Dataset_Features/Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt'
    
    (_, train_labels), _, _ = classical_split_awa2_features(
        features_path,  # Non ci servono le feature per questo script
        labels_path,    # Non ci servono le label complete, solo i train_labels
        test_size=0.2,
        val_size=0.1,
        random_seed=42
    )

    print(f"Trovati {len(relevant_concepts)} concetti. Dimensione matrice: {relevant_concepts_matrix.shape}")

    # 2. Costruisci Gerarchia
    print(f"Costruzione gerarchia usando wordnet...")
    G, new_parents = build_wordnet_hierarchy(relevant_concepts)

    hierarchy_supervision = build_supervisions_from_graph(G)

    nome_file_json = "supervisioni_gerarchia.json"

    print(f"Salvato con successo in {nome_file_json}")
    

    new_parents.append("Animal") # Assicuriamoci che la radice sia inclusa nei concetti finali

    print(f"Trovati {len(new_parents)} nuovi concetti padre.")

    # 3. Aggiorna Matrice
    print("Propagazione delle appartenenze di classe (Bottom-Up)...")
    new_matrix, all_concepts = update_incidence_matrix(relevant_concepts_matrix, relevant_concepts, new_parents, G)

    numerical_supervision(hierarchy_supervision, all_concepts)

    # --- 1. SALVATAGGIO IN JSON ---
    with open(nome_file_json, 'w') as f:
        # json.dump prende la tua lista e la scrive direttamente nel file
        json.dump(hierarchy_supervision, f, indent=4) # indent=4 lo rende leggibile e "a capo"
    
    # Salva la nuova matrice
    np.savetxt("AwA2_Dataset_Labels/Animals_with_Attributes2/extended_matrix.txt", new_matrix, fmt='%d')
    with open("AwA2_Dataset_Labels/Animals_with_Attributes2/extended_concepts.txt", "w") as f:
        for i, c in enumerate(all_concepts):
            f.write(f"{i+1} {c}\n")
    print("Nuova matrice e lista concetti salvate ('extended_matrix.txt', 'extended_concepts.txt').")

    # 4. Visualizza
    print("Generazione PDF dell'albero gerarchico...")
    visualize_tree(G, relevant_concepts)

    df_graph = compute_conditional_probabilities(G)

    df_data = compute_empirical_matrix(new_matrix, train_labels, all_concepts)

    df_final = fuse_probability_matrices(df_graph, df_data)

    # --- SALVATAGGIO IN TXT ---
    # Salviamo arrotondando a 4 decimali per leggibilità
    df_final.round(4).to_csv('AwA2_Dataset_Labels/Animals_with_Attributes2/V_gt.csv', sep='\t')
    print(f"Matrice salvata in testo in: AwA2_Dataset_Labels/Animals_with_Attributes2/V_gt.csv")

    # --- SALVATAGGIO IN PDF (Heatmap) ---
    plt.figure(figsize=(24, 20))
    # Usiamo seaborn per una mappa di calore molto leggibile
    ax = sns.heatmap(df_final, cmap="YlGnBu", annot=False, fmt=".2f", 
                     cbar_kws={'label': 'P( x | y )'}, vmin=0.0, vmax=1.0)
    
    plt.title("Matrice di Probabilità Condizionata P(x|y)", fontsize=24, pad=20)
    plt.ylabel("Concetto X (Dipendente)", fontsize=16)
    plt.xlabel("Concetto Y (Condizionante)", fontsize=16)
    
    # Ruotiamo le etichette per renderle leggibili se ci sono tanti nodi
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('prob_heatmap.pdf', format='pdf', dpi=300)
    plt.close()
    print(f"Heatmap PDF salvata in: prob_heatmap.pdf")

if __name__ == "__main__":
    main()