import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import wordnet as wn
from networkx.drawing.nx_pydot import graphviz_layout

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
    'domestic'
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
        'bipedal': 'leg',
        'quadrapedal': 'leg',
        'arm': 'upper_body_part',
        'flys': 'wings',
        'hops': 'leg',
        'walks': 'leg',
        'leg': 'lower_body_part',
        'flippers': 'swim',
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
                
    return G, list(new_parents_set)


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

def main():
    parser = argparse.ArgumentParser(description="Proietta concetti AwA2 su un Knowledge Graph.")
    parser.add_argument('--concepts', type=str, required=True, help="File txt con i concetti AwA2 (es. predicates.txt)")
    parser.add_argument('--matrix', type=str, required=True, help="File della matrice binaria (es. predicate-matrix-binary.txt)")
    args = parser.parse_args()

    # 1. Carica i dati
    print("Caricamento dati...")
    original_concepts = load_awa2_concepts(args.concepts)
    original_concepts = [c for c in original_concepts if c not in NOISY_CONCEPTS] # Filtriamo i concetti rumorosi prima di costruire la gerarchia
    original_matrix = load_awa2_matrix(args.matrix)
    original_matrix = original_matrix[:, :len(original_concepts)]  # Assicuriamoci che la matrice corrisponda ai concetti filtrati
    print(f"Trovati {len(original_concepts)} concetti. Dimensione matrice: {original_matrix.shape}")

    # 2. Costruisci Gerarchia
    print(f"Costruzione gerarchia usando wordnet...")
    G, new_parents = build_wordnet_hierarchy(original_concepts)

    
    print(f"Trovati {len(new_parents)} nuovi concetti padre.")

    # 3. Aggiorna Matrice
    print("Propagazione delle appartenenze di classe (Bottom-Up)...")
    new_matrix, all_concepts = update_incidence_matrix(original_matrix, original_concepts, new_parents, G)
    
    # Salva la nuova matrice
    np.savetxt("AwA2_Dataset_Labels/Animals_with_Attributes2/extended_matrix.txt", new_matrix, fmt='%d')
    with open("AwA2_Dataset_Labels/Animals_with_Attributes2/extended_concepts.txt", "w") as f:
        for i, c in enumerate(all_concepts):
            f.write(f"{i+1} {c}\n")
    print("Nuova matrice e lista concetti salvate ('extended_matrix.txt', 'extended_concepts.txt').")

    # 4. Visualizza
    print("Generazione PDF dell'albero gerarchico...")
    visualize_tree(G, original_concepts)

if __name__ == "__main__":
    main()