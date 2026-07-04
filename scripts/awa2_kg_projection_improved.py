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
from itertools import combinations
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from src.utils.dataset import classical_split_awa2_features

# ==========================================
# 1. PRUNING E SNELLIMENTO (Merge Concepts)
# ==========================================
# Invece di scartare solo il rumore, uniamo concetti simili per ridurre la dimensionalità 
# (Il Dynamic Box adora la bassa dimensionalità).
MERGE_MAPPING = {
    # Denti
    'strainteeth': 'teeth', 'meatteeth': 'teeth', 'chewteeth': 'teeth', 'buckteeth': 'teeth',
    # Piedi/Zampe
    'paws': 'foot', 'hooves': 'foot', 'claws': 'foot', 'pads': 'foot', 'talons': 'foot',
    # Pattern e colori complessi
    'patches': 'pattern', 'spots': 'pattern', 'stripes': 'pattern',
    # Dimensioni
    'big': 'size_large', 'bulbous': 'size_large', 'small': 'size_small', 'lean': 'size_small',
    # Locomozione unita
    'bipedal': 'walks', 'quadrapedal': 'walks', 'hops': 'walks'
}

NOISY_CONCEPTS = {
    'smelly', 'fast', 'slow', 'strong', 'weak', 'muscle', 'agility',
    'active', 'inactive', 'nocturnal', 'hibernate', 'fierce', 'timid', 
    'smart', 'group', 'solitary', 'tunnels', 'domestic', 'swims',
    'fish', 'meat', 'plankton', 'vegetation', 'insects', 
    'forager', 'grazer', 'hunter', 'scavenger', 'skimmer', 'stalker',
    'newworld', 'oldworld', 'arctic', 'coastal', 'desert', 'bush', 
    'plains', 'forest', 'fields', 'jungle', 'mountains', 'ocean', 
    'ground', 'water', 'tree', 'cave', 'nestspot'
}

# ==========================================
# 2. DEFINIZIONE DELLE GERARCHIE (Ablazione)
# ==========================================
# Regole per forzare l'albero. Sceglieremo quali usare in base al tipo di esperimento.
CUSTOM_HIERARCHY_VISUAL = {
    'teeth': 'mouth', 'mouth': 'head', 'horns': 'head', 'tusks': 'mouth',
    'head': 'body', 'foot': 'leg', 'leg': 'body', 'arm': 'body', 'tail': 'body',
    'flys': 'wings', 'wings': 'body', 'hairless': 'skin', 'furry': 'skin', 'toughskin': 'skin'
}

# La tassonomia ontologica viene in gran parte risolta da WordNet, 
# ma qui forziamo le radici principali se WordNet fallisce.
CUSTOM_HIERARCHY_ONTOLOGICAL = {
    'mammal': 'animal', 'bird': 'animal', 'fish': 'animal', 'reptile': 'animal', 'amphibian': 'animal'
}

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
        'water_sport', 'sport', 'locomotion', 'motion', 'nestspot', 'abstinence', 'agent',
        'binary_compound', 'body_of_water', 'carriage', 'change_magnitude', 'chemical_agent',
        'climatic_zone', 'collection', 'comedian', 'compound', 'confinement', 'decrease',
        'food', 'geographical_area', 'geological_formation', 'literate', 'living_thing',
        'location', 'natural_elevation', 'organ', 'organism', 'pain', 'passage',
        'passageway', 'pedestrian', 'people', 'performer', 'reader', 'rest', 'self-denial',
        'servant', 'skilled_worker', 'sleep', 'solid', 'subjugation', 'thing', 'tract', 'traveler',
        'woody_plant'
}


def load_and_merge_awa2_concepts(filepath, matrix_filepath):
    """Carica i concetti, li snellisce (merge) e restituisce la matrice aggiornata."""
    original_concepts = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                original_concepts.append(parts[1])
                
    original_matrix = np.loadtxt(matrix_filepath, dtype=int)
    num_classes = original_matrix.shape[0]
    
    # Rimuoviamo i rumorosi
    valid_idx = [i for i, c in enumerate(original_concepts) if c not in NOISY_CONCEPTS]
    valid_concepts = [original_concepts[i] for i in valid_idx]
    valid_matrix = original_matrix[:, valid_idx]
    
    # Applichiamo il Merge
    merged_concepts_set = set()
    for c in valid_concepts:
        merged_concepts_set.add(MERGE_MAPPING.get(c, c))
        
    merged_concepts = sorted(list(merged_concepts_set))
    merged_matrix = np.zeros((num_classes, len(merged_concepts)), dtype=int)
    
    for i, old_c in enumerate(valid_concepts):
        new_c = MERGE_MAPPING.get(old_c, old_c)
        new_idx = merged_concepts.index(new_c)
        # Logical OR per unire le colonne (es. se ha 'paws' o 'claws' ha 'foot')
        merged_matrix[:, new_idx] = np.logical_or(merged_matrix[:, new_idx], valid_matrix[:, i]).astype(int)
        
    return merged_concepts, merged_matrix


def build_specific_hierarchy(concepts, mode):
    """
    Costruisce l'albero. 
    mode può essere: 'visual', 'ontological', 'hybrid'
    """
    nltk.download('wordnet', quiet=True)
    G = nx.DiGraph()
    root_node = "Animal" if mode != 'visual' else "Body" # Radice diversa per visivo
    G.add_node(root_node)
    
    new_parents_set = set()
    mapping_to_use = {}
    
    if mode == 'visual':
        mapping_to_use = CUSTOM_HIERARCHY_VISUAL
    elif mode == 'ontological':
        mapping_to_use = CUSTOM_HIERARCHY_ONTOLOGICAL
    else: # hybrid
        mapping_to_use = {**CUSTOM_HIERARCHY_VISUAL, **CUSTOM_HIERARCHY_ONTOLOGICAL}
        
    all_starting_nodes = set(concepts).union(mapping_to_use.keys())

    for concept in all_starting_nodes:
        if concept not in concepts: new_parents_set.add(concept)
        if not G.has_node(concept): G.add_node(concept)
        
        # 1. Regole Custom
        if concept in mapping_to_use:
            curr = concept
            visited = set()
            while curr in mapping_to_use and curr not in visited:
                visited.add(curr)
                parent = mapping_to_use[curr]
                G.add_edge(parent, curr)
                if parent not in concepts: new_parents_set.add(parent)
                curr = parent
            if not G.has_edge(root_node, curr) and curr != root_node:
                G.add_edge(root_node, curr)
            continue
            
        # 2. WordNet (Solo per Ontological e Hybrid)
        if mode in ['ontological', 'hybrid']:
            synsets = wn.synsets(concept.replace('+', '_').replace('-', '_'))
            if synsets:
                path = synsets[0].hypernym_paths()[0]
                path_rev = list(reversed(path))
                prev = concept
                for i in range(1, len(path_rev)):
                    parent = path_rev[i].name().split('.')[0]
                    # Filtro base per evitare rumore di wordnet
                    if parent in NOISY_PARENTS: break
                    G.add_edge(parent, prev)
                    if parent not in concepts: new_parents_set.add(parent)
                    prev = parent
                G.add_edge(root_node, prev)
            else:
                G.add_edge(root_node, concept)
        else:
            # Se siamo in visual e non c'è regola, attacchiamo alla radice 'Body'
            G.add_edge(root_node, concept)
            
    return G, sorted(list(new_parents_set))


# ==========================================
# 3. STRATEGIE DI DISGIUNZIONE (Mutual Exclusivity)
# ==========================================
def build_supervisions_data_driven(G, full_matrix, all_concepts, strategy='data_driven'):
    """
    Genera le regole. 
    'strict': i figli dello stesso nodo sono sempre disgiunti (sbagliato per attributi visivi).
    'data_driven': due figli sono disgiunti SOLO SE nessuna classe nella matrice li ha entrambi a 1.
    """
    supervisions = []
    concept_to_idx = {c: i for i, c in enumerate(all_concepts)}
    
    # 1. Inclusione (1.0)
    for parent in G.nodes():
        for child in nx.descendants(G, parent):
            supervisions.append((parent, child, 1.0))
            
    # 2. Disgiunzione (0.0)
    for parent in G.nodes():
        children = list(G.successors(parent))
        if len(children) > 1:
            for c1, c2 in combinations(children, 2):
                if strategy == 'strict':
                    # Logica ingenua
                    supervisions.append((c1, c2, 0.0))
                    supervisions.append((c2, c1, 0.0))
                elif strategy == 'data_driven':
                    # Logica sicura basata sui dati
                    idx1 = concept_to_idx.get(c1)
                    idx2 = concept_to_idx.get(c2)
                    if idx1 is not None and idx2 is not None:
                        # Controlliamo se esiste almeno una riga in cui entrambi sono 1
                        co_occur = np.sum((full_matrix[:, idx1] == 1) & (full_matrix[:, idx2] == 1))
                        if co_occur == 0:
                            # Sono disgiunti solo se non co-occorrono mai
                            supervisions.append((c1, c2, 0.0))
                            supervisions.append((c2, c1, 0.0))
                            
    return sorted(list(set(supervisions)))


def update_incidence_matrix(base_matrix, base_concepts, new_parents, G):
    """(Invariato dalla tua versione, fa l'OR logico bottom-up)"""
    num_classes = base_matrix.shape[0]
    num_base = len(base_concepts)
    num_new = len(new_parents)
    
    new_matrix = np.zeros((num_classes, num_base + num_new), dtype=int)
    new_matrix[:, :num_base] = base_matrix
    
    all_concepts = base_concepts + new_parents
    concept_to_idx = {c: idx for idx, c in enumerate(all_concepts)}
    
    try:
        nodes_reversed = reversed(list(nx.topological_sort(G)))
    except nx.NetworkXUnfeasible:
        nodes_reversed = all_concepts

    for node in nodes_reversed:
        if node in concept_to_idx:
            children = list(G.successors(node))
            child_indices = [concept_to_idx[c] for c in children if c in concept_to_idx]
            if child_indices:
                parent_idx = concept_to_idx[node]
                children_cols = new_matrix[:, child_indices]
                parent_col = np.any(children_cols == 1, axis=1).astype(int)
                new_matrix[:, parent_idx] = np.logical_or(new_matrix[:, parent_idx], parent_col).astype(int)

    return new_matrix, all_concepts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--concepts', type=str, required=True)
    parser.add_argument('--matrix', type=str, required=True)
    # Aggiunti argomenti per controllare gli esperimenti
    parser.add_argument('--mode', type=str, choices=['visual', 'ontological', 'hybrid'], default='hybrid')
    parser.add_argument('--disjunction', type=str, choices=['strict', 'data_driven'], default='data_driven')
    args = parser.parse_args()

    print(f"--- AVVIO GENERAZIONE DATASET ---")
    print(f"Modalità Gerarchia: {args.mode.upper()}")
    print(f"Strategia Disgiunzione: {args.disjunction.upper()}")

    # 1. Carica e fa il Merge (Pruning)
    base_concepts, base_matrix = load_and_merge_awa2_concepts(args.concepts, args.matrix)
    print(f"Concetti base ridotti a: {len(base_concepts)} tramite pruning.")

    # 2. Costruisce l'albero specifico
    G, new_parents = build_specific_hierarchy(base_concepts, args.mode)
    print(f"Trovati {len(new_parents)} nuovi concetti padre.")

    # 3. Aggiorna Matrice
    full_matrix, all_concepts = update_incidence_matrix(base_matrix, base_concepts, new_parents, G)

    # 4. Genera Supervisioni usando la strategia dati
    hierarchy_supervision = build_supervisions_data_driven(G, full_matrix, all_concepts, args.disjunction)

    # ... [RESTO DEL TUO CODICE PER SALVATAGGIO JSON E PDF INVARIATO] ...
    # Assicurati di cambiare dinamicamente i nomi dei file salvati in base a args.mode!
    
    # Esempio:
    prefix = f"AwA2_{args.mode}_{args.disjunction}"
    # np.savetxt(f"{prefix}_extended_matrix.txt", full_matrix, fmt='%d')
    # ... eccetera ...
    
    print(f"Completato. Usa questo dataset per misurare l'ablazione del Concept Predictor.")

if __name__ == "__main__":
    main()