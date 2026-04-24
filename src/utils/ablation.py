import torch

def sabotage_concept_matrix(class_concept_matrix, class_names, concept_names, modifications):
    """
    Modifica una matrice classe-concetto inserendo rumore mirato.
    
    class_concept_matrix: Tensor (num_classes, num_concepts)
    class_names: Lista di stringhe con i nomi delle classi
    concept_names: Lista di stringhe con i nomi dei concetti
    modifications: Lista di tuple nel formato (nome_classe, nome_concetto, nuovo_valore)
                   es: [('Gabbiano', 'Zampa_palmata', 0.0), ('Corvo', 'Nero', 0.0)]
                   
    Ritorna: Una copia della matrice originale con le modifiche applicate.
    """
    # Cloniamo la matrice per non inquinare la Ground Truth originale
    sabotaged_matrix = class_concept_matrix.clone().float()
    
    modifiche_effettuate = 0
    
    for target_class, target_concept, new_value in modifications:
        # 1. Trova l'indice della classe
        if target_class in class_names:
            c_idx = class_names.index(target_class)
        else:
            print(f"⚠️ Avviso: Classe '{target_class}' non trovata. Ignoro.")
            continue
            
        # 2. Trova l'indice del concetto
        if target_concept in concept_names:
            k_idx = concept_names.index(target_concept)
        else:
            print(f"⚠️ Avviso: Concetto '{target_concept}' non trovato. Ignoro.")
            continue
            
        # 3. Applica il sabotaggio
        old_value = sabotaged_matrix[c_idx, k_idx].item()
        sabotaged_matrix[c_idx, k_idx] = float(new_value)
        modifiche_effettuate += 1
        
        print(f"Sabotaggio: {target_class} -> {target_concept} | Vecchio: {old_value} -> Nuovo: {new_value}")
        
    print(f"Completato: {modifiche_effettuate} modifiche applicate.")
    return sabotaged_matrix