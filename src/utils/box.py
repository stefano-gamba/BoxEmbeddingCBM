import torch
from box_embeddings.parameterizations import MinDeltaBoxTensor, BoxTensor
from box_embeddings.modules.intersection import HardIntersection
from box_embeddings.modules.volume import SoftVolume
import numpy as np

def get_box_dict(model, id2concept):
    """
    Estrae tutti i box appresi dal modello e crea una mappatura diretta.
    Restituisce un dizionario nel formato {concetto (str): box (MinDeltaBoxTensor)}.
    """
    # Mettiamo il modello in eval mode per sicurezza
    model.eval()
    
    dizionario_box = {}
    
    # Disabilitiamo il calcolo dei gradienti dato che stiamo solo estraendo dati
    with torch.no_grad():
        for idx, concept_name in id2concept.items():
            # 1. Creiamo un tensore contenente l'ID del concetto
            tensor_id = torch.tensor([idx], dtype=torch.long)
            
            # 2. Estraiamo il parametro base theta (le word embeddings raw)
            theta = model.embeddings(tensor_id).view(-1, 2, model.dim)
            
            # 3. Lo incapsuliamo nel MinDeltaBoxTensor
            box = MinDeltaBoxTensor(theta)
            
            # 4. Salviamo nel dizionario
            dizionario_box[concept_name] = box
            
    return dizionario_box


# ==========================================
# PREPARAZIONE DEI DATI E DEI BOX
# ==========================================
def prepara_tensore_box(dict_boxes, concept2id):
    """
    Estrae i parametri (theta) dal dizionario dei box e crea un singolo
    tensore fisso. Non calcoleremo i gradienti per questi box.
    """
    num_concepts = len(concept2id)
    # Assumiamo che il dizionario contenga oggetti MinDeltaBoxTensor
    # il cui attributo "data" (o la property corrispondente) sia theta (dimensione 2*D)
    
    # Prendiamo un box a caso per capirne la dimensione
    sample_box = next(iter(dict_boxes.values()))
    # In box_embeddings, the parameters are usually accessible. 
    # Assumendo che il tensore theta originale fosse salvato o recuperabile:
    # (qui simulo estraendo z e Z concatenati, che equivalgono a 2*D parametri)
    box_dim = sample_box.z.size(-1) * 2
    
    boxes_tensor = torch.zeros((num_concepts, box_dim))
    
    for concept_name, idx in concept2id.items():
        box = dict_boxes[concept_name]
        # Concateniamo min (z) e max (Z) per avere i parametri geometrici completi
        theta = torch.cat([box.z.squeeze(), box.Z.squeeze()], dim=-1)
        boxes_tensor[idx] = theta
        
    return boxes_tensor.detach() # .detach() assicura che i box siano congelati


def calcola_matrice_probabilita(boxes_tensor):
    """
    Calcola la matrice delle probabilità condizionate P(i|j)
    usando l'API della libreria Box Embeddings.
    Shape di output: (num_concepts, num_concepts)
    """
    num_concepts = boxes_tensor.size(0)

    box_dim = boxes_tensor.size(-1) // 2
    
    # Prepariamo i tensori per il broadcasting: vogliamo combinare ogni box con tutti gli altri
    # theta_i rappresenta l'ipotesi (intersezione), theta_j rappresenta la premessa (condizionante)
    theta_i = boxes_tensor.unsqueeze(1).expand(-1, num_concepts, -1) 
    theta_j = boxes_tensor.unsqueeze(0).expand(num_concepts, -1, -1) 

    theta_i = theta_i.view(num_concepts, num_concepts, 2, box_dim)
    theta_j = theta_j.view(num_concepts, num_concepts, 2, box_dim)
    
    box_i = BoxTensor(theta_i)
    box_j = BoxTensor(theta_j)
    
    # Inizializziamo i moduli operativi della libreria
    intersection_op = HardIntersection()
    volume_op = SoftVolume(volume_temperature=1.0) # Restituisce il volume logaritmico di default
    
    # 1. Intersezione
    box_int = intersection_op(box_i, box_j)
    
    # 2. Volumi Logaritmici
    log_vol_int = volume_op(box_int)
    log_vol_j = volume_op(box_j)
    
    # 3. Probabilità condizionata: P(i|j) = exp(log(Vol_int) - log(Vol_j))
    log_p_i_given_j = log_vol_int - log_vol_j
    
    # Esponenziale e clamp per rimuovere eventuali artefatti numerici minimi oltre 1.0
    p_i_given_j = torch.exp(log_p_i_given_j)
    prob_matrix = torch.clamp(p_i_given_j, min=0.0, max=1.0)
    
    return prob_matrix


def apply_logical_smoothing(concept_labels, smoothing_matrix, threshold=0.5):
    """
    Applica lo smoothing logico ai concept_labels.
    
    Args:
        concept_labels: Tensore di shape [batch_size, num_concepts]
        smoothing_matrix: Matrice di probabilità condizionate [num_concepts, num_concepts]
                          Si assume che smoothing_matrix[i, j] rappresenti P(i|j).
        threshold: Soglia oltre la quale considerare un concetto attivo (default 0.5).
        
    Returns:
        smoothed_labels: Tensore della stessa shape di concept_labels con le correzioni.
    """

    # 1. Creiamo una maschera binaria dalla matrice di smoothing
    # S_mask[i, j] sarà 1 se P(i|j) > 0.5, altrimenti 0
    S_mask = (smoothing_matrix > threshold).float()

    # 2. Capiamo quali concetti "j" sono attualmente attivi nel batch
    # (utile soprattutto se in ingresso hai probabilità continue e non ancora binarie)
    C_active = (concept_labels > threshold).float()

    # 3. Calcoliamo i concetti da attivare tramite moltiplicazione tra matrici.
    # Moltiplicando C_active [B, N] per la trasposta di S_mask [N, N],
    # otteniamo una matrice [B, N] dove i valori > 0 indicano che c'è almeno 
    # un concetto j attivo che "accende" il concetto i.
    triggered_concepts = torch.matmul(C_active, S_mask.t()) > 0

    # 4. Uniamo i concetti originali con quelli nuovi attivati.
    # Usiamo torch.max: se un concetto era già alto, mantiene il suo valore.
    # Se era basso ma è stato "triggerato" dalla regola, viene forzato a 1.0.
    smoothed_labels = torch.max(concept_labels, triggered_concepts.float())

    return smoothed_labels

def calculate_concept_heights(concept2id, relations_json):
    """
    Calcola l'altezza di ciascun concetto basandosi sulle relazioni di implicazione.
    
    Args:
        concept2id: Dizionario {nome_concetto: id_vettore}
        relations_json: Lista di tuple/liste es. [['mano', 'dito', 1], ['animale', 'cane', 1]]
    
    Returns:
        Lista di interi 'concept_heights' ordinata secondo gli ID di concept2id.
    """
    # 1. Inizializziamo una struttura per mappare ogni padre ai suoi figli diretti
    parent_to_children = {concept: [] for concept in concept2id}
    
    # 2. Popoliamo il grafo invertito (da padre a figli) usando solo le implicazioni (valore 1)
    for parent, child, is_implied in relations_json:
        if is_implied == 1:
            if parent in parent_to_children and child in parent_to_children:
                parent_to_children[parent].append(child)
    
    # 3. Funzione ricorsiva con memoizzazione per calcolare l'altezza di un singolo nodo
    memo = {}
    
    def compute_height(node):
        if node in memo:
            return memo[node]
        
        children = parent_to_children[node]
        # Se il nodo non ha figli, è una foglia della gerarchia -> altezza = 0
        if not children:
            memo[node] = 0
            return 0
        
        # Altrimenti, l'altezza è 1 + la massima altezza tra i suoi figli
        height = 1 + max(compute_height(child) for child in children)
        memo[node] = height
        return height

    # 4. Creiamo il vettore finale allocando gli spazi in base al numero di concetti
    concept_heights = [0] * len(concept2id)
    
    # 5. Riempiamo il vettore assicurandoci di rispettare l'id associato a ogni concetto
    for concept, c_id in concept2id.items():
        concept_heights[c_id] = compute_height(concept)
        
    return concept_heights

def compute_stratified_concept_accuracy(all_preds_binary, all_gts, concept_heights, id2concept):
    """
    Calcola l'accuratezza dei concetti divisa per livello gerarchico (altezza),
    stampando il dettaglio di ogni singolo concetto ordinato per accuratezza.
    
    Args:
        all_preds_binary (np.ndarray o torch.Tensor): Predizioni binarie (N_samples, N_concepts)
        all_gts (np.ndarray o torch.Tensor): Ground truth (N_samples, N_concepts)
        concept_heights (list o np.ndarray): Altezze pre-calcolate dei concetti (len = N_concepts)
        id2concept (dict): Dizionario che mappa l'ID (int) al nome del concetto (str)
        
    Returns:
        dict: Dizionario con l'accuratezza per ogni livello di altezza e il dettaglio dei concetti.
    """
    if torch.is_tensor(all_preds_binary):
        all_preds_binary = all_preds_binary.cpu().numpy()
    if torch.is_tensor(all_gts):
        all_gts = all_gts.cpu().numpy()
        
    heights_arr = np.array(concept_heights)
    unique_heights = np.unique(heights_arr)
    num_samples = all_preds_binary.shape[0]
    
    stratified_results = {}
    
    print("\n" + "="*60)
    print(" ACCURATEZZA CONCETTI STRATIFICATA PER ALTEZZA")
    print("="*60)
    
    for h in sorted(unique_heights, reverse=True):
        # 1. Trova gli indici dei concetti a questa altezza
        indices = np.where(heights_arr == h)[0]
        
        if len(indices) == 0:
            continue
            
        # 2. Estrai le sottomatrici per questo livello
        preds_h = all_preds_binary[:, indices]
        gts_h = all_gts[:, indices]
        
        # 3. Calcola l'accuratezza globale del livello
        correct_total = (preds_h == gts_h).sum()
        accuracy_level = (correct_total / preds_h.size) * 100
        
        # 4. Calcola l'accuratezza per OGNI SINGOLO concetto (colonna per colonna)
        # sum(axis=0) conta i True per ogni colonna
        correct_per_concept = (preds_h == gts_h).sum(axis=0) 
        acc_per_concept = (correct_per_concept / num_samples) * 100
        
        # 5. Raccogliamo i risultati in una lista di tuple (nome, accuratezza)
        concept_details = []
        for i, concept_idx in enumerate(indices):
            concept_name = id2concept[concept_idx]
            concept_details.append((concept_name, acc_per_concept[i]))
            
        # Ordiniamo dal peggiore al migliore per evidenziare i concetti problematici
        concept_details.sort(key=lambda x: x[1])
        
        stratified_results[h] = {
            'accuracy': accuracy_level,
            'num_concepts': len(indices),
            'details': concept_details
        }
        
        # --- STAMPA FORMATTATA ---
        level_name = "Padri/Radici" if h == max(unique_heights) else "Foglie/Micro" if h == min(unique_heights) else "Intermedio"
        print(f"\n► Livello {h:2d} ({level_name}) | N. Concetti: {len(indices)} | Acc. Media: {accuracy_level:.2f}%")
        print("-" * 60)
        
        # Stampiamo i concetti incolonnati
        for nome, acc in concept_details:
            # Un piccolo alert visivo se l'accuratezza è sotto una certa soglia (es. 85%)
            alert = " <--- LEAKAGE SOSPETTO" if acc < 85.0 else ""
            print(f"    {nome:<25} : {acc:6.2f}% {alert}")
            
    print("\n" + "="*60 + "\n")
    return stratified_results

import torch

def compute_concept_implications(y_train, class_concept_matrix):
    """
    Calcola la matrice di probabilità condizionate P(c_i | c_j) in puro PyTorch.
    """
    # 0. Controlli di sicurezza sui tipi
    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train)
    
    if not isinstance(class_concept_matrix, torch.Tensor):
        class_concept_matrix = torch.tensor(class_concept_matrix, dtype=torch.float32)
    else:
        # Assicuriamoci che sia float per le moltiplicazioni di probabilità
        class_concept_matrix = class_concept_matrix.float()

    if y_train.min() > 0:
        y_train = y_train - y_train.min()

    num_classes, num_concepts = class_concept_matrix.shape

    # 1. Calcola P(y)
    # torch.bincount richiede tensori interi 1D (long)
    class_counts = torch.bincount(y_train.long(), minlength=num_classes)
    P_y = class_counts.float() / len(y_train)

    # 2. Calcola P(c_j)
    P_c = P_y @ class_concept_matrix

    # 3. Calcola P(c_i, c_j)
    P_y_diag = torch.diag(P_y)
    P_ij = class_concept_matrix.T @ P_y_diag @ class_concept_matrix

    # 4. Calcola P(c_i | c_j) = P(c_i, c_j) / P(c_j)
    P_i_given_j = torch.zeros_like(P_ij)
    
    # Maschera booleana per evitare la divisione per zero
    mask = P_c > 0 
    
    # Eseguiamo la divisione solo sulle colonne (concetti j) dove P(c_j) > 0.
    # unsqueeze(0) serve per trasmettere (broadcast) correttamente la divisione sulle righe.
    P_i_given_j[:, mask] = P_ij[:, mask] / P_c[mask].unsqueeze(0)

    return P_i_given_j