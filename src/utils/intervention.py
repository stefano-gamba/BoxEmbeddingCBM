import torch

def generate_intervention_mask(concept_probs, strategy="random", k=5, group_indices=None):
    """
    Genera una maschera binaria per il Test Time Intervention.
    
    Argomenti:
        concept_probs (Tensor): Probabilità dei concetti (batch_size, num_concepts)
        strategy (str): "random", "uncertain" o "group"
        k (int): Numero di concetti da correggere (per "random" e "uncertain")
        group_indices (list): Indici dei concetti da correggere (per "group")
    """
    batch_size, num_concepts = concept_probs.shape
    mask = torch.zeros_like(concept_probs)
    
    # Se k è 0, restituisce una maschera vuota (nessun intervento)
    if k == 0 and strategy != "group":
        return mask

    if strategy == "random":
        # Sceglie k concetti a caso per ogni sample nel batch
        for i in range(batch_size):
            random_idx = torch.randperm(num_concepts)[:k]
            mask[i, random_idx] = 1.0

    elif strategy == "uncertain":
        # Trova i k concetti più incerti (probabilità più vicina a 0.5)
        # Calcoliamo la distanza assoluta da 0.5 (più è piccola, più è incerto)
        distance_from_half = torch.abs(concept_probs - 0.5)
        
        # Prendiamo gli indici con le distanze MINORI (largest=False)
        _, uncertain_idx = torch.topk(distance_from_half, k, dim=1, largest=False)
        
        # Scrive 1.0 negli indici selezionati
        mask.scatter_(1, uncertain_idx, 1.0)

    elif strategy == "group":
        # Interviene sempre e solo sugli indici passati
        if group_indices is not None:
            mask[:, group_indices] = 1.0
        else:
            raise ValueError("Per la strategia 'group' devi passare una lista di 'group_indices'.")

    return mask.to(concept_probs.device)