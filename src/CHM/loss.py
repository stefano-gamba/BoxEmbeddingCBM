import torch
import torch.nn.functional as F

def hierarchical_concept_loss(c_probs, prob_matrix):
    """
    Calcola la loss gerarchica basata sui box embedding statici.
    
    Args:
        c_probs: Tensore (batch_size, num_concepts) con le probabilità predette.
        prob_matrix: Tensore (num_concepts, num_concepts) dove [i, j] = P(c_i | c_j).
    """
    # Espandiamo c_probs per calcolare tutte le coppie (c_j - c_i)
    # c_j avrà shape (batch, 1, num_concepts)
    # c_i avrà shape (batch, num_concepts, 1)
    c_j = c_probs.unsqueeze(1) 
    c_i = c_probs.unsqueeze(2) 
    
    # diff[batch, i, j] = c_probs[batch, j] - c_probs[batch, i]
    diff = c_j - c_i
    
    # Penalizziamo solo quando c_j > c_i (violazione logica)
    violation = F.relu(diff) ** 2
    
    # Moltiplichiamo per P(c_i | c_j). 
    # prob_matrix.unsqueeze(0) assicura il corretto broadcasting sul batch.
    weighted_violation = prob_matrix.unsqueeze(0) * violation
    
    # Media sul batch e somma su tutte le coppie di concetti
    return weighted_violation.sum(dim=(1, 2)).mean()

def compute_hierarchical_weights(concept_heights, alpha=0.5, device="cpu"):
    """
    Calcola i pesi per la W-BCE basandosi sull'altezza nella gerarchia.
    
    Args:
        concept_heights: Lista o tensore con l'altezza di ogni concetto 
                         (es. radice = altezza max, foglie = 1 o 0).
        alpha: Iperparametro che controlla il decadimento del peso.
    """
    heights_tensor = torch.tensor(concept_heights, dtype=torch.float32)
    
    # Decadimento esponenziale come suggerito dal paper
    weights = torch.exp(alpha * heights_tensor)
    
    # Opzionale: normalizzare i pesi per evitare che i gradienti esplodano
    # weights = weights / weights.mean() 
    
    return weights.to(device)

def weighted_concept_loss(c_logits, c_gt, weights):
    """
    Calcola la Weighted Binary Cross-Entropy.
    """
    # Usiamo reduction='none' per applicare il peso specifico per ogni concetto
    loss_matrix = F.binary_cross_entropy_with_logits(c_logits, c_gt, reduction='none')
    
    # Moltiplichiamo per i pesi (broadcasting lungo il batch)
    weighted_loss = loss_matrix * weights.unsqueeze(0)
    
    # Media finale
    return weighted_loss.mean()