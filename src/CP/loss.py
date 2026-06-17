import torch

def masked_mse_loss(preds, targets):
    """
    Calcola l'MSE ignorando i valori NaN nei target.
    preds: Tensore di shape (batch_size, 10)
    targets: Tensore di shape (batch_size, 10), può contenere float('nan')
    """
    # Creiamo una maschera booleana: True dove il target è valido
    mask = ~torch.isnan(targets)
    
    if mask.sum() == 0:
        # Se un intero batch è vuoto (raro), restituiamo 0 con gradiente
        return torch.tensor(0.0, device=preds.device, requires_grad=True)
    
    # Calcoliamo l'errore quadratico solo sugli elementi validi
    loss = torch.nn.functional.mse_loss(preds[mask], targets[mask], reduction='mean')
    return loss