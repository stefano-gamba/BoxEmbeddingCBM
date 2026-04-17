import torch

def train_concept_predictor(model, train_loader, val_loader, incidence_matrix, 
                            optimizer, criterion, epochs, device):
    """
    model: Il modulo h -> c_logits
    incidence_matrix: Tensor (num_classes, num_concepts) con la GT binaria
    """
    
    model.to(device)
    incidence_matrix = incidence_matrix.to(device)
    
    history = {
        'train': {'tot_loss': [], 'acc': []},
        'val':   {'tot_loss': [], 'acc': []}
    }

    for epoch in range(epochs):
        # --- Fase di Training ---
        model.train()
        train_loss, train_correct, total_elements = 0.0, 0, 0
        
        for h, y in train_loader:
            h, y = h.to(device), y.to(device).long().view(-1) - 1 # Assumiamo che le classi siano 1-indexed, quindi convertiamo a 0-indexed
            
            # Mappiamo le label della classe ai concetti tramite la matrice 
            c_gt = incidence_matrix[y].float() 
            
            optimizer.zero_grad()
            _, c_logits = model(h) # Assumendo che il modello h->c restituisca i logit
            loss = criterion(c_logits, c_gt)
            
            loss.backward()
            optimizer.step()
            
            # Metriche
            train_loss += loss.item() * h.size(0)
            # Accuratezza: soglia a 0 sui logit (equivale a 0.5 dopo la sigmoid) [cite: 112]
            preds = (c_logits > 0).float()
            train_correct += (preds == c_gt).sum().item()
            total_elements += h.size(0) * c_gt.size(1) # num_samples * num_concepts

        # --- Fase di Validation ---
        model.eval()
        val_loss, val_correct, val_total_elements = 0.0, 0, 0
        
        with torch.no_grad():
            for h, y in val_loader:
                h, y = h.to(device), y.to(device).long().view(-1) - 1
                c_gt = incidence_matrix[y].float()
                
                _, c_logits = model(h)
                loss = criterion(c_logits, c_gt)
                
                val_loss += loss.item() * h.size(0)
                preds = (c_logits > 0).float()
                val_correct += (preds == c_gt).sum().item()
                val_total_elements += h.size(0) * c_gt.size(1)

        t_batches = len(train_loader)
        v_batches = len(val_loader)
        
        history['train']['tot_loss'].append(train_loss / t_batches)
        history['train']['acc'].append(train_correct / total_elements)
        history['val']['tot_loss'].append(val_loss / v_batches)
        history['val']['acc'].append(val_correct / val_total_elements)

        print(f"Loss: {history['train']['tot_loss'][-1]:.4f} | Acc: {history['train']['acc'][-1]*100:.4f} "
              f"|| Val Loss: {history['val']['tot_loss'][-1]:.4f} | Val Acc: {history['val']['acc'][-1]*100:.4f}")

    return history