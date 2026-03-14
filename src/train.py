import torch
import torch.nn.functional as F
from src.model import BoxEmbeddingCBM

def train(model: BoxEmbeddingCBM, dataloader, optimizer, class_concept_matrix, hierarchy_gt, EPOCHS, device):
    """
    Addestra il BoxEmbeddingCBM in modalità Multi-Task.
    
    Args:
        model: L'istanza di BoxEmbeddingCBM.
        dataloader: DataLoader che restituisce (features, labels).
        optimizer: L'ottimizzatore (es. Adam).
        class_concept_matrix: Tensore [num_classes, num_concepts] (valori 0.0 o 1.0).
        hierarchy_gt: Lista di tuple (target_id, source_id, target_prob).
        EPOCHS: Numero di epoche.
        device: 'cpu' o 'cuda'.
    """
    print(f"Inizio addestramento su dispositivo: {device.upper()}")
    print("="*50)
    
    # Spostiamo il modello e la matrice di mappatura sul device corretto
    model = model.to(device)
    class_concept_matrix = class_concept_matrix.to(device)
    
    # Pesi per bilanciare le loss (da "tunnare" in base al dataset)
    W_TASK = 2.0      # Peso per la predizione della classe finale
    W_ACT = 1.0       # Peso per l'attivazione corretta dei concetti
    W_HIER = 1.0      # Peso per la gerarchia logica
    W_VOL = 0.1       # Peso per l'anti-collasso dei volumi
    
    for epoch in range(EPOCHS):
        model.train() # Impostiamo il modello in modalità training
        
        epoch_loss = 0.0
        epoch_task_loss = 0.0
        epoch_act_loss = 0.0
        epoch_hier_loss = 0.0
        
        for batch_idx, (features, labels) in enumerate(dataloader):
            # 1. Prepariamo i dati
            features = features.to(device)
            # Assicuriamoci che le label siano indici interi 1D per mappare la matrice
            labels = labels.to(device).long().squeeze() 
            
            # --- LA MAGIA DELLA MATRICE DI INCIDENZA ---
            # Estraiamo al volo le etichette dei concetti per l'intero batch!
            # Shape: (batch_size, num_concepts)
            concept_labels = class_concept_matrix[labels]
            
            optimizer.zero_grad()
            
            # 2. Forward Pass
            outputs = model(features)
            
            # 3. Calcolo delle Loss Individuali
            
            # A. Task Loss (Predizione della classe dell'immagine)
            # Convertiamo le label nel formato corretto per BCE (float)
            task_labels = labels.float().unsqueeze(1) 
            task_loss = F.binary_cross_entropy(outputs["task_probs"], task_labels)
            
            # B. Concept Activation Loss
            act_loss = F.binary_cross_entropy(outputs["concept_probs"], concept_labels)
            
            # C. Hierarchy Loss (Estraiamo le probabilità dalla matrice KxK)
            hier_loss = 0.0
            batch_size = features.size(0)
            
            for target_id, source_id, target_prob in hierarchy_gt:
                pred_prob = outputs["cond_prob_matrix"][:, target_id, source_id]
                target_tensor = torch.full((batch_size,), target_prob, dtype=torch.float32, device=device)
                hier_loss += F.binary_cross_entropy(pred_prob, target_tensor)
                
            # D. Volume Regularization (Anti-collasso)
            vol_loss = 0.0
            # Usiamo model.k per sapere quanti concetti ci sono e model.volume_op per calcolare
            for i in range(1, model.k): 
                vol_loss -= model.volume_op(outputs["boxes"][i]).mean()
                
            # 4. Loss Totale Ponderata
            loss = (W_TASK * task_loss) + (W_ACT * act_loss) + (W_HIER * hier_loss) + (W_VOL * vol_loss)
            
            # 5. Backpropagation e Ottimizzazione
            loss.backward()
            optimizer.step()
            
            # Aggiorniamo le statistiche
            epoch_loss += loss.item()
            epoch_task_loss += task_loss.item()
            epoch_act_loss += act_loss.item()
            epoch_hier_loss += hier_loss.item()
            
        # Logging a fine epoca (calcoliamo le medie sul numero di batch)
        num_batches = len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoca [{epoch+1:3d}/{EPOCHS}] | "
                  f"Loss Tot: {epoch_loss/num_batches:.4f} | "
                  f"Task: {epoch_task_loss/num_batches:.4f} | "
                  f"Act: {epoch_act_loss/num_batches:.4f} | "
                  f"Hier: {epoch_hier_loss/num_batches:.4f}")

    print("="*50)
    print("Addestramento completato con successo! 🎉")