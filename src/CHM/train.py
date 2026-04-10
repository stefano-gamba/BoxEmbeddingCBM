import torch
import torch.nn as nn
from torch.optim import Adam
from src.CHM.model import BoxHierarchyModel
from src.CHM.model import calcola_matrice_probabilita

def train_box(
        model: BoxHierarchyModel,
        optimizer: Adam,
        criterion: nn.BCELoss,
        dataset: list,
        concept2id: dict,
        id2concept: dict, 
        EPOCHS=100,  
    ):

    num_concepts = len(concept2id)
    print(f"Trovati {num_concepts} concetti unici e {len(dataset)} relazioni supervisionate.")

    # Creazione Tensori Pytorch
    tensor_i = torch.tensor([x[0] for x in dataset], dtype=torch.long)
    tensor_j = torch.tensor([x[1] for x in dataset], dtype=torch.long)
    targets = torch.tensor([x[2] for x in dataset], dtype=torch.float32)

    print("\nInizio Addestramento...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass su tutto il dataset
        probabilities = model(tensor_i, tensor_j)
        
        # Calcolo Loss
        loss = criterion(probabilities, targets)
        
        # Backward e step
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoca [{epoch}/{EPOCHS}] | Loss: {loss.item():.4f}")

    print("\nAddestramento completato!")
    
    # ==========================================
    # 4. VALUTAZIONE DI ESEMPIO
    # ==========================================
    model.eval()
    with torch.no_grad():
        print("\nVerifica di alcune predizioni post-addestramento:")
        sample_indices = [0, len(dataset)//2, -1] # Prendiamo il primo, quello a metà e l'ultimo
        for idx in sample_indices:
            i, j, target = dataset[idx]
            p = model(torch.tensor([i]), torch.tensor([j])).item()
            print(f"Relazione: {id2concept[i]} | {id2concept[j]}")
            print(f" - P_teorica: {target:.1f} -> P_predetta: {p:.4f}")


def train_cbm_classifier(
        model,
        train_dataloader,
        val_dataloader,
        optimizer, 
        criterion, 
        class_concept_matrix,
        boxes_tensor,
        EPOCHS=100,
        device="cpu",
        info="boxes",
    ):
    """
    dataset_classificazione: Lista di tuple (classe_target, vettore_concetti_binario)
    es: [(0, [1, 0, 1, ...]), (1, [0, 1, 1, ...])]
    """

    model.to(device)
    class_concept_matrix = class_concept_matrix.to(device)
    boxes_tensor = boxes_tensor.to(device)

    if info == "rel_matrix":
        with torch.no_grad():
            prob_matrix = calcola_matrice_probabilita(boxes_tensor)
            prob_matrix = prob_matrix.to(device)
    
    
    history = {
        'train': {'tot_loss': [], 'acc': []},
        'val':   {'tot_loss': [], 'acc': []}
    }

    best_val_acc = 0.0

    print("Inizio addestramento del classificatore (c -> y)...")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_correct, train_samples = 0, 0
        
        for features, labels in train_dataloader:
            features = features.to(device)
            labels = labels.to(device).long().view(-1) - 1 # Assumiamo che le classi siano 1-indexed, quindi convertiamo a 0-indexed
            concept_labels = class_concept_matrix[labels].float()
            
            optimizer.zero_grad()
            
            # Formattiamo target e concetti (Ground Truth)
            c_true = concept_labels.unsqueeze(-1) # shape: (batch_size, num_concepts, 1)
            
            # 3. CORE DEL CBM-IBRIDO: Scaliamo i box embedding con la ground truth
            # Effettuiamo un broadcasting: moltiplichiamo c_true (1 o 0) per i box
            # shape finale: (1, num_concepts, box_dim)
            if info == "boxes":
                scaled_info = c_true * boxes_tensor.unsqueeze(0)
            elif info == "rel_matrix":
                scaled_info = c_true * prob_matrix.unsqueeze(0)

            # 4. Forward pass
            logits = model(scaled_info)

            # 5. Calcolo Loss e Backpropagation
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calcolo accuratezza
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_samples += labels.size(0)
        
        model.eval()
        val_loss = 0.0
        val_correct, val_samples = 0, 0
        
        with torch.no_grad():
            for features, labels in val_dataloader:
                features = features.to(device)
                labels = labels.to(device).long().view(-1) - 1
                concept_labels = class_concept_matrix[labels].float()
                
                c_true = concept_labels.unsqueeze(-1)

                if info == "boxes":
                    scaled_info = c_true * boxes_tensor.unsqueeze(0)
                elif info == "rel_matrix":
                    scaled_info = c_true * prob_matrix.unsqueeze(0)
                
                logits = model(scaled_info)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_samples += labels.size(0)
                
        # Statistiche epoca
        t_batches = len(train_dataloader)
        v_batches = len(val_dataloader)
        
        history['train']['tot_loss'].append(train_loss / t_batches)
        history['train']['acc'].append(train_correct / train_samples)
        history['val']['tot_loss'].append(val_loss / v_batches)
        history['val']['acc'].append(val_correct / val_samples)

        print(f"Epoca {epoch+1:3d}/{EPOCHS} | "
              f"TRAIN: Loss={history['train']['tot_loss'][-1]:.3f}, Acc={history['train']['acc'][-1]*100:.1f}% | "
              f"VAL: Loss={history['val']['tot_loss'][-1]:.3f}, Acc={history['val']['acc'][-1]*100:.1f}%")


    print("Addestramento completato.")
    return history