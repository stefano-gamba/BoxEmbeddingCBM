import torch
import torch.nn as nn
from torch.optim import Adam
from src.CHM.model import BoxHierarchyModel
from src.CHM.model import calcola_matrice_probabilita
import matplotlib.pyplot as plt

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
        bipolar=False,
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
            prob_matrix.fill_diagonal_(0.0)
    
    
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

            # Trasformiamo i 0 in -1, e lasciamo gli 1 come 1.
            # La formula (x * 2) - 1 fa esattamente questo: (0*2)-1 = -1 | (1*2)-1 = 1
            if bipolar:
                concept_labels = concept_labels * 2 - 1
            
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
            elif info == 'concepts':
                scaled_info = c_true

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

                if bipolar:
                    concept_labels = concept_labels * 2 - 1
                
                c_true = concept_labels.unsqueeze(-1)

                if info == "boxes":
                    scaled_info = c_true * boxes_tensor.unsqueeze(0)
                elif info == "rel_matrix":
                    scaled_info = c_true * prob_matrix.unsqueeze(0)
                elif info == 'concepts':
                    scaled_info = c_true
                
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


def sequential_training(
        classifier, 
        concept_predictor, 
        train_loader, 
        val_loader, 
        class_concept_matrix, 
        boxes_tensor,
        optimizer_cls,
        optimizer_concept,
        criterion_cls,
        criterion_concept,
        epochs, 
        device,
        info="boxes",
        bipolar=False
):
    """
    Esegue l'addestramento Sequential Bottleneck in due fasi:
    Fase 1: Addestra il predittore di concetti sulle feature h usando la Ground Truth c.
    Fase 2: Addestra il classificatore usando le PREDIZIONI dei concetti.
    """
    
    # =====================================================================
    # FASE 1: Addestramento del Predittore di Concetti (h -> c)
    # Il modello impara a mappare le feature nei concetti in modo indipendente.
    # =====================================================================
    print("========== FASE 1: Addestramento Predittore Concetti (h -> c) ==========")
    history_concept = train_concept_predictor(
        concept_predictor, 
        train_loader, 
        val_loader, 
        class_concept_matrix, 
        optimizer_concept, 
        criterion_concept, 
        epochs, # Nota: potresti voler sdoppiare epochs_concept ed epochs_cls
        device
    )
    
    # =====================================================================
    # FASE 2: Addestramento del Classificatore (c_pred -> y)
    # Il classificatore impara ad adattarsi alle predizioni imperfette del bottleneck.
    # =====================================================================
    print("\n========== FASE 2: Addestramento Classificatore Sequenziale (c_pred -> y) ==========")
    
    classifier.to(device)
    # Mettiamo il predittore in eval() per non aggiornare più i suoi pesi
    concept_predictor.eval() 
    
    boxes_tensor = boxes_tensor.to(device)

    if info == "rel_matrix":
        with torch.no_grad():
            prob_matrix = calcola_matrice_probabilita(boxes_tensor)
            prob_matrix = prob_matrix.to(device)
            prob_matrix.fill_diagonal_(0.0)
            
    history_cls = {
        'train': {'tot_loss': [], 'acc': []},
        'val':   {'tot_loss': [], 'acc': []}
    }

    for epoch in range(1, epochs + 1):
        classifier.train()
        train_loss = 0.0
        train_correct, train_samples = 0, 0
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device).long().view(-1) - 1 

            # -------------------------------------------------------------
            # MODIFICA SEQUENTIAL: Otteniamo le predizioni dal modello h -> c
            # anziché usare class_concept_matrix[labels]
            # -------------------------------------------------------------
            with torch.no_grad():
                # Assumiamo che il predittore restituisca (probs, logits)
                c_probs, _ = concept_predictor(features)
            
            # Usiamo le probabilità (valori tra 0 e 1) per il mascheramento soft
            concept_preds = c_probs

            if bipolar:
                # Mappa le probabilità da [0, 1] a [-1, 1] 
                # Se p=0 -> -1, p=1 -> 1, p=0.5 -> 0
                concept_preds = concept_preds * 2 - 1
            
            optimizer_cls.zero_grad()
            
            # Formattiamo i concetti predetti
            c_pred_expanded = concept_preds.unsqueeze(-1) # shape: (batch_size, num_concepts, 1)
            
            # Scaliamo i box embedding / matrici con le PREDIZIONI
            if info == "boxes":
                scaled_info = c_pred_expanded * boxes_tensor.unsqueeze(0)
            elif info == "rel_matrix":
                scaled_info = c_pred_expanded * prob_matrix.unsqueeze(0)
            elif info == 'concepts':
                scaled_info = c_pred_expanded

            # Forward pass del classificatore
            logits = classifier(scaled_info)

            # Calcolo Loss e Backpropagation (solo per il classificatore)
            loss = criterion_cls(logits, labels)
            loss.backward()
            optimizer_cls.step()
            
            train_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_samples += labels.size(0)
        
        # --- Fase di Validation Classificatore ---
        classifier.eval()
        val_loss = 0.0
        val_correct, val_samples = 0, 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device).long().view(-1) - 1
                
                # Otteniamo i concetti predetti sul set di validazione
                c_probs, _ = concept_predictor(features)
                concept_preds = c_probs

                if bipolar:
                    concept_preds = concept_preds * 2 - 1
                
                c_pred_expanded = concept_preds.unsqueeze(-1)

                if info == "boxes":
                    scaled_info = c_pred_expanded * boxes_tensor.unsqueeze(0)
                elif info == "rel_matrix":
                    scaled_info = c_pred_expanded * prob_matrix.unsqueeze(0)
                elif info == 'concepts':
                    scaled_info = c_pred_expanded
                
                logits = classifier(scaled_info)
                loss = criterion_cls(logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_samples += labels.size(0)
                
        # Statistiche epoca Classificatore
        t_batches = len(train_loader)
        v_batches = len(val_loader)
        
        history_cls['train']['tot_loss'].append(train_loss / t_batches)
        history_cls['train']['acc'].append(train_correct / train_samples)
        history_cls['val']['tot_loss'].append(val_loss / v_batches)
        history_cls['val']['acc'].append(val_correct / val_samples)

        print(f"Epoca {epoch:3d}/{epochs} | "
              f"TRAIN: Loss={history_cls['train']['tot_loss'][-1]:.3f}, Acc={history_cls['train']['acc'][-1]*100:.1f}% | "
              f"VAL: Loss={history_cls['val']['tot_loss'][-1]:.3f}, Acc={history_cls['val']['acc'][-1]*100:.1f}%")

    print("\nAddestramento Sequenziale completato.")
    return history_concept, history_cls


def plot_history(history):
    epochs = range(1, len(history['train']['tot_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        
    # --- Grafico 1: Loss di Train vs Validazione ---
    ax1.plot(epochs, history['train']['tot_loss'], label='Train Loss Totale', color='blue', linewidth=2)
    ax1.plot(epochs, history['val']['tot_loss'], label='Val Loss Totale', color='red', linewidth=2)
        
        
    ax1.set_title('Curve di Loss (Train vs Val)', fontsize=14)
    ax1.set_xlabel('Epoche', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
        
    # --- Grafico 2: Accuratezza Multi-Classe ---
    ax2.plot(epochs, history['train']['acc'], label='Train Accuracy', color='green', linewidth=2)
    ax2.plot(epochs, history['val']['acc'], label='Val Accuracy', color='orange', linewidth=2)
        
    ax2.set_title('Accuratezza di Classificazione', fontsize=14)
    ax2.set_xlabel('Epoche', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    plt.show()