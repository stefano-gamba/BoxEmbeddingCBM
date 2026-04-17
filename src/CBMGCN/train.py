import torch
from src.CBMGCN.model import ConceptGNNClassifier
from src.utils.box import calcola_matrice_probabilita
from src.CP.train import train_concept_predictor

def train_gnn_cbm_classifier(
        model: ConceptGNNClassifier,
        train_dataloader,
        val_dataloader,
        optimizer, 
        criterion, 
        class_concept_matrix,
        boxes_tensor,
        EPOCHS=20,
        device="cpu",
        info="boxes",
        bipolar=False,
    ):
    """
    Funzione di training dedicata per il ConceptGNNClassifier.
    La differenza principale è che la prob_matrix viene SEMPRE calcolata
    poiché serve come matrice di adiacenza (struttura del grafo) per i layer GCN.
    """

    model.to(device)
    class_concept_matrix = class_concept_matrix.to(device)
    boxes_tensor = boxes_tensor.to(device)

    # 1. Calcolo OBBLIGATORIO della matrice di probabilità (Adjacency Matrix)
    with torch.no_grad():
        # Assicurati di avere 'calcola_matrice_probabilita' a scope
        prob_matrix = calcola_matrice_probabilita(boxes_tensor).to(device)
        prob_matrix.fill_diagonal_(0.0) # Evitiamo self-loop doppi (li gestisce il modello)
    
    history = {
        'train': {'tot_loss': [], 'acc': []},
        'val':   {'tot_loss': [], 'acc': []}
    }

    print("Inizio addestramento del GNN CBM (c -> grafo -> y)...")
    
    for epoch in range(1, EPOCHS + 1):
        # --- FASE DI TRAINING ---
        model.train()
        train_loss = 0.0
        train_correct, train_samples = 0, 0
        
        for features, labels in train_dataloader:
            features = features.to(device)
            labels = labels.to(device).long().view(-1) - 1 
            
            concept_labels = class_concept_matrix[labels].float()

            if bipolar:
                concept_labels = concept_labels * 2 - 1
            
            optimizer.zero_grad()
            
            c_true = concept_labels.unsqueeze(-1) # shape: (batch, num_concepts, 1)
            
            # Feature dei nodi (X)
            if info == "boxes":
                scaled_info = c_true * boxes_tensor.unsqueeze(0)
            elif info == 'concepts':
                scaled_info = c_true

            # 2. FORWARD PASS MODIFICATO: passiamo sia le feature (scaled_info) che il grafo (prob_matrix)
            logits = model(scaled_info, prob_matrix)

            # Calcolo Loss e Backpropagation
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_samples += labels.size(0)
        
        # --- FASE DI VALIDATION ---
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
                elif info == 'concepts':
                    scaled_info = c_true
                
                # FORWARD PASS MODIFICATO per Validation
                logits = model(scaled_info, prob_matrix)
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

        print(f"Epoca {epoch:3d}/{EPOCHS} | "
              f"TRAIN: Loss={history['train']['tot_loss'][-1]:.3f}, Acc={history['train']['acc'][-1]*100:.1f}% | "
              f"VAL: Loss={history['val']['tot_loss'][-1]:.3f}, Acc={history['val']['acc'][-1]*100:.1f}%")

    print("Addestramento GNN completato.")
    return history

def sequential_training_gnn(
        classifier: ConceptGNNClassifier, 
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
            elif info == 'concepts':
                scaled_info = c_pred_expanded

            # Forward pass del classificatore
            logits = classifier(scaled_info, prob_matrix)

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
                elif info == 'concepts':
                    scaled_info = c_pred_expanded
                
                logits = classifier(scaled_info, prob_matrix)
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