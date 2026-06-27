import torch
import torch.nn as nn
from torch.optim import Adam
from src.BOX.model import BoxHierarchyModel, BoxHierarchyModelJoint
from src.utils.dataset import ConceptImplicationDataset
from torch.utils.data import DataLoader

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

def train_box_empirical(model, ground_truth_matrix, optimizer, criterion, epochs=100, batch_size=256):
    # Setup del device (GPU se disponibile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ground_truth_matrix = ground_truth_matrix.to(device)
    
    # Inizializza Dataset e DataLoader
    dataset = ConceptImplicationDataset(ground_truth_matrix)
    # Se il numero di concetti è piccolo (es. 50x50 = 2500 coppie), 
    # puoi anche omettere il dataloader e processare tutto in un batch,
    # ma il DataLoader è la best practice.
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_fit_loss = 0.0
        total_reg_loss = 0.0
        
        for batch_i, batch_j, batch_targets in dataloader:
            # Sposta i dati sul device corretto
            batch_i = batch_i.to(device)
            batch_j = batch_j.to(device)
            batch_targets = batch_targets.to(device)
            
            # 1. Azzera i gradienti
            optimizer.zero_grad()
            
            # 2. Forward pass: calcola le probabilità previste
            predictions = model(batch_i, batch_j)
            
            # Assicurati che predictions e targets abbiano la stessa forma (1D)
            predictions = predictions.squeeze()
            
            # 3. Calcola la loss
            loss_fit = criterion(predictions, batch_targets)

            loss_reg = model.get_regularization_loss()

            loss = loss_fit + loss_reg
            
            # 4. Backward pass (calcola gradienti)
            loss.backward()
            
            # 5. Aggiorna i pesi
            optimizer.step()
            
            total_loss += loss.item() * batch_i.size(0)
            total_fit_loss += loss_fit.item() * batch_i.size(0)
            total_reg_loss += loss_reg.item() * batch_i.size(0)
            
        # Calcola la loss media dell'epoca
        avg_loss = total_loss / len(dataset)
        avg_fit = total_fit_loss / len(dataset)
        avg_reg = total_reg_loss / len(dataset)
        
        # Stampa i progressi ogni 10 epoche
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Tot: {avg_loss:.4f} | Fit: {avg_fit:.4f} | Vol: {avg_reg:.4f}")

    print("Addestramento completato!")
    return model

def train_box_joint(
        model: BoxHierarchyModelJoint,
        optimizer: torch.optim.Adam,
        criterion: nn.BCELoss,
        dataset_concepts: list,
        dataset_classes: list,
        concept2id: dict,
        class2id: dict,
        id2concept: dict,
        id2class: dict,
        EPOCHS=100,  
        alpha=1.0 # Peso per bilanciare le due loss
    ):

    print(f"Concetti: {len(concept2id)} | Classi: {len(class2id)}")
    print(f"Supervisioni Concetti: {len(dataset_concepts)} | Supervisioni Classi: {len(dataset_classes)}")

    # Tensori Gerarchia Concetti
    tc_i = torch.tensor([x[0] for x in dataset_concepts], dtype=torch.long)
    tc_j = torch.tensor([x[1] for x in dataset_concepts], dtype=torch.long)
    tc_targets = torch.tensor([x[2] for x in dataset_concepts], dtype=torch.float32)

    # Tensori Appartenenza Classi
    cls_concept_idx = torch.tensor([x[0] for x in dataset_classes], dtype=torch.long)
    cls_class_idx = torch.tensor([x[1] for x in dataset_classes], dtype=torch.long)
    cls_targets = torch.tensor([x[2] for x in dataset_classes], dtype=torch.float32)

    print("\nInizio Addestramento Congiunto...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        
        # 1. Forward e Loss sulla Gerarchia dei Concetti
        p_concepts = model.forward_concepts(tc_i, tc_j)
        loss_concepts = criterion(p_concepts, tc_targets)
        
        # 2. Forward e Loss sull'assegnazione Classi-Concetti
        p_classes = model.forward_classes(cls_concept_idx, cls_class_idx)
        loss_classes = criterion(p_classes, cls_targets)
        
        # 3. Loss Totale = Loss Concetti + (alpha * Loss Classi) + Regolarizzazione
        total_loss = loss_concepts + (alpha * loss_classes) + model.get_regularization_loss()
        
        # Backward e step
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoca [{epoch}/{EPOCHS}] | L_Tot: {total_loss.item():.4f} (L_Conc: {loss_concepts.item():.4f}, L_Cls: {loss_classes.item():.4f})")

    print("\nAddestramento completato!")
    
    # --- VALUTAZIONE DI ESEMPIO ---
    model.eval()
    with torch.no_grad():
        print("\nVerifica di alcune assegnazioni Classe-Concetto:")
        # Testiamo un paio di associazioni positive e negative
        for idx in [0, len(dataset_classes)//2, -1]:
            c_idx, cls_idx, target = dataset_classes[idx]
            p = model.forward_classes(torch.tensor([c_idx]), torch.tensor([cls_idx])).item()
            print(f"Classe: {id2class[cls_idx]} | Concetto: {id2concept[c_idx]}")
            print(f" - P_teorica: {target:.1f} -> P_predetta: {p:.4f}")