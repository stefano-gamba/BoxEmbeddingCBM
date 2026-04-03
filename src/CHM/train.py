import torch
import torch.nn as nn
from torch.optim import Adam
from src.CHM.model import BoxHierarchyModel

def train(
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