import torch
import torch.nn as nn
import torch.optim as optim
from src.model import ConceptBoxModel
import torch.nn.functional as F
import wandb

def train_step(model, optimizer, h, y_true, class_concept_matrix, V_true, lambda_c=1.0, lambda_b=1.0):
    model.train()
    optimizer.zero_grad()

    # class_concept_matrix ha shape (NUM_CLASSES, NUM_CONCEPTS)
    # y_true contiene gli indici delle classi per ogni elemento del batch, shape (BATCH_SIZE,)
    # Indicizzando la matrice con y_true, otteniamo direttamente un tensore di shape (BATCH_SIZE, NUM_CONCEPTS)
    c_true = class_concept_matrix[y_true].float()
    
    # Forward pass: l'input è 'h', le features già estratte dal backbone
    y_hat_logits, p_hat, V_hat = model(h)
    
    # 1. L_task: Cross Entropy sulle classi finali
    # L_task = - sum(y_i * log(y_hat_i))
    loss_task = F.cross_entropy(y_hat_logits, y_true)
    
    # 2. L_concept: Binary Cross Entropy sui concetti
    # L_concepts = - sum(c_i * log(p_hat_i) + (1 - c_i) * log(1 - p_hat_i))
    loss_concept = F.binary_cross_entropy(p_hat, c_true.float())
    
    # 3. L_box
    # L_interactions = MSE(V, V_hat)
    loss_box = F.mse_loss(V_hat, V_true)
    
    # Loss Totale combinata
    loss = loss_task + (lambda_c * loss_concept) + (lambda_b * loss_box)
    
    # Backward e ottimizzazione
    loss.backward()
    optimizer.step()
    
    return loss.item(), loss_task.item(), loss_concept.item(), loss_box.item()


def train_loop(model, dataloader, optimizer, class_concept_matrix, V_gt_matrix, epochs, device):
    # Inizializza il progetto su Weights & Biases
    wandb.init(
        project="concept-box-model",
        config={
            "epochs": epochs,
            "batch_size": dataloader.batch_size,
            "learning_rate": optimizer.param_groups[0]['lr'],
        }
    )

    model = model.to(device)
    class_concept_matrix = class_concept_matrix.to(device)
    V_gt_matrix = V_gt_matrix.to(device)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_task = 0.0
        epoch_concept = 0.0
        epoch_box = 0.0
        
        for batch_h, batch_y in dataloader:
            # Spostiamo i dati del batch sul device corretto
            batch_h = batch_h.to(device)
            batch_y = batch_y.to(device)
            
            # NOTA: Qui estraiamo la V_true per il batch corrente. 
            # Assumendo che V_gt_matrix sia globale (per classe o fissa), 
            # la adatteremo alla dimensione del batch.
            # Se V_true è fissa per tutti (es. ontologia globale), espandiamola:
            batch_V_true = V_gt_matrix.unsqueeze(0).expand(batch_h.size(0), -1, -1).to(device)
            
            # Eseguiamo lo step di addestramento
            loss, l_task, l_concept, l_box = train_step(
                model, optimizer, batch_h, batch_y, 
                class_concept_matrix, batch_V_true
            )
            
            # Accumuliamo le loss per calcolare la media dell'epoca
            epoch_loss += loss
            epoch_task += l_task
            epoch_concept += l_concept
            epoch_box += l_box
            
        # Calcoliamo le medie
        num_batches = len(dataloader)
        avg_loss = epoch_loss / num_batches
        avg_task = epoch_task / num_batches
        avg_concept = epoch_concept / num_batches
        avg_box = epoch_box / num_batches
        
        # Log su wandb
        wandb.log({
            "epoch": epoch + 1,
            "loss/total": avg_loss,
            "loss/task": avg_task,
            "loss/concept": avg_concept,
            "loss/box_interactions": avg_box
        })
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} | Task: {avg_task:.4f} | Concept: {avg_concept:.4f} | Box: {avg_box:.4f}")
        
    wandb.finish()