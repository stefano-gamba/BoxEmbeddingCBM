import torch
import torch.nn as nn
import torch.optim as optim
from src.model import ConceptBoxModel
import torch.nn.functional as F

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
