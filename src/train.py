import torch
import torch.nn.functional as F
from src.model import BoxEmbeddingCBM
import os
import matplotlib.pyplot as plt

def train(
        model: BoxEmbeddingCBM, 
        dataloader, 
        optimizer, 
        class_concept_matrix, 
        hierarchy_gt, 
        EPOCHS, 
        device,
        save_dir="checkpoints", 
        save_interval=50
):
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

    # Creiamo la cartella per i checkpoint se non esiste
    os.makedirs(save_dir, exist_ok=True)
    
    # Spostiamo il modello e la matrice di mappatura sul device corretto
    model = model.to(device)
    class_concept_matrix = class_concept_matrix.to(device)
    
    # Pesi per bilanciare le loss (da "tunnare" in base al dataset)
    W_TASK = 2.0      # Peso per la predizione della classe finale
    W_ACT = 1.0       # Peso per l'attivazione corretta dei concetti
    W_HIER = 1.0      # Peso per la gerarchia logica
    W_VOL = 0.1       # Peso per l'anti-collasso dei volumi

    history = {
        'tot_loss': [], 'task_loss': [], 'act_loss': [], 
        'hier_loss': [], 'vol_loss': [], 'task_acc': []
    }
    
    for epoch in range(EPOCHS):
        model.train() # Impostiamo il modello in modalità training
        
        epoch_loss = 0.0
        epoch_task_loss = 0.0
        epoch_act_loss = 0.0
        epoch_hier_loss = 0.0
        
        epoch_vol_loss = 0.0
        epoch_corrects = 0
        epoch_samples = 0
        
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
            task_labels = labels.long()
            task_loss = F.cross_entropy(outputs["task_logits"], task_labels)
            
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

            epoch_vol_loss += vol_loss.item()

            # Se la probabilità >= 0.5, la predizione è 1, altrimenti 0
            preds = torch.argmax(outputs["task_logits"], dim=1)
            epoch_corrects += (preds == task_labels).sum().item()
            epoch_samples += task_labels.size(0)

        num_batches = len(dataloader)
        history['tot_loss'].append(epoch_loss / num_batches)
        history['task_loss'].append(epoch_task_loss / num_batches)
        history['act_loss'].append(epoch_act_loss / num_batches)
        history['hier_loss'].append(epoch_hier_loss / num_batches)
        history['vol_loss'].append(epoch_vol_loss / num_batches)
        history['task_acc'].append(epoch_corrects / epoch_samples)
            
        # Logging a fine epoca (calcoliamo le medie sul numero di batch)
        num_batches = len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoca [{epoch+1:3d}/{EPOCHS}] | "
                  f"Loss Tot: {epoch_loss/num_batches:.4f} | "
                  f"Task: {epoch_task_loss/num_batches:.4f} | "
                  f"Act: {epoch_act_loss/num_batches:.4f} | "
                  f"Hier: {epoch_hier_loss/num_batches:.4f}")
        
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss / num_batches
            }, checkpoint_path)
            print(f"-> Checkpoint salvato: {checkpoint_path}")

    final_path = os.path.join(save_dir, "model_final.pt")
    torch.save(model.state_dict(), final_path)
    print("="*50)
    print(f"Addestramento completato! 🎉 Modello finale salvato in: {final_path}")
    return history


def plot_training_history(history):
    """
    Disegna due grafici: uno per le loss (totale e separate) e uno per l'accuratezza.
    """
    epochs = range(1, len(history['tot_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # --- Primo Grafico: Le Loss ---
    ax1.plot(epochs, history['tot_loss'], label='Totale (Pesata)', color='black', linewidth=2.5)
    ax1.plot(epochs, history['task_loss'], label='Task Loss', linestyle='--')
    ax1.plot(epochs, history['act_loss'], label='Activation Loss', linestyle='--')
    ax1.plot(epochs, history['hier_loss'], label='Hierarchy Loss', linestyle='--')
    ax1.plot(epochs, history['vol_loss'], label='Volume Loss', linestyle=':', alpha=0.7)
    
    ax1.set_title('Andamento delle Componenti della Loss', fontsize=14)
    ax1.set_xlabel('Epoche', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # --- Secondo Grafico: Accuratezza ---
    ax2.plot(epochs, history['task_acc'], label='Task Accuracy', color='green', linewidth=2.5)
    ax2.set_title('Accuratezza di Classificazione (Task)', fontsize=14)
    ax2.set_xlabel('Epoche', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_ylim(0, 1.05) # L'accuratezza va da 0 a 1
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()


def train_and_validate(
        model: BoxEmbeddingCBM, 
        train_dataloader, 
        val_dataloader, 
        optimizer, 
        class_concept_matrix, 
        hierarchy_gt, 
        EPOCHS, 
        device, 
        save_dir="checkpoints", 
        save_interval=10
):
    print(f"Inizio Addestramento e Validazione su: {device.upper()}")
    print("="*60)
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    class_concept_matrix = class_concept_matrix.to(device)
    
    W_TASK, W_ACT, W_HIER, W_VOL = 2.0, 1.0, 1.0, 0.1
    
    # Dizionario storico sdoppiato (Train e Val)
    history = {
        'train': {'tot_loss': [], 'task_loss': [], 'act_loss': [], 'hier_loss': [], 'acc': []},
        'val':   {'tot_loss': [], 'task_loss': [], 'act_loss': [], 'hier_loss': [], 'acc': []}
    }
    
    best_val_acc = 0.0 # Per salvare il modello migliore in assoluto
    
    for epoch in range(EPOCHS):
        # ==========================================
        #               FASE DI TRAINING
        # ==========================================
        model.train()
        train_loss, train_task, train_act, train_hier = 0.0, 0.0, 0.0, 0.0
        train_correct, train_samples = 0, 0
        
        for features, labels in train_dataloader:
            features = features.to(device)
            labels = labels.to(device).long().squeeze()
            concept_labels = class_concept_matrix[labels]
            
            optimizer.zero_grad()
            outputs = model(features)
            
            # Loss Multi-Classe
            task_loss = F.cross_entropy(outputs["task_logits"], labels)
            act_loss = F.binary_cross_entropy(outputs["concept_probs"], concept_labels)
            
            hier_loss = 0.0
            for tgt, src, prob in hierarchy_gt:
                pred_prob = outputs["cond_prob_matrix"][:, tgt, src]
                tgt_tensor = torch.full((features.size(0),), prob, dtype=torch.float32, device=device)
                hier_loss += F.binary_cross_entropy(pred_prob, tgt_tensor)
                
            vol_loss = 0.0
            for i in range(1, model.k): 
                vol_loss -= model.volume_op(outputs["boxes"][i]).mean()
                
            loss = (W_TASK * task_loss) + (W_ACT * act_loss) + (W_HIER * hier_loss) + (W_VOL * vol_loss)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_task += task_loss.item()
            train_act += act_loss.item()
            train_hier += hier_loss.item()
            
            # Accuratezza Multi-Classe (Argmax)
            preds = torch.argmax(outputs["task_logits"], dim=1)
            train_correct += (preds == labels).sum().item()
            train_samples += labels.size(0)
            
        # ==========================================
        #              FASE DI VALIDAZIONE
        # ==========================================
        model.eval()
        val_loss, val_task, val_act, val_hier = 0.0, 0.0, 0.0, 0.0
        val_correct, val_samples = 0, 0
        
        with torch.no_grad():
            for features, labels in val_dataloader:
                features = features.to(device)
                labels = labels.to(device).long().squeeze()
                concept_labels = class_concept_matrix[labels]
                
                outputs = model(features)
                
                # Loss
                task_loss = F.cross_entropy(outputs["task_logits"], labels)
                act_loss = F.binary_cross_entropy(outputs["concept_probs"], concept_labels)
                
                hier_loss = 0.0
                for tgt, src, prob in hierarchy_gt:
                    pred_prob = outputs["cond_prob_matrix"][:, tgt, src]
                    tgt_tensor = torch.full((features.size(0),), prob, dtype=torch.float32, device=device)
                    hier_loss += F.binary_cross_entropy(pred_prob, tgt_tensor)
                    
                vol_loss = 0.0
                for i in range(1, model.k): 
                    vol_loss -= model.volume_op(outputs["boxes"][i]).mean()
                    
                loss = (W_TASK * task_loss) + (W_ACT * act_loss) + (W_HIER * hier_loss) + (W_VOL * vol_loss)
                
                val_loss += loss.item()
                val_task += task_loss.item()
                val_act += act_loss.item()
                val_hier += hier_loss.item()
                
                # Accuratezza
                preds = torch.argmax(outputs["task_logits"], dim=1)
                val_correct += (preds == labels).sum().item()
                val_samples += labels.size(0)
                
        # ==========================================
        #          LOGGING E SALVATAGGIO
        # ==========================================
        # Aggiorniamo la history calcolando le medie
        t_batches = len(train_dataloader)
        v_batches = len(val_dataloader)
        
        history['train']['tot_loss'].append(train_loss / t_batches)
        history['train']['task_loss'].append(train_task / t_batches)
        history['train']['acc'].append(train_correct / train_samples)
        
        history['val']['tot_loss'].append(val_loss / v_batches)
        history['val']['task_loss'].append(val_task / v_batches)
        history['val']['acc'].append(val_correct / val_samples)
        
        # Stampa a schermo riassuntiva
        print(f"Epoca {epoch+1:3d}/{EPOCHS} | "
              f"TRAIN: Loss={history['train']['tot_loss'][-1]:.3f}, Acc={history['train']['acc'][-1]*100:.1f}% | "
              f"VAL: Loss={history['val']['tot_loss'][-1]:.3f}, Acc={history['val']['acc'][-1]*100:.1f}%")

        # Salvataggio automatico del modello migliore
        current_val_acc = history['val']['acc'][-1]
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            best_path = os.path.join(save_dir, "model_best.pt")
            torch.save(model.state_dict(), best_path)
            
        # Salvataggio checkpoint periodico
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"checkpoint_ep_{epoch+1}.pt"))

    print("="*60)
    print(f"Fine! Miglior accuratezza in validazione: {best_val_acc*100:.2f}% (Salvato come 'model_best.pt')")
    return history


def plot_train_val_history(history):
    epochs = range(1, len(history['train']['tot_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # --- Grafico 1: Loss di Train vs Validazione ---
    ax1.plot(epochs, history['train']['tot_loss'], label='Train Loss Totale', color='blue', linewidth=2)
    ax1.plot(epochs, history['val']['tot_loss'], label='Val Loss Totale', color='red', linewidth=2)
    
    # Possiamo mostrare anche solo la Task Loss tratteggiata per capire come si comporta il task principale
    ax1.plot(epochs, history['train']['task_loss'], label='Train Task Loss', color='blue', linestyle='--', alpha=0.6)
    ax1.plot(epochs, history['val']['task_loss'], label='Val Task Loss', color='red', linestyle='--', alpha=0.6)
    
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