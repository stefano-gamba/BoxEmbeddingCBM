import torch
import torch.nn.functional as F
from src.model import BoxEmbeddingCBM
import os
import matplotlib.pyplot as plt
import optuna
import torch.optim as optim
from optuna.trial import Trial
from torch.utils.data import DataLoader

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
    print(f"Inizio Addestramento e Validazione su: {str(device).upper()}")
    print("="*60)
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    class_concept_matrix = class_concept_matrix.to(device)
    
    W_TASK, W_ACT, W_HIER, W_VOL = 2.0, 1.0, 1.0, 0.0
    
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
            labels = labels.to(device).long().view(-1) - 1

            concept_labels = class_concept_matrix[labels].float()
            
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
                labels = labels.to(device).long().view(-1) - 1
                concept_labels = class_concept_matrix[labels].float()
                
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


def train_and_validate_optuna(
        model, 
        train_dataloader, 
        val_dataloader, 
        optimizer, 
        class_concept_matrix, 
        hierarchy_gt, 
        EPOCHS, 
        device,
        W_TASK=2.0, W_ACT=1.0, W_HIER=1.0, W_VOL=0.0,
        trial: Trial = None,
        save_dir="checkpoints", 
        save_interval=10
):
    
    print(f"Inizio Addestramento su: {str(device).upper()} | Task:{W_TASK:.2f}, Act:{W_ACT:.2f}, Hier:{W_HIER:.2f}, Vol:{W_VOL:.2f}")
    print(f"Inizio Addestramento e Validazione su: {str(device).upper()}")
    print("="*60)
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    class_concept_matrix = class_concept_matrix.to(device)

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
            labels = labels.to(device).long().view(-1) - 1

            concept_labels = class_concept_matrix[labels].float()
            
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
                labels = labels.to(device).long().view(-1) - 1
                concept_labels = class_concept_matrix[labels].float()
                
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
        
    current_val_acc = history['val']['acc'][-1]
        
    # PRUNING
    if trial is not None:
        # Comunichiamo a Optuna l'accuratezza corrente a questa epoca
        trial.report(current_val_acc, epoch)
        # Se Optuna capisce che questa run sta andando troppo male rispetto alle altre, la taglia
        if trial.should_prune():
            print(f"Trial potato (pruned) all'epoca {epoch+1}!")
            raise optuna.exceptions.TrialPruned()

    if current_val_acc > best_val_acc:
        best_val_acc = current_val_acc
        best_path = os.path.join(save_dir, "model_best.pt")
        torch.save(model.state_dict(), best_path)

    return history


def objective(
        trial: Trial, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        class_concept_matrix, 
        hierarchy_gt, 
        device,
        save_dir: str
):
    
    w_task = trial.suggest_float("w_task", 0.5, 2.0)
    w_act  = trial.suggest_float("w_act", 1.0, 3.0)
    w_hier = trial.suggest_float("w_hier", 1.0, 2.0)
    #w_vol  = trial.suggest_float("w_vol", 0.0, 0.1)
    
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = 1e-5

    images, labels = next(iter(train_loader))
    box_dim = trial.suggest_int("box_dim", 4, 16)
    BATCH_SIZE = 256
    LATENT_DIM = images.shape[1]
    print(f"Dimensione features: {LATENT_DIM}")
    NUM_CONCEPTS = 50
    print(f"Numero di concetti: {NUM_CONCEPTS}")
    NUM_CLASSES = 50
    print(f"Numero di classi: {NUM_CLASSES}")

    model = BoxEmbeddingCBM(LATENT_DIM, NUM_CONCEPTS, box_dim, NUM_CLASSES).to(device) 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    EPOCHS_TUNING = 10
    
    history = train_and_validate_optuna(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        class_concept_matrix=class_concept_matrix,
        hierarchy_gt=hierarchy_gt,
        EPOCHS=EPOCHS_TUNING,
        device=device,
        W_TASK=w_task, 
        W_ACT=w_act, 
        W_HIER=w_hier, 
        #W_VOL=w_vol,
        trial=trial,          
        save_dir=f"{save_dir}{trial.number}"
    )
    
    # 4. Optuna deve sapere qual è il valore finale da massimizzare
    best_acc = max(history['val']['acc'])
    return best_acc