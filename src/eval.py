import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def test(model, dataloader, class_concept_matrix, hierarchy_gt, device):
    """
    Valuta il BoxEmbeddingCBM sul set di Test e plotta i risultati.
    Non richiede l'optimizer perché non aggiorniamo i pesi!
    """
    print(f"\nInizio valutazione sul set di TEST ({str(device).upper()})...")
    print("="*50)
    
    model = model.to(device)
    class_concept_matrix = class_concept_matrix.to(device)
    
    # Pesi della loss (devono essere uguali a quelli del training)
    W_TASK, W_ACT, W_HIER, W_VOL = 2.0, 1.0, 1.0, 0.0
    
    # Inizializziamo i contatori
    test_loss = 0.0
    test_task_loss = 0.0
    test_act_loss = 0.0
    test_hier_loss = 0.0
    test_vol_loss = 0.0
    
    correct_preds = 0
    total_samples = 0
    
    # Liste per salvare tutte le predizioni reali e previste (per la Confusion Matrix)
    all_true_labels = []
    all_pred_labels = []
    
    # FONDAMENTALE: Mettiamo il modello in modalità valutazione
    model.eval()
    
    # FONDAMENTALE: Spegniamo il calcolo dei gradienti per risparmiare memoria e tempo
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device).long().view(-1) - 1
            concept_labels = class_concept_matrix[labels].float()
            
            # Forward Pass
            outputs = model(features)
            
            # --- Calcolo Loss ---
            task_labels = labels.long() 
            task_loss = F.cross_entropy(outputs["task_logits"], task_labels)
            act_loss = F.binary_cross_entropy(outputs["concept_probs"], concept_labels)
            
            hier_loss = 0.0
            batch_size = features.size(0)
            for target_id, source_id, target_prob in hierarchy_gt:
                pred_prob = outputs["cond_prob_matrix"][:, target_id, source_id]
                target_tensor = torch.full((batch_size,), target_prob, dtype=torch.float32, device=device)
                hier_loss += F.binary_cross_entropy(pred_prob, target_tensor)
                
            vol_loss = 0.0
            for i in range(1, model.k): 
                vol_loss -= model.volume_op(outputs["boxes"][i]).mean()
                
            loss = (W_TASK * task_loss) + (W_ACT * act_loss) + (W_HIER * hier_loss) + (W_VOL * vol_loss)
            
            # --- Aggiornamento Statistiche ---
            test_loss += loss.item()
            test_task_loss += task_loss.item()
            test_act_loss += act_loss.item()
            test_hier_loss += hier_loss.item()
            test_vol_loss += vol_loss.item()
            
            # --- Calcolo Accuratezza ---
            preds = torch.argmax(outputs["task_logits"], dim=1)
            correct_preds += (preds == task_labels).sum().item()
            total_samples += task_labels.size(0)
            
            # Salviamo per il plot finale
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(preds.squeeze().cpu().numpy())

    # Calcoliamo le medie finali
    num_batches = len(dataloader)
    avg_loss = test_loss / num_batches
    avg_task = test_task_loss / num_batches
    avg_act = test_act_loss / num_batches
    avg_hier = test_hier_loss / num_batches
    avg_vol = test_vol_loss / num_batches
    accuracy = correct_preds / total_samples
    
    print(f"Test Completato! | Accuracy Finale: {accuracy*100:.2f}% | Loss Totale: {avg_loss:.4f}\n")
    
    # ==========================================
    # --- FASE DI PLOTTING DEI RISULTATI ---
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Grafico a Barre: Scomposizione della Loss
    loss_names = ['Task Loss', 'Act Loss', 'Hier Loss', 'Vol Loss']
    loss_values = [avg_task, avg_act, avg_hier, avg_vol]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax1.bar(loss_names, loss_values, color=colors, alpha=0.8)
    ax1.set_title('Scomposizione della Test Loss', fontsize=14)
    ax1.set_ylabel('Valore Loss', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. Matrice di Confusione FIXATA
    # Fissiamo il numero totale di classi (model.k dovrebbe essere 50)
    num_classi_totali = model.k 
    
    # Forziamo sklearn a creare una matrice 50x50, anche se alcune classi non compaiono nel test
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=np.arange(num_classi_totali))
    
    # Creiamo etichette corte (es. "C0", "C1", "C2") per risparmiare spazio visivo
    etichette_assi = [f'C{i}' for i in range(num_classi_totali)]
    
    # Passiamo le etichette direttamente a seaborn tramite xticklabels/yticklabels
    sns.heatmap(cm, annot=False, cmap='Blues', ax=ax2, cbar=True,
                xticklabels=etichette_assi, yticklabels=etichette_assi)
    
    # Ruotiamo leggermente le etichette per farle entrare meglio
    ax2.tick_params(axis='x', rotation=90, labelsize=8)
    ax2.tick_params(axis='y', rotation=0, labelsize=8)
    
    ax2.set_title(f'Matrice di Confusione (Acc: {accuracy*100:.1f}%)', fontsize=14)
    ax2.set_xlabel('Classe Predetta', fontsize=12)
    ax2.set_ylabel('Classe Reale', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'tot_loss': avg_loss,
        'task_loss': avg_task,
        'act_loss': avg_act,
        'hier_loss': avg_hier
    }