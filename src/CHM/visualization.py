import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from box_embeddings.parameterizations import MinDeltaBoxTensor
import numpy as np
import seaborn as sns
import torch

def visualize_concept_hierarchy(model, id2concept, concept2id, concept_i, concept_j):
    """
    Estrae i parametri dei box dal modello addestrato e li visualizza in 2 modi:
    1. Intervalli 1D per tutte le dimensioni (stile Figura 3 del paper)
    2. Proiezione 2D delle prime due dimensioni
    """
    model.eval()
    
    # Recuperiamo gli ID
    idx_i = concept2id[concept_i]
    idx_j = concept2id[concept_j]
    
    with torch.no_grad():
        # Otteniamo i box passando per i layer
        theta_i = model.embeddings(torch.tensor([idx_i])).view(-1, 2, 32)
        theta_j = model.embeddings(torch.tensor([idx_j])).view(-1, 2, 32)
        
        box_parent = MinDeltaBoxTensor(theta_i)
        box_child = MinDeltaBoxTensor(theta_j)
        
        # Estraiamo le coordinate z (min) e Z (max) come indicato nel paper
        z_p = box_parent.z.squeeze().numpy()
        Z_p = box_parent.Z.squeeze().numpy()
        
        z_f = box_child.z.squeeze().numpy()
        Z_f = box_child.Z.squeeze().numpy()

    dim_totali = len(z_p)

    # Creiamo una figura con due subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ==========================================
    # GRAFICO 1: Intervalli su tutte le dimensioni (Stile Fig. 3)
    # ==========================================
    y_pos = range(dim_totali)
    
    # Disegniamo i segmenti del padre (più spessi e in arancione scuro)
    for i in range(dim_totali):
        ax1.plot([z_p[i], Z_p[i]], [i, i], color='orange', linewidth=6, alpha=0.6, label='i' if i==0 else "")
        
    # Disegniamo i segmenti del figlio (più sottili e in blu)
    for i in range(dim_totali):
        ax1.plot([z_f[i], Z_f[i]], [i, i], color='teal', linewidth=3, label='j' if i==0 else "")
        
    ax1.set_yticks(y_pos)
    ax1.set_ylabel("Dimensioni")
    ax1.set_xlabel("Coordinate")
    ax1.set_title("Intervalli (min/max) per ogni dimensione")
    ax1.legend()

    # ==========================================
    # GRAFICO 2: Proiezione 2D delle prime due dimensioni
    # ==========================================
    dim_x, dim_y = 0, 1 # Scegliamo le prime due dimensioni da proiettare
    
    # Rettangolo Padre
    width_i = Z_p[dim_x] - z_p[dim_x]
    height_i = Z_p[dim_y] - z_p[dim_y]
    rect_i = patches.Rectangle((z_p[dim_x], z_p[dim_y]), width_i, height_i, 
                                   linewidth=2, edgecolor='orange', facecolor='orange', alpha=0.3, label=concept_i)
    
    # Rettangolo Figlio
    width_j = Z_f[dim_x] - z_f[dim_x]
    height_j = Z_f[dim_y] - z_f[dim_y]
    rect_j = patches.Rectangle((z_f[dim_x], z_f[dim_y]), width_j, height_j, 
                                    linewidth=2, edgecolor='teal', facecolor='teal', alpha=0.5, label=concept_j)
    
    ax2.add_patch(rect_i)
    ax2.add_patch(rect_j)
    
    # Impostiamo i limiti del grafico in base alle coordinate
    margin_x = max(width_i, width_j) * 0.5
    margin_y = max(height_i, height_j) * 0.5
    ax2.set_xlim(min(z_p[dim_x], z_f[dim_x]) - margin_x, max(Z_p[dim_x], Z_f[dim_x]) + margin_x)
    ax2.set_ylim(min(z_p[dim_y], z_f[dim_y]) - margin_y, max(Z_p[dim_y], Z_f[dim_y]) + margin_y)
    
    ax2.set_xlabel(f"Dimensione {dim_x}")
    ax2.set_ylabel(f"Dimensione {dim_y}")
    ax2.set_title("Proiezione Box in 2D")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def visualize_heatmap_compare(model, dataset, concept2id, id2concept):
    """
    Costruisce e visualizza due heatmap: una per la Ground Truth e una per
    le predizioni del modello.
    L'asse Y (righe) rappresenta il concetto i (Padre / Contenitore).
    L'asse X (colonne) rappresenta il concetto j (Figlio / Contenuto).
    """
    model.eval()
    num_concepts = len(concept2id)
    
    # Inizializziamo le matrici N x N con zeri
    matrice_gt = np.zeros((num_concepts, num_concepts))
    matrice_pred = np.zeros((num_concepts, num_concepts))
    
    # 1. Popoliamo la matrice Ground Truth usando il dataset
    for i, j, target in dataset:
        matrice_gt[i, j] = target
        
    # 2. Popoliamo la matrice delle Predizioni interrogando il modello
    with torch.no_grad():
        for i in range(num_concepts):
            for j in range(num_concepts):
                # P(i|j) - probabilità che j sia contenuto in i
                tensor_i = torch.tensor([i], dtype=torch.long)
                tensor_j = torch.tensor([j], dtype=torch.long)
                
                prob = model(tensor_i, tensor_j).item()
                matrice_pred[i, j] = prob
                
    # Prepariamo le etichette per gli assi (in ordine di ID)
    labels = [id2concept[idx] for idx in range(num_concepts)]
    
    # 3. Creazione del plot affiancato
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Confronto Heatmap: P(i | j) - Probabilità che 'j' sia contenuto in 'i'", fontsize=16)
    
    # Heatmap Ground Truth
    sns.heatmap(matrice_gt, 
                xticklabels=labels, yticklabels=labels, 
                ax=axes[0], cmap='Blues', 
                cbar_kws={'label': 'Probabilità'})
    axes[0].set_title("Ground Truth (Dati JSON)")
    axes[0].set_xlabel("Concetto j (Figlio / Contenuto)")
    axes[0].set_ylabel("Concetto i (Padre / Contenitore)")
    
    # Heatmap Predizioni Modello
    sns.heatmap(matrice_pred, 
                xticklabels=labels, yticklabels=labels, 
                ax=axes[1], cmap='Blues', 
                cbar_kws={'label': 'Probabilità'})
    axes[1].set_title("Predizioni del Modello")
    axes[1].set_xlabel("Concetto j (Figlio / Contenuto)")
    axes[1].set_ylabel("Concetto i (Padre / Contenitore)")
    
    # Ruotiamo le etichette per renderle leggibili
    plt.setp(axes[0].get_xticklabels(), rotation=90, ha="right", fontsize=8)
    plt.setp(axes[0].get_yticklabels(), rotation=0, fontsize=8)
    plt.setp(axes[1].get_xticklabels(), rotation=90, ha="right", fontsize=8)
    plt.setp(axes[1].get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.show()


def visualize_all_boxes_2d(model, id2concept, dim_x=0, dim_y=1, figsize=(16, 12)):
    """
    Estrae tutti i box dal modello e li proietta su un piano 2D scegliendo due dimensioni.
    """
    model.eval()
    num_concepts = len(id2concept)
    
    with torch.no_grad():
        # Otteniamo i parametri di *tutti* i concetti contemporaneamente
        all_ids = torch.arange(num_concepts, dtype=torch.long)
        thetas = model.embeddings(all_ids).view(-1, 2, 32)
        
        # Convertiamo in Box
        all_boxes = MinDeltaBoxTensor(thetas)
        
        # Estraiamo le matrici (N_concetti, N_dimensioni)
        z_all = all_boxes.z.numpy()
        Z_all = all_boxes.Z.numpy()

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(f"Proiezione Globale 2D (Dimensioni: {dim_x} e {dim_y})", fontsize=16)

    # Variabili per calcolare i limiti del grafico
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # Utilizziamo una colormap per assegnare colori diversi ai box
    cmap = plt.get_cmap("tab20")

    for i in range(num_concepts):
        concept_name = id2concept[i]
        
        # Coordinate per il concetto corrente
        x_min, y_min = z_all[i, dim_x], z_all[i, dim_y]
        x_max, y_max = Z_all[i, dim_x], Z_all[i, dim_y]
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Aggiorniamo i limiti per il plot
        min_x = min(min_x, x_min)
        min_y = min(min_y, y_min)
        max_x = max(max_x, x_max)
        max_y = max(max_y, y_max)
        
        # Evidenziamo il concetto root (es. "Animal") con un colore/stile speciale
        if concept_name.lower() == "animal":
            edgecolor = 'red'
            linewidth = 4
            alpha = 0.1
            zorder = 1 # Lo mettiamo sul fondo
        else:
            edgecolor = cmap(i % 20)
            linewidth = 2
            alpha = 0.3
            zorder = 2 # Sopra il padre

        # Creiamo il rettangolo
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=linewidth, edgecolor=edgecolor, facecolor=edgecolor, 
            alpha=alpha, zorder=zorder
        )
        ax.add_patch(rect)
        
        # Aggiungiamo il testo al centro del box
        # Usiamo una dimensione del font piccola per evitare troppo disordine
        cx = x_min + width / 2.0
        cy = y_min + height / 2.0
        ax.text(cx, cy, concept_name, ha='center', va='center', 
                fontsize=8, zorder=3, color='black',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    # Aggiungiamo un po' di margine ai limiti degli assi
    margin_x = (max_x - min_x) * 0.1
    margin_y = (max_y - min_y) * 0.1
    ax.set_xlim(min_x - margin_x, max_x + margin_x)
    ax.set_ylim(min_y - margin_y, max_y + margin_y)
    
    ax.set_xlabel(f"Valori Dimensione {dim_x}")
    ax.set_ylabel(f"Valori Dimensione {dim_y}")
    
    # Aggiungiamo una griglia per facilitare la lettura
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Esempio di utilizzo:
# Puoi cambiare dim_x e dim_y (da 0 a 31) per esplorare la "forma" da diverse angolazioni
# visualizza_tutti_i_box_2d(model, id2concept, dim_x=0, dim_y=1)
# visualizza_tutti_i_box_2d(model, id2concept, dim_x=2, dim_y=3)
