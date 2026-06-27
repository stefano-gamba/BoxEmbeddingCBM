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
        theta_i = model.embeddings(torch.tensor([idx_i])).view(-1, 2, model.dim)
        theta_j = model.embeddings(torch.tensor([idx_j])).view(-1, 2, model.dim)
        
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
        thetas = model.embeddings(all_ids).view(-1, 2, model.dim)
        
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


def visualize_box_pair(model, name_i, name_j, dict2id_i, dict2id_j, type_i='concept', type_j='concept'):
    """
    Visualizza la relazione tra due box qualsiasi (Concetto-Concetto, Classe-Concetto, ecc.)
    type_i e type_j devono essere 'concept' o 'class'.
    """
    model.eval()
    
    idx_i = dict2id_i[name_i]
    idx_j = dict2id_j[name_j]
    
    def get_box(idx, type_flag):
        emb_layer = model.concept_embeddings if type_flag == 'concept' else model.class_embeddings
        theta = emb_layer(torch.tensor([idx])).view(-1, 2, model.dim)
        return MinDeltaBoxTensor(theta)
    
    with torch.no_grad():
        box_parent = get_box(idx_i, type_i)
        box_child = get_box(idx_j, type_j)
        
        z_p, Z_p = box_parent.z.squeeze().numpy(), box_parent.Z.squeeze().numpy()
        z_f, Z_f = box_child.z.squeeze().numpy(), box_child.Z.squeeze().numpy()

    dim_totali = len(z_p)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- GRAFICO 1: Intervalli 1D ---
    y_pos = range(dim_totali)
    for i in range(dim_totali):
        ax1.plot([z_p[i], Z_p[i]], [i, i], color='orange', linewidth=6, alpha=0.6, label=name_i if i==0 else "")
        ax1.plot([z_f[i], Z_f[i]], [i, i], color='teal', linewidth=3, label=name_j if i==0 else "")
        
    ax1.set_yticks(y_pos)
    ax1.set_ylabel("Dimensioni")
    ax1.set_xlabel("Coordinate")
    ax1.set_title(f"Intervalli 1D: {name_i} ({type_i}) vs {name_j} ({type_j})")
    ax1.legend()

    # --- GRAFICO 2: Proiezione 2D ---
    dim_x, dim_y = 0, 1 
    
    width_i, height_i = Z_p[dim_x] - z_p[dim_x], Z_p[dim_y] - z_p[dim_y]
    rect_i = patches.Rectangle((z_p[dim_x], z_p[dim_y]), width_i, height_i, 
                               linewidth=2, edgecolor='orange', facecolor='orange', alpha=0.3, label=name_i)
    
    width_j, height_j = Z_f[dim_x] - z_f[dim_x], Z_f[dim_y] - z_f[dim_y]
    rect_j = patches.Rectangle((z_f[dim_x], z_f[dim_y]), width_j, height_j, 
                               linewidth=2, edgecolor='teal', facecolor='teal', alpha=0.5, label=name_j)
    
    ax2.add_patch(rect_i)
    ax2.add_patch(rect_j)
    
    margin_x, margin_y = max(width_i, width_j) * 0.5, max(height_i, height_j) * 0.5
    ax2.set_xlim(min(z_p[dim_x], z_f[dim_x]) - margin_x, max(Z_p[dim_x], Z_f[dim_x]) + margin_x)
    ax2.set_ylim(min(z_p[dim_y], z_f[dim_y]) - margin_y, max(Z_p[dim_y], Z_f[dim_y]) + margin_y)
    
    ax2.set_xlabel(f"Dimensione {dim_x}")
    ax2.set_ylabel(f"Dimensione {dim_y}")
    ax2.set_title("Proiezione Box in 2D")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_class_concept_heatmap(model, dataset_classes, id2concept, id2class):
    """
    Heatmap rettangolare per visualizzare le assegnazioni Classe (X) -> Concetto (Y).
    """
    model.eval()
    num_concepts = len(id2concept)
    num_classes = len(id2class)
    
    # Matrici (N_concetti, N_classi)
    matrice_gt = np.zeros((num_concepts, num_classes))
    matrice_pred = np.zeros((num_concepts, num_classes))
    
    # Popoliamo Ground Truth
    for c_idx, cls_idx, target in dataset_classes:
        matrice_gt[c_idx, cls_idx] = target
        
    # Popoliamo Predizioni Modello
    with torch.no_grad():
        for c_idx in range(num_concepts):
            for cls_idx in range(num_classes):
                t_concept = torch.tensor([c_idx], dtype=torch.long)
                t_class = torch.tensor([cls_idx], dtype=torch.long)
                
                # prob = Vol(Classe interseca Concetto) / Vol(Classe)
                prob = model.forward_classes(t_concept, t_class).item()
                matrice_pred[c_idx, cls_idx] = prob
                
    labels_concepts = [id2concept[idx] for idx in range(num_concepts)]
    labels_classes = [id2class[idx] for idx in range(num_classes)]
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle("Confronto Heatmap: P(Concetto | Classe) - Inclusione delle Classi", fontsize=16)
    
    sns.heatmap(matrice_gt, 
                xticklabels=labels_classes, yticklabels=labels_concepts, 
                ax=axes[0], cmap='Greens', cbar_kws={'label': 'Probabilità GT'})
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel("Classi (Figlio)")
    axes[0].set_ylabel("Concetti (Padre)")
    
    sns.heatmap(matrice_pred, 
                xticklabels=labels_classes, yticklabels=labels_concepts, 
                ax=axes[1], cmap='Greens', cbar_kws={'label': 'Probabilità Predetta'})
    axes[1].set_title("Predizioni del Modello")
    axes[1].set_xlabel("Classi (Figlio)")
    axes[1].set_ylabel("Concetti (Padre)")
    
    # Ottimizzazione visiva per etichette fitte
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=7)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    
    plt.tight_layout()
    plt.show()

def visualize_joint_boxes_2d(model, id2concept, id2class, dim_x=0, dim_y=1, figsize=(18, 14)):
    """
    Proietta TUTTI i box (Concetti e Classi) sullo stesso piano 2D.
    Distingue visivamente le classi (solide/scure) dai concetti (tratteggiati/chiari).
    """
    model.eval()
    num_concepts = len(id2concept)
    num_classes = len(id2class)
    
    def get_all_coords(emb_layer, num_items):
        with torch.no_grad():
            all_ids = torch.arange(num_items, dtype=torch.long)
            thetas = emb_layer(all_ids).view(-1, 2, model.dim)
            boxes = MinDeltaBoxTensor(thetas)
            return boxes.z.numpy(), boxes.Z.numpy()

    z_conc, Z_conc = get_all_coords(model.concept_embeddings, num_concepts)
    z_cls, Z_cls = get_all_coords(model.class_embeddings, num_classes)

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(f"Spazio Latente Congiunto: Concetti e Classi (Dim: {dim_x}, {dim_y})", fontsize=16)

    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # Funzione helper per disegnare un set di box
    def draw_boxes(z, Z, num_items, id2name, is_class=False):
        nonlocal min_x, min_y, max_x, max_y
        cmap = plt.get_cmap("tab20b" if is_class else "tab20c")

        for i in range(num_items):
            name = id2name[i]
            x_min, y_min = z[i, dim_x], z[i, dim_y]
            x_max, y_max = Z[i, dim_x], Z[i, dim_y]
            
            width, height = x_max - x_min, y_max - y_min
            
            min_x, min_y = min(min_x, x_min), min(min_y, y_min)
            max_x, max_y = max(max_x, x_max), max(max_y, y_max)
            
            # Stile differenziato
            if is_class:
                edgecolor = 'black'
                linewidth = 2.5
                linestyle = '-'
                alpha = 0.5
                facecolor = cmap(i % 20)
                zorder = 4 # Classi in primo piano
                font_weight = 'bold'
            else:
                edgecolor = cmap(i % 20)
                linewidth = 1.5
                linestyle = '--'
                alpha = 0.15
                facecolor = 'none' # Concetti trasparenti all'interno
                zorder = 2
                font_weight = 'normal'

            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=linewidth, linestyle=linestyle, edgecolor=edgecolor, 
                facecolor=facecolor, alpha=alpha, zorder=zorder
            )
            ax.add_patch(rect)
            
            cx, cy = x_min + width / 2.0, y_min + height / 2.0
            ax.text(cx, cy, name, ha='center', va='center', 
                    fontsize=8, zorder=zorder+1, color='black', weight=font_weight,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))

    # Disegniamo prima i concetti, poi le classi sopra
    draw_boxes(z_conc, Z_conc, num_concepts, id2concept, is_class=False)
    draw_boxes(z_cls, Z_cls, num_classes, id2class, is_class=True)

    margin_x, margin_y = (max_x - min_x) * 0.1, (max_y - min_y) * 0.1
    ax.set_xlim(min_x - margin_x, max_x + margin_x)
    ax.set_ylim(min_y - margin_y, max_y + margin_y)
    
    ax.set_xlabel(f"Coordinate Dim {dim_x}")
    ax.set_ylabel(f"Coordinate Dim {dim_y}")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Legenda manuale
    import matplotlib.lines as mlines
    class_line = mlines.Line2D([], [], color='black', linewidth=2.5, label='Box Classi')
    concept_line = mlines.Line2D([], [], color='gray', linewidth=1.5, linestyle='--', label='Box Concetti')
    ax.legend(handles=[class_line, concept_line], loc='upper right')
    
    plt.tight_layout()
    plt.show()