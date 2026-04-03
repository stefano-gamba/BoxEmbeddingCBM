import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from box_embeddings.parameterizations import MinDeltaBoxTensor

def visualize_concept_hierarchy(model, id2concept, concept2id, parent_concept, child_concept):
    """
    Estrae i parametri dei box dal modello addestrato e li visualizza in 2 modi:
    1. Intervalli 1D per tutte le dimensioni (stile Figura 3 del paper)
    2. Proiezione 2D delle prime due dimensioni
    """
    model.eval()
    
    # Recuperiamo gli ID
    idx_parent = concept2id[parent_concept]
    idx_child = concept2id[child_concept]
    
    with torch.no_grad():
        # Otteniamo i box passando per i layer
        theta_parent = model.embeddings(torch.tensor([idx_parent])).view(-1, 2, 32)
        theta_child = model.embeddings(torch.tensor([idx_child])).view(-1, 2, 32)
        
        box_parent = MinDeltaBoxTensor(theta_parent)
        box_child = MinDeltaBoxTensor(theta_child)
        
        # Estraiamo le coordinate z (min) e Z (max) come indicato nel paper
        z_p = box_parent.z.squeeze().numpy()
        Z_p = box_parent.Z.squeeze().numpy()
        
        z_f = box_child.z.squeeze().numpy()
        Z_f = box_child.Z.squeeze().numpy()

    dim_totali = len(z_p)

    # Creiamo una figura con due subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Visualizzazione Gerarchia: {child_concept} dovrebbe essere dentro {parent_concept}")

    # ==========================================
    # GRAFICO 1: Intervalli su tutte le dimensioni (Stile Fig. 3)
    # ==========================================
    y_pos = range(dim_totali)
    
    # Disegniamo i segmenti del padre (più spessi e in arancione scuro)
    for i in range(dim_totali):
        ax1.plot([z_p[i], Z_p[i]], [i, i], color='orange', linewidth=6, alpha=0.6, label='Padre' if i==0 else "")
        
    # Disegniamo i segmenti del figlio (più sottili e in blu)
    for i in range(dim_totali):
        ax1.plot([z_f[i], Z_f[i]], [i, i], color='teal', linewidth=3, label='Figlio' if i==0 else "")
        
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
    width_p = Z_p[dim_x] - z_p[dim_x]
    height_p = Z_p[dim_y] - z_p[dim_y]
    rect_parent = patches.Rectangle((z_p[dim_x], z_p[dim_y]), width_p, height_p, 
                                   linewidth=2, edgecolor='orange', facecolor='orange', alpha=0.3, label=parent_concept)
    
    # Rettangolo Figlio
    width_c = Z_f[dim_x] - z_f[dim_x]
    height_c = Z_f[dim_y] - z_f[dim_y]
    rect_child = patches.Rectangle((z_f[dim_x], z_f[dim_y]), width_c, height_c, 
                                    linewidth=2, edgecolor='teal', facecolor='teal', alpha=0.5, label=child_concept)
    
    ax2.add_patch(rect_parent)
    ax2.add_patch(rect_child)
    
    # Impostiamo i limiti del grafico in base alle coordinate
    margin_x = max(width_p, width_c) * 0.5
    margin_y = max(height_p, height_c) * 0.5
    ax2.set_xlim(min(z_p[dim_x], z_f[dim_x]) - margin_x, max(Z_p[dim_x], Z_f[dim_x]) + margin_x)
    ax2.set_ylim(min(z_p[dim_y], z_f[dim_y]) - margin_y, max(Z_p[dim_y], Z_f[dim_y]) + margin_y)
    
    ax2.set_xlabel(f"Dimensione {dim_x}")
    ax2.set_ylabel(f"Dimensione {dim_y}")
    ax2.set_title("Proiezione Box in 2D")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
