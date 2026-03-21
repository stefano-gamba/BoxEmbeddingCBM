import torch
from box_embeddings.parameterizations import MinDeltaBoxTensor
from src.model import BoxEmbeddingCBM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


def explain_prediction(model: BoxEmbeddingCBM, image_index, features, k, concept_names=None, target_class=None):
    """
    Spiega la decisione del modello per una singola immagine, 
    adattata per la classificazione multiclasse.
    """
    if concept_names is None:
        concept_names = [f"C{i}" for i in range(k)]
        
    print(f"\n{'='*50}")
    print(f" SPIEGAZIONE DECISIONE PER L'IMMAGINE {image_index}")
    print(f"{'='*50}")
    
    with torch.no_grad():
        device = next(model.parameters()).device
        
        # 1. Forward pass concepts
        feat = features[image_index].unsqueeze(0).to(device)
        
        boxes = []
        logits = []
        for i in range(k):
            theta_i = model.projectors[i](feat)
            box_i = MinDeltaBoxTensor(theta_i.view(1, 2, model.num_dims))
            boxes.append(box_i)
            
            coords = torch.cat([box_i.z, box_i.Z], dim=-1)
            logits.append(model.prob_predictors[i](coords).squeeze(-1))
            
        logits_tensor = torch.stack(logits, dim=1)
        concept_probs = torch.sigmoid(logits_tensor)[0]
        
        # 2. Matrice delle Relazioni
        cond_prob_matrix = torch.zeros((k, k), device=device)
        for i in range(k):
            for j in range(k):
                int_box = model.intersection_op(boxes[i], boxes[j])
                prob = torch.exp(model.volume_op(int_box) - model.volume_op(boxes[j]))
                cond_prob_matrix[i, j] = torch.clamp(prob, 1e-6, 1.0 - 1e-6)[0]
                
        # 3. Box Scalati
        scaled_coords_list = []
        for i in range(k):
            p = concept_probs[i]
            z_scaled = boxes[i].z[0] * p
            Z_scaled = boxes[i].Z[0] * p
            scaled_coords_list.append(torch.cat([z_scaled, Z_scaled], dim=-1))
            
        flat_scaled_boxes = torch.cat(scaled_coords_list, dim=-1).unsqueeze(0)
        flat_relation_matrix = cond_prob_matrix.view(1, k * k)
        
        # 4. Predizione Finale Multiclasse
        logit_boxes = model.clf_boxes(flat_scaled_boxes)
        logit_rels = model.clf_relations(flat_relation_matrix)
        
        # Calcoliamo le probabilità per tutte le classi (es. 50 classi)
        all_probs = torch.softmax(logit_boxes + logit_rels)[0] 
        
        # Se target_class non è specificata, spieghiamo la classe con la probabilità più alta
        if target_class is None:
            target_class = torch.argmax(all_probs).item()
            
        final_prob = all_probs[target_class].item()
        
        print(f"CLASSE ANALIZZATA: {target_class}")
        print(f"PROBABILITÀ PER LA CLASSE {target_class}: {final_prob:.4f}\n")
        
        # --- ESTRAZIONE DEI CONTRIBUTI ---
        
        print(f"--- 1. CONTRIBUTI DEI CONCETTI ALLA CLASSE {target_class} ---")
        # FIX CHIAVE: Selezioniamo solo i pesi associati alla target_class
        box_weights = model.clf_boxes.weight[target_class] 
        concept_contributions = []
        
        for i in range(k):
            coords_i = scaled_coords_list[i]
            weights_i = box_weights[i*4 : (i+1)*4]
            
            contrib = torch.dot(coords_i, weights_i).item()
            concept_contributions.append((concept_names[i], contrib, concept_probs[i].item()))
            
        concept_contributions.sort(key=lambda x: x[1], reverse=True)
        for name, contrib, prob in concept_contributions:
            segno = "+" if contrib > 0 else ""
            print(f"{name:5} | Attivazione: {prob:.2f} | Contributo: {segno}{contrib:.4f}")
            
            
        print(f"\n--- 2. CONTRIBUTI DELLE RELAZIONI ALLA CLASSE {target_class} ---")
        # FIX CHIAVE: Selezioniamo solo i pesi associati alla target_class
        rel_weights = model.clf_relations.weight[target_class] 
        relation_contributions = []
        
        for i in range(k):
            for j in range(k):
                if i == j: continue 
                
                idx = i * k + j
                weight = rel_weights[idx].item()
                prob = cond_prob_matrix[i, j].item()
                
                contrib = prob * weight
                if abs(contrib) > 0.01: 
                    relation_contributions.append((concept_names[i], concept_names[j], contrib, prob))
                    
        relation_contributions.sort(key=lambda x: x[2], reverse=True)
        for target, source, contrib, prob in relation_contributions:
            segno = "+" if contrib > 0 else ""
            print(f"P({target} | {source}) = {prob:.2f} | Contributo: {segno}{contrib:.4f}")


def visualize_ontology_box(model, dataloader, device, concept_names=None):
    """
    Estrae la gerarchia globale calcolando la media dei box generati
    sull'intero dataloader e plottando sia la matrice che i rettangoli 2D.
    """
    print("Estrazione delle geometrie latenti in corso...")
    model.eval()

    all_cond_probs = []
    all_z = []
    all_Z = []

    with torch.no_grad():
        for features, _ in dataloader:
            features = features.to(device)
            outputs = model(features)

            # Salviamo la matrice di probabilità condizionata [batch, K, K]
            all_cond_probs.append(outputs["cond_prob_matrix"].cpu())

            # Estraiamo i limiti z e Z per tutti i K box del batch
            # Shape di batch_z: [batch_size, num_concepts, num_dims]
            batch_z = torch.stack([box.z for box in outputs["boxes"]], dim=1)
            batch_Z = torch.stack([box.Z for box in outputs["boxes"]], dim=1)

            all_z.append(batch_z.cpu())
            all_Z.append(batch_Z.cpu())

    # --- 1. Calcoliamo i "Box Medi" (Gli Archetipi dei Concetti) ---
    mean_cond_prob = torch.cat(all_cond_probs, dim=0).mean(dim=0).numpy() # Shape [K, K]
    global_z = torch.cat(all_z, dim=0).mean(dim=0).numpy()                # Shape [K, 16]
    global_Z = torch.cat(all_Z, dim=0).mean(dim=0).numpy()                # Shape [K, 16]

    # Se non passiamo i nomi, usiamo C0, C1, ecc.
    if concept_names is None:
        concept_names = [f"C{i}" for i in range(model.k)]

    # ==========================================
    # --- FASE DI PLOTTING ---
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- PLOT 1: Matrice di Inclusione (Analitica) ---
    # Questa matrice deve assomigliare alla tua Ground Truth di networkx!
    sns.heatmap(mean_cond_prob, cmap='viridis', ax=ax1,
                xticklabels=concept_names, yticklabels=concept_names)
    ax1.set_title("Matrice di Inclusione P(Target | Source)", fontsize=14)
    ax1.set_xlabel("Concetto Source (Contenuto)")
    ax1.set_ylabel("Concetto Target (Contenitore)")

    # --- PLOT 2: Sezione 2D dei Box (Spaziale) ---
    # Trovare le 2 dimensioni con più varianza per un bel grafico
    centri = (global_z + global_Z) / 2
    varianze = np.var(centri, axis=0)
    dim_x, dim_y = np.argsort(varianze)[-2:] # Prende gli indici delle 2 dim maggiori

    ax2.set_title(f"Spazio Latente: Sezione 2D (Dimensioni {dim_x} e {dim_y})", fontsize=14)

    cmap = plt.get_cmap('tab20') # Colori per i box

    for i in range(model.k):
        # Calcoliamo larghezza, altezza e punto in basso a sinistra nel piano 2D scelto
        width = global_Z[i, dim_x] - global_z[i, dim_x]
        height = global_Z[i, dim_y] - global_z[i, dim_y]
        x_min = global_z[i, dim_x]
        y_min = global_z[i, dim_y]

        # Disegniamo il rettangolo
        rect = patches.Rectangle((x_min, y_min), width, height,
                                 linewidth=2, edgecolor=cmap(i % 20),
                                 facecolor=cmap(i % 20), alpha=0.3)
        ax2.add_patch(rect)

        # Aggiungiamo il nome del concetto al centro del box
        ax2.text(x_min + width/2, y_min + height/2, concept_names[i],
                 ha='center', va='center', fontsize=9, fontweight='bold', color='#333333')

    # Adattiamo gli assi per far rientrare tutti i box
    ax2.autoscale_view()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def heatmap_hierarchy_ground_truth(hierarchy_gt, k_concepts, concept_names):
    """
    Crea e plotta la matrice di inclusione direttamente dalla lista di supervisione.
    """
    # 1. Creiamo una matrice K x K riempita di "Vuoti" (NaN)
    # Così distingueremo le regole esplicite (1.0 o 0.0) dalle coppie senza regole
    gt_matrix = np.full((k_concepts, k_concepts), np.nan)

    # 2. Popoliamo la matrice con le tue tuple
    for target_id, source_id, prob in hierarchy_gt:
        gt_matrix[target_id, source_id] = prob

    # 3. Plot della Heatmap
    plt.figure(figsize=(14, 12))

    # Usiamo una colormap che metta in risalto l'1.0 e lo 0.0
    # I valori NaN verranno colorati di un grigio chiaro di sfondo
    cmap = sns.color_palette("viridis", as_cmap=True)
    cmap.set_bad(color='lightgray')

    ax = sns.heatmap(gt_matrix, cmap=cmap, vmin=0.0, vmax=1.0,
                     xticklabels=concept_names, yticklabels=concept_names,
                     cbar_kws={'label': 'Probabilità (1.0 = Contiene, 0.0 = Disgiunto)'})

    ax.set_title("La Vera Ground Truth (Regole Passate alla Rete)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Concetto Source (Contenuto / Figlio)", fontsize=12)
    ax.set_ylabel("Concetto Target (Contenitore / Padre)", fontsize=12)

    # Ruotiamo le etichette per leggibilità
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    plt.show()