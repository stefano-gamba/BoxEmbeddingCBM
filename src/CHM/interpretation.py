import torch
import numpy as np
from src.utils.box import calcola_matrice_probabilita, apply_logical_smoothing
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calculate_ablation_contributions(model, concept_input, pred_idx, device):
    """
    Calcola il contributo di ogni concetto tramite Ablation Geometrica (Leave-One-Out).
    Inverte la probabilità di un concetto alla volta (1 - p) e calcola il calo del logit.
    """
    num_concepts = concept_input.size(0)
    contributions = torch.zeros(num_concepts, device=device)
    
    with torch.no_grad():
        # 1. Calcoliamo il logit base (intatto)
        base_logits = model(concept_input.unsqueeze(0))
        base_logit = base_logits[0, pred_idx].item()
        
        # 2. Ablation Loop
        for i in range(num_concepts):
            p_ablated = concept_input.clone()
            
            # Invertiamo la probabilità: se era 0.9 -> 0.1, se era 0.0 -> 1.0
            p_ablated[i] = 1.0 - p_ablated[i] 
            
            # Calcoliamo il nuovo logit con l'informazione alterata
            ablated_logits = model(p_ablated.unsqueeze(0))
            ablated_logit = ablated_logits[0, pred_idx].item()
            
            # Il contributo è la differenza (quanto il logit è peggiorato)
            contributions[i] = base_logit - ablated_logit
            
    return contributions

def explain_prediction(
        model, 
        test_dataloader, 
        concept_names, 
        class_names, 
        class_concept_matrix, 
        boxes_tensor=None, 
        prob_matrix=None, 
        top_k=10, 
        device="cpu", 
        info_type='boxes',
        target_class=None, 
        bipolar=False, 
        concept_predictor=None,
        logical_smoothing=False,
        alpha=0.5,
    ):
    """
    Spiega la predizione del modello visualizzando un grafico a barre dei contributi.
    Supporta sia i modelli a layer lineare (pesi) che il Dynamic Box (tramite Ablation).
    """
    model.eval()
    model.to(device)
    if concept_predictor is not None:
        concept_predictor.eval()
        concept_predictor.to(device)
    
    label_idx = None
    target_features = None
    
    # ==========================================
    # 1. Ricerca nel dataloader
    # ==========================================
    if target_class is not None:
        if isinstance(target_class, str):
            if target_class in class_names:
                target_idx = class_names.index(target_class)
            else:
                raise ValueError(f"Classe '{target_class}' non trovata in class_names.")
        elif isinstance(target_class, int):
            target_idx = target_class
        else:
            raise TypeError("target_class deve essere str o int.")

        found = False
        for batch_features, batch_labels in test_dataloader:
            for i in range(len(batch_labels)):
                current_idx = batch_labels[i].item() - 1 
                
                if current_idx == target_idx:
                    label_idx = current_idx
                    target_features = batch_features[i].unsqueeze(0)
                    found = True
                    break
            if found:
                break
                
        if not found:
            raise ValueError(f"Nessun sample trovato per la classe target: {target_class}")
            
    else:
        features, labels = next(iter(test_dataloader))
        label_idx = labels[0].item() - 1
        target_features = features[0].unsqueeze(0)
    
    # ==========================================
    # 2. Ottenimento dei Concetti
    # ==========================================
    concept_gt_original = class_concept_matrix[label_idx].to(device).float()

    with torch.no_grad():
        if info_type == 'rel_matrix' or info_type == 'all' or logical_smoothing:
            if prob_matrix is None and boxes_tensor is not None:
                prob_matrix = calcola_matrice_probabilita(boxes_tensor.to(device))
                if not logical_smoothing:
                    prob_matrix.fill_diagonal_(0.0)
                    
        if concept_predictor is not None:
            c_probs, _ = concept_predictor(target_features.to(device))
            concept_base = c_probs.squeeze(0) 
        else:
            concept_base = concept_gt_original

        if logical_smoothing:
            concept_base = apply_logical_smoothing(concept_base, prob_matrix, alpha).squeeze(0)

        if bipolar:
            concept_input = concept_base * 2 - 1
        else:
            concept_input = concept_base
        
        # ==========================================
        # 3. Costruzione Input scalato e Predizione
        # ==========================================
        if info_type == 'dynamic_box':
            scaled_input = concept_input # Il DB accetta direttamente le probabilità
            input_flat = scaled_input.view(1, -1)
        elif info_type == 'boxes':
            if boxes_tensor is None:
                raise ValueError("Il modello richiede 'boxes_tensor'.")
            scaled_input = concept_input.unsqueeze(-1) * boxes_tensor.to(device)
            input_flat = scaled_input.view(1, -1)
        elif info_type == 'rel_matrix':
            joint_activation = concept_input.unsqueeze(1) * concept_input.unsqueeze(0)
            scaled_input = joint_activation * prob_matrix.to(device).unsqueeze(0)
            input_flat = scaled_input.view(1, -1)
        elif info_type == 'all':
            pass
        elif info_type == 'concepts':
            scaled_input = concept_input.unsqueeze(-1) 
            input_flat = scaled_input.view(1, -1)
        else:
            raise ValueError(f"Tipo info '{info_type}' non riconosciuto.")

        # Predizione Base
        logits = model(scaled_input.unsqueeze(0) if info_type != 'dynamic_box' else input_flat)
        pred_idx = torch.argmax(logits, dim=1).item()
        num_concepts = len(concept_names)

        # ========================================================
        # 4. CALCOLO CONTRIBUTI: LINEARE vs DYNAMIC BOX (ABLATION)
        # ========================================================
        if info_type == 'dynamic_box':
            target_contributions = calculate_ablation_contributions(
                model=model, 
                concept_input=concept_input, 
                pred_idx=pred_idx, 
                device=device
            )
            title = f"Top e Bottom {top_k} Contributi (Ablation Geometrica B_img)"
            
        else:
            # Calcolo Lineare Classico
            weights = model.classifier.weight[pred_idx] 
            contributions = weights * input_flat.squeeze(0)
            
            if info_type == 'boxes':
                box_dim = boxes_tensor.shape[1]
                target_contributions = contributions.view(num_concepts, box_dim).sum(dim=1)
                title = f"{top_k} Top and Bottom Concepts Contribution"
            elif info_type == 'rel_matrix':
                target_contributions = contributions
                title = f"Top and Bottom {top_k} Contributi Relazionali"
            elif info_type == 'concepts':
                target_contributions = contributions
                title = f"{top_k} Top and Bottom Concepts Contribution"

        # ==========================================
        # 5. Aggregazione ed Estrazione
        # ==========================================
        plot_labels = []
        plot_values = []

        def format_label(concept_idx, label_string):
            is_present_gt = concept_gt_original[concept_idx].item() > 0.5
            gt_text = "GT: Present" if is_present_gt else "GT: Absent"
            
            if concept_predictor is not None:
                pred_prob = concept_base[concept_idx].item()
                return f"{label_string} (Pred: {pred_prob:.2f} | {gt_text})"
            else:
                status_text = "Present" if is_present_gt else "Absent"
                return f"{label_string} ({status_text})"

        k_to_extract = min(top_k, target_contributions.size(0))
        top_vals, top_idx = torch.topk(target_contributions, k_to_extract)
        bottom_vals, bottom_idx = torch.topk(target_contributions, k_to_extract, largest=False)
        
        all_vals = torch.cat([top_vals, bottom_vals]).cpu().tolist()
        all_idx = torch.cat([top_idx, bottom_idx]).cpu().tolist()

        seen_labels = set()
        for val, idx in zip(all_vals, all_idx):
            if info_type == 'rel_matrix':
                i = idx // num_concepts
                j = idx % num_concepts
                concept_gt_idx = i
                label_str = f"P({concept_names[i]}|{concept_names[j]})"
            else:
                concept_gt_idx = idx
                label_str = concept_names[idx]
                
            if label_str not in seen_labels:
                seen_labels.add(label_str)
                plot_labels.append(format_label(concept_gt_idx, label_str))
                plot_values.append(val)

        plot_values = np.array(plot_values)

    # ==========================================
    # 6. Visualizzazione Grafica
    # ==========================================
    plt.figure(figsize=(12, 8))
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in plot_values]
    
    y_pos = np.arange(len(plot_labels))
    plt.barh(y_pos, plot_values, color=colors, align='center', alpha=0.8)
    plt.yticks(y_pos, plot_labels, fontsize=9) 
    plt.gca().invert_yaxis() 
    
    plt.axvline(0, color='black', linewidth=0.8) 
    
    mode_text = "Modalità: SEQUENTIAL (Usa probabilità predette)" if concept_predictor else "Modalità: ORACLE (Usa Ground Truth)"
    plt.xlabel('Impact on Logit' if info_type == 'dynamic_box' else 'Contribute to Logit (Weight * Feature)')
    plt.title(f"{title}\nPrediction: {class_names[pred_idx]} | Real: {class_names[label_idx]}\n")

    plt.tight_layout()
    plt.show()

    return pred_idx == label_idx


def plot_logical_violations(model_linear, class_concept_matrix, class_names, concept_names, tolerance=0.1):
    """
    Analizza i pesi del layer lineare per trovare Violazioni Logiche rispetto alla Ground Truth.
    Una violazione avviene se:
    - GT = 1 (Il concetto c'è) MA Peso < -tolerance (Il modello lo penalizza)
    - GT = 0 (Il concetto non c'è) MA Peso > tolerance (Il modello lo premia)
    """
    # Estraiamo la matrice dei pesi del layer lineare: shape (num_classes, num_concepts)
    # Assumiamo che info_type fosse 'concepts' per un confronto 1:1. 
    # Se usavi 'boxes' con dimensione > 1, dovresti sommare i pesi lungo la dimensione del box.
    weights = model_linear.classifier.weight.detach().cpu()
    c_gt = class_concept_matrix.cpu().float()
    
    num_classes, num_concepts = weights.shape
    
    # Maschere per le violazioni
    # 1. Penalizza un concetto che dovrebbe esserci (Falso Negativo Logico)
    fn_violations = (c_gt == 1.0) & (weights < -tolerance)
    
    # 2. Premia un concetto che NON dovrebbe esserci (Falso Positivo Logico)
    fp_violations = (c_gt == 0.0) & (weights > tolerance)
    
    # Calcoliamo le violazioni totali per classe
    total_violations_per_class = (fn_violations | fp_violations).sum(dim=1).numpy()
    
    # Ordinamo le classi in base al numero di violazioni
    sorted_indices = np.argsort(total_violations_per_class)[::-1] # Decrescente
    
    # Prendiamo le 15 peggiori per il grafico per non sovraffollarlo
    top_k = min(15, num_classes)
    worst_classes_idx = sorted_indices[:top_k]
    
    worst_classes_names = [class_names[i] for i in worst_classes_idx]
    fn_counts = fn_violations.sum(dim=1).numpy()[worst_classes_idx]
    fp_counts = fp_violations.sum(dim=1).numpy()[worst_classes_idx]
    
    # Stampa un esempio concreto della classe peggiore
    worst_class = worst_classes_idx[0]
    print(f"--- ANALISI DELLA CLASSE PEGGIORE: {class_names[worst_class]} ---")
    print("PRESENT Concepts (GT=1) but PENALIZED by Linear Layer (Weights < 0):")
    for c in range(num_concepts):
        if fn_violations[worst_class, c]:
            print(f"  - {concept_names[c]} (Peso: {weights[worst_class, c]:.3f})")
            
    print("\nConcetti ASSENTI (GT=0) ma PREMIATI dal Linear Layer (Pesi > 0):")
    for c in range(num_concepts):
        if fp_violations[worst_class, c]:
            print(f"  - {concept_names[c]} (Peso: {weights[worst_class, c]:.3f})")

    # --- PLOT BAR CHART ---
    x = np.arange(top_k)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, fn_counts, width, label='Penalizes true concepts (Logical FN)', color='#e74c3c')
    rects2 = ax.bar(x + width/2, fp_counts, width, label='Rewards false concepts (Logical FP)', color='#f39c12')
    
    ax.set_ylabel('Number of Logical Violations (Wrong Weights)')
    ax.set_title(f'{top_k} Classes with more Logical Violations')
    ax.set_xticks(x)
    ax.set_xticklabels(worst_classes_names, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    total_violations = total_violations_per_class.sum()
    print(f"\nViolazioni Logiche Totali del Linear Layer su tutto il dataset: {total_violations}")
    print(f"Violazioni Totali del Dynamic Box: 0 (Per definizione geometrica)")
    
    return total_violations


def explain_dynamic_box_error(
    model, 
    test_dataloader, 
    concept_names, 
    class_names, 
    class_concept_matrix, 
    concept_predictor,
    top_k=10, 
    device="cpu", 
    target_class=None
):
    """
    Trova un'inferenza vera (misclassificata) del Dynamic Box e calcola tramite Ablation 
    quali concetti predetti erroneamente dalla CNN hanno "distrutto" il volume 
    di intersezione geometrica con la classe reale.
    """
    model.eval()
    model.to(device)
    concept_predictor.eval()
    concept_predictor.to(device)
    
    found = False
    target_features, label_idx, pred_idx = None, None, None
    base_probs, base_logits = None, None
    
    # ==========================================
    # 1. Ricerca di un sample misclassificato
    # ==========================================
    for batch_features, batch_labels in test_dataloader:
        batch_features = batch_features.to(device)
        
        with torch.no_grad():
            c_probs, _ = concept_predictor(batch_features)
            # Il Dynamic Box accetta direttamente le probabilità
            logits = model(c_probs)
            preds = torch.argmax(logits, dim=1)
        
        for i in range(len(batch_labels)):
            # Adattare l'indice in base all'implementazione (spesso le label sono 1-based o 0-based)
            current_label = batch_labels[i].item() 
            current_pred = preds[i].item()
            
            # Filtro opzionale per cercare l'errore su una classe specifica (es. "giant+panda")
            if target_class is not None:
                t_idx = class_names.index(target_class) if isinstance(target_class, str) else target_class
                if current_label != t_idx:
                    continue
            
            # Se troviamo un errore, fermiamo la ricerca
            if current_label != current_pred:
                target_features = batch_features[i].unsqueeze(0)
                label_idx = current_label
                pred_idx = current_pred
                base_probs = c_probs[i].unsqueeze(0)
                base_logits = logits[i].unsqueeze(0)
                found = True
                break
        if found:
            break
            
    if not found:
        raise ValueError("Nessun sample misclassificato trovato con i criteri specificati.")
        
    print(f"Errore trovato in test: Reale='{class_names[label_idx]}' | Predetto='{class_names[pred_idx]}'")
    
    # ==========================================
    # 2. Calcolo Ablation Geometrica
    # ==========================================
    gt_concepts = class_concept_matrix[label_idx].to(device).float()
    num_concepts = len(concept_names)
    
    impact_on_true_class = []
    
    with torch.no_grad():
        base_true_logit = base_logits[0, label_idx].item()
        
        for c_idx in range(num_concepts):
            # Creiamo una copia temporanea delle probabilità
            ablated_probs = base_probs.clone()
            
            # Sostituiamo la predizione della CNN per il singolo concetto con la Ground Truth
            ablated_probs[0, c_idx] = gt_concepts[c_idx]
            
            # Ricalcoliamo il forward pass attraverso il Dynamic Box
            ablated_logits = model(ablated_probs)
            ablated_true_logit = ablated_logits[0, label_idx].item()
            
            # Delta = LogVolume(Corretto) - LogVolume(Originale Errato)
            # Se Delta è > 0, l'errore sul concetto stava "schiacciando" il volume della classe vera.
            delta_true = ablated_true_logit - base_true_logit
            impact_on_true_class.append(delta_true)
            
    # ==========================================
    # 3. Estrazione e formattazione dei risultati
    # ==========================================
    impact_on_true_class = np.array(impact_on_true_class)
    
    # Ordiniamo per mostrare i concetti che hanno causato la maggior perdita di volume
    sorted_indices = np.argsort(impact_on_true_class)[::-1]
    top_indices = sorted_indices[:top_k]
    
    plot_labels = []
    plot_values = []
    
    for idx in top_indices:
        val = impact_on_true_class[idx]
        pred_val = base_probs[0, idx].item()
        gt_val = gt_concepts[idx].item()
        
        status_gt = "Present" if gt_val > 0.5 else "Absent"
        # Mostriamo il concetto, cosa ha predetto la CNN e cosa era in realtà
        label_str = f"{concept_names[idx]} (Pred: {pred_val:.2f} | GT: {status_gt})"
        
        plot_labels.append(label_str)
        plot_values.append(val)
        
    # ==========================================
    # 4. Visualizzazione Grafica
    # ==========================================
    plt.figure(figsize=(12, 8))
    
    # Usiamo il rosso per evidenziare i "colpevoli" del collasso logico del box
    colors = ['#e74c3c' if val > 0 else '#3498db' for val in plot_values]
    
    y_pos = np.arange(len(plot_labels))
    plt.barh(y_pos, plot_values, color=colors, align='center', alpha=0.8)
    plt.yticks(y_pos, plot_labels, fontsize=10)
    plt.gca().invert_yaxis()
    
    plt.axvline(0, color='black', linewidth=0.8)
    
    plt.xlabel('Recupero del Log-Volume (Impatto dell\'errore CNN)')
    plt.title(f"Dynamic Box Error: Concetti fatali per la classe Reale\n"
              f"Predizione: {class_names[pred_idx]} | Reale: {class_names[label_idx]}\n"
              f"(Barre rosse = Volume ripristinato forzando il concetto alla GT)")
    
    plt.tight_layout()
    plt.show()
    
    return pred_idx, label_idx


def visualize_dynamic_box_error_2d(
    model, 
    test_dataloader, 
    concept_names, 
    class_names, 
    class_concept_matrix, 
    concept_predictor,
    dim_x=0, 
    dim_y=1, 
    device="cpu",
    target_class=None
):
    """
    Trova un'inferenza errata del Dynamic Box, identifica il concetto "colpevole" (tramite Ablation)
    e plotta in 2D la collisione spaziale:
    - Box della classe Vera (Verde)
    - Box della classe Errata (Rosso)
    - B_img dinamico collassato (Grigio/Tratteggiato)
    - Box del concetto fallato che ha "tirato" B_img fuori rotta (Blu)
    """
    model.eval()
    model.to(device)
    concept_predictor.eval()
    concept_predictor.to(device)
    
    found = False
    base_probs, target_features = None, None
    label_idx, pred_idx = None, None
    
    # ==========================================
    # 1. Ricerca dell'errore e calcolo ablation
    # ==========================================
    for batch_features, batch_labels in test_dataloader:
        batch_features = batch_features.to(device)
        with torch.no_grad():
            c_probs, _ = concept_predictor(batch_features)
            logits = model(c_probs)
            preds = torch.argmax(logits, dim=1)
        
        for i in range(len(batch_labels)):
            current_label = batch_labels[i].item() 
            current_pred = preds[i].item()
            
            if target_class is not None:
                t_idx = class_names.index(target_class) if isinstance(target_class, str) else target_class
                if current_label != t_idx:
                    continue
            
            if current_label != current_pred:
                target_features = batch_features[i].unsqueeze(0)
                label_idx = current_label
                pred_idx = current_pred
                base_probs = c_probs[i].unsqueeze(0)
                base_logits = logits[i].unsqueeze(0)
                found = True
                break
        if found:
            break
            
    if not found:
        raise ValueError("Nessun sample misclassificato trovato.")
        
    print(f"Plotting Error: True='{class_names[label_idx]}' | Pred='{class_names[pred_idx]}'")

    # Ablation per trovare il colpevole (il concetto errato con più impatto)
    gt_concepts = class_concept_matrix[label_idx].to(device).float()
    num_concepts = len(concept_names)
    impact_on_true_class = []
    
    with torch.no_grad():
        base_true_logit = base_logits[0, label_idx].item()
        for c_idx in range(num_concepts):
            ablated_probs = base_probs.clone()
            ablated_probs[0, c_idx] = gt_concepts[c_idx]
            ablated_logits = model(ablated_probs)
            delta = ablated_logits[0, label_idx].item() - base_true_logit
            impact_on_true_class.append(delta)
            
    impact_on_true_class = np.array(impact_on_true_class)
    culprit_idx = np.argmax(impact_on_true_class) # Il concetto che ha distrutto più volume
    culprit_name = concept_names[culprit_idx]
    
    # ==========================================
    # 2. Estrazione parametri spaziali
    # ==========================================
    # Estraiamo i box base dalla fase 1 (utilizzando model.concept_theta e model.class_theta se il modello è Dynamic Box)
    try:
        c_theta = model.concept_theta.cpu()
        cls_theta = model.class_theta.cpu()
    except AttributeError:
        # Fallback se le variabili si chiamano diversamente nel tuo wrapper
        c_theta = model.concept_embeddings.weight.detach().cpu()
        cls_theta = model.class_embeddings.weight.detach().cpu()
        
    dim = c_theta.size(-1) // 2
    M = getattr(model, 'M', 10.0) # Confini universo per B_img
    
    p = base_probs.cpu().squeeze()
    
    # Ricostruzione B_img
    z_c = c_theta[:, :dim]
    Z_c = z_c + c_theta[:, dim:]
    
    z_relaxed = z_c * p.unsqueeze(-1) + (-M) * (1 - p).unsqueeze(-1)
    Z_relaxed = Z_c * p.unsqueeze(-1) + (M) * (1 - p).unsqueeze(-1)
    
    z_img = torch.max(z_relaxed, dim=0)[0].numpy()
    Z_img = torch.min(Z_relaxed, dim=0)[0].numpy()
    
    # Estrazione Box Classe Vera
    z_true = cls_theta[label_idx, :dim].numpy()
    Z_true = z_true + cls_theta[label_idx, dim:].numpy()
    
    # Estrazione Box Classe Errata (Predetta)
    z_pred = cls_theta[pred_idx, :dim].numpy()
    Z_pred = z_pred + cls_theta[pred_idx, dim:].numpy()
    
    # Estrazione Box Concetto "Colpevole"
    z_culp = z_c[culprit_idx].numpy()
    Z_culp = Z_c[culprit_idx].numpy()

    # ==========================================
    # 3. Disegno del Plot 2D
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def add_box(z, Z, color, label, linestyle='-', alpha=0.3, zorder=2):
        x, y = z[dim_x], z[dim_y]
        w, h = Z[dim_x] - x, Z[dim_y] - y
        rect = patches.Rectangle((x, y), w, h, linewidth=2, linestyle=linestyle, 
                                 edgecolor=color, facecolor=color, alpha=alpha, zorder=zorder, label=label)
        ax.add_patch(rect)
        return x, y, x+w, y+h

    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    
    # Disegniamo i box in ordine di profondità (zorder)
    bounds = []
    bounds.append(add_box(z_true, Z_true, 'green', f"True Class: {class_names[label_idx]}", alpha=0.4, zorder=1))
    bounds.append(add_box(z_pred, Z_pred, 'red', f"Pred Class: {class_names[pred_idx]}", alpha=0.4, zorder=2))
    bounds.append(add_box(z_culp, Z_culp, 'blue', f"Culprit Concept: {culprit_name}", linestyle='-.', alpha=0.1, zorder=3))
    
    # Il B_img (che ha causato l'errore)
    # Se z > Z, c'è un collasso totale (volume negativo o zero)
    w_img = Z_img[dim_x] - z_img[dim_x]
    h_img = Z_img[dim_y] - z_img[dim_y]
    if w_img > 0 and h_img > 0:
        bounds.append(add_box(z_img, Z_img, 'gray', "B_img (Dynamic Image Box)", linestyle='--', alpha=0.5, zorder=4))
    else:
        # Se B_img è collassato in questa proiezione, lo plottiamo come un punto/linea per evidenziarlo
        x, y = z_img[dim_x], z_img[dim_y]
        w = max(w_img, 0.05) 
        h = max(h_img, 0.05)
        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='black', hatch='//', facecolor='none', 
                                 alpha=0.8, zorder=5, label="B_img (Collapsed/Fragmented)")
        ax.add_patch(rect)
        bounds.append((x, y, x+w, y+h))
        
    for b in bounds:
        min_x, min_y = min(min_x, b[0]), min(min_y, b[1])
        max_x, max_y = max(max_x, b[2]), max(max_y, b[3])
        
    margin_x = (max_x - min_x) * 0.2
    margin_y = (max_y - min_y) * 0.2
    ax.set_xlim(min_x - margin_x, max_x + margin_x)
    ax.set_ylim(min_y - margin_y, max_y + margin_y)
    
    ax.set_xlabel(f"Dimension {dim_x}")
    ax.set_ylabel(f"Dimension {dim_y}")
    ax.set_title("Visualizzazione 2D del Fallimento Geometrico (Dynamic Box)")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()