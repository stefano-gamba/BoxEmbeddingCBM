import torch
import numpy as np
from src.utils.box import calcola_matrice_probabilita, apply_logical_smoothing
import matplotlib.pyplot as plt

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
    Se concept_predictor è fornito (Sequential Mode), usa le predizioni dei concetti 
    e mostra la probabilità stimata confrontandola con la Ground Truth.
    Se concept_predictor è None (Oracle Mode), usa la Ground Truth.
    """
    model.eval()
    model.to(device)
    if concept_predictor is not None:
        concept_predictor.eval()
        concept_predictor.to(device)
    
    label_idx = None
    target_features = None # Novità: dobbiamo salvare le features (h) per il concept_predictor
    
    # 1. Ricerca nel dataloader
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
                    target_features = batch_features[i].unsqueeze(0) # Salviamo l'input dell'immagine
                    found = True
                    break
            if found:
                break
                
        if not found:
            raise ValueError(f"Nessun sample trovato per la classe target: {target_class}")
            
    else:
        # Se non specifichiamo la classe, prendiamo il primo elemento del batch
        features, labels = next(iter(test_dataloader))
        label_idx = labels[0].item() - 1
        target_features = features[0].unsqueeze(0)
    
    
    # 2. Ottenimento dei Concetti (Oracle vs Sequential)
    # Salviamo SEMPRE la Ground Truth per visualizzarla nei label del grafico
    concept_gt_original = class_concept_matrix[label_idx].to(device).float()

    with torch.no_grad():
        if info_type == 'rel_matrix' or info_type == 'all' or logical_smoothing:
            if prob_matrix is None and boxes_tensor is not None:
                # Assicurati di avere la tua funzione calcola_matrice_probabilita disponibile
                prob_matrix = calcola_matrice_probabilita(boxes_tensor.to(device))
                if not logical_smoothing:
                    prob_matrix.fill_diagonal_(0.0)
        if concept_predictor is not None:
            # SEQUENTIAL MODE: Estraiamo le probabilità predette dal modello h -> c
            c_probs, _ = concept_predictor(target_features.to(device))
            if logical_smoothing:
                concept_base = apply_logical_smoothing(c_probs.squeeze(0), prob_matrix, alpha).squeeze(0) # shape: (num_concepts,)
            else:
                concept_base = c_probs.squeeze(0) # shape: (num_concepts,)
        else:
            # ORACLE MODE: Usiamo la Ground Truth direttamente
            concept_base = concept_gt_original

        # Applichiamo la trasformazione bipolare se richiesta [0, 1] -> [-1, 1]
        if bipolar:
            concept_input = concept_base * 2 - 1
        else:
            concept_input = concept_base
        
        # 3. Costruzione Input scalato
        if info_type == 'boxes':
            if boxes_tensor is None:
                raise ValueError("Il modello richiede 'boxes_tensor'.")
            scaled_input = concept_input.unsqueeze(-1) * boxes_tensor.to(device)
            input_flat = scaled_input.view(1, -1)
            
        elif info_type == 'rel_matrix' or info_type == 'all' or logical_smoothing:
            scaled_input = concept_input.unsqueeze(-1) * prob_matrix.to(device)
            input_flat = scaled_input.view(1, -1)
            
        elif info_type == 'concepts':
            scaled_input = concept_input.unsqueeze(-1) 
            input_flat = scaled_input.view(1, -1)
        else:
            raise ValueError(f"Tipo info '{info_type}' non riconosciuto.")

        # 4. Predizione Finale
        logits = model(scaled_input.unsqueeze(0))
        pred_idx = torch.argmax(logits, dim=1).item()
        
        # 5. Calcolo contributi al logit (assumendo layer lineare come finale)
        weights = model.classifier.weight[pred_idx] 
        contributions = weights * input_flat.squeeze(0)
        
        plot_labels = []
        plot_values = []

        # Funzione di supporto aggiornata per mostrare predizione vs realtà
        def format_label(concept_idx, label_string):
            is_present_gt = concept_gt_original[concept_idx].item() > 0.5
            gt_text = "GT: Presente" if is_present_gt else "GT: Assente"
            
            if concept_predictor is not None:
                pred_prob = concept_base[concept_idx].item()
                return f"{label_string} (Pred: {pred_prob:.2f} | {gt_text})"
            else:
                status_text = "Presente" if is_present_gt else "Assente"
                return f"{label_string} ({status_text})"

        # --- Logica di aggregazione top_k identica alla tua versione ---
        if info_type == 'boxes':
            box_dim = boxes_tensor.shape[1]
            num_concepts = len(concept_names)
            concept_contributions = contributions.view(num_concepts, box_dim).sum(dim=1)
            
            values, indices = torch.topk(concept_contributions, min(top_k, num_concepts))
            for i in indices:
                plot_labels.append(format_label(i, concept_names[i]))
            plot_values = values.cpu().numpy()
            title = f"Top {top_k} Contributi dei Concetti (Box)"
            
        elif info_type == 'rel_matrix':
            num_concepts = len(concept_names)
            top_vals, top_flat_indices = torch.topk(contributions, min(top_k, num_concepts**2)) 
            bottom_vals, bottom_flat_indices = torch.topk(contributions, min(top_k, num_concepts**2), largest=False) 
            
            for val, flat_idx in zip(top_vals, top_flat_indices):
                i = flat_idx // num_concepts
                j = flat_idx % num_concepts
                label_str = f"P({concept_names[i]}|{concept_names[j]})"
                plot_labels.append(format_label(i, label_str))
                plot_values.append(val.item())
                
            for val, flat_idx in zip(bottom_vals, bottom_flat_indices):
                i = flat_idx // num_concepts
                j = flat_idx % num_concepts
                label_str = f"P({concept_names[i]}|{concept_names[j]})"
                plot_labels.append(format_label(i, label_str))
                plot_values.append(val.item())
                
            title = f"Top {top_k} e Bottom {top_k} Contributi Relazionali"
            plot_values = np.array(plot_values)
            
        elif info_type == 'concepts':
            num_concepts = len(concept_names)
            top_vals, top_indices = torch.topk(contributions, min(top_k, num_concepts))
            for i in top_indices:
                plot_labels.append(format_label(i, concept_names[i]))
            plot_values = top_vals.cpu().numpy()
            title = f"Top {top_k} Contributi dei Concetti (Presenza)"

    # 6. Visualizzazione Grafica
    plt.figure(figsize=(12, 8))
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in plot_values]
    
    y_pos = np.arange(len(plot_labels))
    plt.barh(y_pos, plot_values, color=colors, align='center', alpha=0.8)
    plt.yticks(y_pos, plot_labels, fontsize=9) 
    plt.gca().invert_yaxis() 
    
    plt.axvline(0, color='black', linewidth=0.8) 
    
    # Aggiungiamo un sottotitolo se siamo in modalità Sequenziale per chiarezza
    mode_text = "Modalità: SEQUENTIAL (Usa probabilità predette)" if concept_predictor else "Modalità: ORACLE (Usa Ground Truth)"
    plt.xlabel('Contributo al Logit (Forza della decisione)')
    plt.title(f"{title}\nPredizione: {class_names[pred_idx]} | Reale: {class_names[label_idx]}\n{mode_text}")

    plt.tight_layout()
    plt.show()

    return pred_idx == label_idx