import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.CHM.model import calcola_matrice_probabilita

def explain_prediction(model, test_dataloader, concept_names, class_names, 
                       class_concept_matrix, boxes_tensor=None, prob_matrix=None, 
                       top_k=10, device="cpu", info_type='boxes'):
    """
    Spiega la predizione del modello visualizzando un grafico a barre dei contributi.
    Usa automaticamente il tipo di info ('boxes' o 'rel_matrix') definito nel modello.
    """
    model.eval()
    model.to(device)
    
    # 2. Estrazione di un sample dal test set
    features, labels = next(iter(test_dataloader))
    features, labels = features.to(device), labels.to(device)
    # Assumiamo 1-indexed -> 0-indexed
    label_idx = labels[0].item() - 1
    
    # 3. Preparazione Ground Truth dei concetti
    concept_gt = class_concept_matrix[label_idx].to(device).float()
    
    with torch.no_grad():
        # Costruiamo l'input corretto in base a cosa si aspetta il classificatore
        if info_type == 'boxes':
            if boxes_tensor is None:
                raise ValueError("Il modello richiede 'boxes_tensor' per l'interpretazione.")
            # Shape: (num_concepts, box_dim)
            scaled_input = concept_gt.unsqueeze(-1) * boxes_tensor.to(device)
            input_flat = scaled_input.view(1, -1)
        elif info_type == 'rel_matrix':
            if prob_matrix is None and boxes_tensor is not None:
                prob_matrix = calcola_matrice_probabilita(boxes_tensor.to(device))
                prob_matrix.fill_diagonal_(0.0)
            elif prob_matrix is None:
                raise ValueError("Il modello richiede 'prob_matrix' o 'boxes_tensor'.")
            
            # Shape: (num_concepts, num_concepts)
            scaled_input = concept_gt.unsqueeze(-1) * prob_matrix.to(device)
            input_flat = scaled_input.view(1, -1)
        elif info_type == 'concepts':
            scaled_input = concept_gt.unsqueeze(-1) # shape: (num_concepts, 1)
            input_flat = scaled_input.view(1, -1)
        else:
            raise ValueError(f"Tipo info '{info_type}' non riconosciuto.")

        # 4. Predizione
        # scaled_input.unsqueeze(0) aggiunge la dimensione del batch -> (1, ...)
        logits = model(scaled_input.unsqueeze(0))
        pred_idx = torch.argmax(logits, dim=1).item()
        
        # 5. Calcolo contributi (Peso * Valore Input) per la classe predetta
        weights = model.classifier.weight[pred_idx] 
        contributions = weights * input_flat.squeeze(0)
        
        plot_labels = []
        plot_values = []

        if info_type == 'boxes':
            box_dim = boxes_tensor.shape[1]
            num_concepts = len(concept_names)
            # Sommiamo i contributi di tutte le dimensioni del box per ogni concetto
            concept_contributions = contributions.view(num_concepts, box_dim).sum(dim=1)
            
            values, indices = torch.topk(concept_contributions, min(top_k, num_concepts))
            plot_labels = [concept_names[i] for i in indices]
            plot_values = values.cpu().numpy()
            title = f"Top {top_k} Contributi dei Concetti (Box)"
        elif info_type == 'rel_matrix':
            num_concepts = len(concept_names)
            # Ogni feature è una relazione P(i|j)
            top_vals, top_flat_indices = torch.topk(contributions, min(top_k, num_concepts**2))
            
            for val, flat_idx in zip(top_vals, top_flat_indices):
                i = flat_idx // num_concepts
                j = flat_idx % num_concepts
                plot_labels.append(f"P({concept_names[i]}|{concept_names[j]})")
                plot_values.append(val.item())
            title = f"Top {top_k} Contributi delle Relazioni Probabilistiche"
            plot_values = np.array(plot_values)
        elif info_type == 'concepts':
            num_concepts = len(concept_names)
            top_vals, top_indices = torch.topk(contributions, min(top_k, num_concepts))
            plot_labels = [concept_names[i] for i in top_indices]
            plot_values = top_vals.cpu().numpy()
            title = f"Top {top_k} Contributi dei Concetti (Solo Presenza)"

    # 6. Visualizzazione Grafica
    plt.figure(figsize=(10, 8))
    # Coloriamo in base al segno: verde per contributi positivi, rosso per negativi
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in plot_values]
    
    y_pos = np.arange(len(plot_labels))
    plt.barh(y_pos, plot_values, color=colors, align='center', alpha=0.8)
    plt.yticks(y_pos, plot_labels)
    plt.gca().invert_yaxis() # Metodo per avere il più importante in alto
    
    plt.axvline(0, color='black', linewidth=0.8) # Linea dello zero
    plt.xlabel('Contributo al Logit (Forza della decisione)')
    plt.title(f"{title}\nPredizione: {class_names[pred_idx]} | Reale: {class_names[label_idx]}")
    
    # Aggiunta dei valori numerici accanto alle barre
    for i, v in enumerate(plot_values):
        plt.text(v + (0.01 if v > 0 else -0.05), i, f'{v:.3f}', 
                 color='black', va='center', fontweight='bold' if abs(v) == max(abs(plot_values)) else 'normal')

    plt.tight_layout()
    plt.show()

    return pred_idx == label_idx