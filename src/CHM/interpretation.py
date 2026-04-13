import torch
import numpy as np
from src.CHM.model import calcola_matrice_probabilita
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def explain_prediction(model, test_dataloader, concept_names, class_names, 
                       class_concept_matrix, boxes_tensor=None, prob_matrix=None, 
                       top_k=10, device="cpu", info_type='boxes',
                       target_class=None, bipolar=False):
    """
    Spiega la predizione del modello visualizzando un grafico a barre dei contributi,
    evidenziando esplicitamente se il concetto base era presente o assente nell'input.
    """
    model.eval()
    model.to(device)
    
    # 1. Ricerca nel dataloader
    label_idx = None
    
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
                    found = True
                    break
            if found:
                break
                
        if not found:
            raise ValueError(f"Nessun sample trovato per la classe target: {target_class}")
            
    else:
        features, labels = next(iter(test_dataloader))
        label_idx = labels[0].item() - 1
    
    
    # 2. Preparazione Ground Truth dei concetti
    # Salviamo la GT binaria originale (0 o 1) per usarla nelle etichette
    concept_gt_original = class_concept_matrix[label_idx].to(device).float()
    
    # Applichiamo la trasformazione per l'input del modello
    if bipolar:
        concept_gt = concept_gt_original * 2 - 1
    else:
        concept_gt = concept_gt_original
    
    with torch.no_grad():
        # 3. Costruzione Input
        if info_type == 'boxes':
            if boxes_tensor is None:
                raise ValueError("Il modello richiede 'boxes_tensor'.")
            scaled_input = concept_gt.unsqueeze(-1) * boxes_tensor.to(device)
            input_flat = scaled_input.view(1, -1)
        elif info_type == 'rel_matrix':
            if prob_matrix is None and boxes_tensor is not None:
                prob_matrix = calcola_matrice_probabilita(boxes_tensor.to(device))
                prob_matrix.fill_diagonal_(0.0)
            elif prob_matrix is None:
                raise ValueError("Il modello richiede 'prob_matrix' o 'boxes_tensor'.")
            scaled_input = concept_gt.unsqueeze(-1) * prob_matrix.to(device)
            input_flat = scaled_input.view(1, -1)
        elif info_type == 'concepts':
            scaled_input = concept_gt.unsqueeze(-1) 
            input_flat = scaled_input.view(1, -1)
        else:
            raise ValueError(f"Tipo info '{info_type}' non riconosciuto.")

        # 4. Predizione
        logits = model(scaled_input.unsqueeze(0))
        pred_idx = torch.argmax(logits, dim=1).item()
        
        # 5. Calcolo contributi
        weights = model.classifier.weight[pred_idx] 
        contributions = weights * input_flat.squeeze(0)
        
        plot_labels = []
        plot_values = []

        # Funzione di supporto per formattare il nome del concetto con lo stato (Presente/Assente)
        def format_label(concept_idx, label_string):
            is_present = concept_gt_original[concept_idx].item() > 0.5
            status_text = "(Presente)" if is_present else "(Assente)"
            return f"{label_string} {status_text}"

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
                # Il concetto che maschera l'input è "i" (la riga della matrice)
                label_str = f"P({concept_names[i]}|{concept_names[j]})"
                plot_labels.append(format_label(i, label_str))
                plot_values.append(val.item())
                
            for val, flat_idx in zip(bottom_vals, bottom_flat_indices):
                i = flat_idx // num_concepts
                j = flat_idx % num_concepts
                label_str = f"P({concept_names[i]}|{concept_names[j]})"
                plot_labels.append(format_label(i, label_str))
                plot_values.append(val.item())
                
            title = f"Top {top_k} e Bottom {top_k} Contributi delle Relazioni Probabilistiche"
            plot_values = np.array(plot_values)
            
        elif info_type == 'concepts':
            num_concepts = len(concept_names)
            top_vals, top_indices = torch.topk(contributions, min(top_k, num_concepts))
            for i in top_indices:
                plot_labels.append(format_label(i, concept_names[i]))
            plot_values = top_vals.cpu().numpy()
            title = f"Top {top_k} Contributi dei Concetti (Solo Presenza)"

    # 6. Visualizzazione Grafica
    plt.figure(figsize=(12, 8))
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in plot_values]
    
    y_pos = np.arange(len(plot_labels))
    plt.barh(y_pos, plot_values, color=colors, align='center', alpha=0.8)
    plt.yticks(y_pos, plot_labels, fontsize=9) # Font leggermente più piccolo per far stare il testo extra
    plt.gca().invert_yaxis() 
    
    plt.axvline(0, color='black', linewidth=0.8) 
    plt.xlabel('Contributo al Logit (Forza della decisione)')
    plt.title(f"{title}\nPredizione: {class_names[pred_idx]} | Reale: {class_names[label_idx]}")

    plt.tight_layout()
    plt.show()

    return pred_idx == label_idx


def visualizza_separabilita(
        model, 
        test_dataloader, 
        class_concept_matrix, 
        boxes_tensor, 
        device="cpu", 
        info='boxes', 
        bipolar=False,
        projection_method='tnse'
):
    model.eval()
    all_inputs = []
    all_labels = []
    model.to(device)
    class_concept_matrix = class_concept_matrix.to(device)
    boxes_tensor = boxes_tensor.to(device)
    
    with torch.no_grad():
        for _, labels in test_dataloader:
            # Creiamo l'input (scaled boxes)
            labels_0idx = labels - 1
            concept_gt = class_concept_matrix[labels_0idx]

            if bipolar:
                concept_gt = concept_gt * 2 - 1
            
            if info == 'boxes':
                scaled_input = concept_gt.unsqueeze(-1) * boxes_tensor.unsqueeze(0)
            elif info == 'rel_matrix':
                with torch.no_grad():
                    prob_matrix = calcola_matrice_probabilita(boxes_tensor)
                    prob_matrix.fill_diagonal_(0.0)
                scaled_input = concept_gt.unsqueeze(-1) * prob_matrix.unsqueeze(0)
            elif info == 'concepts':
                scaled_input = concept_gt.unsqueeze(-1)
            else:
                raise ValueError(f"Tipo info '{info}' non riconosciuto per la visualizzazione.")
            input_flat = scaled_input.view(len(labels), -1)
            
            all_inputs.append(input_flat.cpu())
            all_labels.append(labels_0idx.cpu())
            
    X = torch.cat(all_inputs).numpy()
    y = torch.cat(all_labels).numpy()
    
    if projection_method == 'tsne':
        proj = TSNE(n_components=2, random_state=42)
    elif projection_method == 'pca':
        proj = PCA(n_components=2, random_state=42)

    X_2d = proj.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab20', alpha=0.6)
    plt.title(f"Visualizzazione {projection_method} degli input al Classificatore")
    plt.colorbar(scatter)
    plt.show()

def shuffle_test(model, test_dataloader, class_concept_matrix, boxes_tensor, prob_matrix=None, info='boxes', bipolar=False):
    model.eval()
    device = next(model.parameters()).device

    if info == 'boxes':
        if boxes_tensor is None:
            raise ValueError("Il modello richiede 'boxes_tensor' per il shuffle test.")
        boxes_tensor = boxes_tensor.to(device)
    elif info == 'rel_matrix':
        if prob_matrix is None and boxes_tensor is not None:
            prob_matrix = calcola_matrice_probabilita(boxes_tensor.to(device))
            prob_matrix.fill_diagonal_(0.0)
        elif prob_matrix is None:
            raise ValueError("Il modello richiede 'prob_matrix' o 'boxes_tensor' per il shuffle test.")
        prob_matrix = prob_matrix.to(device)
    elif info == 'concepts':
        pass
    else:
        raise ValueError(f"Tipo info '{info}' non riconosciuto per il shuffle test.")
    
    all_preds = []
    all_shuffled_labels = []
    
    with torch.no_grad():
        for _, labels in test_dataloader:
            labels_0idx = labels.long() - 1
            
            # 1. INPUT ORIGINALE: Concetti corretti per l'animale reale
            concept_gt = class_concept_matrix[labels_0idx].to(device)

            if bipolar:
                concept_gt = concept_gt * 2 - 1
            

            if info == 'boxes':
                scaled_input = concept_gt.unsqueeze(-1) * boxes_tensor.to(device).unsqueeze(0)
            elif info == 'rel_matrix':
                scaled_input = concept_gt.unsqueeze(-1) * prob_matrix.to(device).unsqueeze(0)
            elif info == 'concepts':
                scaled_input = concept_gt.unsqueeze(-1).float()
            else:
                raise ValueError(f"Tipo info '{info}' non riconosciuto per il shuffle test.")
            
            input_flat = scaled_input.view(len(labels), -1)
            
            # 2. PREDIZIONE: Il modello riceve i concetti corretti
            logits = model(input_flat)
            preds = torch.argmax(logits, dim=1).cpu()
            
            # 3. SHUFFLE DELLE ETICHETTE: Rompiamo il legame
            # Creiamo un ordine casuale per le etichette di questo batch
            shuffled_labels = labels_0idx[torch.randperm(len(labels_0idx))].cpu()
            
            all_preds.append(preds)
            all_shuffled_labels.append(shuffled_labels)
            
    all_preds = torch.cat(all_preds).numpy()
    all_shuffled_labels = torch.cat(all_shuffled_labels).numpy()
    
    # 4. CALCOLO ACCURACY
    # Qui l'accuratezza DEVE crollare vicino alla probabilità casuale
    accuracy = np.mean(all_preds == all_shuffled_labels)
    num_classes = class_concept_matrix.shape[0]
    chance_level = 1.0 / num_classes
    
    print(f"--- RISULTATI SHUFFLE TEST ---")
    print(f"Accuratezza Shufflata: {accuracy:.4f}")
    print(f"Livello del caso (1/{num_classes}): {chance_level:.4f}")
        
    return accuracy