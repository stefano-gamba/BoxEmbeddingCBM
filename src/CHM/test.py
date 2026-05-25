import torch
from sklearn.metrics import classification_report
from src.utils.box import calcola_matrice_probabilita, apply_logical_smoothing
import numpy as np

from src.utils.intervention import generate_intervention_mask

def test_cbm_classifier(
        model, 
        test_dataloader, 
        class_concept_matrix,
        boxes_tensor,
        device="cpu",
        info="boxes",
        bipolar=False,
        oracle=False,
        concept_predictor=None,
        smoothing_logic=False,
        alpha=0.5, 
        ablation=False,
        intervention_strategy=None, # "random", "uncertain", "group" o None
        k_interventions=5,          
        group_indices=None
    ):
    
    model.eval()
    model.to(device)
    class_concept_matrix = class_concept_matrix.to(device)
    
    # --- FIX: GESTIONE SEPARATA DELLE MATRICI ---
    if info == "rel_matrix" or info == "all" or smoothing_logic:
        with torch.no_grad():
            base_prob_matrix = calcola_matrice_probabilita(boxes_tensor).to(device)
            
            # 1. Matrice per lo Smoothing (Diagonale INTATTA)
            smoothing_matrix = base_prob_matrix.clone()
            
            # 2. Matrice per il Modello (Diagonale AZZERATA, come in training)
            model_prob_matrix = base_prob_matrix.clone()
            model_prob_matrix.fill_diagonal_(0.0)
    
    test_correct = 0
    test_samples = 0
    all_preds = []
    all_labels = []

    all_concept_preds = []
    all_concept_probs = []
    all_concept_trues = []
    
    print("Inizio valutazione sul Test Set...")
    
    with torch.no_grad():
        for features, labels in test_dataloader:
            features = features.to(device)
            labels = labels.to(device).long().view(-1) - 1
            true_concepts_batch = class_concept_matrix[labels].float()
            
            if oracle:
                concept_labels = class_concept_matrix[labels].float()
            else:
                with torch.no_grad():
                    concept_labels, _  = concept_predictor(features)

            
            if intervention_strategy is not None:
                # 1. Genera la maschera dinamicamente per questo batch
                mask = generate_intervention_mask(
                    concept_probs=concept_labels, 
                    strategy=intervention_strategy, 
                    k=k_interventions,
                    group_indices=group_indices
                )
                
                # 2. Applica l'intervento
                # c_final = (predetto * non_intervenuti) + (reale * intervenuti)
                concept_labels = (concept_labels * (1 - mask)) + (true_concepts_batch * mask)

            if ablation:
                indices_to_keep = [i for i in range(55) if i not in [39,40,41,42,43]]
                concept_labels = concept_labels[:, indices_to_keep]
                true_concepts_batch = true_concepts_batch[:, indices_to_keep]

            # --- APPLICAZIONE DELLO SMOOTHING E BINARIZZAZIONE ---
            if smoothing_logic:
                # Usiamo smoothing_matrix (con diagonale a 1)
                concept_labels = apply_logical_smoothing(concept_labels, smoothing_matrix, alpha, ablation)
                # Binarizziamo per non sconvolgere il layer lineare
                concept_labels = (concept_labels > 0.5).float()
            
            # Assumendo che concept_labels sia continuo (es. probabilità), lo binarizziamo per l'analisi
            binary_preds = (concept_labels > 0.5).float() 
            all_concept_preds.extend(binary_preds.cpu().numpy())
            all_concept_trues.extend(true_concepts_batch.cpu().numpy())
            all_concept_probs.extend(concept_labels.cpu().numpy())

            if bipolar:
                concept_labels = concept_labels * 2 - 1
            
            c_true = concept_labels.unsqueeze(-1)

            # --- CREAZIONE DELL'INPUT (Usa model_prob_matrix!) ---
            if info == "boxes":
                scaled_info = c_true * boxes_tensor.unsqueeze(0)
            elif info == "rel_matrix":
                joint_activation = concept_labels.unsqueeze(2) * concept_labels.unsqueeze(1)
                # FIX: Usiamo model_prob_matrix (con diagonale a 0)
                scaled_info = joint_activation * model_prob_matrix.unsqueeze(0) 
            elif info == 'concepts':
                scaled_info = c_true
            
            logits = model(scaled_info)
            preds = torch.argmax(logits, dim=1)
            
            test_correct += (preds == labels).sum().item()
            test_samples += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = (test_correct / test_samples) * 100
    print(f"\nAccuratezza Totale: {accuracy:.2f}%")
    return accuracy, np.array(all_preds), np.array(all_labels), np.array(all_concept_preds), np.array(all_concept_trues), np.array(all_concept_probs)


def test_sequential_cbm(
        classifier, 
        concept_predictor, 
        test_dataloader, 
        boxes_tensor,
        class_concept_matrix=None, # Opzionale, utile solo se vuoi stampare il report o calcolare la concept accuracy
        device="cpu",
        info="boxes",
        bipolar=False,
        logical_smoothing=False,
        alpha=0.5,
):
    """
    Testa il Concept Bottleneck Classifier in modalità Sequenziale.
    Il flusso è strettamente: feature -> concept_pred -> y_pred.
    Non usa MAI la ground truth dei concetti per le predizioni finali.
    """
    
    print("Inizio valutazione sul Test Set (Modalità Sequenziale)...")
    
    # Mettiamo entrambi i modelli in modalità valutazione
    classifier.eval()
    concept_predictor.eval()
    
    classifier.to(device)
    concept_predictor.to(device)
    boxes_tensor = boxes_tensor.to(device)

    # Pre-calcolo della matrice di probabilità (se richiesta)
    if info == "rel_matrix" or info == "all" or logical_smoothing:
        with torch.no_grad():
            prob_matrix = calcola_matrice_probabilita(boxes_tensor) # Assicurati di avere questa funzione a scope
            prob_matrix = prob_matrix.to(device)
            if not logical_smoothing:
                prob_matrix.fill_diagonal_(0.0)
    
    test_correct = 0
    test_samples = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_dataloader:
            features = features.to(device)
            # Portiamo le classi a 0-indexed
            labels = labels.to(device).long().view(-1) - 1
            
            # -------------------------------------------------------------
            # STEP 1: h -> c_pred (Estrazione dei concetti predetti)
            # -------------------------------------------------------------
            c_probs, _ = concept_predictor(features)

            if logical_smoothing:
                concept_preds = apply_logical_smoothing(c_probs, prob_matrix, alpha)
            else:
                concept_preds = c_probs
            
            if bipolar:
                # Scaliamo le probabilità [0, 1] in [-1, 1]
                concept_preds = concept_preds * 2 - 1
            
            # Espandiamo per il broadcasting: (batch_size, num_concepts, 1)
            c_pred_expanded = concept_preds.unsqueeze(-1)
            
            # -------------------------------------------------------------
            # STEP 2: Mascheramento Soft (Scaling)
            # -------------------------------------------------------------
            if info == "boxes":
                scaled_info = c_pred_expanded * boxes_tensor.unsqueeze(0)
            elif info == "rel_matrix":
                joint_activation = concept_preds.unsqueeze(2) * concept_preds.unsqueeze(1)
                scaled_info = joint_activation * prob_matrix.unsqueeze(0)
            elif info == 'concepts':
                scaled_info = c_pred_expanded
            elif info == 'all':
                scaled_concepts = c_pred_expanded 
                joint_activation = concept_preds.unsqueeze(2) * concept_preds.unsqueeze(1)
                scaled_rel = joint_activation * prob_matrix.unsqueeze(0)
                scaled_info = (scaled_concepts, scaled_rel)
            
            # -------------------------------------------------------------
            # STEP 3: c_pred -> y_pred (Predizione Finale)
            # -------------------------------------------------------------
            logits = classifier(scaled_info)
            preds = torch.argmax(logits, dim=1)
            
            # Aggiornamento metriche
            test_correct += (preds == labels).sum().item()
            test_samples += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calcolo Accuratezza Totale
    accuracy = (test_correct / test_samples) * 100
    print(f"\nRisultati Test Set (Sequenziale):")
    print(f"Accuratezza Totale: {accuracy:.2f}% ({test_correct}/{test_samples})")
    
    # Classification Report (opzionale)
    if class_concept_matrix is not None:
        print("\nClassification Report (prime 10 classi):")
        labels_to_print = list(range(min(10, class_concept_matrix.size(0))))
        print(classification_report(all_labels, all_preds, labels=labels_to_print, zero_division=0))
    
    return accuracy, all_preds, all_labels