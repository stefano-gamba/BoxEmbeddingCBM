import torch
from sklearn.metrics import classification_report
from src.utils.box import calcola_matrice_probabilita, apply_logical_smoothing

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
    ):
    """
    Testa il Concept Bottleneck Classifier usando la ground truth dei concetti.
    
    Argomenti:
        model: Il modello ConceptBottleneckClassifier addestrato.
        test_dataloader: Dataloader del set di test.
        class_concept_matrix: Matrice (num_classes, num_concepts) con le annotazioni GT.
        prob_matrix: Matrice (num_concepts, num_concepts) con le P(i|j) pre-calcolate dai box.
    """
    model.eval()
    model.to(device)
    class_concept_matrix = class_concept_matrix.to(device)
    

    if info == "rel_matrix" or info == "all":
        # Pre-calcolo della matrice di probabilità
        with torch.no_grad():
            prob_matrix = calcola_matrice_probabilita(boxes_tensor)
            prob_matrix = prob_matrix.to(device)
            prob_matrix.fill_diagonal_(0.0)
    
    test_correct = 0
    test_samples = 0
    
    all_preds = []
    all_labels = []
    
    print("Inizio valutazione sul Test Set...")
    
    with torch.no_grad():
        for features, labels in test_dataloader:
            features = features.to(device)
            # Assumiamo che le classi siano 1-indexed (es. 1 a 200 come nel dataset CUB), le portiamo a 0-indexed
            labels = labels.to(device).long().view(-1) - 1
            
            # 1. ORACLE TEST: Recuperiamo la Ground Truth dei concetti
            if oracle:
                concept_labels = class_concept_matrix[labels].float()
            else:
                # Se non siamo in modalità oracle, possiamo comunque testare con i concetti predetti (opzionale)
                with torch.no_grad():
                    concept_labels, _  = concept_predictor(features) # Supponendo che il modello restituisca anche i logit dei concetti

            if bipolar:
                concept_labels = concept_labels * 2 - 1
            
            
            # 2. Mascheramento (Broadcasting)
            # c_true shape: (batch_size, num_concepts, 1)
            c_true = concept_labels.unsqueeze(-1)

            if info == "boxes":
                scaled_info = c_true * boxes_tensor.unsqueeze(0)
            elif info == "rel_matrix":
                joint_activation = concept_labels.unsqueeze(2) * concept_labels.unsqueeze(1)
                scaled_info = joint_activation * prob_matrix.unsqueeze(0)
            elif info == 'concepts':
                scaled_info = c_true
            
            
            # 3. Predizione
            logits = model(scaled_info)
            preds = torch.argmax(logits, dim=1)
            
            # 4. Aggiornamento metriche
            test_correct += (preds == labels).sum().item()
            test_samples += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calcolo Accuratezza
    accuracy = (test_correct / test_samples) * 100
    print(f"\nRisultati Test Set:")
    print(f"Accuratezza Totale: {accuracy:.2f}% ({test_correct}/{test_samples})")
    
    # Restituisce anche un report più dettagliato se hai classi sbilanciate
    print("\nClassification Report (prime 10 classi):")
    # (limitiamo la stampa alle prime 10 classi per leggibilità se il dataset è molto grande)
    labels_to_print = list(range(min(10, class_concept_matrix.size(0))))
    print(classification_report(all_labels, all_preds, labels=labels_to_print, zero_division=0))
    
    return accuracy, all_preds, all_labels


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