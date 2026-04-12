import torch
from sklearn.metrics import classification_report, confusion_matrix
from src.CHM.model import calcola_matrice_probabilita
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def test_cbm_classifier(
        model, 
        test_dataloader, 
        class_concept_matrix,
        boxes_tensor,
        device="cpu",
        info="boxes",
        bipolar=False,
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
    

    if info == "rel_matrix":
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
            concept_labels = class_concept_matrix[labels].float()

            if bipolar:
                concept_labels = concept_labels * 2 - 1
            
            
            # 2. Mascheramento (Broadcasting)
            # c_true shape: (batch_size, num_concepts, 1)
            c_true = concept_labels.unsqueeze(-1)

            if info == "boxes":
                scaled_info = c_true * boxes_tensor.unsqueeze(0)
            elif info == "rel_matrix":
                scaled_info = c_true * prob_matrix.unsqueeze(0)
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


def plot_test_results(accuracy, preds, labels, class_names=None, figsize=(12, 10)):
    """
    Visualizza i risultati del test set.
    
    Argomenti:
        accuracy: Valore float dell'accuratezza (restituito da test_cbm_classifier).
        preds: Lista/Array delle predizioni.
        labels: Lista/Array delle etichette reali.
        class_names: Lista opzionale di stringhe con i nomi delle classi.
    """
    # 1. Calcolo della Confusion Matrix
    cm = confusion_matrix(labels, preds)
    # Normalizzazione per visualizzare le percentuali di correttezza per riga
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    
    # 2. Heatmap della Confusion Matrix
    # Se ci sono molte classi (es. 50+), usiamo annot=False per leggibilità
    sns.heatmap(cm_norm, 
                annot=len(cm) < 20, # Annotazioni solo se le classi sono poche
                fmt=".2f", 
                cmap="Blues", 
                xticklabels=class_names if class_names else "auto", 
                yticklabels=class_names if class_names else "auto")
    
    plt.title(f"Confusion Matrix Normalizzata (Accuratezza Totale: {accuracy:.2f}%)", fontsize=15)
    plt.ylabel('Classe Reale (Ground Truth)')
    plt.xlabel('Classe Predetta')
    
    # Se i nomi delle classi sono lunghi, ruotiamo le etichette
    if class_names:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()

    # 3. Visualizzazione degli errori principali (Top Misclassifications)
    if class_names:
        print("\nAnalisi degli Errori Principali:")
        errors = np.where(np.array(preds) != np.array(labels))[0]
        if len(errors) > 0:
            error_counts = {}
            for idx in errors:
                pair = (class_names[labels[idx]], class_names[preds[idx]])
                error_counts[pair] = error_counts.get(pair, 0) + 1
            
            # Ordiniamo per frequenza di errore
            sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
            for (real, pred), count in sorted_errors[:5]: # Mostriamo i primi 5
                print(f" - {count} volte: '{real}' è stato scambiato per '{pred}'")