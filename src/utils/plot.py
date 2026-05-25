import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

from src.CHM.test import test_cbm_classifier

def plot_history(history):
    epochs = range(1, len(history['train']['tot_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        
    # --- Grafico 1: Loss di Train vs Validazione ---
    ax1.plot(epochs, history['train']['tot_loss'], label='Train Loss Totale', color='blue', linewidth=2)
    ax1.plot(epochs, history['val']['tot_loss'], label='Val Loss Totale', color='red', linewidth=2)
        
        
    ax1.set_title('Curve di Loss (Train vs Val)', fontsize=14)
    ax1.set_xlabel('Epoche', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
        
    # --- Grafico 2: Accuratezza Multi-Classe ---
    ax2.plot(epochs, history['train']['acc'], label='Train Accuracy', color='green', linewidth=2)
    ax2.plot(epochs, history['val']['acc'], label='Val Accuracy', color='orange', linewidth=2)
        
    ax2.set_title('Accuratezza di Classificazione', fontsize=14)
    ax2.set_xlabel('Epoche', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    plt.show()


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


def plot_concept_error_heatmap(labels, concept_preds, concept_trues, class_names=None, concept_names=None, figsize=(15, 8)):
    """
    Visualizza una Heatmap dove le righe sono le classi e le colonne i concetti.
    Il colore indica la percentuale di errore (0 = sempre giusto, 1 = sempre sbagliato).
    """
    num_classes = len(np.unique(labels))
    num_concepts = concept_preds.shape[1]
    
    # Matrice vuota per gli errori: (num_classes, num_concepts)
    error_matrix = np.zeros((num_classes, num_concepts))
    
    # Calcoliamo se la predizione del concetto è sbagliata (1 = Errore, 0 = Corretto)
    concept_errors = (concept_preds != concept_trues).astype(float)
    
    for c in range(num_classes):
        # Troviamo tutti gli esempi appartenenti alla classe 'c'
        idx = (labels == c)
        if np.sum(idx) > 0:
            # Calcoliamo l'errore medio per ogni concetto per questa classe
            error_matrix[c] = np.mean(concept_errors[idx], axis=0)
            
    plt.figure(figsize=figsize)
    sns.heatmap(error_matrix, 
                cmap="Reds",  # Rosso = più errori
                xticklabels=concept_names if concept_names else "auto", 
                yticklabels=class_names if class_names else "auto")
    
    plt.title("Tasso di Errore dei Concetti per Classe", fontsize=15)
    plt.ylabel('Classe')
    plt.xlabel('Concetti')
    
    if concept_names:
        plt.xticks(rotation=90, ha='center')
    if class_names:
        plt.yticks(rotation=0)
        
    plt.tight_layout()
    plt.show()

def analyze_misclassifications_concepts(preds, labels, concept_preds, concept_trues, class_names, concept_names, num_examples=3):
    """
    Prende esempi in cui la classe è stata predetta male e mostra quali concetti
    hanno causato l'errore.
    """
    # Trova gli indici in cui il task finale ha fallito
    error_indices = np.where(preds != labels)[0]
    
    if len(error_indices) == 0:
        print("Nessun errore! Modello perfetto.")
        return
        
    print(f"\n--- ANALISI DEI CONCETTI SUGLI ERRORI (Mostrando {min(num_examples, len(error_indices))} esempi) ---")
    
    # Prendiamo i primi N errori
    for i, idx in enumerate(error_indices[:num_examples]):
        real_class = class_names[labels[idx]] if class_names else labels[idx]
        pred_class = class_names[preds[idx]] if class_names else preds[idx]
        
        print(f"\n[Esempio Errato #{i+1} - Indice Batch: {idx}]")
        print(f"Classe Reale: '{real_class}' ---> Classe Predetta: '{pred_class}'")
        print("Concetti Sbagliati (Falsi Positivi / Falsi Negativi):")
        
        # Quali concetti ha sbagliato per questo specifico esempio?
        wrong_concepts_idx = np.where(concept_preds[idx] != concept_trues[idx])[0]
        
        if len(wrong_concepts_idx) == 0:
            print("  -> STRANO: Ha predetto tutti i concetti perfettamente, ma ha sbagliato la classe finale!")
            print("  -> Questo suggerisce un problema nel layer finale (Bottleneck non puro o pesi mal calibrati).")
        else:
            for c_idx in wrong_concepts_idx:
                c_name = concept_names[c_idx] if concept_names else f"Concetto_{c_idx}"
                c_true_val = bool(concept_trues[idx][c_idx])
                c_pred_val = bool(concept_preds[idx][c_idx])
                
                if c_true_val == True and c_pred_val == False:
                    print(f"  - [{c_name}] Falso Negativo: Doveva essere VERO, il modello ha detto FALSO.")
                elif c_true_val == False and c_pred_val == True:
                    print(f"  - [{c_name}] Falso Positivo: Doveva essere FALSO, il modello ha detto VERO.")


def plot_concept_uncertainty_heatmap(labels, concept_probs, class_names=None, concept_names=None, figsize=(15, 8)):
    """
    Visualizza una Heatmap dell'incertezza del modello.
    Un valore vicino a 1 (scuro) indica che la probabilità media assegnata 
    dal modello per quel concetto in quella classe è intorno a 0.5.
    """
    num_classes = len(np.unique(labels))
    num_concepts = concept_probs.shape[1]
    
    # Matrice vuota per l'incertezza: (num_classes, num_concepts)
    uncertainty_matrix = np.zeros((num_classes, num_concepts))
    
    # Calcoliamo lo score di incertezza per ogni singola predizione
    # Formula: 1 - 2 * |p - 0.5|
    uncertainty_scores = 1.0 - 2.0 * np.abs(concept_probs - 0.5)
    
    for c in range(num_classes):
        # Troviamo tutti gli esempi appartenenti alla classe 'c'
        idx = (labels == c)
        if np.sum(idx) > 0:
            # Calcoliamo l'incertezza media per ogni concetto per questa classe
            uncertainty_matrix[c] = np.mean(uncertainty_scores[idx], axis=0)
            
    plt.figure(figsize=figsize)
    
    # Usiamo una palette viola per distinguerla da quella rossa degli errori
    sns.heatmap(uncertainty_matrix, 
                cmap="Purples", 
                vmin=0.0, vmax=1.0, # Fissiamo la scala da 0 (Certo) a 1 (Incerto)
                xticklabels=concept_names if concept_names else "auto", 
                yticklabels=class_names if class_names else "auto")
    
    plt.title("Mappa dell'Incertezza: Concetti predetti vicini a 0.5", fontsize=15)
    plt.ylabel('Classe Reale')
    plt.xlabel('Concetti')
    
    if concept_names:
        plt.xticks(rotation=90, ha='center')
    if class_names:
        plt.yticks(rotation=0)
        
    plt.tight_layout()
    plt.show()
    
    return uncertainty_matrix

def plot_intervention_curve(
        k_values, 
        model, 
        test_loader, 
        class_concept_matrix, 
        boxes_tensor,
        device="cpu",
        info="boxes",
        concept_predictor=None
):
    results_random = []
    results_uncertain = []
    results_random_feedback = []
    results_uncertain_feedback = []

    for k in k_values:
        print(f"\n--- Valutazione con k={k} interventi ---")
        
        # Test Intervento Casuale
        acc_random, _, _, _, _, _ = test_cbm_classifier(
            model, test_loader, class_concept_matrix, boxes_tensor, device=device,
            intervention_strategy="random", k_interventions=k, info=info, concept_predictor=concept_predictor
        )
        results_random.append(acc_random)
        
        # Test Intervento Incertezza (Spesso ha performance migliori all'inizio!)
        acc_uncertain, _, _, _, _, _ = test_cbm_classifier(
            model, test_loader, class_concept_matrix, boxes_tensor, device=device,
            intervention_strategy="uncertain", k_interventions=k, info=info, concept_predictor=concept_predictor
        )
        results_uncertain.append(acc_uncertain)

        acc_random_feedback, _, _, _, _, _ = test_cbm_classifier(
            model, test_loader, class_concept_matrix, boxes_tensor, device=device,
            intervention_strategy="random", k_interventions=k, info=info, concept_predictor=concept_predictor,
            smoothing_logic=True, alpha=0.8
        )
        results_random_feedback.append(acc_random_feedback)

        acc_uncertain_feedback, _, _, _, _, _ = test_cbm_classifier(
            model, test_loader, class_concept_matrix, boxes_tensor, device=device,
            intervention_strategy="uncertain", k_interventions=k, info=info, concept_predictor=concept_predictor,
            smoothing_logic=True, alpha=0.8
        )
        results_uncertain_feedback.append(acc_uncertain_feedback)

    # Plot dei risultati
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, results_random, marker='o', label='Intervento Casuale')
    plt.plot(k_values, results_uncertain, marker='s', label='Intervento su Incertezza')
    plt.plot(k_values, results_random_feedback, marker='^', label='Intervento Casuale con Feedback')
    plt.plot(k_values, results_uncertain_feedback, marker='v', label='Intervento su Incertezza con Feedback')

    plt.title('Test Time Intervention Curve')
    plt.xlabel('Numero di Concetti Corretti (k)')
    plt.ylabel('Accuratezza del Classificatore (%)')
    plt.grid(True)
    plt.legend()
    plt.show()