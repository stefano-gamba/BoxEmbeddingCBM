import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

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