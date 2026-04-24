import torch

def test_concept_predictor(model, test_loader, incidence_matrix, criterion, device):
    """
    Testa il modulo estrattore di concetti (h -> c_logits) sul test set.
    Restituisce loss, accuratezza globale e i tensori per analisi successive.
    """
    model.eval()
    model.to(device)
    incidence_matrix = incidence_matrix.to(device)
    
    test_loss = 0.0
    test_correct = 0
    total_elements = 0
    
    # Liste per salvare le predizioni e fare analisi post-hoc (es. c_old vs c_new)
    all_preds_probs = []
    all_preds_binary = []
    all_gts = []

    print("Inizio valutazione del Concept Predictor sul Test Set...")

    with torch.no_grad():
        for h, y in test_loader:
            # 1. Preparazione input e label
            h = h.to(device)
            y = y.to(device).long().view(-1) - 1 # Da 1-indexed a 0-indexed
            c_gt = incidence_matrix[y].float()
            
            # 2. Forward pass
            # Assumiamo che model(h) ritorni (c_probs, c_logits)
            c_probs, c_logits = model(h) 
            
            # 3. Calcolo Loss
            loss = criterion(c_logits, c_gt)
            # Moltiplichiamo per la batch_size attuale per avere la loss totale pesata
            test_loss += loss.item() * h.size(0) 
            
            # 4. Calcolo Accuratezza
            # I logit > 0 corrispondono a probabilità > 0.5
            preds_binary = (c_logits > 0).float()
            test_correct += (preds_binary == c_gt).sum().item()
            total_elements += h.size(0) * c_gt.size(1) # campioni * numero di concetti
            
            # 5. Salvataggio batch
            all_preds_probs.append(c_probs.cpu())
            all_preds_binary.append(preds_binary.cpu())
            all_gts.append(c_gt.cpu())

    # Calcolo metriche finali sull'intero dataset
    # Uso len(test_loader.dataset) al posto dei batch per una media matematicamente perfetta
    num_samples = len(test_loader.dataset)
    avg_loss = test_loss / num_samples
    avg_acc = test_correct / total_elements

    print("-" * 50)
    print(f"TEST CONCEPT PREDICTOR COMPLETATO")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Concept Accuracy (Globale): {avg_acc * 100:.2f}%")
    print("-" * 50)

    # Concateniamo i risultati in tensori unici per facilitare le analisi future
    all_preds_probs = torch.cat(all_preds_probs, dim=0)
    all_preds_binary = torch.cat(all_preds_binary, dim=0)
    all_gts = torch.cat(all_gts, dim=0)

    return avg_loss, avg_acc, all_preds_probs, all_preds_binary, all_gts