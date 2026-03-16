import torch
from box_embeddings.parameterizations import MinDeltaBoxTensor
from src.model import BoxEmbeddingCBM

def explain_prediction(model: BoxEmbeddingCBM, image_index, features, k, concept_names=None):
    """
    Spiega la decisione del modello per una singola immagine del batch.
    """
    if concept_names is None:
        concept_names = [f"C{i}" for i in range(k)]
        
    print(f"\n{'='*50}")
    print(f" SPIEGAZIONE DECISIONE PER L'IMMAGINE {image_index}")
    print(f"{'='*50}")
    
    with torch.no_grad():
        # 1. Ricalcoliamo il forward pass solo per questa immagine
        feat = features[image_index].unsqueeze(0) # Shape: (1, feature_dim)
        
        boxes = []
        logits = []
        for i in range(k):
            theta_i = model.projectors[i](feat)
            box_i = MinDeltaBoxTensor(theta_i.view(1, 2, model.num_dims))
            boxes.append(box_i)
            
            coords = torch.cat([box_i.z, box_i.Z], dim=-1)
            logits.append(model.prob_predictors[i](coords).squeeze(-1))
            
        logits_tensor = torch.stack(logits, dim=1)
        concept_probs = torch.sigmoid(logits_tensor)[0] # Probabilità per l'immagine
        
        # 2. Ricalcoliamo la Matrice delle Relazioni (P(Ci | Cj))
        cond_prob_matrix = torch.zeros((k, k))
        for i in range(k):
            for j in range(k):
                int_box = model.intersection_op(boxes[i], boxes[j])
                prob = torch.exp(model.volume_op(int_box) - model.volume_op(boxes[j]))
                cond_prob_matrix[i, j] = torch.clamp(prob, 1e-6, 1.0 - 1e-6)[0]
                
        # 3. Ricalcoliamo i Box Scalati
        scaled_coords_list = []
        for i in range(k):
            p = concept_probs[i]
            z_scaled = boxes[i].z[0] * p
            Z_scaled = boxes[i].Z[0] * p
            scaled_coords_list.append(torch.cat([z_scaled, Z_scaled], dim=-1))
            
        flat_scaled_boxes = torch.cat(scaled_coords_list, dim=-1).unsqueeze(0)
        flat_relation_matrix = cond_prob_matrix.view(1, k * k)
        
        # 4. Predizione Finale
        logit_boxes = model.clf_boxes(flat_scaled_boxes)
        logit_rels = model.clf_relations(flat_relation_matrix)
        final_prob = torch.sigmoid(logit_boxes + logit_rels).item()
        
        print(f"PREDIZIONE TASK FINALE: {final_prob:.4f} (1.0 = Positivo, 0.0 = Negativo)\n")
        
        # --- ESTRAZIONE DEI CONTRIBUTI ---
        
        print("--- 1. CONTRIBUTI DEI SINGOLI CONCETTI (Presenza & Geometria) ---")
        box_weights = model.clf_boxes.weight[0] # Shape: (k * 4)
        concept_contributions = []
        
        for i in range(k):
            # Estraiamo le 4 coordinate scalate del concetto e i 4 pesi associati
            coords_i = scaled_coords_list[i]
            weights_i = box_weights[i*4 : (i+1)*4]
            
            # Il contributo è il prodotto scalare: sum(coordinata * peso)
            contrib = torch.dot(coords_i, weights_i).item()
            concept_contributions.append((concept_names[i], contrib, concept_probs[i].item()))
            
        # Ordiniamo dal contributo più positivo a quello più negativo
        concept_contributions.sort(key=lambda x: x[1], reverse=True)
        for name, contrib, prob in concept_contributions:
            segno = "+" if contrib > 0 else ""
            print(f"{name:5} | Attivazione: {prob:.2f} | Contributo al Task: {segno}{contrib:.4f}")
            
            
        print("\n--- 2. CONTRIBUTI DELLE RELAZIONI LOGICHE (P(Target | Source)) ---")
        rel_weights = model.clf_relations.weight[0] # Shape: (k * k)
        relation_contributions = []
        
        for i in range(k):
            for j in range(k):
                if i == j: continue # Ignoriamo la relazione di un concetto con se stesso (sempre 1.0)
                
                idx = i * k + j
                weight = rel_weights[idx].item()
                prob = cond_prob_matrix[i, j].item()
                
                contrib = prob * weight
                # Filtriamo solo le relazioni che hanno una probabilità e un peso rilevanti
                if abs(contrib) > 0.01: 
                    relation_contributions.append((concept_names[i], concept_names[j], contrib, prob))
                    
        # Ordiniamo per contributo
        relation_contributions.sort(key=lambda x: x[2], reverse=True)
        for target, source, contrib, prob in relation_contributions:
            segno = "+" if contrib > 0 else ""
            print(f"P({target} | {source}) = {prob:.2f} | Contributo al Task: {segno}{contrib:.4f}")