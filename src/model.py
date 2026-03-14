import torch
import torch.nn as nn
from box_embeddings.parameterizations import MinDeltaBoxTensor
from box_embeddings.modules.intersection import GumbelIntersection
from box_embeddings.modules.volume import SoftVolume

class BoxEmbeddingCBM(nn.Module):
    """
    Concept Bottleneck Model basato su Box Embeddings.
    Estrae concetti geometrici, calcola le loro relazioni logiche 
    e predice un task a valle usando sia l'attivazione che la gerarchia.
    """
    def __init__(
        self, 
        feature_dim: int, 
        num_concepts: int, 
        num_dims: int = 2, 
        num_classes: int = 1,
        vol_temp: float = 0.5,
        int_temp: float = 0.1
    ):
        super(BoxEmbeddingCBM, self).__init__()
        
        self.k = num_concepts
        self.num_dims = num_dims
        
        # --- 1. ESTRATTORI (Feature -> Geometria -> Probabilità) ---
        # Proiettano le feature nello spazio dei Box (z, Z)
        self.projectors = nn.ModuleList([
            nn.Linear(in_features=feature_dim, out_features=2 * num_dims) 
            for _ in range(self.k)
        ])
        
        # Predicono l'attivazione del concetto guardando SOLO la sua geometria
        self.prob_predictors = nn.ModuleList([
            nn.Linear(in_features=2 * num_dims, out_features=1) 
            for _ in range(self.k)
        ])
        
        # --- 2. MODULI GEOMETRICI ---
        self.intersection_op = GumbelIntersection(intersection_temperature=int_temp)
        self.volume_op = SoftVolume(volume_temperature=vol_temp)
        
        # --- 3. CLASSIFICATORI TASK FINALE ---
        # Prende i box scalati (gating): k concetti * 2 coordinate * num_dims
        self.clf_boxes = nn.Linear(in_features=self.k * 2 * self.num_dims, out_features=num_classes)
        
        # Prende la matrice delle relazioni K x K
        self.clf_relations = nn.Linear(in_features=self.k * self.k, out_features=num_classes)

    def forward(self, features):
        batch_size = features.size(0)
        
        boxes = []
        logits = []
        
        # --- FASE 1: Creazione Box e Probabilità di Attivazione ---
        for i in range(self.k):
            # A. Generiamo il box
            theta_i = self.projectors[i](features)
            box_i = MinDeltaBoxTensor(theta_i.view(batch_size, 2, self.num_dims))
            boxes.append(box_i)
            
            # B. Calcoliamo la probabilità di presenza (gating)
            coords = torch.cat([box_i.z, box_i.Z], dim=-1)
            logit_i = self.prob_predictors[i](coords).squeeze(-1)
            logits.append(logit_i)
            
        logits_tensor = torch.stack(logits, dim=1)
        concept_probs = torch.sigmoid(logits_tensor) # Shape: (batch_size, k)
        
        # --- FASE 2: Matrice delle Relazioni Gerarchiche P(C_i | C_j) ---
        matrix_rows = []
        for i in range(self.k): # Target
            row = []
            for j in range(self.k): # Source
                int_box = self.intersection_op(boxes[i], boxes[j])
                # Calcolo di P(C_i | C_j) usando i volumi
                prob = torch.exp(self.volume_op(int_box) - self.volume_op(boxes[j]))
                row.append(torch.clamp(prob, 1e-6, 1.0 - 1e-6))
            matrix_rows.append(torch.stack(row, dim=1))
            
        cond_prob_matrix = torch.stack(matrix_rows, dim=1) # Shape: (batch_size, k, k)
        
        # --- FASE 3: Gating Geometrico (Box Scalati) ---
        scaled_coords_list = []
        for i in range(self.k):
            p = concept_probs[:, i].unsqueeze(-1) # Espandiamo per il broadcasting
            z_scaled = boxes[i].z * p
            Z_scaled = boxes[i].Z * p
            scaled_coords_list.append(torch.cat([z_scaled, Z_scaled], dim=-1))
            
        flat_scaled_boxes = torch.cat(scaled_coords_list, dim=-1) # Shape: (batch_size, k * 2 * num_dims)
        
        # --- FASE 4: Classificazione Task Finale ---
        flat_relation_matrix = cond_prob_matrix.view(batch_size, self.k * self.k)
        
        task_logit_boxes = self.clf_boxes(flat_scaled_boxes)
        task_logit_rels = self.clf_relations(flat_relation_matrix)
        
        # Uniamo i due segnali (puoi anche usare pesi appresi qui in futuro)
        final_task_logit = task_logit_boxes + task_logit_rels
        final_task_prob = torch.sigmoid(final_task_logit)
        
        # Restituiamo un dizionario con tutto il necessario per Loss e Interpretabilità
        return {
            "task_probs": final_task_prob,
            "concept_probs": concept_probs,
            "cond_prob_matrix": cond_prob_matrix,
            "boxes": boxes, # Lista di MinDeltaBoxTensor (utile per la vol_loss anti-collasso)
        }