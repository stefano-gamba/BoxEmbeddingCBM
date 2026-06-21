import torch
import torch.nn as nn
from box_embeddings.parameterizations import MinDeltaBoxTensor
from box_embeddings.modules.intersection import HardIntersection, GumbelIntersection
from box_embeddings.modules.volume import SoftVolume, BesselApproxVolume
from box_embeddings.modules.regularization import L2SideBoxRegularizer

# ========================================================
# TRAINING A PRIORI BOX EMBEDDING DA USARE COME CB NEL CBM
# ========================================================

class BoxHierarchyModel(nn.Module):
    def __init__(self, num_concepts, dim=32, soft_intersection=False):
        super().__init__()
        self.dim = dim # <-- Salviamo dim per usarlo nel forward
        
        # Ogni box necessita di 2*dim parametri (dim per z_min, dim per delta)
        self.embeddings = nn.Embedding(num_concepts, 2 * dim)
        
        # 1. Centri: Sfalsati per rompere la simmetria
        nn.init.uniform_(self.embeddings.weight.data[:, :dim], -0.01, 0.01)
        
        # 2. Lati: Molto grandi in modo che partano tutti sovrapposti ma scentrati
        nn.init.constant_(self.embeddings.weight.data[:, dim:], 2.0)
        
        if soft_intersection:
            self.intersection = GumbelIntersection(intersection_temperature=0.001)
        else:
            self.intersection = HardIntersection()
        #self.volume = SoftVolume(volume_temperature=0.1)
        self.volume = BesselApproxVolume(
            volume_temperature=0.1, 
            intersection_temperature=0.01 # Deve combaciare con l'intersezione
        )
        self.regularizer = L2SideBoxRegularizer(log_scale=False, weight=0.0)

    def forward(self, idx_i, idx_j):
        # 1. Recupero parametri (theta) dalle word embeddings e reshaping
        # Da (batch_size, 64) a (batch_size, 2, 32)
        theta_i = self.embeddings(idx_i).view(-1, 2, self.dim)
        theta_j = self.embeddings(idx_j).view(-1, 2, self.dim)
        
        # 2. Conversione in Box validi (assicura lati non negativi)
        box_i = MinDeltaBoxTensor(theta_i)
        box_j = MinDeltaBoxTensor(theta_j)
        
        # 3. Intersezione
        box_int = self.intersection(box_i, box_j)
        
        # 4. Calcolo volumi logaritmici
        log_vol_j = self.volume(box_j)
        log_vol_int = self.volume(box_int)
        
        # 5. Probabilità P(i|j) = Vol(i ∩ j) / Vol(j)
        log_p = log_vol_int - log_vol_j
        
        # Ritorniamo la probabilità limitata
        p = torch.exp(log_p)
        p = torch.clamp(p, min=1e-7, max=1.0 - 1e-7)
        
        return p
    
    def get_regularization_loss(self):
        """Calcola la penalità L2 sui lati di tutti i box."""
        all_theta = self.embeddings.weight.view(-1, 2, self.dim)
        all_boxes = MinDeltaBoxTensor(all_theta)
        
        # Il regolarizzatore calcola la penalità e la moltiplica automaticamente
        # per il parametro 'weight=0.01' impostato nell'__init__
        return self.regularizer(all_boxes)
    
# ==========================================
# DEFINIZIONE DEL CLASSIFICATORE (c -> y)
# ==========================================
class ConceptBottleneckClassifier(nn.Module):
    def __init__(self, num_concepts, box_dim, num_classes, info="boxes"):
        super().__init__()
        self.info = info
        
        if self.info == "boxes":
            input_size = num_concepts * box_dim
        elif self.info == "rel_matrix":
            input_size = num_concepts * num_concepts
        elif self.info == "concepts":
            input_size = num_concepts
        elif self.info == "all":
            input_size = num_concepts + (num_concepts * num_concepts)
        
        self.classifier = nn.Linear(input_size, num_classes)

    def forward(self, scaled_info):
        """
        Input:
            scaled_info: Tensore di shape (batch_size, num_concepts, box_dim)
                         rappresenta i box embedding attivati/disattivati.
        Output:
            logits: Shape (batch_size, num_classes)
        """

        if self.info == "all":
            # 1. Disimballiamo la tupla
            c_info, rel_info = scaled_info
            
            # 2. Appiattiamo i due tensori separatamente
            c_flat = c_info.view(c_info.size(0), -1)     # Shape: (batch, num_concepts)
            rel_flat = rel_info.view(rel_info.size(0), -1) # Shape: (batch, num_concepts^2)
            
            # 3. Concateniamo lungo la dimensione delle feature (dim=1)
            flattened_features = torch.cat([c_flat, rel_flat], dim=1) # Shape: (batch, input_size)
            
        else:
            # Comportamento standard per 'boxes', 'concepts', 'rel_matrix'
            # Appiattiamo l'input per il layer lineare: 
            # da (batch, num_concepts, box_dim) a (batch, num_concepts * box_dim)
            flattened_features = scaled_info.view(scaled_info.size(0), -1)
        
        # Calcoliamo i logit della classe
        logits = self.classifier(flattened_features)
        return logits
    

