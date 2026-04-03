import torch
import torch.nn as nn
from box_embeddings.parameterizations import MinDeltaBoxTensor
from box_embeddings.modules.intersection import HardIntersection
from box_embeddings.modules.volume import SoftVolume

class BoxHierarchyModel(nn.Module):
    def __init__(self, num_concepts, dim=32):
        super().__init__()
        self.dim = dim # <-- Salviamo dim per usarlo nel forward
        
        # Ogni box necessita di 2*dim parametri (dim per z_min, dim per delta)
        self.embeddings = nn.Embedding(num_concepts, 2 * dim)
        
        # Inizializziamo i parametri con una distribuzione uniforme
        nn.init.uniform_(self.embeddings.weight, -0.5, 0.5)
        
        self.intersection = HardIntersection()
        self.volume = SoftVolume(volume_temperature=1.0) 

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