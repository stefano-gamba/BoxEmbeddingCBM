import torch
import torch.nn as nn
from box_embeddings.parameterizations import MinDeltaBoxTensor
from box_embeddings.modules.intersection import HardIntersection, GumbelIntersection
from box_embeddings.modules.volume import SoftVolume, BesselApproxVolume
from box_embeddings.modules.regularization import L2SideBoxRegularizer

class BoxHierarchyModel(nn.Module):
    def __init__(self, num_concepts, dim=32, soft_intersection=False, bessel_volume=False, volume_temperature=0.1, simple_init=True):
        super().__init__()
        self.dim = dim # <-- Salviamo dim per usarlo nel forward
        
        # Ogni box necessita di 2*dim parametri (dim per z_min, dim per delta)
        self.embeddings = nn.Embedding(num_concepts, 2 * dim)
        
        if simple_init:
            nn.init.uniform_(self.embeddings.weight, -0.5, 0.5)
        else:
            # 1. Centri: Sfalsati per rompere la simmetria
            nn.init.uniform_(self.embeddings.weight.data[:, :dim], -0.01, 0.01)
            # 2. Lati: Molto grandi in modo che partano tutti sovrapposti ma scentrati
            nn.init.constant_(self.embeddings.weight.data[:, dim:], 2.0)
        
        if soft_intersection:
            self.intersection = GumbelIntersection(intersection_temperature=0.001)
        else:
            self.intersection = HardIntersection()
        
        if bessel_volume:
    
            self.volume = BesselApproxVolume(
                volume_temperature=volume_temperature, 
                intersection_temperature=0.01 # Deve combaciare con l'intersezione
            )
        else:
            self.volume = SoftVolume(volume_temperature=volume_temperature)

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


class BoxHierarchyModelJoint(nn.Module):
    def __init__(
            self, 
            num_concepts, 
            num_classes, 
            dim=32, 
            soft_intersection=False, 
            bessel_volume=False, 
            volume_temperature=0.1, 
            simple_init=True
        ):
        super().__init__()
        self.dim = dim
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.concept_embeddings = nn.Embedding(num_concepts, 2 * dim)
        self.class_embeddings = nn.Embedding(num_classes, 2 * dim)
        
        if not simple_init:
            # TABELLA 1: Box dei Concetti
            nn.init.uniform_(self.concept_embeddings.weight.data[:, :dim], -0.01, 0.01)
            nn.init.constant_(self.concept_embeddings.weight.data[:, dim:], 2.0)
        
            # TABELLA 2: Box delle Classi
            nn.init.uniform_(self.class_embeddings.weight.data[:, :dim], -0.01, 0.01)
            nn.init.constant_(self.class_embeddings.weight.data[:, dim:], 2.0)
        else:
            nn.init.uniform_(self.concept_embeddings.weight, -0.5, 0.5)
            nn.init.uniform_(self.class_embeddings.weight, -0.5, 0.5)
        
        if soft_intersection:
            self.intersection = GumbelIntersection(intersection_temperature=0.001)
        else:
            self.intersection = HardIntersection()
        
        if bessel_volume:
            self.volume = BesselApproxVolume(
                volume_temperature=volume_temperature, 
                intersection_temperature=0.01
            )
        else:
            self.volume = SoftVolume(volume_temperature=volume_temperature)
        
        self.regularizer = L2SideBoxRegularizer(log_scale=False, weight=0.0)

    def _compute_prob(self, box_container, box_contained):
        """Metodo di supporto per calcolare P(contained | container) = Vol(int) / Vol(contained)"""
        box_int = self.intersection(box_container, box_contained)
        log_vol_contained = self.volume(box_contained)
        log_vol_int = self.volume(box_int)
        
        log_p = log_vol_int - log_vol_contained
        p = torch.exp(log_p)
        return torch.clamp(p, min=1e-7, max=1.0 - 1e-7)

    def forward_concepts(self, idx_i, idx_j):
        """
        Ottimizza la gerarchia tra concetti. 
        idx_i: concetto che contiene (Container)
        idx_j: concetto contenuto (Contained)
        """
        theta_i = self.concept_embeddings(idx_i).view(-1, 2, self.dim)
        theta_j = self.concept_embeddings(idx_j).view(-1, 2, self.dim)
        
        box_i = MinDeltaBoxTensor(theta_i)
        box_j = MinDeltaBoxTensor(theta_j)
        
        return self._compute_prob(box_container=box_i, box_contained=box_j)
    
    def forward_classes(self, idx_concept, idx_class):
        """
        Ottimizza l'appartenenza della classe al concetto.
        Il concetto fa da Container, la classe fa da Contained.
        """
        theta_concept = self.concept_embeddings(idx_concept).view(-1, 2, self.dim)
        theta_class = self.class_embeddings(idx_class).view(-1, 2, self.dim)
        
        box_concept = MinDeltaBoxTensor(theta_concept)
        box_class = MinDeltaBoxTensor(theta_class)
        
        return self._compute_prob(box_container=box_concept, box_contained=box_class)

    def get_regularization_loss(self):
        """Calcola la penalità su entrambi gli spazi."""
        all_concept_theta = self.concept_embeddings.weight.view(-1, 2, self.dim)
        all_class_theta = self.class_embeddings.weight.view(-1, 2, self.dim)
        
        reg_concepts = self.regularizer(MinDeltaBoxTensor(all_concept_theta))
        reg_classes = self.regularizer(MinDeltaBoxTensor(all_class_theta))
        return reg_concepts + reg_classes