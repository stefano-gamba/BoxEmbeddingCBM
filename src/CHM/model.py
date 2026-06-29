import torch
import torch.nn as nn
from box_embeddings.parameterizations import MinDeltaBoxTensor
from box_embeddings.modules.intersection import GumbelIntersection
from box_embeddings.modules.volume import BesselApproxVolume

class ConceptBottleneckClassifier(nn.Module):
    def __init__(self, num_concepts, box_dim, num_classes, info="boxes", 
                 geometric_weights=None, concept_embeddings=None, class_embeddings=None):
        super().__init__()
        self.info = info
        
        # ==========================================
        # NUOVA MODALITÀ: BOX DELL'IMMAGINE DINAMICO
        # ==========================================
        if self.info == "dynamic_box":
            if concept_embeddings is None or class_embeddings is None:
                raise ValueError("Devi fornire concept_embeddings e class_embeddings (i pesi .weight.detach())")
            
            # Registriamo i pesi originali congelati dalla Fase 1
            self.register_buffer("concept_theta", concept_embeddings.detach())
            self.register_buffer("class_theta", class_embeddings.detach())
            
            # Moduli per il calcolo delle intersezioni finali
            self.intersection = GumbelIntersection(intersection_temperature=0.001)
            self.volume = BesselApproxVolume(volume_temperature=0.1, intersection_temperature=0.01)
            
            # M definisce i "confini dell'universo" per ignorare i concetti non presenti
            self.M = 10.0 

        # ==========================================
        # VECCHIE MODALITÀ 
        # ==========================================
        elif self.info == "geometric":
            if geometric_weights is None:
                raise ValueError("Devi fornire 'geometric_weights' se info='geometric'")
            self.register_buffer("geometric_weights", geometric_weights)
            
        elif self.info == "boxes":
            input_size = num_concepts * box_dim
            self.classifier = nn.Linear(input_size, num_classes)
        elif self.info == "rel_matrix":
            input_size = num_concepts * num_concepts
            self.classifier = nn.Linear(input_size, num_classes)
        elif self.info == "concepts":
            input_size = num_concepts
            self.classifier = nn.Linear(input_size, num_classes)
        elif self.info == "all":
            input_size = num_concepts + (num_concepts * num_concepts)
            self.classifier = nn.Linear(input_size, num_classes)

    def forward(self, input_data):
        
        if self.info == "dynamic_box":
            # input_data è semplicemente il vettore di probabilità (batch_size, num_concepts)
            batch_size = input_data.size(0)
            dim = self.concept_theta.size(-1) // 2
            
            # (batch, num_concepts, 1)
            p = input_data.unsqueeze(-1) 
            
            # --- 1. RICOSTRUZIONE BOX DEI CONCETTI ---
            # shape: (1, num_concepts, 2, dim)
            c_theta = self.concept_theta.view(1, -1, 2, dim) 
            z_c = c_theta[:, :, 0, :] 
            delta_c = c_theta[:, :, 1, :]
            Z_c = z_c + delta_c # Calcoliamo Z_max esatto
            
            # --- 2. SCOLPIRE LO SPAZIO (RILASSAMENTO) ---
            # Espandiamo e mascheriamo dinamicamente in base alla certezza della CNN
            z_relaxed = z_c * p + (-self.M) * (1 - p)
            Z_relaxed = Z_c * p + (self.M) * (1 - p)
            
            # --- 3. CREAZIONE DI B_img ---
            # L'intersezione di tutti i concetti è il max dei minimi e il min dei massimi
            z_img = torch.max(z_relaxed, dim=1)[0] # shape: (batch_size, dim)
            Z_img = torch.min(Z_relaxed, dim=1)[0] 
            
            # Riconvertiamo nel formato (z, delta)
            delta_img = Z_img - z_img
            # Salvataggio di sicurezza: se la CNN predice concetti logicamente impossibili/disgiunti, 
            # delta_img potrebbe diventare negativo. Clamp previene NaN.
            delta_img = torch.clamp(delta_img, min=1e-7) 
            
            theta_img = torch.stack([z_img, delta_img], dim=-2) # (batch, 2, dim)
            b_img = MinDeltaBoxTensor(theta_img)
            
            # --- 4. MATCHING CON LE CLASSI FINALI ---
            # Prepariamo le classi: (1, num_classes, 2, dim) espanso al batch
            cls_theta = self.class_theta.view(1, -1, 2, dim).expand(batch_size, -1, -1, -1)
            b_classes = MinDeltaBoxTensor(cls_theta)
            
            # Espandiamo B_img per matchare la dimensione delle classi: (batch, num_classes, 2, dim)
            b_img_expanded = MinDeltaBoxTensor(theta_img.unsqueeze(1).expand(-1, cls_theta.size(1), -1, -1))
            
            # Calcoliamo l'intersezione soft finale e i volumi
            b_int = self.intersection(b_classes, b_img_expanded)
            log_vol_int = self.volume(b_int)
            log_vol_class = self.volume(b_classes)
            
            # P(B_img | Class) = Vol(Classe ∩ B_img) / Vol(Classe)
            # Manteniamo il denominatore sulla classe per evitare la "Penalità di volume"
            log_p = log_vol_int - log_vol_class
            
            # Ritorniamo i logit. Usa torch.exp(log_p) se la loss si aspetta probabilità assolute.
            return log_p

        # ==========================================
        # VECCHI FORWARD PASS 
        # ==========================================
        elif self.info == "geometric":
            return torch.matmul(input_data, self.geometric_weights.t())
        else:
            if self.info == "all":
                c_info, rel_info = input_data
                c_flat = c_info.view(c_info.size(0), -1)
                rel_flat = rel_info.view(rel_info.size(0), -1)
                flattened_features = torch.cat([c_flat, rel_flat], dim=1)
            else:
                flattened_features = input_data.view(input_data.size(0), -1)
            
            return self.classifier(flattened_features)