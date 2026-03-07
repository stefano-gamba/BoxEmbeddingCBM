import torch
import torch.nn as nn
import torch.nn.functional as F

class ConceptBoxModel(nn.Module):
    def __init__(self, latent_dim, num_concepts, box_dim, num_classes):
        super(ConceptBoxModel, self).__init__()
        self.num_concepts = num_concepts
        self.box_dim = box_dim
        
        # Proiezione Latent Space -> Single Box Embedding
        # Creiamo un FC layer separato per il centro (x_m) e per l'offset (Delta) DI OGNI CONCETTO
        self.box_center_projs = nn.ModuleList([
            nn.Linear(latent_dim, box_dim) for _ in range(num_concepts)
        ])
        self.box_offset_projs = nn.ModuleList([
            nn.Linear(latent_dim, box_dim) for _ in range(num_concepts)
        ])
        
        # Allineamento: calcolo probabilità attivazione concetto (p_hat)
        # Anche qui, usiamo un layer separato per ogni concetto per dedurre la sua probabilità
        self.concept_prob_projs = nn.ModuleList([
            nn.Linear(box_dim * 2, 1) for _ in range(num_concepts)
        ])
        
        # Predizione finale
        self.fc1 = nn.Linear(num_concepts * box_dim * 2, num_classes)
        self.fc2 = nn.Linear(num_concepts * num_concepts, num_classes)

    def _calc_volume(self, offset):
        return torch.prod(offset + 1e-6, dim=-1)

    def _calc_intersection(self, center_i, offset_i, center_j, offset_j):
        min_i, max_i = center_i - offset_i, center_i + offset_i
        min_j, max_j = center_j - offset_j, center_j + offset_j
        
        inter_min = torch.max(min_i, min_j)
        inter_max = torch.min(max_i, max_j)
        
        inter_offset = F.relu(inter_max - inter_min) / 2.0
        return self._calc_volume(inter_offset)

    def forward(self, h):
        batch_size = h.size(0)
        
        # 1. Proiezione separata per ogni concetto
        x_m_list = []
        delta_list = []
        for i in range(self.num_concepts):
            x_m_i = self.box_center_projs[i](h)
            # Softplus per garantire offset positivi
            delta_i = F.softplus(self.box_offset_projs[i](h)) 
            x_m_list.append(x_m_i)
            delta_list.append(delta_i)
            
        # Stack per ottenere tensori di shape (B, num_concepts, box_dim)
        x_m = torch.stack(x_m_list, dim=1)
        delta = torch.stack(delta_list, dim=1)
        
        # 2. Allineamento (Probabilità Concetti) separato
        p_hat_list = []
        for i in range(self.num_concepts):
            # Concateniamo centro e offset del singolo concetto
            box_coords_i = torch.cat([x_m_list[i], delta_list[i]], dim=-1)
            p_i = torch.sigmoid(self.concept_prob_projs[i](box_coords_i))
            p_hat_list.append(p_i)
            
        # Shape: (B, num_concepts)
        p_hat = torch.cat(p_hat_list, dim=-1)
        box_coords = torch.cat([x_m, delta], dim=-1)
        
        # 3. Calcolo Gerarchia (Matrice V)
        V = torch.zeros((batch_size, self.num_concepts, self.num_concepts), device=h.device)
        for i in range(self.num_concepts):
            for j in range(self.num_concepts):
                vol_inter = self._calc_intersection(x_m[:, i, :], delta[:, i, :], 
                                                    x_m[:, j, :], delta[:, j, :])
                vol_j = self._calc_volume(delta[:, j, :])
                V[:, i, j] = vol_inter / vol_j
                
        # 4. Predizione
        aligned_boxes = box_coords * p_hat.unsqueeze(-1)
        aligned_boxes_flat = aligned_boxes.view(batch_size, -1)
        V_flat = V.view(batch_size, -1)
        
        out_fc1 = self.fc1(aligned_boxes_flat)
        out_fc2 = self.fc2(V_flat)
        y_hat = out_fc1 + out_fc2
        
        return y_hat, p_hat, V