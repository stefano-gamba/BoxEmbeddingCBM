import torch
import torch.nn as nn
    
# ==========================================
# DEFINIZIONE DEL CLASSIFICATORE (c -> y)
# ==========================================
class ConceptBottleneckClassifier(nn.Module):
    def __init__(self, num_concepts, box_dim, num_classes, info="boxes", geometric_weights=None):
        super().__init__()
        self.info = info
        
        if self.info == "geometric":
            if geometric_weights is None:
                raise ValueError("Devi fornire 'geometric_weights' se info='geometric'")
            # Registriamo la matrice W (shape: num_classes, num_concepts) come buffer fisso
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
        if self.info == "geometric":
            # Per la logica geometrica, input_data è semplicemente il vettore
            # delle probabilità dei concetti di shape (batch_size, num_concepts)
            # Effettuiamo: P(c) * W^T
            logits = torch.matmul(input_data, self.geometric_weights.t())
            return logits

        # --- Vecchia logica per i layer lineari ---
        if self.info == "all":
            c_info, rel_info = input_data
            c_flat = c_info.view(c_info.size(0), -1)
            rel_flat = rel_info.view(rel_info.size(0), -1)
            flattened_features = torch.cat([c_flat, rel_flat], dim=1)
        else:
            flattened_features = input_data.view(input_data.size(0), -1)
        
        logits = self.classifier(flattened_features)
        return logits