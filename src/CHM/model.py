import torch
import torch.nn as nn
    
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
    

