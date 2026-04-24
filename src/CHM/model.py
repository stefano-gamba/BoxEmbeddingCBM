import torch
import torch.nn as nn
from box_embeddings.parameterizations import MinDeltaBoxTensor
from box_embeddings.modules.intersection import HardIntersection
from box_embeddings.modules.volume import SoftVolume

# ========================================================
# TRAINING A PRIORI BOX EMBEDDING DA USARE COME CB NEL CBM
# ========================================================

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

class ConceptPredictor(nn.Module):
    def __init__(self, in_features, num_concepts, is_binary=True):
        """
        in_features: La dimensione del tuo vettore di feature 'h'
        num_concepts: Il numero totale di concetti 'c' da predire
        is_binary: True se i concetti sono binari (es. presenza/assenza di un attributo), 
                   False se sono valori continui (regressione).
        """
        super(ConceptPredictor, self).__init__()
        
        # Un singolo layer lineare per mappare le features ai logits dei concetti.
        # Se le tue features 'h' sono molto complesse, potresti usare un piccolo MLP.
        self.linear = nn.Linear(in_features, num_concepts)
        self.is_binary = is_binary

    def forward(self, h):
        logits = self.linear(h)
        
        # Se i concetti sono di classificazione binaria
        # usiamo una sigmoide per ottenere probabilità [0, 1].
        # Se è regressione, restituiamo direttamente i logits.
        if self.is_binary:
            c_pred = torch.sigmoid(logits)
        else:
            c_pred = logits
            
        return c_pred, logits # Spesso è utile restituire anche i logits per la loss function
    

