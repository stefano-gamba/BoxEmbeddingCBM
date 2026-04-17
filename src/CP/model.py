import torch
import torch.nn as nn

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