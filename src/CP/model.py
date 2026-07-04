import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, resnet101, ResNet18_Weights, ResNet101_Weights
from tqdm import tqdm

class ConceptPredictor(nn.Module):
    def __init__(self, in_features, num_concepts, is_binary=True, backbone=False):
        """
        in_features: La dimensione del tuo vettore di feature 'h'
        num_concepts: Il numero totale di concetti 'c' da predire
        is_binary: True se i concetti sono binari (es. presenza/assenza di un attributo), 
                   False se sono valori continui (regressione).
        """
        super(ConceptPredictor, self).__init__()

        self.has_backbone = backbone

        if backbone:
            self.backbone = resnet101(weights=ResNet101_Weights.DEFAULT)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_ftrs, num_concepts)
        else:
            # Un singolo layer lineare per mappare le features ai logits dei concetti.
            # Se le tue features 'h' sono molto complesse, potresti usare un piccolo MLP.
            self.linear = nn.Linear(in_features, num_concepts)
        self.is_binary = is_binary

    def forward(self, h):
        if self.has_backbone:
            logits = self.backbone(h)
        else:
            logits = self.linear(h)
        
        # Se i concetti sono di classificazione binaria
        # usiamo una sigmoide per ottenere probabilità [0, 1].
        # Se è regressione, restituiamo direttamente i logits.
        if self.is_binary:
            c_pred = torch.sigmoid(logits)
        else:
            c_pred = logits
            
        return c_pred, logits
    
    def unfreeze_backbone(self, num_blocks_to_unfreeze=1):
        """
        Scongela gradualmente la backbone della ResNet partendo dal fondo.
        
        Args:
            num_blocks_to_unfreeze (int): Numero di blocchi da scongelare.
                0: Scongela solo l'ultimo layer lineare (fc).
                1: Scongela fc + layer4 (consigliato).
                2: Scongela fc + layer4 + layer3.
                ...
        """
        if not self.has_backbone:
            print("Nessuna backbone da scongelare. Il modello è solo un layer lineare.")
            return

        # 1. Congeliamo TUTTA la backbone per sicurezza
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. Scongeliamo SEMPRE l'ultimo layer (fc)
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
            
        if num_blocks_to_unfreeze >= 1:
            # Scongela il layer4
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
                
        if num_blocks_to_unfreeze >= 2:
            # Scongela il layer3
            for param in self.backbone.layer3.parameters():
                param.requires_grad = True
                
        if num_blocks_to_unfreeze >= 3:
            # Scongela il layer2
            for param in self.backbone.layer2.parameters():
                param.requires_grad = True
        
        if num_blocks_to_unfreeze >= 4:
            # Scongela il layer1
            for param in self.backbone.layer1.parameters():
                param.requires_grad = True
        
        # Stampiamo un recap per sicurezza
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Backbone sbloccata (Blocks: {num_blocks_to_unfreeze}). Parametri addestrabili: {trainable_params:,}")


class OAIConceptPredictor(nn.Module):
    def __init__(self, num_concepts=10):
        super(OAIConceptPredictor, self).__init__()
        
        # Caricamento della backbone pre-addestrata
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Sostituzione del fully connected layer finale
        # num_ftrs per ResNet-18 è tipicamente 512
        num_ftrs = self.backbone.fc.in_features
        
        # Output lineare continuo (Regressione) per i 10 concetti
        self.backbone.fc = nn.Linear(num_ftrs, num_concepts)

    def forward(self, x):
        # x shape: (batch_size, 3, 224, 224) assumendo crop standard
        # Ritorna: (batch_size, 10)
        return self.backbone(x)