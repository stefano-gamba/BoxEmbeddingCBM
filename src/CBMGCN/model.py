import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # W: La matrice dei pesi per trasformare le feature del nodo
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        """
        x shape: (batch_size, num_nodes, in_features)
        adj shape: (num_nodes, num_nodes)
        """
        # 1. Trasformazione delle feature: X * W
        support = self.linear(x) 
        
        # 2. Message Passing: A * (XW)
        # adj viene propagata su tutto il batch tramite broadcasting
        output = torch.matmul(adj, support) 
        return output

class ConceptGNNClassifier(nn.Module):
    def __init__(self, num_concepts, in_features, hidden_dim, num_classes):
        """
        in_features: 1 (se usi gli scalari) oppure box_dim (se usi i box)
        """
        super().__init__()
        
        # Due layer di Graph Convolution
        self.gcn1 = DenseGCNLayer(in_features, hidden_dim)
        self.gcn2 = DenseGCNLayer(hidden_dim, hidden_dim)

        self.node_reducer = nn.Linear(hidden_dim, 1)
        
        # Layer di classificazione finale
        self.classifier = nn.Linear(num_concepts, num_classes)

    def forward(self, scaled_info, prob_matrix):
        """
        scaled_info: Le feature dei nodi (batch_size, num_concepts, in_features)
                     Possono essere le tue probabilità (unsqueeze) o i box scalati.
        prob_matrix: La matrice delle relazioni (num_concepts, num_concepts)
        """
        # STEP 0: Preparazione della matrice di adiacenza
        # È fondamentale aggiungere l'identità (self-loops) alla prob_matrix.
        # Senza questo, un nodo passerebbe le sue informazioni ai vicini ma dimenticherebbe le proprie!
        I = torch.eye(prob_matrix.size(0), device=prob_matrix.device)
        A_tilde = prob_matrix + I
        
        # STEP 1: Passaggio nei layer GCN
        # (batch_size, num_concepts, hidden_dim)
        x = F.relu(self.gcn1(scaled_info, A_tilde)) 
        x = F.relu(self.gcn2(x, A_tilde))
        
        # STEP 2: Readout (Aggregazione Globale)
        # Comprimiamo le feature nascoste di ogni nodo a un singolo scalare
        concept_scalars = self.node_reducer(x).squeeze(-1) 
        
        # STEP 3: Classificazione
        logits = self.classifier(concept_scalars)
        return logits
    
