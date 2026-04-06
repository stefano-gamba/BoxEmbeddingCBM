import torch
import torch.nn as nn
from box_embeddings.parameterizations import MinDeltaBoxTensor, BoxTensor
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
        self.classifier = nn.Linear(input_size, num_classes)

    def forward(self, scaled_info):
        """
        Input:
            scaled_info: Tensore di shape (batch_size, num_concepts, box_dim)
                         rappresenta i box embedding attivati/disattivati.
        Output:
            logits: Shape (batch_size, num_classes)
        """
        # Appiattiamo l'input per il layer lineare: 
        # da (batch, num_concepts, box_dim) a (batch, num_concepts * box_dim)
        flattened_features = scaled_info.view(scaled_info.size(0), -1)
        
        # Calcoliamo i logit della classe
        logits = self.classifier(flattened_features)
        return logits
    

def get_box_dict(model, id2concept):
    """
    Estrae tutti i box appresi dal modello e crea una mappatura diretta.
    Restituisce un dizionario nel formato {concetto (str): box (MinDeltaBoxTensor)}.
    """
    # Mettiamo il modello in eval mode per sicurezza
    model.eval()
    
    dizionario_box = {}
    
    # Disabilitiamo il calcolo dei gradienti dato che stiamo solo estraendo dati
    with torch.no_grad():
        for idx, concept_name in id2concept.items():
            # 1. Creiamo un tensore contenente l'ID del concetto
            tensor_id = torch.tensor([idx], dtype=torch.long)
            
            # 2. Estraiamo il parametro base theta (le word embeddings raw)
            theta = model.embeddings(tensor_id).view(-1, 2, model.dim)
            
            # 3. Lo incapsuliamo nel MinDeltaBoxTensor
            box = MinDeltaBoxTensor(theta)
            
            # 4. Salviamo nel dizionario
            dizionario_box[concept_name] = box
            
    return dizionario_box


# ==========================================
# PREPARAZIONE DEI DATI E DEI BOX
# ==========================================
def prepara_tensore_box(dict_boxes, concept2id):
    """
    Estrae i parametri (theta) dal dizionario dei box e crea un singolo
    tensore fisso. Non calcoleremo i gradienti per questi box.
    """
    num_concepts = len(concept2id)
    # Assumiamo che il dizionario contenga oggetti MinDeltaBoxTensor
    # il cui attributo "data" (o la property corrispondente) sia theta (dimensione 2*D)
    
    # Prendiamo un box a caso per capirne la dimensione
    sample_box = next(iter(dict_boxes.values()))
    # In box_embeddings, the parameters are usually accessible. 
    # Assumendo che il tensore theta originale fosse salvato o recuperabile:
    # (qui simulo estraendo z e Z concatenati, che equivalgono a 2*D parametri)
    box_dim = sample_box.z.size(-1) * 2
    
    boxes_tensor = torch.zeros((num_concepts, box_dim))
    
    for concept_name, idx in concept2id.items():
        box = dict_boxes[concept_name]
        # Concateniamo min (z) e max (Z) per avere i parametri geometrici completi
        theta = torch.cat([box.z.squeeze(), box.Z.squeeze()], dim=-1)
        boxes_tensor[idx] = theta
        
    return boxes_tensor.detach() # .detach() assicura che i box siano congelati


def calcola_matrice_probabilita(boxes_tensor):
    """
    Calcola la matrice delle probabilità condizionate P(i|j)
    usando l'API della libreria Box Embeddings.
    Shape di output: (num_concepts, num_concepts)
    """
    num_concepts = boxes_tensor.size(0)

    box_dim = boxes_tensor.size(-1) // 2
    
    # Prepariamo i tensori per il broadcasting: vogliamo combinare ogni box con tutti gli altri
    # theta_i rappresenta l'ipotesi (intersezione), theta_j rappresenta la premessa (condizionante)
    theta_i = boxes_tensor.unsqueeze(1).expand(-1, num_concepts, -1) 
    theta_j = boxes_tensor.unsqueeze(0).expand(num_concepts, -1, -1) 

    theta_i = theta_i.view(num_concepts, num_concepts, 2, box_dim)
    theta_j = theta_j.view(num_concepts, num_concepts, 2, box_dim)
    
    box_i = BoxTensor(theta_i)
    box_j = BoxTensor(theta_j)
    
    # Inizializziamo i moduli operativi della libreria
    intersection_op = HardIntersection()
    volume_op = SoftVolume(volume_temperature=1.0) # Restituisce il volume logaritmico di default
    
    # 1. Intersezione
    box_int = intersection_op(box_i, box_j)
    
    # 2. Volumi Logaritmici
    log_vol_int = volume_op(box_int)
    log_vol_j = volume_op(box_j)
    
    # 3. Probabilità condizionata: P(i|j) = exp(log(Vol_int) - log(Vol_j))
    log_p_i_given_j = log_vol_int - log_vol_j
    
    # Esponenziale e clamp per rimuovere eventuali artefatti numerici minimi oltre 1.0
    p_i_given_j = torch.exp(log_p_i_given_j)
    prob_matrix = torch.clamp(p_i_given_j, min=0.0, max=1.0)
    
    return prob_matrix