import torch
from box_embeddings.parameterizations import MinDeltaBoxTensor, BoxTensor
from box_embeddings.modules.intersection import HardIntersection
from box_embeddings.modules.volume import SoftVolume

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


def apply_logical_smoothing(c_pred, prob_matrix, alpha=0.5):
    """
    Applica il Logical Smoothing per correggere le probabilità dei concetti
    usando la matrice delle relazioni gerarchiche.
    
    c_pred: shape (batch_size, num_concepts) - Le probabilità dalla CNN
    prob_matrix: shape (num_concepts, num_concepts) - Matrice P(colonna | riga)
    alpha: float tra 0 e 1. 
           1.0 = Fiducia totale nella CNN (nessuna correzione)
           0.0 = Fiducia totale nella logica della matrice
           0.5 = Bilanciamento a metà
    """
    # 1. Calcoliamo i valori "attesi" secondo la logica (Prodotto Matriciale)
    # Moltiplicare (batch, num_concepts) per (num_concepts, num_concepts)
    # restituisce (batch, num_concepts)
    c_expected = torch.matmul(c_pred, prob_matrix)
    
    # 2. Limitiamo le probabilità attese al range [0, 1]
    c_expected = torch.clamp(c_expected, min=0.0, max=1.0)
    
    # 3. Interpolazione Lineare (Smoothing)
    c_smoothed = (alpha * c_pred) + ((1.0 - alpha) * c_expected)
    
    return c_smoothed