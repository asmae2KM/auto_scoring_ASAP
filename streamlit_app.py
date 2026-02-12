import streamlit as st
import torch
import torch.nn as nn
import pickle
import os
from sentence_transformers import SentenceTransformer
import re


# Hyperparamètres par question
hyperparams = {
    1: {'lr': 0.001, 'dropout': 0.4, 'epochs': 100, 'batch_size': 32, 'label_smoothing': 0.1},
    2: {'lr': 0.001, 'dropout': 0.4, 'epochs': 100, 'batch_size': 32, 'label_smoothing': 0.1},
    3: {'lr': 0.0005, 'dropout': 0.5, 'epochs': 150, 'batch_size': 16, 'label_smoothing': 0.15},  # Spécial Q3
    4: {'lr': 0.001, 'dropout': 0.3, 'epochs': 100, 'batch_size': 32, 'label_smoothing': 0.1},
    5: {'lr': 0.001, 'dropout': 0.3, 'epochs': 100, 'batch_size': 32, 'label_smoothing': 0.1},
    6: {'lr': 0.001, 'dropout': 0.3, 'epochs': 100, 'batch_size': 32, 'label_smoothing': 0.1},
    7: {'lr': 0.001, 'dropout': 0.4, 'epochs': 100, 'batch_size': 32, 'label_smoothing': 0.1},
    8: {'lr': 0.001, 'dropout': 0.4, 'epochs': 100, 'batch_size': 32, 'label_smoothing': 0.1},
    9: {'lr': 0.001, 'dropout': 0.4, 'epochs': 100, 'batch_size': 32, 'label_smoothing': 0.1},
    10: {'lr': 0.001, 'dropout': 0.4, 'epochs': 100, 'batch_size': 32, 'label_smoothing': 0.1},
}

# --- CONFIGURATION ---
st.set_page_config(page_title="ASAP Scoring System", page_icon="📝")

# --- ARCHITECTURE EXACTE DU NOTEBOOK ---
class ASAPScorer(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=5, dropout_rate=0.3):
        super(ASAPScorer, self).__init__()
        
        # Couche 1 : Entrée (768) -> 512
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        # Couche 2 : 512 -> 256
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        # Couche 3 : 256 -> 128
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Couche 4 : 128 -> 64
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        
        # Couche de sortie
        self.fc_out = nn.Linear(64, num_classes)
        
        # CONNEXION RÉSIDUELLE : 
        # L'erreur dit [128, 512], donc elle part de la sortie de fc1 (512) 
        # vers la sortie de fc3 (128)
        self.residual_proj = nn.Linear(512, 128) 
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 1. Passage dans la première couche
        out1 = self.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out1)
        
        # 2. Branche résiduelle partant de out1 (taille 512)
        residual = self.residual_proj(out1) # dimension 128
        
        # 3. Suite du flux principal
        out = self.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        
        out = self.relu(self.bn3(self.fc3(out)))
        
        # 4. Fusion avec le résidu (128 + 128)
        out += residual
        out = self.dropout(out)
        
        # 5. Fin du réseau
        out = self.relu(self.bn4(self.fc4(out)))
        out = self.fc_out(out)
        return out
@st.cache_resource
def load_resources():
    sbert = SentenceTransformer('all-mpnet-base-v2')
    with open('models/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return sbert, metadata

def clean_text(text):
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"\-]', '', text)
    return text


# --- INTERFACE ---
st.title("📝 ASAP : Scoring Automatique")

try:
    sbert_model, metadata = load_resources()
    ##########################################
    # --- DICTIONNAIRE DES QUESTIONS (Basé sur vos documents) ---
    QUESTION_MAP = {
        1: "Science - Acid Rain: Describe additional information needed to replicate the experiment.",
        2: "Science - Polymer: Draw a conclusion and describe two ways to improve the design.",
        3: "ELA - Koala/Panda: Demonstrate exploration of ideas presented in the text.",
        4: "ELA - Invasive Species: Demonstrate a critical stance and evaluate implicit information.",
        5: "Biology - Protein Synthesis: List and describe four major steps involved.",
        6: "Biology - Cell Membrane: List and describe three processes of substance movement.",
        7: "English - Trait of Rose: Identify a trait of Rose with supporting details from the story.",
        8: "English - Mr. Leonard: Explain the effect of background information on Paul.",
        9: "English - Orbiting Junk: Describe how the author organizes the article with details.",
        10: "Science - Doghouse: Choose a color and explain its effect using experimental results."
    }

    # --- DANS LA SIDEBAR ---
    st.sidebar.header("Configuration")

    # Sélection par intitulé lisible
    selected_label = st.sidebar.selectbox(
        "Choisissez le sujet de la question :",
        options=list(QUESTION_MAP.values())
    )

    # Récupération du q_id correspondant à l'intitulé
    q_id = [k for k, v in QUESTION_MAP.items() if v == selected_label][0]

    # Récupération des infos de performance du modèle depuis metadata
    results_list = metadata.get('question_results', [])
    q_info = next((item for item in results_list if item.get('Set') == q_id or item.get('Question') == q_id), None)
##############################
    if q_info:
        st.sidebar.divider()
        st.sidebar.write(f"**Performances du modèle Q{q_id} :**")
        st.sidebar.metric("Kappa (QWK)", f"{(q_info.get('QWK', 0)*100):.2f}%")
        st.sidebar.metric("Précision", f"{(q_info.get('Accuracy', 0)*100):.2f}%")

    # --- AFFICHAGE CENTRAL ---
    st.subheader(f"Question sélectionnée : {selected_label}")
    st.info(f"💡 Vous évaluez actuellement les réponses pour le set n°{q_id}.")

    # Zone de saisie
    user_input = st.text_area("✍️ Entrez la réponse de l'étudiant :", height=200, 
                            placeholder="Saisissez le texte ici pour obtenir une évaluation automatique...")
##############################

    if st.button("Évaluer"):
        if not user_input.strip():
            st.warning("Entrez un texte.")
        else:
            with st.spinner("Analyse en cours..."):
                # 1. Préparation de l'entrée
                cleaned = clean_text(user_input)
                embedding = torch.tensor(sbert_model.encode([cleaned])).float()
                
                # 2. Détermination dynamique de num_classes
                # Dans le bloc 'if st.button("Évaluer"):'
                model_path = f'models/model_q{q_id}.pth'

                if os.path.exists(model_path):
                    # Charger les poids
                    state_dict = torch.load(model_path, map_location='cpu')
                    
                    # Détecter automatiquement le nombre de classes via le bias de la couche de sortie
                    num_classes_found = state_dict['fc_out.bias'].shape[0]
                    
                    # Récupérer le dropout du dictionnaire hyperparams (défini dans votre notebook)
                    current_dropout = hyperparams.get(q_id, {}).get('dropout', 0.4)
                    
                    # Initialiser le modèle avec les bonnes dimensions
                    model = ASAPScorer(
                        input_dim=768, 
                        hidden_dim=512, 
                        num_classes=num_classes_found, 
                        dropout_rate=current_dropout
                    )
                    
                    # Charger les poids dans l'architecture
                    model.load_state_dict(state_dict)
                    model.eval()
                    
                    # Prédiction
                    with torch.no_grad():
                        output = model(embedding)
                        prediction = torch.argmax(output, dim=1).item()
                    
                    st.success(f"### Score prédit : {prediction}")
                    ##########################################
                else:
                    st.error(f"Le fichier {model_path} est introuvable.")
except Exception as e:
    st.error(f"Détail de l'erreur : {e}")