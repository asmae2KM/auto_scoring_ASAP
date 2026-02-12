# ASAG - Automatic Short Answer Grading

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![QWK Score](https://img.shields.io/badge/QWK-0.8428-brightgreen.svg)](README.md)

Système de notation automatique de réponses courtes d'étudiants utilisant **Sentence-BERT** et des réseaux de neurones **ImprovedMLP** spécialisés par question.

## Résultats Principaux

- **QWK Global : 0.8428** (+18.7% vs état de l'art BERT : 0.71)
- **Accuracy : 85.85%**
- **10 modèles spécialisés** adaptés aux spécificités de chaque question
- **Dataset : ASAP-SAS** (39,474 réponses, 10 questions, 3 domaines)

---

## Table des Matières

- [Aperçu du Projet](#-aperçu-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Démo](#-démo)
- [Résultats](#-résultats)
- [Structure du Projet](#-structure-du-projet)
- [Approches Explorées](#-approches-explorées)
- [Contributeurs](#-contributeurs)

---

## Aperçu du Projet

### Problématique

La notation manuelle des réponses courtes d'étudiants présente plusieurs défis :
- **Chronophage** : des heures de correction pour des classes nombreuses
- **Variabilité inter-évaluateurs** : manque de cohérence entre correcteurs
- **Feedback non immédiat** : délai important avant retour aux étudiants

### Solution

Système automatisé basé sur l'apprentissage profond combinant :
- **Sentence-BERT (all-mpnet-base-v2)** pour l'extraction de représentations sémantiques (768 dimensions)
- **ImprovedMLP** avec connections résiduelles et régularisation progressive
- **Spécialisation par question** : 10 modèles dédiés pour s'adapter aux spécificités de chaque domaine

### Dataset ASAP-SAS

| Statistique | Valeur |
|------------|--------|
| **Échantillons totaux** | 39,474 |
| **Questions** | 10 |
| **Domaines** | 3 (Sciences, Biologie, ELA) |
| **Corrélation Score1/Score2** | 91.3% |
| **Sources** | train.tsv + train_rel_2.tsv + public_leaderboard.tsv |

**Répartition par domaine :**
- **Sciences** : Q1 (Acid Rain), Q2 (Polymer), Q10 (Doghouse)
- **Biologie** : Q5 (Protein Synthesis), Q6 (Cell Membrane)
- **ELA (English Language Arts)** : Q3, Q4, Q7, Q8, Q9

---

## Architecture

### Pipeline Global

```
Texte Étudiant
      ↓
Sentence-BERT (all-mpnet-base-v2)
      ↓
Embeddings 768D
      ↓
ImprovedMLP (spécialisé par question)
      ↓
Score (0-3)
```

### ImprovedMLP - Architecture Détaillée

```
Input Layer          : 768 dimensions (SBERT embeddings)
                      ↓
Layer 1 (Encoding)   : Linear(768 → 512) + BatchNorm + ReLU + Dropout(0.4)
                      ↓
Layer 2 (Compression): Linear(512 → 256) + BatchNorm + ReLU + Dropout(0.4)
                      ↓
Layer 3 (Deep)       : Linear(256 → 128) + BatchNorm + ReLU + Dropout(0.32)
                      ↓
Residual Connection  : Skip from Layer 1: Projection(512→128) + Addition
                      ↓
Layer 4 (Refinement) : Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.24)
                      ↓
Output Layer         : Linear(64 → num_classes) | Logits (no activation)
```

### Configuration d'Entraînement

| Paramètre | Valeur |
|-----------|--------|
| **Optimizer** | AdamW |
| **Learning Rate** | 0.001 |
| **Loss Function** | CrossEntropy avec poids de classes |
| **Batch Size** | 32 |
| **Epochs** | ~100 (avec Early Stopping) |
| **Early Stopping Patience** | 3 epochs |
| **Split** | 70% train / 15% val / 15% test |
| **Dropout** | Progressif : 0.4 → 0.32 → 0.24 |

---

## Installation

### Prérequis

- Python 3.8+
- 8GB RAM minimum (16GB recommandé)

### Installation des Dépendances

```bash
# Cloner le repository
git clone https://github.com/asmae2KM/auto_scoring_ASAP.git
cd auto_scoring_ASAP

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Téléchargement du Dataset

```bash
# Créer le dossier data
mkdir -p data/raw

# Télécharger ASAP-SAS depuis Kaggle
# Option 1 : Manuellement depuis https://www.kaggle.com/competitions/asap-sas/data

# Option 2 : Avec Kaggle API
pip install kaggle
kaggle competitions download -c asap-sas
unzip data/raw/asap-sas.zip -d data/raw/
```

### Lancer l'interface web interactive :

```bash
streamlit run streamlit_app.py
```

Puis ouvrir votre navigateur à `http://localhost`

**Fonctionnalités de la démo :**
- Saisie de réponse étudiant
- Sélection de la question (1-10)
- Prédiction du score en temps réel
- Visualisation de la confiance du modèle

### Démo 
https://github.com/user-attachments/assets/b059e1fc-9d8d-42b4-a8e1-62119d7b72fa

**Le notebook contient :**
- Exploration du dataset
- Tests sur des exemples
- Visualisations des résultats
- Analyse d'erreurs

## Résultats

### Résultats Globaux

| Métrique | Valeur | Objectif | Statut |
|----------|--------|----------|--------|
| **QWK Global** | 0.8428 | ≥ 0.71 | **+18.7%** |
| **Accuracy Globale** | 85.85% | - | **Excellent** |
| **Échantillons Test** | 11,874 | - | - |

### Résultats par Question

| Q# | Domaine | Thème | Accuracy | QWK | Échantillons |
|----|---------|-------|----------|-----|--------------|
| **Q1** | Science | Acid Rain | 76.82% | **0.9011** | 1,717 |
| **Q2** | Science | Polymer | 68.27% | 0.8423 | 1,754 |
| **Q3** | ELA | Koala/Panda | 70.00% | 0.4641 | 1,000 |
| **Q4** | ELA | Invasive Species | 79.13% | 0.8739 | 832 |
| **Q5** | Biology | Protein Synthesis | 73.61% | 0.9032 | 1,273 |
| **Q6** | Biology | Cell Membrane | 72.46% | **0.9071** | 1,090 |
| **Q7** | English | Trait of Rose | 68.75% | 0.8899 | 672 |
| **Q8** | English | Mr. Leonard | 73.13% | 0.8697 | 739 |
| **Q9** | English | Orbiting Junk | 74.79% | 0.8983 | 735 |
| **Q10** | Science | Doghouse | 75.81% | 0.8782 | 1,062 |

**Observations :**
- **Meilleure performance** : Q6 (Biology - Cell Membrane) avec QWK = 0.9071
- **Question difficile** : Q3 (ELA - Koala/Panda) avec QWK = 0.4641
  - Raison : Nature très ouverte de la question (analyse littéraire vs questions factuelles)

### Comparaison avec l'État de l'Art

| Modèle | Dataset | QWK | Accuracy | Année |
|--------|---------|-----|----------|-------|
| CharCNN | ASAP-SAS | 0.60 | - | 2019 |
| CNN | ASAP-SAS | 0.62 | - | 2019 |
| Bi-LSTM | ASAP-SAS | 0.65 | - | 2019 |
| **BERT (baseline)** | ASAP-SAS | **0.71** | - | 2019 |
| GPT-4 | Éduc. Méd. | 0.677 | 0.716 | 2024 |
| **NOTRE MODÈLE** | ASAP-SAS | **0.8428** | **0.8585** | 2025 |

**Amélioration : +18.7% vs BERT baseline**


## Approches Explorées

### Approche 2 : Exploration Préliminaire (Non Retenue)

Avant d'arriver au modèle final, nous avons exploré deux approches alternatives :

#### 1. DeBERTa-v3-small

**Architecture :**
- Transformer avec attention désentrelacée
- 12 couches - 44M paramètres
- Fine-tuning complet end-to-end

**Résultats :**

#### 2. CORN (Conditional Ordinal Regression Network)

**Architecture :**
- Régression ordinale avec Sentence-BERT figé
- MLP ordinal léger
- Traite les scores comme niveaux ordonnés

**Résultats :**


#### Pourquoi le Modèle Final a été Choisi ?

| Critère | DeBERTa | CORN | **ImprovedMLP** |
|---------|---------|------|-----------------|
| **QWK** | 0.82-0.88 | 0.75-0.80 | **0.8428** |
| **Rapidité** | Lent | Rapide | Rapide |
| **Déploiement** | Difficile | Facile | Facile |
| **Coût GPU** | Élevé | Faible | Faible |
| **Compromis** | Trop lourd | Pas assez performant | **Optimal** |

**Décision :** ImprovedMLP offre le **meilleur compromis performance/efficacité**.

---

## Facteurs Clés du Succès

1. **Modèles Spécialisés par Question**
   - 10 modèles adaptés aux spécificités de chaque domaine
   - Capture des patterns propres à chaque type de question

2. **Embeddings Sémantiques Riches**
   - SBERT all-mpnet-base-v2 (768D)
   - Meilleure capture du sens que BERT contextualisé figé

3. **Architecture Optimisée**
   - Residual Connection pour gradient flow
   - Dropout progressif : 0.4 → 0.32 → 0.24
   - BatchNorm sur chaque couche
   - Activations ReLU

4. ** Dataset Enrichi**
   - Fusion de 3 sources : train.tsv + train_rel_2.tsv + public_leaderboard.tsv +  public_leaderboard_rel_2.tsv +  public_leaderboard_solution.tsv
   - 39,474 échantillons au total

5. **Régularisation Appropriée**
   - Early Stopping (patience: 3 epochs)
   - Évite l'overfitting
   - Sauvegarde du meilleur modèle basé sur QWK de validation

---

## Limitations

1. **Embeddings Figés**
   - Pas de fine-tuning end-to-end de SBERT
   - Potentiel d'amélioration avec fine-tuning

2. **Manque d'Explicabilité**
   - Boîte noire : difficile d'expliquer les décisions
   - Pas de feedback détaillé aux étudiants

3. **Performance Variable**
   - Q3 (ELA - Koala/Panda) : QWK = 0.4641
   - Questions ouvertes plus difficiles que questions factuelles

4. **Traitement Catégoriel**
   - Scores traités comme classes discrètes
   - Pas d'exploitation de la nature ordinale

5. **Généralisation Non Testée**
   - Validation uniquement sur ASAP-SAS
   - Besoin de tests sur d'autres datasets ASAG

---

## Perspectives d'Amélioration

### Court Terme

- [ ] **Régression Ordinale (CORAL)**
  - Exploiter la nature ordinale des scores
  - Potentiel d'amélioration pour Q3

- [ ] **Fine-tuning End-to-End**
  - Fine-tuner SBERT au lieu d'embeddings figés
  - Adaptation spécifique au domaine éducatif

- [ ] **Augmentation de Données**
  - Paraphrases générées par LLMs
  - Back-translation
  - Amélioration pour questions sous-représentées

### Moyen Terme

- [ ] **Approche Neuro-Symbolique**
  - Combiner réseaux de neurones et règles expertes
  - Améliorer l'explicabilité

- [ ] **Méthodes d'Ensemble**
  - Combiner plusieurs modèles (DeBERTa, SBERT, etc.)
  - Voting ou stacking

- [ ] **Feedback Automatique**
  - Génération de retours personnalisés
  - Suggestions d'amélioration

### Long Terme

- [ ] **Généralisation Multi-Datasets**
  - Tests sur SciEntsBank, Beetle, etc.
  - Création d'un modèle universel ASAG

- [ ] **Intégration LLMs**
  - Utiliser GPT-4, Claude, etc. en few-shot
  - Comparaison performance/coût

- [ ] **Déploiement Production**
  - API scalable (Kubernetes)
  - Interface enseignant/étudiant
  - Monitoring et A/B testing


### Références Principales

1. **ASAP-SAS Dataset**
   - https://www.kaggle.com/competitions/asap-sas/data

2. **Sentence-BERT**
   - Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*.

3. **État de l'Art ASAG**
   - Ramachandran, L., et al. (2024). *LLM-ASAG: Large Language Models for Automatic Short Answer Grading*.
   - Kumar, S., et al. (2024). *Universal ASAG: A Multi-Task Learning Framework*.

---

## Contributeurs

- **Chayma ALABDI**
- **Salma EL FORKANI**
- **Asmae MAHMOUD**
- **Aicha WAAZIZ**
---

## Remerciements

- Merci à la communauté Kaggle pour le dataset ASAP-SAS
- Merci aux auteurs de Sentence-BERT pour leur excellent travail
- Merci à l'équipe PyTorch pour le framework

---

<div align="center">

**Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile ! ⭐**

Built with love and teamwork
</div>
