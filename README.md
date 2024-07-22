# <span style='background:blue'>Contexte</span>

L'entreprise **"Prêt à dépenser"** souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un **algorithme de classification** en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

Dasn un deuxième temps, il faudrait mettre en  place un POC (Proof of Concept) pour démontrer une nouvelle méthode de classification d'image en Computer vision.


# <span style='background:blue'>Missions</span>

1/ Construire un **dashboard** interactif basé sur le modèle de scoring déployé sur l’API

2/ Réaliser une veille des outils de Data Science et démontrer un **POC**.


# <span style='background:blue'>Dataset</span>

Source : Non précisée

# <span style='background:blue'>Fichiers du dépôt</span>

- Dossier **Modele_ML** : dossier contenant les modèles de Deep Learning (VGG16 et Yolo_V8)

- **berthe_pierrick_2_dashboard_manuel_explicatif_042024.pdf** : PDF expliquant l'utilisation du dashboard streamlit

- **berthe_pierrick_3_notebook_veille_042024.ipynb** : Notebook de veille scientifique pour le POC en Computer Vision (classification d'image pour VGG16 Vs. Yolo_V8)

- **dashboard.py** : Script python pour le dashboard Streamlit, déployé sur le cloud à l'adresse:
https://projet8ocrdatascientist-pierrick-berthe.streamlit.app/

- **validation_on_test_dataset.yaml** : fichier YAML pour la classification d'image avec Yolo_V8

- **berthe_pierrick_5_presentation_042024.pdf** : Présentation des résultats


# <span style='background:blue'>Auteur</span>

Pierrick BERTHE<br>
Avril 2024
