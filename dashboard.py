"""
Description: Ce fichier contient le dashboard de l'application Streamlit.

Author: Pierrick Berthe
Date: 2024-04-30
"""


# ============== étape 1 : Importation des librairies ====================

import sys
import time
import pandas as pd
import numpy as np
import requests
import os
import shap
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import json
import PIL
from PIL import Image
from io import BytesIO
import plotly
import plotly.graph_objects as go
import seaborn as sns

# Versions
print("\nVersion des librairies utilisees :")
print("Python        : " + sys.version)
print("Io            : No module version")
print("Json          : No module version")
print("Matplotlib    : " + matplotlib.__version__)
print("Numpy         : " + np.__version__)
print("Os            : No module version")
print("Pandas        : " + pd.__version__)
print("PIL           : " + PIL.__version__)
print("Plotly        : " + plotly.__version__)
print("Requests      : " + requests.__version__)
print("Shap          : " + shap.__version__)
print("Streamlit     : " + st.__version__)
print("\n")


# ================= étape 2 : Chemins environnement ========================

# Indicateur pour savoir si l'API est sur le cloud ou en local
IS_API_ON_CLOUD = True  

# Titre de l'application
st.title('Projet 8\n')
st.title('Réalisez un Dashboard\n')
st.write('Formation Data Scientist de Pierrick BERTHE\n')

# Affichage le chemin du répertoire courant
print("os.getcwd():",os.getcwd(), "\n")

# URL de l'API Flask (local ou distant)
if IS_API_ON_CLOUD:
    URL_API = 'https://api-pb-e85d72620dec.herokuapp.com/'
else:
    URL_API = 'http://127.0.0.1:5000'
print("URL_API:",URL_API, "\n")

# URL de l'API pour les requêtes POST
URL_API_CLIENT_SELECTION = f'{URL_API}/client_selection'
URL_API_CLIENT_EXTRACTION= f'{URL_API}/client_extraction'
URL_API_PREDICT = f'{URL_API}/predict'
URL_API_FI_LOCALE= f'{URL_API}/feature_importance_locale'
URL_API_FI_GLOBALE= f'{URL_API}/feature_importance_globale'
URL_API_FEAT_PLOT= f'{URL_API}/feature_plot'
URL_API_FEAT_SELECTION= f'{URL_API}/feature_selection'
URL_API_FEAT_EXTRACTION= f'{URL_API}/feat_extraction'


# ====================== étape 3 : Fonctions ============================

def fetch_data_and_client_selection(url, key_unique=None):
    """
    Récupère les données et crée une liste déroulante pour sélectionner
    un client.
    """
    # Envoi de la requête POST
    response = requests.post(url)

    # SI requete OK => extraction, transformation et affichage des données
    if response.status_code == 200:
        sk_id_curr_all = response.json()
        sk_id_curr_all = pd.Series(sk_id_curr_all)
        client_id = st.selectbox(
            'Choisir un id de client :',
            sk_id_curr_all.unique(),
            key=key_unique
        )
        return int(client_id)
    else:
        st.write("Erreur lors de la récupération des id des clients.")
        return None


def fetch_feat_name_and_feat_selection(url, axe, key_unique=None):
    """
    Récupère les noms des features et crée une liste déroulante pour
    sélectionner une feature.
    """
    # Envoi de la requête POST
    response = requests.post(url)

    # SI requete OK => extraction, transformation et affichage des données
    if response.status_code == 200:
        feat_name_all = response.json()
        feat_name_all_df = pd.Series(feat_name_all)
        feat_name = st.selectbox(
            f"Choisir une feature à afficher sur l'axe des {axe} :",
            feat_name_all_df.unique(),
            key=key_unique
        )
        return str(feat_name)

    else:
        st.write("Erreur lors de la récupération des noms des features.")
        return None


def get_client_data(url, client_id):
    """
    Récupère les données d'un client spécifique.
    """
    # Envoi de la requête POST
    response = requests.post(url, json={'client_id': client_id})

    # SI requete OK => load json, verif type str, transformation en df
    if response.status_code == 200:

        # load json
        response_dict = json.loads(response.text)

        # Vérification que 'client_data' est une chaîne JSON
        if isinstance(response_dict["client_data"], str):
            client_data = pd.read_json(
                response_dict["client_data"],
                orient='records'
            )
            client_data_brutes = pd.read_json(
                response_dict["client_data_brutes"],
                orient='records'
            )

            # Affichage des données du client
            st.title(f'Client sélectionné : {client_id}')

            # Vérification que 'SK_ID_CURR' est une colonne du DataFrame
            if 'SK_ID_CURR' in client_data.columns:
                return (
                    client_data.drop(columns=['SK_ID_CURR']),
                    client_data_brutes.drop(columns=['SK_ID_CURR', 'TARGET'])
                )
            else:
                st.write("La colonne 'SK_ID_CURR' n'est pas dans le df.")
                return client_data, client_data_brutes

        else:
            st.write("'client_data' n'est pas une chaîne JSON.")
            return None, None

    else:
        st.write("Erreur lors de la récupération des données du client.")
        return None, None


def get_feat_data(url, feature_name, axe):
    """
    Récupère les données d'une feature spécifique.
    """
    # Envoi de la requête POST
    response = requests.post(url, json={'feature_name': feature_name})

    # SI requete OK => load json, verif type str, transformation en df
    if response.status_code == 200:

        # load json
        response_dict = json.loads(response.text)

        # Vérification que 'client_data' est une chaîne JSON
        if isinstance(response_dict["feat_data_brutes"], str):
            feat_data_brutes = pd.read_json(
                response_dict["feat_data_brutes"],
                orient='records'
            )

            # Affichage des données du client
            st.write(f'Feature sélectionné pour axe {axe} : {feature_name}')

            return feat_data_brutes

        else:
            st.write("'feat_data_brutes' n'est pas une chaîne JSON.")
            return None

    else:
        st.write("Erreur lors de la récupération des données de la feature.")
        return None


def request_prediction(url, data):
    """
    Envoie une requête POST de prédiction à un service web.
    """
    # ESSAI de la requête POST
    try:
        response = requests.post(url, json=data.to_dict())
        response.raise_for_status()
        return response.json()

    # Gestion des autres erreurs
    except Exception as err:
        print(f'Erreur dans la requête POST de prediction: {err}')
        return None


def request_fi_locale(url, data):
    """
    Récupère le calcul de la feature importance locale et l'affiche
    """
    # Supprimer la col TARGET si elle existe
    if 'TARGET' in data.columns:
        data = data.drop(columns=['TARGET'])

    # ESSAI de la requête POST
    try:
        print('request.post pret a lancer')
        response = requests.post(url, json=data.to_dict())
        print('request.post_effectued')
        response.raise_for_status()
        return response.json()

    # Gestion des autres erreurs
    except Exception as err:
        print(f'Erreur dans la requête POST de FI locale: {err}')
        return None


def calcule_fi_globale(url):
    """
    Récupère le calcul de la feature importance globale et affiche le 
    summary plot en image. 
    """
    # Envoi de la requête POST
    response = requests.post(url)

    # SI requete OK => extraction, ouverture image et affichage
    if response.status_code == 200:
        image_bytes = BytesIO(response.content)
        img = Image.open(image_bytes)
        st.image(img)

    else:
        print("Erreur lors de la récupération de l'image.")


def get_feature_plot(url, client_id, feat_to_display):
    """
    Récupère les données d'un client spécifique pour une feature donnée ainsi
    que les distributions des données des 2 classes pour cette feature.
    """
    # Envoi de la requête POST
    response = requests.post(
        url,
        json={
            'client_id': client_id,
            'feat_to_display': feat_to_display
        },
        timeout=10
    )

    # SI requete OK => load json, verif type str, transformation en df
    if response.status_code == 200:

        # load json
        response_dict = json.loads(response.text)

        # Vérification que 'response_dict' est une float
        if isinstance(response_dict, dict):

            # Création du DataFrame
            data_for_plot = pd.DataFrame({
                'client_data': [response_dict['client_data']],
                'client_0_data': [response_dict['client_0_data']],
                'client_1_data': [response_dict['client_1_data']]
            })

            return data_for_plot

        else:
            st.write("'response_dict' n'est pas une dict.")
            return None

    else:
        st.write("Erreur lors de la récupération des données du client.")
        return None


def afficher_hist(client_data, client_id, url, key_unique=None):
    """
    Cette procédure affiche un histogramme de la distribution des données
    pour une feature sélectionnée.
    """

    # Sélection de la feature à afficher
    feat_to_display = st.selectbox(
        'Choisir la feature à visualiser :',
        client_data.columns,
        key=key_unique
    )

    # Récupération des données du client
    data_for_plot = get_feature_plot(
        url,
        client_id,
        feat_to_display
    )

    # Affichage de l'histogramme
    fig, ax = plt.subplots(figsize=(10, 8))

    # Affichage de la distribution pour la classe 0 en vert
    sns.kdeplot(
        data_for_plot['client_0_data'][0],
        color='green',
        fill=True,
        label='Clients accordés'
    )

    # Affichage de la distribution pour la classe 1 en rouge
    sns.kdeplot(
        data_for_plot['client_1_data'][0],
        color='red',
        fill=True,
        label='Clients refusés'
    )

    # Ajout d'une ligne verticale pour la valeur du client en bleu
    plt.axvline(
        data_for_plot['client_data'][0],
        color='blue',
        linestyle='dashed',
        linewidth=2,
        label='Client sélectionné'
    )

    # Ajout de la légende et des titres
    plt.legend()
    ax.set_title(
        f'Distribution de \n{feat_to_display}',
        fontsize=24,
        fontweight='bold'
    )
    ax.set_xlabel('Valeur', fontsize=24)

    # Affichage du tracé dans Streamlit
    st.pyplot(fig)


# ============= étape 4 : Fonction principale du dashboard ==================

def main():
    """
    Fonction principale de l'application Streamlit.
    """
    # Création des onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "Prédiction client",
        "DataViz features",
        "Analyse bivariée",
        "Feature importance Globale"
    ])

    # Page 1
    with tab1:
        st.title('Prédiction client\n')

        # Récupération des clients et sélection d'un client
        client_id_p1 = fetch_data_and_client_selection(
            URL_API_CLIENT_SELECTION,
            key_unique='select_client_id_1'
        )
        
        # Récupération des données du client
        client_data_p1, client_data_brutes_p1 = get_client_data(
            URL_API_CLIENT_EXTRACTION,
            client_id_p1
        )

        # Sélection des features à afficher
        feat_to_display = st.multiselect(
            'Choisir les features à afficher :',
            client_data_p1.columns
        )

        # Affichage uniquement des colonnes sélectionnées
        if feat_to_display:
            st.dataframe(client_data_brutes_p1[feat_to_display])
        else:
            st.dataframe(client_data_brutes_p1)

        # Bouton pour calculer la prédiction => envoi de la requête POST
        if st.button('Calculer la prédiction'):
            response = request_prediction(URL_API_PREDICT, client_data_p1)

            # Affichage d'une erreur si la prédiction est None
            if response is None:
                st.write('Erreur de prédiction\n')

            # Affichage de la prédiction en français
            else:
                if response["prediction"] == 0:
                    st.markdown(
                        '<div style="background-color: #98FB98; padding: 10px;'
                        'border-radius: 5px; color: #000000;"'
                        '>Le prêt est accordé.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div style="background-color: #FF6347; padding: 10px;'
                        'border-radius: 5px; color: #000000;"'
                        '>Le prêt n\'est pas accordé.</div>',
                        unsafe_allow_html=True
                    )

                # Affichage de la probabilité de refus
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=response['proba_1'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Score de risque de crédit", 'font': {
                        'size': 24
                    }},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "white"},
                        'steps': [
                            {
                                'range': [0, response['seuil_predict']],
                                'color': "green"
                            },
                            {
                                'range': [response['seuil_predict'], 1],
                                'color': "red"
                            }
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 5},
                            'thickness': 1,
                            'value': response['seuil_predict']
                        }
                    }
                ))

                st.plotly_chart(fig)

                # Afficher la valeur de seuil_predict
                st.write(
                    f"Seuil de risque accord/refus : {response['seuil_predict']}")

                print('len(client_data_p1):', len(client_data_p1))
                print('client_data_p1:', client_data_p1)

                # Calcule et affichage de feature importance locale
                st.title('Feature importance locale :')
                st.write(
                    'Les 5 features impactant le plus le score de ce client'
                )
                response = request_fi_locale(URL_API_FI_LOCALE, client_data_p1)

                # Affichage d'une erreur si la prédiction est None
                if response is None:
                    st.write('Erreur de feature importance locale\n')

                else :
                    # Transformation données pour SHAP en array puis en df
                    shap_values_subset_array = np.array(
                        response['fi_locale_subset']['shap_values_subset']
                    )

                    client_data_subset_df = pd.DataFrame(
                        [response['fi_locale_subset']['client_data_subset']],
                        columns=response['fi_locale_subset']['top_features']
                    )

                    # Affichage feature importance locale
                    shap.force_plot(
                        response["explainer"][1],
                        shap_values_subset_array,
                        client_data_subset_df,
                        matplotlib=True
                    )

                # Obtention de la figure actuelle et affichage streamlit
                fig = plt.gcf()
                st.pyplot(fig)

                st.write('Les features en bleu entrainent une diminution du score de risque de crédit tandis que celles en rouge entrainent une augmentation. ')

    # Page 2
    with tab2:
        st.title('DataViz features\n')

        # Récupération des clients et sélection d'un client
        client_id_p2 = fetch_data_and_client_selection(
            URL_API_CLIENT_SELECTION,
            key_unique='select_client_id_2'
        )

        # Récupération des données du client
        client_data_p2, client_data_brutes_p2 = get_client_data(
            URL_API_CLIENT_EXTRACTION,
            client_id_p2
        )

        # Création des colonnes
        col1, col2 = st.columns(2)

        # Colonne 1 : (gauche)
        with col1:
            st.title('Feature n°1\n')
            afficher_hist(
                client_data_brutes_p2,
                client_id_p2,
                URL_API_FEAT_PLOT,
                key_unique='histo_1'
            )

        # Colonne 2 : (droite)
        with col2:
            st.title('Feature n°2\n')
            afficher_hist(
                client_data_brutes_p2,
                client_id_p2,
                URL_API_FEAT_PLOT,
                key_unique='histo_2'
            )

    # Page 3
    with tab3:
        st.title('Analyse bivariée')

        # Récupération nom de feature et sélection d'une feature axe X et Y
        feature_name_x = fetch_feat_name_and_feat_selection(
            URL_API_FEAT_SELECTION,
            axe='X',
            key_unique='select_feat_name_x',
        )

        feature_name_y = fetch_feat_name_and_feat_selection(
            URL_API_FEAT_SELECTION,
            axe='Y',
            key_unique='select_feat_name_y',
        )

        # Récupération des données de la feature de l'axe X et Y
        feat_data_brutes_x = get_feat_data(
            URL_API_FEAT_EXTRACTION,
            feature_name_x,
            'X'
        )

        feat_data_brutes_y = get_feat_data(
            URL_API_FEAT_EXTRACTION,
            feature_name_y,
            'Y'
        )

        # Création du scatter plot avec ligne de corrélation
        fig, ax = plt.subplots()
        sns.regplot(
            x=feat_data_brutes_x,
            y=feat_data_brutes_y,
            ax=ax,
            line_kws={"color": "red"}
        )        
        ax.set_title('Analyse bivariée')
        ax.set_xlabel(feature_name_x)
        ax.set_ylabel(feature_name_y)

        # Affichage du plot dans Streamlit
        st.pyplot(fig)


    # Page 4
    with tab4:
        st.title('Feature importance Globale\n')

        st.write('Les features ayant le plus d’impact sur le jeux de données de tous les clients sont présentés ici par ordre d’importance (du haut vers le bas).')

        st.write('La disposition des couleurs de point (rouge à bleu) permet de comprendre si la feature est corrélée positivement ou négativement à la prédiction de score de risque de crédit.')

        st.write('Si les points rouges (valeurs les plus hautes) sont à droite alors cette feature est corrélée positivement au score de risque de crédit.')

        st.write('Si les points bleus (valeurs les plus basses) sont à droites alors cette feature est corrélée négativement au score de risque de crédit.')

        # Calcule et affichage de feature importance globale
        calcule_fi_globale(URL_API_FI_GLOBALE)


# =================== étape 5 : Run du dashboard ==========================

if __name__ == '__main__':
    main()
