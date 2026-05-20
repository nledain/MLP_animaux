#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:26:22 2026

@author: anoukvalabregue
"""

#%%
#Fonction

import os
import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.ndimage import zoom 


def mono (donnee):
    """
    Transforme un signal stéréo en signal mono
    prends en arguement un array a deux colonnes et renvoie un array en 
    une colonne
    """
    donnee_mono = (donnee[:,0].copy() +donnee[:,1].copy())/2 #on prends la moyenne du signal 
    
    return donnee_mono

dossier_entree = r"C:\Users\titi\OneDrive - INSA Lyon\S2\Projet\data\nael"

#listes vides 
liste_images = []
liste_labels = []

for racine, dossiers, fichiers in os.walk(dossier_entree):
    for fichier in fichiers:
        # reconstruire le chemin complet du fichier audio + nom (sous doss)
        chemin_audio = os.path.join(racine, fichier)
        nom_animal = os.path.basename(racine)
        
        try:
            # son et générer le spectrogramme
            samplerate, data = wavfile.read(chemin_audio)
            data = data * 1.0 #pour passer en float
            
            # transformation du signal en mono si nécessaire
            if len(data.shape) > 1:
                data = mono(data)
            
            #  spectrogramme (avec une figure temporaire)
           
            px,freq, t, im = plt.specgram(data, Fs = samplerate) #calcul automatiquement la transformé de Fourier en SFT
            plt.close()
            #regarder ce que renvoie exactement specgram
            
            # normalisation
            px_db = 10 * np.log10(px + 1e-10) #décibels 
            px_norm = (px_db - np.min(px_db)) / (np.max(px_db) - np.min(px_db)) #z-score
            
            # découpe temporelle (0.75s)
            somme = np.sum(px_norm, axis=0)
            mx = int(np.argmax(somme)) # ajout int / on  prends la valeur max d'une colonne
            
            duree_cible = 1 
            pas = float(t[1]-t[0]) # pour avoir float pur 
            nb_pix_cible = int(round(duree_cible / pas)) #nombre de pixels necessaire pour avoir 0.75s
            taille_temps = int(px_norm.shape[1]) 
            
            start = int(mx)
            end = int(start + nb_pix_cible)
            
            if end > taille_temps:
                end = taille_temps
                start = np.maximum(0, end - nb_pix_cible) #je comprends pas l'intêret ?????
          
            px_coupee = px_norm[:, start:end]
            
            # --- HARMONISATION DE LA TAILLE EN MÉMOIRE ---
            # On force chaque spectrogramme à faire exactement 129 lignes par 65 colonnes
            TAILLE_CIBLE = (129, 65)
            if px_coupee.shape != TAILLE_CIBLE:
                px_coupee2 = zoom(px_coupee, (TAILLE_CIBLE[0] / px_coupee.shape[0], TAILLE_CIBLE[1] / px_coupee.shape[1]))
            
            #ajout aux listes pour les utiliser par la suite
            liste_images.append(px_coupee2)
            liste_labels.append(nom_animal)
        
        except Exception as e :
            print(f"Erreur sur le fichier {fichier} : {e}")
        
#conversion on numpy pour la suite 
X = np.array(liste_images)
Y = np.array(liste_labels)

def permutation(x,y):
    n = len(x) #prends le nombre d'image
    n_perm = np.random.permutation(n)
    x_f = x[n_perm]
    y_f = y[n_perm]
    
    return x_f,y_f

X_f, Y_f = permutation(X,Y)

print("Spectrogrammes terminés")
print(f"Forme de la variable X (Images) : {X.shape}")
print(f"Forme de la variable Y (Labels) : {Y.shape}")


def stockage (img,labels):
    np.save("labels.npy",labels)
    np.save("images.npy",img)
    return "fichiers enregistrés"

#stockage(X_f,Y_f)




