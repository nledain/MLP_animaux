\#!/usr/bin/env python3

# \-*- coding: utf-8 -*-

"""
Created on Wed May 20 10:53:47 2026

@author: anoukvalabregue
"""

# MLP Animaux B4

## Projet

L'objectif du projet est de créer un algorithme permettant d'identifier le type d'animal en fonction de l'enregistrement du son émis par celui-ci.

## Librairies

Pour executer notre programme, les librairies/modules suivants sont requises :

* **os** : permet de naviguer dans l'arborescence des fichiers
* **numpy** : permet manipulation des matrices d'images et les opérations mathématiques
* **matplotlib.pyplot** : pour générer les spectrogrammes visuellement'
* **scipy.io.wavfile** : necessaire pour lire les fichiers audio au format wav et extraire leurs fréquences d'échantillionnage (issue de la librairie scipy)
* **scipy.ndimage.zoom** : permet de redimensionner les images dans une dimension uniforme (issue de la librairie scipy)
* **sklearn.dataset :** permet d'importer les images de minst (!installez la librairie scikit-learn!)

## Données



Les données sont issues de la banques Kaggle "Sound Classification of Animal Voice" publié par "Rushi Balaji Putthewa". https://www.kaggle.com/datasets/rushibalajiputthewad/sound-classification-of-animal-voice

Il contient les dons des différents animaux. Pour des raisons de performance, nous n'avons sélectionné que 5 animaux parmi les 13 disponibles. 

Pour chaque animal, 40 sons servirons de données d'entrainement et 10 de données de test.





## workflow d'utilisation du code



Le dossier de Projet est composé de deux fichiers .py, deux dossiers de données, un dossier de documentation et le fichier readme.

fonctions.py contient toutes les fonctions nécessaires au lancement du code. 

main.py contient le code principal. Il est composé de trois parties :

* partie 0 : contient les librairies et set la seed pour la reproductibilité
* partie 1 : cette partie permet de récupérer les sons d'animaux et de les transformer en spectrogramme normalisé et scalé
* partie 2 : cette partie lance l'autoencodeur et le MLP



ATTENTION : dû à de très faible résultat des réseaux de neurones avec les spectrogrammes, il est préférable de tester le réseaux avec le dataset mninst (voir ci-dessous)



1. ouvrir le code de main.py
2. modifier la variable "chemin\_data" et la remplacer par le chemin jusqu'au dossier "2026\_PIML\_B4-main"(ou autre si vous avez changer le nom du dossier projet).
3. modifier la variable "DATASET" (partie 3) pour tester le réseau vec les spectrogrammes ("mnist") ou les spectrogrammes("spectres")



indication : vous pouvez visualiser les données avec plt.imshow(images\_train\[n]) ou plt.imshow(images\_test\[n]), avec n l'index de l'image.



## Pré-traitement Audio

Le script parcours un dossier contenant les fichiers audio .wav triés par sous dossiers représentant les différentes classes d'animaux.
Le traitement se fait en plusieurs étapes

* **Conversion en mono** : fusionne les pistes si le fichier audio est en stéréo en faisant la moyenne des deux pour garder un signal 1D
* **Génération du spectrogramme** : utilisation de la fonction "plt.specgram()" pour extraire px (la matrice de densité spectrale = intensité du son), freq (un array 1D avec les fréquences donc le Y du spectrogramme), t (array 1D avec les X du spectrogramme) et im le graphique du spectrogramme
* **Normalisation et découpage temporel** :
* passage en **décibels** : application de la formule mathématiques
* **normalisation** : les valeurs des pixels sont ramenés entre 0 et 1
* \*\* découpe temporel\*\* : identifier le max d'intensité comme le début de l'enregistrement et le couper ensuite 1sec après : l'objectif est d'éliminer le bruit du début d'enregistrement et faire commencer tous les cris dès le début de l'enregistrement
* **Standardisation** : afin de garantir que toutes les entrées possèdent les mêmes dimensions pour le réseau de neurones : utilisation de la fonction "zoom" de **SciPy** force chaque matrice à avoir une taille fixe d'ici 129 lignes par 65 colonnes '
* **Sauvegarde** : enregistrement des images coupées dans des listes initialement vides puis la fonction stockage exporte les données traitées sous forme de fichiers Numpy
* *image.npy* contient la matrice globale des images de spectrogrammes
* *labels.npy* contient les étiquettes correspondantes aux noms des animaux

## Auto encodeur:

### Principe



L'auto encodeur sert à réduire les dimensions des images d'entrainement et de test afin qu'elles soient admise dans le mlp classificateur après. Il se constitue en 3 parties:

1. l'encodeur, un réseau neuronale simple avec un nombre de neurones décroissant couche après couche qui va "encoder" l'information et la réduire en dimensions du nombre de neurones de la dernière couche de cette étape.
2. Le goulot, c'est la dernière couche de l'encodeur, et la couche du milieu de l'auto encodeur dans sa globalité, c'est ici que l'information est le plus réduit, et si lors de la reconstruction, l'image est récrée assez fidèlement, alors cela signifie qu'au goulot, l'information est réduite en dimensions sans trop de perte, après entrainement, lors des test, c'est les coefficients de cette couche que l'on récupèrera et considéra comme la nouvelle image pour le mlp classificateur.

3\) Le décodeur, est un réseau de neurones avec des tailles de couches croissantes, qui est symétrique à l'encodeur et reforme donc une image à partir des coefs du goulot

Dans l'algo, la mse regarde si l'image reconstruite est proche de l'image de base (MSE classique), le forward, le backward ainsi que l'update des poids fonctionne de la même manière qu'un mlp classique



### Fonctionnement de l'entrainement



pour l'entrainer on fonctionne en Mini-batch, avec la fonction du même nom, qui prendra en entrée X, la matrice des images normalisé sur lequel l'algo s'entrainera, et Y la matrice des résultats, qui vaut dans ce cas est aussi X.

Pour entrainer l'auto-encodeur: On initialise d'abord W et B avec la fonction init\_params, qui crée les matrices W et B, adapté selon l'architecture du réseau demandé en entrée (avec layer\_size), un type d'initialisation (adapté selon la fonction d'activation du réseau avec init\_type) et la seed utilisé pour l'aléatoire (seed)
On l'entraine ensuite avec Mini\_Batch, qui peut varier selon les paramètres suivants:

\-nb\_epoch= le nombre d'epoch demander pour l'entrainement

\-Lr: le Learning rate, arbitraire selon le modèle

\-nom\_tech\_Loss : le nom de la fonction de loss, possible entre (cross\_entropy ou MSE\_reconstruction)

\-nom\_tech\_activation: le nom de la fonction d'activation utilisé pour les couches cachées, possible entre (relu, sigmoid ou softmax)

\-act\_finale: Le nom de la fonction d'activation de la couche finale (Relu, sigmoid ou softmax)

\-Batch\_size: taille des groupes fait par l'algo pour réaliser la backpropagation, adaptable selon capacité de la machine



Pour vérifier si l'entrainement a bien fonctionné, on utiise la fonction vue\_reconstruite qui permet de regarder visuellement si quelques images utilisé pour l'entrainement, on été bien reconstruite par le modèle (ainsi que la loss bien sûr):

\-act, la fonction d'activation des couches cachés, possible entre Relu, sigmoid et softmax

\-act\_finale: Le nom de la fonction d'activation de la couche finale (Relu, sigmoid ou softmax)

\-n: Le nombre d'image que l'on affiche avec son image reconstruite pour regarder la qualité de la reconstuction



### Fonctionnement du modèle entrainé:

Une fois l'auto-encodeur entrainé, lorsqu'on lui envoi une image, on considère qu'il réduit l'image à la dimension du goulot sans perdre beaucoup d'information de cette image, pour l'utiliser, on lui donne donc une image, dans la fonction encode qui représente la première partie décroissante jusqu'au goulot, qui fait le forward jusqu'au goulot avec les poids et biais entrainés, cette fonction encode fonctionne avec les paramètres variables suivants:

\-act, la fonction d'activation des couches cachés de l'auto-encodeur, possible de choisir entre Relu, sigmoid et softmax



## MLP classique:



### Principe:



Le mlp est un algorithme de classification supervisé qui fonctionne en couche de neurones décroissantes, la dernière couches ayant comme nombre de neurones, le nombre de classes à classifier. Son principe d'entrainement est simple:

1)Forward, le réseau fait une prédiction à partir de tout ses neurones et ses coefs associés

2)Loss, le réseau calcule sa perte en comparant les bonnes des mauvaises prédictions

3\) Backpropagation, le réseau revient en arrière et regarde couche après couche d'où vient l'erreur (descente de gradient) et quels coefs sont responsable de l'erreur

4)Update: le réseau modifie légèrement (Learning rate) les coefs selon leurs importance dans l'erreur pour ajuster et être plus précis à sa prochaine prédiction



Le MLP classificateur sert à classifier les images compressées par l’auto-encodeur dans les différentes classes d’animaux. Contrairement à l’auto-encodeur, son objectif n’est pas de reconstruire une image mais de prédire à quelle classe appartient l’image donnée en entrée. Le réseau est constitué de plusieurs couches de neurones entièrement connectées :



Les couches cachées, qui réalisent progressivement une extraction et une transformation des informations importantes contenues dans les coefficients issus du goulot de l’auto-encodeur. Chaque couche applique une transformation linéaire suivie d’une fonction d’activation non linéaire afin de permettre au réseau d’apprendre des relations complexes entre les données et les classes.

La couche de sortie, qui contient autant de neurones que de classes d’animaux à prédire. Cette dernière couche utilise généralement la fonction softmax afin de transformer les sorties du réseau en probabilités. Chaque neurone représente alors la probabilité que l’image appartienne à une classe donnée, et la prédiction finale du modèle correspond à la classe ayant la probabilité la plus élevée.



Dans l’algorithme, le forward réalise les calculs couche après couche afin de produire les probabilités de sortie du modèle. La fonction de loss compare ensuite ces probabilités avec les vraies classes des images. Le backward calcule les gradients nécessaires à la correction des poids grâce à la backpropagation, puis la fonction update modifie les poids et biais afin d’améliorer progressivement les prédictions du réseau.



### Fonctionnement de l'entrainement:



Pour entraîner le MLP classificateur, on utilise également un fonctionnement en Mini-Batch avec la fonction Mini\_Batch. Les données d’entrée X correspondent ici aux coefficients compressés récupérés au niveau du goulot de l’auto-encodeur, tandis que Y correspond aux labels des classes des images.



Avant l’entrainement, on initialise les poids W et biais B avec la fonction init\_params, qui crée automatiquement les matrices adaptées à l’architecture du réseau demandée dans layer\_sizes. Cette initialisation dépend :



\-de la taille des couches demandées

\-du type d’initialisation choisi avec init\_type

\-de la seed utilisée pour l’aléatoire



Le modèle est ensuite entraîné avec Mini\_Batch, qui peut varier selon plusieurs paramètres :



nb\_epoch : nombre d’epochs d’entrainement du modèle

Lr : learning rate utilisé pour les mises à jour des poids

nom\_tech\_Loss : fonction de loss utilisée, généralement cross\_entropy pour la classification

nom\_tech\_activation : fonction d’activation des couches cachées, possible entre Relu ou sigmoid

act\_finale : fonction d’activation finale, généralement softmax pour obtenir des probabilités de classes

batch\_size : taille des groupes utilisés pour la backpropagation, dépendant des capacités de calcul de la machine



Pendant l’entrainement, les images sont mélangées aléatoirement puis séparées en mini-batchs. Pour chaque mini-batch :



le réseau réalise un forward pour produire les probabilités de chaque classe

la loss calcule l’erreur entre les prédictions et les vraies classes

le backward calcule les gradients de tous les poids du réseau

update applique les corrections aux poids et biais



Ce processus est répété pour tous les mini-batchs et toutes les epochs afin d’améliorer progressivement les performances du modèle.



### Fonctionnement du modèle entraîné:



Une fois entraîné, le MLP classificateur reçoit en entrée les coefficients compressés produits par l’encodeur de l’auto-encodeur. Ces coefficients représentent une version réduite mais informative des images originales.



Le réseau effectue alors un forward complet à travers toutes ses couches jusqu’à la couche finale softmax, qui produit les probabilités d’appartenance aux différentes classes. La classe prédite correspond à celle ayant la probabilité maximale.



Le modèle peut ensuite être évalué grâce à différentes métriques :



l’accuracy, qui mesure le pourcentage global de bonnes prédictions

le recall, qui mesure la capacité du modèle à retrouver les vrais exemples d’une classe

le F1-score, qui combine précision et recall afin d’obtenir une mesure plus robuste des performances, particulièrement utile si les classes sont déséquilibrées.

