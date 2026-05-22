# ── fonctions.py ─────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.ndimage import zoom
import os 

# ── Prétraitement audio ───────────────────────────────────────────────────────
def mono (donnee):
    """
    Objectif : Convertir un signal audio stéréo en signal mono en calculant la moyenne des deux canaux.

    Entrées : donnee (numpy array 2D avec deux colonnes représentant les canaux stéréo).

    Return : Un numpy array 1D correspondant au signal mono.
    """
    donnee_mono = (donnee[:,0].copy() +donnee[:,1].copy())/2 #on prends la moyenne du signal 
    
    return donnee_mono

def permutation(x,y,seed):
    """
   Objectif : Mélanger aléatoirement deux tableaux d'images et de labels tout en conservant la correspondance entre eux.

    Entrées : x (les données/images), y (les labels correspondants), seed (un générateur numpy rng pour la reproductibilité).
    
    Return : x_f, y_f (les deux tableaux mélangés de manière identique).
    """
    n = len(x) #prends le nombre d'image
    n_perm = seed.permutation(n)
    x_f = x[n_perm]
    y_f = y[n_perm]
    
    return x_f,y_f

def stockage (img,labels,dossier_entree,fichier_img,fichier_label):
    """
    Objectif : Sauvegarder les tableaux de données et de labels dans des fichiers .npy sur le disque.

    Entrées : img (array des images), labels (array des labels), dossier_entree (chemin du dossier de destination), fichier_img (nom du fichier image), fichier_label (nom du fichier label).

    Return : Rien (affiche un message de confirmation dans la console).
    """
    path_label = os.path.join(dossier_entree,fichier_label)
    path_image = os.path.join(dossier_entree,fichier_img)
    np.save(path_label.replace('\\', '/'),labels)
    np.save(path_image.replace('\\', '/'),img)

    return print("fichiers enregistrés")

def spectrogramme(dossier_entree,liste_images,liste_labels,dossier):
    """
    Objectif : Parcourir un dossier d'audios, générer des spectrogrammes normalisés (convertis en décibels et coupés sur 1 seconde autour du pic d'intensité), les redimensionner à une taille fixe (129x65) et les associer à leur classe animale.

    Entrées : dossier_entree (chemin racine), liste_images et liste_labels (listes vides à remplir), dossier (sous-dossier contenant les classes/audios).

    Return : liste_images, liste_labels (deux numpy arrays contenant respectivement les matrices 2D des spectrogrammes et les chaînes de caractères des noms d'animaux).
    """
    path = os.path.join(dossier_entree,dossier)
    for racine, dossiers, fichiers in os.walk(path):
        for fichier in fichiers:
            # reconstruire le chemin complet du fichier audio + nom (sous doss)
            chemin_audio = os.path.join(racine, fichier)
            nom_animal = os.path.basename(racine)
            
            try:
                # extraire le son et générer le spectrogramme
                samplerate, data = wavfile.read(chemin_audio)
                data = data * 1.0 #float 
                
                # transformation du signal en mono si nécessaire
                if len(data.shape) > 1:
                    data = mono(data)
                
                #  spectrogramme 
                px, freq, t, im = plt.specgram(data, Fs=samplerate)
                plt.close()
                """
                px = matrice de densité spectrale = intensité du son 
                freq = array 1D avec les Y du spectro 
                t = array 1D avec les X du spectro 
                im = graphique du spectro 
                """
                
                # normalisation
                px_db = 10 * np.log10(px + 1e-9) #décibel
                px_norm = (px_db - np.min(px_db)) / (np.max(px_db) - np.min(px_db))
                
                # découpe temporelle (1s)
                somme = np.sum(px_norm, axis=0)
                mx = int(np.argmax(somme)) # ajout int
                
                duree_cible = 1
                pas = float(t[1]-t[0]) # pour avoir float pur 
                nb_pix_cible = int(round(duree_cible / pas))
                taille_temps = int(px_norm.shape[1])
                
                start = int(mx)
                end = int(start + nb_pix_cible)
                
                if end > taille_temps:
                    end = taille_temps
                    start = np.maximum(0, end - nb_pix_cible)
                    
                px_coupee = px_norm[:, start:end]
                
                # On force chaque spectrogramme à faire exactement 129 lignes par 65 colonnes
                TAILLE_CIBLE = (129, 65)
                if px_coupee.shape != TAILLE_CIBLE:
                    px_coupee = zoom(px_coupee, (TAILLE_CIBLE[0] / px_coupee.shape[0], TAILLE_CIBLE[1] / px_coupee.shape[1]))
                #ratio de zoom pour axe 0 et 1 
                
                liste_images.append(px_coupee)
                liste_labels.append(nom_animal)
            
            except Exception as erreur :
                print(f"Erreur sur le fichier {fichier} : {erreur}")
                
    liste_images = np.array(liste_images)
    liste_labels = np.array(liste_labels)
                
    return liste_images,liste_labels
# ── Activations ───────────────────────────────────────────────────────────────
def Relu(z):
    """
    Applique la fonction Relu à un np.array
    """
    return np.maximum(0,z)

def dRelu(z):
    """
    Calcule la dérivé de la fonction Relu
    """
    return (z > 0).astype(float)

def sigmoid(z):
    """
    Applique la fonction Sigmoide à un np.array
    """
    return 1/(1+np.exp(-z))

def dsigmoid(z):
    """
    Calcule la dérivé de la fonction Sigmoïde
    """
    s = sigmoid(z)
    return s * (1 - s)


def softmax(Z):    
    """
    Objectif : Convertir les scores bruts de la dernière couche en probabilités (somme = 1),
    ce qui permet d'interpréter la sortie du réseau comme une distribution de probabilité
    sur les classes. Z-np.max(Z) =astuce numérique pour éviter l'overflow (stabilité de l'exponentielle).

    Entrées : Z (numpy array 2D de shape (N, n_classes), valeurs pré-activation de la dernière couche).

    Return : Un numpy array de même shape où chaque ligne somme à 1 (probabilités par classe).
    """
    Z = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=1, keepdims=True)

# ── Initialisation ────────────────────────────────────────────────────────────
def init_params(layer_sizes, seed=None):
    """
    Objectif : Initialiser les poids d'un réseau de neurones avec une distribution uniforme entre -0.5 et 0.5, et les biais à zéro.

    Entrées : layer_sizes (liste du nombre de neurones par couche), seed (entier).

    Return : W, B (deux dictionnaires contenant respectivement les matrices de poids et les vecteurs de biais pour chaque couche).
    """
    if seed is not None:
        np.random.seed(seed)
        
    W = {}
    B = {}
    L = len(layer_sizes) - 1
    
    for l in range(1, L + 1):
        n_prev = layer_sizes[l - 1]
        n_curr = layer_sizes[l]
        
        # Logique de la 2ème fonction : tirage uniforme dans [0, 1[ puis on soustrait 0.5
        # Format (n_prev, n_curr) et dictionnaires pour respecter les sorties de la 1ère fonction
        W[l] = np.random.rand(n_prev, n_curr) - 0.5
        
        # Biais à zéro, format (1, n_curr) comme dans la 1ère fonction
        B[l] = np.zeros((1, n_curr))
        
    return W, B


def init_params2(layer_sizes, seed=None):
    """
    Objectif : Initialiser les poids d'un réseau avec une méthode avancé, spécifique à Relu

    Entrées : layer_sizes (liste des tailles de couches), seed (entier).

    Return : W, B (dictionnaires des poids et biais).
    """
    if seed is not None:
        np.random.seed(seed)
    W = {}
    B = {}
    L = len(layer_sizes) - 1
    for l in range(1, L + 1):
        n_prev = layer_sizes[l - 1]
        n_curr = layer_sizes[l]
        # ==========================================
        # INITIALISATION DES POIDS
        # ==========================================
        W[l] = np.random.randn(n_prev, n_curr) * np.sqrt(2 / n_prev)
        # ==========================================
        # BIAIS
        # ==========================================
        B[l] = np.zeros((1, n_curr))
    return W, B

# ── Forward ───────────────────────────────────────────────────────────────────
def linear_forward(X, W, B, act,act_finale):
    """
    Objectif : Permet de réalsier une forward propagation.

    Entrées : X (données d'entrée), W, B (dictionnaires des paramètres), act (fonction d'activation des couches cachées), act_finale (fonction d'activation de la dernière couche).

    Return : Zs, As (deux dictionnaires stockant respectivement les valeurs pré-activation Z et post-activation A pour chaque couche).
    """
    Zs={}
    As={0: X}
    L=len(W) #nombre de couche de neurones dans le réseau
    for l in range(1, L + 1):
        A_prev = As[l - 1]
        Z = np.dot(A_prev, W[l]) + B[l]
        Zs[l] = Z
        if l < L:
            As[l] = act(Z)          # couche normales
        else:
            As[l] = act_finale(Z)      # couche de sortie
    return Zs, As

# ── Loss ──────────────────────────────────────────────────────────────────────
def cross_entropy(Y, A_L):
    """
    Objectif : Calculer la fonction de coût Cross-Entropy et son gradient.

    Entrées : Y (vrais labels), A_L (prédictions du réseau après softmax).

    Return : J (valeur de la loss), dZ_L (matrice des gradients à la dernière couche).
    """
    eps = 1e-8
    N = Y.shape[0]
    J = -(1 / N) * np.sum(Y * np.log(A_L + eps))
    dZ_L = (A_L - Y) / N   # divisé par N car on normalise
    return J, dZ_L 

'''
encodeur type: 512-128-64-128-512
 L'auto-encodeur se code avec les différentes fonctions:
 
 -init_parameter: rien à changer par rapport au mlp, permet d'initialiser W et B
 
 -linear forward: on l'utilise aussi mais avec la sigmoid à la fin pas la softmax
 
 -backward: même utilisation pour regarder en arrière les poids etc
 
 -update: même utilisation pour update les poids dans les couches précédentes
 
 -Fonction de loss: Remplace cross_entropy. Compare l'image reconstruite à l'originale.
 Retourne la loss et le gradient dZ pour lancer la backprop.
 
 -Mini-Batch: ça va être la pipeline d'entrainement utilisé mais la sortie ça ne sera pas des labels Y mais simplement X
 
 -Encode: Va couper le réseau au milieu (au goulot) à la fin et récuperer l'information sous image réduite.
 Fait tourner le forward uniquement jusqu'au goulot (couche du milieu). Retourne la représentation compressée 
 
 -decode:Fait tourner le forward depuis le goulot jusqu'à la sortie. Utile pour visualiser ce que le réseau reconstruit et vérifier qu'il apprend bien.
 
 -vue_reconstruite: fonction pas absolument nécéssaire mais j'ai galérer à debuger sans elle donc elle sert de fou, 
 Affiche côte à côte n images originales et leurs reconstructions.si les reconstructions sont floues ou fausses, le réseau n'apprend pas.
 '''
 
def MSE_reconstruction(X, X_reconstruit, Z_L):  #ADAPTE POUR SIGMOID CAR MEILLEUR FONCTION D'ACTIVATION POUR AUTO-ENCODEUR
    """
    Objectif : Calculer l'erreur quadratique moyenne entre l'image d'origine et celle générée par l'auto-encodeur, ainsi que le gradient associé avec une dérivée Sigmoïde.

    Entrées : X (données d'origine), X_reconstruit (sortie de l'auto-encodeur), Z_L (valeurs pré-activation de la dernière couche).

    Return : loss (scalaire), dZ (matrice des gradients).
    """
    N = X.shape[0]          # nombre d'exemples dans le batch
    nb_pixels = X.shape[1]  # 16384 pour du 128x128
    
    loss = (1/(N * nb_pixels)) * np.sum((X - X_reconstruit)**2)
    dA = (2 / (N * nb_pixels)) * (X_reconstruit - X)
    dZ=dA*dsigmoid(Z_L)
    return loss, dZ

def Loss_choisi(nom_technique, Y_r, Y_p, Z_final=None):
    """
    Objectif : Dispatcher vers la bonne fonction de loss selon la tâche
    (classification avec cross_entropy, ou reconstruction avec MSE_reconstruction).

    Entrées : nom_technique (référence à la fonction de loss), Y_r (valeurs réelles),
    Y_p (valeurs prédites / sortie du réseau), Z_final (valeurs pré-activation de la
    dernière couche, requis uniquement pour MSE_reconstruction).

    Return : J (valeur scalaire de la loss), dZ_L (gradient à la dernière couche
    pour lancer la backpropagation).
    """
    if nom_technique==cross_entropy:
        J, dZ_L=nom_technique(Y_r,Y_p)
    elif nom_technique==MSE_reconstruction:
        J,dZ_L=nom_technique(Y_r,Y_p, Z_final)
    else:
        print("La fonction de loss choisi n'est pas encore bien codé car pas adapté")
        print("Donc toute la suite peut être considéré faux")
    return J, dZ_L  #On récup la valeur de la loss et DA

# ── Backward ──────────────────────────────────────────────────────────────────
def activation_backward(nom_act, Z):
    """
    Objectif : Dispatcher vers la bonne dérivée d'activation lors de la backpropagation,
    en fonction de la fonction d'activation utilisée en forward.

    Entrées : nom_act (référence à la fonction d'activation, ex: Relu ou sigmoid),
    Z (numpy array des valeurs pré-activation de la couche concernée).

    Return : numpy array de même shape que Z — la dérivée de l'activation appliquée en Z.
    """
    if nom_act == Relu:
        return dRelu(Z)
    elif nom_act == sigmoid:
        return dsigmoid(Z)
    else:
        
        raise ValueError("Activation non reconnue.")
        
'''
Y: Les réultats
W: Dico des poids par couche
Zs: Dico des val prédit dans chaque neurone (avant application de non-linearité) par couche
As: Dico des val prédites après softmax par couche(pour mlp sinon c'est après sigmoid pour l'auto-encodeur)
dZ_L: Liste de la variation des val prédites à la dernière couche, calculé dans cross_entropy
gradients_W: Dico des dw par couche
gradients_B: Dico des db par couche
Fonctionnement: on parcourt le réseau à l'envers puisqu'on remonte les erreurs
                à chaque couche n, on choppe le A de la couche n-1,avec, on calcule
                dw de la couche (liste de couche et une valeur par neurone)
'''

def backward(Y, W, Zs, As, dZ_L, nom_act,clip=None):#Fonction complexe, remonte chaque couche depuis la fin, pour trouver les dZ, les dA, les dw et db
    """
    Objectif : Effectuer la rétropropagation du gradient à travers toutes les couches
    du réseau, en remontant de la dernière couche vers la première pour calculer
    les gradients dW et dB de chaque couche.

    Fonctionnement couche par couche (de L à 1) :
      - On reçoit dZ_L (gradient sur la sortie, calculé par la loss).
      - Pour chaque couche l :
          dW[l] = A[l-1].T · dZ[l]        (gradient des poids)
          dB[l] = somme(dZ[l])             (gradient des biais)
          dA[l-1] = dZ[l] · W[l].T        (on propage l'erreur vers la couche précédente)
          dZ[l-1] = dA[l-1] * f'(Z[l-1])  (on applique la dérivée de l'activation)
      - Le clip optionnel borne dZ dans [-clip, +clip] à chaque couche pour éviter
        l'explosion du gradient sur des réseaux profonds ou des données bruitées.

    Entrées :
      Y        — labels réels (non utilisés directement ici, le gradient vient de dZ_L)
      W        — dictionnaire des poids par couche
      Zs       — dictionnaire des valeurs pré-activation Z par couche (issu de linear_forward)
      As       — dictionnaire des valeurs post-activation A par couche (As[0] = X)
      dZ_L     — gradient à la dernière couche, calculé par la fonction de loss
      nom_act  — référence à la fonction d'activation des couches cachées (ex: Relu)
      clip     — valeur de clipping du gradient (float ou None)

    Return : gradients_W, gradients_B (deux dictionnaires indexés par numéro de couche).
    """
    
    gradients_W = {}
    gradients_B = {}
    L = len(W)
    dZ_current = dZ_L
    for l in reversed(range(1, L + 1)):
        if clip is not None:
            dZ_current = np.clip(dZ_current, -clip, clip)# on clip car le gradient peut exploser et faire exploser la loss
        A_prev = As[l - 1]
        gradients_W[l] = np.dot(A_prev.T, dZ_current)
        gradients_B[l] = np.sum(dZ_current, axis=0, keepdims=True)
        if l > 1:
            dA_prev = np.dot(dZ_current, W[l].T)
            dZ_current = dA_prev * activation_backward(nom_act, Zs[l - 1])

    return gradients_W, gradients_B

# ── Update ────────────────────────────────────────────────────────────────────
def update(dw,db,w,b, Lr):
    """
    Objectif : Mettre à jour les poids et biais du réseau en appliquant la règle
    de descente de gradient : nouveau_poids = ancien_poids - taux_apprentissage * gradient.

    Entrées : dw, db (dictionnaires des gradients calculés par backward()),
    w, b (dictionnaires des poids et biais actuels), Lr (taux d'apprentissage, float).

    Return : W_new, B_new (nouveaux dictionnaires de poids et biais mis à jour).
    """
    L=len(w)
    W_new={}
    B_new={}
    for i in range(1,L+1):
        W_new[i]=w[i]-Lr*dw[i]
        B_new[i]=b[i]-Lr*db[i]
    return W_new,B_new

# ── Entraînement ──────────────────────────────────────────────────────────────
def Batch_complet(X,nb_epoch, Lr, Y,W,B, Nom_tech_Loss,Nom_tech_activation,act_finale):
    """
    Objectif : Entraîner le réseau en mode batch complet (full-batch gradient descent) :
    à chaque epoch, on calcule le gradient sur l'intégralité du dataset avant de
    mettre à jour les poids une seule fois.

    Entrées :
      X                — données d'entrée complètes (shape N × nb_pixels)
      nb_epoch         — nombre total d'epochs d'entraînement
      Lr               — taux d'apprentissage (learning rate)
      Y                — labels / cibles (shape N × n_classes pour classification, N × nb_pixels pour AE)
      W, B             — dictionnaires des poids et biais initiaux
      Nom_tech_Loss    — référence à la fonction de loss (cross_entropy ou MSE_reconstruction)
      Nom_tech_activation — référence à la fonction d'activation des couches cachées
      act_finale       — référence à la fonction d'activation de la couche de sortie

    Return : W, B (poids entraînés), cout (liste des valeurs de loss par epoch).
    """
    cout=[]
    L=len(W)
    for i in range(nb_epoch):
        Zs, As=linear_forward(X,W,B,Nom_tech_activation,act_finale)
        J,dZ_L=Loss_choisi(Nom_tech_Loss, Y,As[L])
        gradients_w,gradients_b=backward(Y,W,Zs,As,dZ_L,Nom_tech_activation)
        W,B=update(gradients_w,gradients_b,W,B,Lr)
        cout.append(J)
        if i % 5 == 0:
            print(f"Epoch {i}/{nb_epoch} — Loss: {J:.4f}")
    return W,B, cout

def Mini_Batch(X, nb_epoch, Lr, Y, W, B, Nom_tech_Loss, Nom_tech_activation,act_finale, batch_size=64, clip=None):
    """
    Objectif : Entraîner le réseau avec la descente de gradient stochastique par mini-batchs :
    à chaque epoch, le dataset est mélangé puis découpé en petits blocs de taille batch_size,
    et les poids sont mis à jour après chaque bloc.

    Pipeline à chaque epoch :
      1. Mélange aléatoire du dataset complet (np.random.permutation).
      2. Pour chaque mini-batch de taille batch_size :
           a. Forward   : linear_forward(X_batch, ...)  → Zs, As
           b. Loss      : Loss_choisi(...)               → J, dZ_L
           c. Backward  : backward(...)                  → gradients (avec clip optionnel)
           d. Update    : update(...)                    → W, B mis à jour
      3. La loss de l'epoch est la moyenne des loss de tous les mini-batchs.
      4. Affichage toutes les 5 epochs.

    Entrées :
      X                   — données d'entrée (shape N × nb_pixels)
      nb_epoch            — nombre total d'epochs
      Lr                  — taux d'apprentissage
      Y                   — labels / cibles
      W, B                — dictionnaires des poids et biais initiaux
      Nom_tech_Loss       — référence à la fonction de loss
      Nom_tech_activation — référence à la fonction d'activation des couches cachées
      act_finale          — référence à la fonction d'activation de la couche de sortie
      batch_size          — nombre d'exemples par mini-batch (défaut : 64)
      clip                — valeur de clipping du gradient (float ou None, défaut : None)

    Return : W, B (poids entraînés), cout (liste des loss moyennes par epoch).
    """
    cout = []
    L = len(W)
    N = X.shape[0]
    for i in range(nb_epoch):
        indices = np.random.permutation(N)
        X_s = X[indices]
        Y_s = Y[indices]
        epoch_loss = []
        for j in range(0, N, batch_size):
            X_batch = X_s[j:j + batch_size]
            Y_batch = Y_s[j:j + batch_size]
            Zs, As = linear_forward(X_batch, W, B, Nom_tech_activation,act_finale)         
            J, dZ_L = Loss_choisi(Nom_tech_Loss, Y_batch, As[L], Zs[L])
            gradients_w, gradients_b = backward(Y_batch, W, Zs, As, dZ_L, Nom_tech_activation, clip=clip)
            W, B = update(gradients_w, gradients_b, W, B, Lr)
            epoch_loss.append(J)
        mean_loss = np.mean(epoch_loss)
        cout.append(mean_loss)
        if i % 5 == 0:
            print(f"Epoch {i}/{nb_epoch} — Loss: {mean_loss:.4f}")
    return W, B, cout

# ── Auto-encodeur ─────────────────────────────────────────────────────────────
def encode(X,W,B, act):
    """
    Objectif : Faire passer les données dans la partie encodeur de l'auto-encodeur,
    c'est-à-dire uniquement jusqu'à la couche centrale (goulot d'étranglement),
    pour obtenir la représentation compressée/latente des données.

    Entrées : X (données d'entrée, shape (N, nb_pixels)), W et B (dictionnaires des poids
    et biais de l'auto-encodeur complet), act (fonction d'activation des couches cachées).

    Return : Un numpy array de shape (N, taille_goulot) — la représentation latente compressée.
    """
    milieu=len(W)//2
    X_encoded={0:X}
    for i in range(1,milieu+1):
        A_prev=X_encoded[i-1]
        Z = np.dot(A_prev, W[i]) + B[i]
        X_encoded[i]=act(Z)
    return X_encoded[milieu]

def decode(X_encoded,W,B,act,act_finale):
    """
    Objectif : Faire passer la représentation latente dans la partie décodeur
    de l'auto-encodeur (de la couche centrale jusqu'à la sortie) pour reconstruire
    l'image d'origine.

    Entrées : X_encoded (représentation latente, shape (N, taille_goulot)), W et B
    (dictionnaires des poids et biais), act (activation des couches cachées),
    act_finale (activation de la couche de sortie, typiquement sigmoid).

    Return : Un numpy array de shape (N, nb_pixels) — les images reconstruites.
    """
    milieu=len(W)//2
    X_decoded={milieu:X_encoded}
    for i in range(milieu+1,len(W)+1):
        A_prev=X_decoded[i-1]
        Z = np.dot(A_prev, W[i]) + B[i]
        if i < len(W):
            X_decoded[i] = act(Z)          # couche normales
        else:
            X_decoded[i] = act_finale(Z)      # couche de sortie
    return X_decoded[len(W)]

def vue_reconstruite(X, W, B, act, act_finale,D, E, n=5):
    """
    Objectif : Vérifier visuellement la qualité de reconstruction de l'auto-encodeur
    en affichant côte à côte n images originales (ligne du haut) et leurs versions
    reconstruites par le réseau (ligne du bas).

    Fonctionnement pour chaque image i :
      1. encode(x_single, W, B, act)          - représentation latente compressée
      2. decode(x_encoded, W, B, act, act_finale) - image reconstruite aplatie
      3. reshape en (D, E) pour l'affichage
      4. Affichage avec imshow (colormap "gray")

    Entrées :
      X         — données de test aplaties (shape N × nb_pixels)
      W, B      — dictionnaires des poids et biais de l'auto-encodeur entraîné
      act       — fonction d'activation des couches cachées (ex: Relu)
      act_finale — fonction d'activation de la couche de sortie (ex: sigmoid)
      D         — hauteur de l'image en pixels (nombre de lignes au reshape)
      E         — largeur de l'image en pixels (nombre de colonnes au reshape)
      n         — nombre d'images à afficher (défaut : 5)

    Return : Rien — affiche directement une figure matplotlib 2×n.
    """
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))
    for i in range(n):
        # reconstruction
        x_single = X[i:i+1]                                    # shape (1, 16384)
        x_encoded = encode(x_single, W, B, act)
        x_decoded = decode(x_encoded, W, B, act, act_finale)   # shape (1, 16384)
        # reshape en image 128x128
        original     = x_single.reshape(D, E)
        reconstruit  = x_decoded.reshape(D, E)
        # ligne du haut : originales
        axes[0, i].imshow(original, cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=12)
        # ligne du bas : reconstruites
        axes[1, i].imshow(reconstruit, cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstruit", fontsize=12)
    plt.tight_layout()
    plt.show()

# ── Évaluation ────────────────────────────────────────────────────────────────
def evaluation(pred, y_test_subset, n_classes, label_to_int=None):
    """
    Objectif : Calculer et afficher les métriques de performance du classifieur :
    accuracy globale, puis précision, recall et F1-score pour chaque classe,
    avec une ligne de moyenne finale.

    Les trois métriques par classe :
      - Précision  = TP / (TP + FP) : parmi tout ce que le modèle a prédit comme
                     classe C, quelle fraction l'était vraiment ? Mesure les FP
      - Recall     = TP / (TP + FN) : parmi tous les vrais exemples de classe C,
                     quelle fraction a été correctement détectée ? Mesure les oublis: FN
      - F1-score   = 2 × (Précision × Recall) / (Précision + Recall) : moyenne harmonique
                     des deux, utile quand les classes sont déséquilibrées.
    Les cas dégénérés (TP+FP=0 ou TP+FN=0) sont gérés en retournant 0 pour éviter
    les divisions par zéro.

    Entrées :
      pred           — array 1D des classes prédites (entiers, issu de np.argmax)
      y_test_subset  — array 1D des vraies classes (entiers)
      n_classes      — nombre total de classes
      label_to_int   — dictionnaire {nom_animal: indice} pour afficher les noms lisibles
                       à la place des indices (optionnel, défaut : affiche les indices)

    Return : Rien — affiche un tableau formaté dans la console.
    """
    accuracy = np.mean(pred == y_test_subset)
    print(f"Accuracy : {accuracy * 100:.2f}%")
    recall_par_classe    = np.zeros(n_classes)
    precision_par_classe = np.zeros(n_classes)
    f1_par_classe        = np.zeros(n_classes)
    for c in range(n_classes):
        TP = np.sum((pred == c) & (y_test_subset == c))
        FP = np.sum((pred == c) & (y_test_subset != c))
        FN = np.sum((pred != c) & (y_test_subset == c))
        precision_par_classe[c] = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall_par_classe[c]    = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_par_classe[c]        = (2 * precision_par_classe[c] * recall_par_classe[c] /
                                   (precision_par_classe[c] + recall_par_classe[c])
                                   if (precision_par_classe[c] + recall_par_classe[c]) > 0 else 0)
    nom = {i: l for l, i in label_to_int.items()} if label_to_int else {i: str(i) for i in range(n_classes)}
    print(f"\n{'Classe':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 47)
    for c in range(n_classes):
        print(f"{nom[c]:<15} {precision_par_classe[c]:>10.3f} {recall_par_classe[c]:>10.3f} {f1_par_classe[c]:>10.3f}")
    print("-" * 47)
    print(f"{'Moyenne':<15} {np.mean(precision_par_classe):>10.3f} {np.mean(recall_par_classe):>10.3f} {np.mean(f1_par_classe):>10.3f}")
    
    