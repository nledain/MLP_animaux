# ── fonctions.py ─────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.ndimage import zoom

# ── Prétraitement audio ───────────────────────────────────────────────────────
def mono (donnee):
    """
    Transforme un signal stéréo en signal mono
    prends en arguement un array a deux colonnes et renvoie un array en 
    une colonne
    """
    donnee_mono = (donnee[:,0].copy() +donnee[:,1].copy())/2 #on prends la moyenne du signal 
    
    return donnee_mono

def permutation(x,y):
    n = len(x) #prends le nombre d'image
    n_perm = np.random.permutation(n)
    x_f = x[n_perm]
    y_f = y[n_perm]
    
    return x_f,y_f

def stockage (img,labels):
    np.save("labels.npy",labels)
    np.save("images.npy",img)
    return "fichiers enregistrés"


# ── Activations ───────────────────────────────────────────────────────────────
def Relu(z):
    return np.maximum(0,z)

def dRelu(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def dsigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)

'''
Prends la valeur prédite par le modèle A et la transforme en proba 
chaque neuronne de la couche finale represente une classe d'animaux,
et sa proba après softmax est donc sa chance d'être de cette classe,
la prédiction fianle du modèle sera celle de la proba la plus haute
'''
def softmax(Z):    
    Z = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=1, keepdims=True)

# ── Initialisation ────────────────────────────────────────────────────────────
def init_params(layer_sizes, init_type="he", seed=None):
    """
    Initialise les poids et biais du réseau.
    Parameters
    ----------
    layer_sizes : list
        Liste contenant le nombre de neurones par couche.
        Exemple :
        [4, 16, 8, 3]
        signifie :
        4 entrées
        -> couche cachée 16 neurones
        -> couche cachée 8 neurones
        -> sortie 3 neurones
    init_type : str
        "he"      -> conseillé pour ReLU
        "xavier"  -> conseillé pour sigmoid/tanh
        "random"  -> petit aléatoire simple
    seed : int ou None
        Pour reproductibilité.
    Returns
    -------
    W : dict
        Dictionnaire des matrices de poids.
    B : dict
        Dictionnaire des biais.
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
        if init_type == "he":
            # très bon pour ReLU
            W[l] = np.random.randn(n_prev, n_curr) * np.sqrt(2 / n_prev)
        elif init_type == "xavier":
            # mieux pour sigmoid/tanh
            W[l] = np.random.randn(n_prev, n_curr) * np.sqrt(1 / n_prev)
        elif init_type == "random":
            W[l] = np.random.randn(n_prev, n_curr) * 0.01
        else:
            raise ValueError("Type d'initialisation inconnu")
        # ==========================================
        # BIAIS
        # ==========================================
        B[l] = np.zeros((1, n_curr))
    return W, B

# ── Forward ───────────────────────────────────────────────────────────────────
def linear_forward(X, W, B, act,act_finale):
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
    Loss + gradient dZ directement pour softmax + cross-entropy.
    dZ_L = A_L - Y  (simplification mathématique exacte)
    Y doit être one-hot, shape (N, C). #Ca jsp ce que ça veut dire par contre
    """
    eps = 1e-8
    N = Y.shape[0]
    J = -(1 / N) * np.sum(Y * np.log(A_L + eps))
    dZ_L = (A_L - Y) / N   # divisé par N car on normalise
    return J, dZ_L 

'''
encodeur type: 4096-512-128-32
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
    N = X.shape[0]          # nombre d'exemples dans le batch
    nb_pixels = X.shape[1]  # 16384 pour du 128x128
    
    loss = (1/(N * nb_pixels)) * np.sum((X - X_reconstruit)**2)
    dA = (2 / (N * nb_pixels)) * (X_reconstruit - X)
    dZ=dA*dsigmoid(Z_L)
    return loss, dZ

def Loss_choisi(nom_technique, Y_r, Y_p, Z_final=None):
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
As: Dico des val prédites après softmax par couche
dZ_L: Liste de la variation des val prédites à la dernière couche, calculé dans cross_entropy
gradients_W: Dico des dw par couche
gradients_B: Dico des db par couche
Fonctionnement: on parcourt le réseau à l'envers puisqu'on remonte les erreurs
                à chaque couche n, on choppe le A de la couche n-1,avec, on calcule
                dw de la couche (liste de couche et une valeur par neurone)
'''

def backward(Y, W, Zs, As, dZ_L, nom_act,clip=None):#Fonction complexe, remonte chaque couche depuis la fin, pour trouver les dZ, les dA, les dw et db
    gradients_W = {}
    gradients_B = {}
    L = len(W)
    dZ_current = dZ_L
    for l in reversed(range(1, L + 1)):
        if clip is not None:
            dZ_current = np.clip(dZ_current, -clip, clip)
        A_prev = As[l - 1]
        gradients_W[l] = np.dot(A_prev.T, dZ_current)
        gradients_B[l] = np.sum(dZ_current, axis=0, keepdims=True)
        if l > 1:
            dA_prev = np.dot(dZ_current, W[l].T)
            dZ_current = dA_prev * activation_backward(nom_act, Zs[l - 1])

    return gradients_W, gradients_B

# ── Update ────────────────────────────────────────────────────────────────────
def update(dw,db,w,b, Lr):
    L=len(w)
    W_new={}
    B_new={}
    for i in range(1,L+1):
        W_new[i]=w[i]-Lr*dw[i]
        B_new[i]=b[i]-Lr*db[i]
    return W_new,B_new

# ── Entraînement ──────────────────────────────────────────────────────────────
def Batch_complet(X,nb_epoch, Lr, Y,W,B, Nom_tech_Loss,Nom_tech_activation,act_finale):
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
    milieu=len(W)//2
    X_encoded={0:X}
    for i in range(1,milieu+1):
        A_prev=X_encoded[i-1]
        Z = np.dot(A_prev, W[i]) + B[i]
        X_encoded[i]=act(Z)
    return X_encoded[milieu]

def decode(X_encoded,W,B,act,act_finale):
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
    Affiche n images originales et leurs reconstructions côte à côte.
    X : batch d'images shape (N, 16384)
    n : nombre d'images à afficher
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