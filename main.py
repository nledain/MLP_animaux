# ── main.py ───────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import os
from fonctions import * # ← importe tout depuis fonctions.py

chemin_data = r"C:\Users\titi\OneDrive - INSA Lyon\S2\Projet" # A CHANGER  !!! prends le dossier complet
rng = np.random.default_rng(42)

# =========================================================
# CHOIX DU DATASET
# =========================================================
DATASET = "mnist"   # ← "mnist" ou "spectres"

# =========================================================
# BLOC 2 — CHARGEMENT DES DONNÉES
# =========================================================
if DATASET == "mnist":
    from sklearn.datasets import fetch_openml
    mnist_data   = fetch_openml('mnist_784', version=1, as_frame=False)
    X_all        = mnist_data.data.astype(np.float32) / 255.0
    Y_all        = mnist_data.target.astype(int)
    X_train      = X_all[:10000]
    X_test       = X_all[60000:62000]
    y_train_raw  = Y_all[:10000]
    y_test_raw   = Y_all[60000:62000]
    Y_train      = np.eye(10)[y_train_raw].astype(np.float32)
    Y_test       = np.eye(10)[y_test_raw].astype(np.float32)
    labels_test_int = y_test_raw
    label_to_int    = {str(i): i for i in range(10)}
    n_classes, nb_pixels, D, E = 10, 784, 28, 28
    N_train, N_test = 10000, 2000

elif DATASET == "spectres":
    path_donnee = [["data","images_train.npy","labels_train.npy"],["data","images_test.npy","labels_test.npy"]] 
    #variable contenant le nom du dossier des données d'entrainement et de test, ainsi que du noms des fichiers de sortie.

    for data in path_donnee:
        #listes vides
        X = []
        Y = []
        
        images,labels= spectrogramme(chemin_data,X,Y,data[0])
        
        images_f,labels_f = permutation(images,labels,rng)
        
        stockage(images_f,labels_f,chemin_data,data[1],data[2])
        
    print("Spectrogrammes terminés")
    print(f"Forme de la variable images_f (Images) : {images_f.shape}")
    print(f"Forme de la variable labels_f (Labels) : {labels_f.shape}")

    

    # ── chargement fichiers .npy ───────────────────────────
    images_train = np.load("images_train.npy")
    images_test  = np.load("images_test.npy")
    labels_train = np.load("labels_train.npy")
    labels_test  = np.load("labels_test.npy")

    N_train   = images_train.shape[0]
    N_test    = images_test.shape[0]
    nb_pixels = images_train.shape[1] * images_train.shape[2]
    n_classes = len(np.unique(labels_train))
    D, E      = images_train.shape[2], images_train.shape[1]

    classes          = sorted(np.unique(labels_train))
    label_to_int     = {label: i for i, label in enumerate(classes)}
    labels_train_int = np.array([label_to_int[l] for l in labels_train])
    labels_test_int  = np.array([label_to_int[l] for l in labels_test])

    X_train = (images_train.reshape(N_train, nb_pixels) / np.float32(255.0))
    X_test  = (images_test.reshape(N_test,   nb_pixels) / np.float32(255.0))
    Y_train = np.eye(n_classes)[labels_train_int].astype(np.float32)
    Y_test  = np.eye(n_classes)[labels_test_int].astype(np.float32)

# =========================================================
# BLOC 3 — AUTO-ENCODEUR
# =========================================================
W_ae, B_ae = init_params2(
    layer_sizes=[nb_pixels, 512, 64, 512, nb_pixels],  # ← nb_pixels automatique
        init_type="he", seed= 42
)

W_ae, B_ae, cout_ae = Mini_Batch(
    X=X_train, Y=X_train, nb_epoch=100, Lr=0.05,
    W=W_ae, B=B_ae, Nom_tech_Loss=MSE_reconstruction,
    Nom_tech_activation=Relu, act_finale=sigmoid,
    batch_size=128, clip=5.0
)

vue_reconstruite(X_test, W_ae, B_ae, act=Relu, act_finale=sigmoid, D=D, E=E, n=5)

plt.plot(cout_ae)
plt.title("Training loss Auto-encodeur")
plt.show()

# =========================================================
# BLOC 4 — ENCODAGE DU DATASET
# =========================================================
X_train_encoded = encode(X_train, W_ae, B_ae, Relu)
X_test_encoded  = encode(X_test,  W_ae, B_ae, Relu)

# =========================================================
# BLOC 5 — MLP CLASSIFIEUR
# =========================================================
W_clf, B_clf = init_params2(
    layer_sizes=[64, 128, 64, n_classes],  # ← n_classes automatique
     init_type="he",seed= 42
)

W_clf, B_clf, cout_clf = Mini_Batch(
    X=X_train_encoded, Y=Y_train, nb_epoch=200, Lr=0.01,
    W=W_clf, B=B_clf, Nom_tech_Loss=cross_entropy,
    Nom_tech_activation=Relu, act_finale=softmax,
    batch_size=64, clip=None
)

# =========================================================
# BLOC 6 — ÉVALUATION
# =========================================================
Zs_test, As_test = linear_forward(X_test_encoded, W_clf, B_clf, Relu, softmax)
L    = len(W_clf)
pred = np.argmax(As_test[L], axis=1)
evaluation(pred, labels_test_int[:len(pred)], n_classes, label_to_int)

plt.plot(cout_clf)
plt.title("Training loss MLP")
plt.show()