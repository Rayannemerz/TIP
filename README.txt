CHALLENGE FOOD-11 - CLASSIFICATION D'IMAGES
=======================================================================

Groupe : MOHAMED TAYEB, LUCAS, RAYAN, TAIGA

Réseau final soumis au leaderboard : SqueezeNet v5 à gel partiel
(0,728139 M paramètres). Score obtenu sur le serveur : Frugality = 0,7159 (F1 = 0,8423).


1. PRÉREQUIS
------------
- MATLAB R2026b
- Deep Learning Toolbox
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox
- Modèles pré-entraînés SqueezeNet et MobileNetV2 (Deep Learning Toolbox Model
  for SqueezeNet Network / for MobileNet-v2 Network)
- Un GPU NVIDIA est fortement conseillé pour les scripts d'entraînement (nous avons
  utilisé une RTX 5060). Sans GPU, ils s'exécutent sur le processeur
  (ExecutionEnvironment="auto"), mais beaucoup plus lentement.


2. ORGANISATION DES DOSSIERS
----------------------------
Le jeu de données n'est pas fourni dans l'archive. Placer le dossier food-11
(avec ses sous-dossiers train et test) à côté du dossier code, et créer un
dossier resultats vide :

  projet/
    code/        scripts .m de l'archive
    food-11/
      train/     un sous-dossier par classe
      test/      3 097 images
    resultats/   réseaux entraînés (.mat) et fichiers JSON


3. EXÉCUTION (réseau final)
---------------------------
Dans MATLAB, se placer dans le dossier code/, puis lancer les scripts dans
cet ordre :

  Étape 1 : etape1_donnees.m
            Découpe le dossier train en 80 % entraînement / 20 % validation
            (rng(0)). Crée food11_datastores.mat dans code/, utilisé par les
            scripts d'entraînement.

  Étape 2 : etape2_entrainement_squeezenet_v5.m
            Entraîne le réseau final (SqueezeNet v5, quelques dizaines de minutes
            sur GPU). Crée squeezenet_v5_finetuned.mat dans code/.
            -> Déplacer ce fichier dans resultats/ avant l'étape 3.

  Étape 3 : etape3_evaluation.m
            - affiche le nombre de paramètres et les métriques sur la validation,
              calculées comme le serveur (exactitude, F1 pondéré, Frugality,
              détail par classe) ;
            - prédit les 3 097 images de test et écrit
              resultats/predictions_squeezenet_v5.json ;
            - vérifie le format du JSON (3 097 entrées, 11 classes officielles).
            La fin attendue est "=> JSON VALIDE".

Les deux réseaux de comparaison (comparaison_mobilenetv2.m et
comparaison_cnn_v3.m) se lancent de la même façon, après l'étape 1.


4. DESCRIPTION DES FICHIERS
---------------------------
etape1_donnees.m
    Découpage train/validation (80/20, rng(0)) commun à tous les entraînements.

etape2_entrainement_squeezenet_v5.m  (RÉSEAU FINAL)
    Transfer learning à partir de SqueezeNet pré-entraîné sur ImageNet.
    Début du réseau gelé (conv1, fire2 à fire5), fin réentraînée (fire6 à fire9),
    nouvelle couche finale 1x1 à 11 classes (apprentissage 5 fois plus rapide),
    ReLU finale supprimée, moyenne globale. Images 227x227, augmentation
    (miroir, rotation ±12°, zoom 0,9-1,1, translations ±15 px), Adam
    (taux 3e-4 divisé par 4 toutes les 10 époques), régularisation L2 2e-4,
    28 époques, meilleur réseau sur la validation conservé.

comparaison_mobilenetv2.m
    MobileNetV2 pré-entraîné sur ImageNet, fine-tuning complet, nouvelle couche
    finale à 11 classes. Images 224x224, SGDM (taux 5e-4), 18 époques.
    Crée mobilenetv2_trained.mat.

comparaison_cnn_v3.m
    Réseau convolutif conçu et entraîné depuis zéro : 4 blocs convolution +
    normalisation + ReLU + max pooling (32, 64, 128, 256 filtres), moyenne
    globale, couche entièrement connectée de 128, dropout. Images 160x160,
    Adam, 45 époques. 0,424 M paramètres. Crée ReseauCNNfromScratch_v3.mat.

etape3_evaluation.m
    Chaîne d'évaluation complète en un seul fichier (fonctions locales en bas
    du fichier). Pour évaluer un autre réseau, modifier uniquement les deux
    paramètres en haut du fichier, par exemple :
        FICHIER_RESEAU = 'mobilenetv2_trained.mat';
        FICHIER_JSON   = 'predictions_mobilenetv2.json';
    (après avoir placé le .mat correspondant dans resultats/).


5. RÉSULTATS (serveur d'évaluation, images de test)
---------------------------------------------------
Réseau                       Paramètres     F1       Frugality
SqueezeNet v5 (final)        0,728 M        0,8423   0,7159
CNN v3 (depuis zéro)         0,424 M        0,7784   0,6890
MobileNetV2 v2               2,255 M        0,8679   0,6600
Référence du sujet (ResNet-18) 11,6 M       0,8192   0,5430

Frugality = F1 - 0,3 / (1 + 1/R), R = nombre de paramètres en millions


6. REMARQUES
------------
- Les scripts doivent être lancés depuis le dossier code/ : c'est là que
  etape1_donnees.m crée food11_datastores.mat et que les scripts d'entraînement
  le recherchent.
- etape1_donnees.m doit être exécuté depuis son fichier, et non par copier-coller
  de ses lignes : il calcule le chemin du jeu de données à partir de son propre
  emplacement.
- Les résultats d'un nouvel entraînement peuvent légèrement différer de ceux
  indiqués ci-dessus (initialisation aléatoire de la nouvelle couche, ordre des
  images, augmentation, calculs sur GPU). Deux entraînements du même script ont
  obtenu 0,7105 et 0,7159 sur le serveur.
