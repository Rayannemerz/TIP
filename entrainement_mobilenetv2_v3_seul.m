% ENTRAINEMENT_MOBILENETV2_V3_SEUL  Fine-tuning de MobileNetV2 sur Food-11, version 3.
% Version autonome : n'utilise aucun autre fichier du projet.
% Sortie : le fichier mobilenetv2_v3_trained.mat (variables net et info),
% enregistré dans le dossier courant de MATLAB.
%
% Diagnostic de la v2 (diapositive "machine learning recipe" du cours) :
% environ 98 % d'exactitude sur le train contre 88 % sur la validation
% -> forte variance (surapprentissage). Remèdes appliqués :
%   1) augmentation de données plus forte (rotation, zoom, translations, cisaillement)
%   2) régularisation : dropout avant la dernière couche, L2 plus fort,
%      lissage des étiquettes (label smoothing)
%   3) hyper-paramètres : taux d'apprentissage initial plus élevé,
%      décroissance plus rapide, arrêt anticipé

clc; close all;

% >>> À ADAPTER : chemin du dossier train (celui qui contient les 11 dossiers de classes)
DOSSIER_TRAIN = fullfile('food-11', 'train');

inputSize = [224 224 3];
numClasses = 11;

%% 1) Données : même découpage que le groupe (rng(0), 80/20 stratifié)
rng(0);
imdsTout = imageDatastore(DOSSIER_TRAIN, IncludeSubfolders=true, LabelSource="foldernames");
[imdsTrain, imdsVal] = splitEachLabel(imdsTout, 0.8, "randomized");
fprintf('%d images d''entraînement, %d images de validation\n', ...
    numel(imdsTrain.Files), numel(imdsVal.Files));

% Remède 1 : augmentation de données plus forte
augmenter = imageDataAugmenter( ...
    RandXReflection=true, ...
    RandRotation=[-20 20], ...
    RandScale=[0.8 1.2], ...
    RandXTranslation=[-20 20], ...
    RandYTranslation=[-20 20], ...
    RandXShear=[-10 10]);
augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    DataAugmentation=augmenter, ColorPreprocessing="gray2rgb");
augVal = augmentedImageDatastore(inputSize(1:2), imdsVal, ColorPreprocessing="gray2rgb");

%% 2) Réseau : MobileNetV2 pré-entraîné, nouvelle tête à 11 classes
% Remède 2 : dropout avant la dernière couche (n'ajoute aucun paramètre)
lgraph = layerGraph(mobilenetv2);
nouvelleTete = [
    dropoutLayer(0.3, Name="dropout_tete")
    fullyConnectedLayer(numClasses, Name="new_fc", ...
        WeightLearnRateFactor=10, BiasLearnRateFactor=10)];
lgraph = replaceLayer(lgraph, "Logits", nouvelleTete);
lgraph = removeLayers(lgraph, "ClassificationLayer_Logits");
net = dlnetwork(lgraph);
fprintf('Paramètres du réseau : %.6f M\n\n', ...
    sum(cellfun(@numel, net.Learnables.Value)) / 1e6);

% Remède 2 (suite) : lissage des étiquettes. La cible "100 % bread" devient
% "90 % bread + 10 % répartis sur les 11 classes" : le réseau devient moins
% sûr de lui, ce qui limite le surapprentissage.
lissage = 0.1;
perte = @(Y, T) crossentropy(Y, (1 - lissage) * T + lissage / numClasses);

%% 3) Entraînement
% Remède 3 : taux initial 1e-3 (au lieu de 5e-4), décroissance tous les
% 5 epochs, L2 5e-4 (au lieu de 2e-4), arrêt anticipé
batchSize = 64;
iterParEpoch = floor(numel(imdsTrain.Files) / batchSize);
options = trainingOptions("sgdm", ...
    InitialLearnRate=1e-3, Momentum=0.9, ...
    LearnRateSchedule="piecewise", LearnRateDropPeriod=5, LearnRateDropFactor=0.2, ...
    L2Regularization=5e-4, ...
    MaxEpochs=15, MiniBatchSize=batchSize, Shuffle="every-epoch", ...
    ValidationData=augVal, ValidationFrequency=iterParEpoch, ...
    ValidationPatience=5, OutputNetwork="best-validation", ...
    Metrics="accuracy", Verbose=true, Plots="training-progress");

[net, info] = trainnet(augTrain, net, perte, options);

%% 4) Sauvegarde
save('mobilenetv2_v3_trained.mat', 'net', 'info');
fprintf('\nTerminé : mobilenetv2_v3_trained.mat enregistré dans %s\n', pwd);
