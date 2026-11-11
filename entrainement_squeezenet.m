clc;
clear;
close all;

fprintf('Chargement des données d''entraînement...\n');

% le dossier avec les images d'entrainement, chaque classe dans son dossier
trainFolder = 'food-11/train';

imdsTrain = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% je separe 80% pour entrainer et 20% pour valider apres
[imdsTrain, imdsValidation] = splitEachLabel(imdsTrain, 0.8, 'randomized');

numClasses = numel(categories(imdsTrain.Labels));
fprintf('Nombre de classes trouvées : %d\n', numClasses); % normalement 11

% avec squeezenet 
fprintf('Chargement du réseau pré-entraîné...\n');
net = squeezenet;

lgraph = layerGraph(net);


newConvLayer = convolution2dLayer(1, numClasses, ...
    'Name', 'new_conv10', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);

lgraph = replaceLayer(lgraph, 'conv10', newConvLayer);

newClassLayer = classificationLayer('Name', 'new_classoutput');
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassLayer);

% la taille d'image que le reseau veut en entrée
inputSize = net.Layers(1).InputSize;

augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandRotation', [-10 10]);

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

% les options d'entrainement, jai mis des valeurs un peu classiques
% pas trop sur si cest les meilleurs params mais ca marche normalement
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 8, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

fprintf('Entraînement en cours... (ça peut prendre du temps)\n');
net = trainNetwork(augimdsTrain, lgraph, options);

fprintf('Sauvegarde du réseau...\n');
save('monReseauSqueezenet.mat', 'net');

% calculer le F1 sur exactement les memes images que moi
save('monSplitValidation.mat', 'imdsValidation');

fprintf('Terminé !\n');
