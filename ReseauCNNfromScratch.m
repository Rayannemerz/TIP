clc;
clear;
close all;

fprintf('debut\n');

trainFolder = 'food-11/train'; % Définit le chemin d'accès au dossier contenant les images d'entraînement
% Crée un datastore pour gérer de grands ensembles d'images.
imdsTrain = imageDatastore(trainFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

[imdsTrain, imdsValidation] = splitEachLabel(imdsTrain, 0.8, 'randomized');
% Calcule le nombre total de classes en comptant le nombre d'étiquettes uniques
numClasses = numel(categories(imdsTrain.Labels));
disp(numClasses)

inputSize = [128 128 3]; % Définit la taille attendue des images en entrée du réseau (128x128 pixels, 3 canaux pour les couleurs RGB)

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain); % Crée un datastore augmenté pour l'entraînement, qui redimensionnera automatiquement
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation); % la même chose pour les images de validation

layers = [
    imageInputLayer(inputSize) % Couche d'entrée

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer % accélère l'apprentissage et stabilise le réseau
    reluLayer % ReLU ( les valeurs négatives à zéro)
    maxPooling2dLayer(2,'Stride',2) % réduit la taille spatiale de l'image par 2

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.5)

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam','MiniBatchSize',32,'MaxEpochs',20,'InitialLearnRate',1e-3,'ValidationData',augimdsValidation,'ValidationFrequency',30,'Verbose',true,'Plots','training-progress');

net = trainNetwork(augimdsTrain,layers,options); % Lance l'entraînement avec nos données, nos couches et nos options

fprintf('fini\n')

save('ReseauCNNfromScratch.mat','net')
save('SplitValidation.mat','imdsValidation')

fprintf('sauvegardé\n')
