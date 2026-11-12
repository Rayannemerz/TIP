```matlab
clc;
clear;
close all;

fprintf('debut\n');

trainFolder = 'food-11/train';

imdsTrain = imageDatastore(trainFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

[imdsTrain, imdsValidation] = splitEachLabel(imdsTrain, 0.8, 'randomized');

numClasses = numel(categories(imdsTrain.Labels));
disp(numClasses)

inputSize = [128 128 3];

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

layers = [
    imageInputLayer(inputSize)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

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

net = trainNetwork(augimdsTrain,layers,options);

fprintf('fini\n')

save('monReseauCNNfromScratch.mat','net')
save('monSplitValidation.mat','imdsValidation')

fprintf('sauvegardé\n')
```
