clearvars -except imdsTrain imdsVal imdsTest;
gpuDevice([]);

if ~exist("imdsTrain", "var")
    load("food11_datastores.mat");
end

numClasses = numel(categories(imdsTrain.Labels));
% 160x160 pour voir les détails
inputSize = [160 160 3];

augmenter = imageDataAugmenter( ...
    RandXReflection=true, ...
    RandRotation=[-10 10], ...
    RandScale=[0.9 1.1], ...
    RandXTranslation=[-10 10], ...
    RandYTranslation=[-10 10]);

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, DataAugmentation=augmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsVal);

% Les 5 mêmes couches que v_prec, mais *2 (32 -> 64 -> 128 -> 256)
layers = [
    imageInputLayer(inputSize, Name="input")

    % Couche 1 (160x160 -> 80x80)
    convolution2dLayer(3, 32, Padding="same", Name="conv1")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, Stride=2)

    % Couche 2 (80x80 -> 40x40)
    convolution2dLayer(3, 64, Padding="same", Name="conv2")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, Stride=2)

    % Couche 3 (40x40 -> 20x20)
    convolution2dLayer(3, 128, Padding="same", Name="conv3")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, Stride=2)

    % Couche 4 (20x20 -> 10x10)
    convolution2dLayer(3, 256, Padding="same", Name="conv4")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, Stride=2)

    % Condense le 10x10x256 en 1x1x256
    globalAveragePooling2dLayer(Name="gap")

    fullyConnectedLayer(128, Name="fc_128")
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.15) 

    fullyConnectedLayer(numClasses, Name="fc_out")
    softmaxLayer(Name="prob")
];

lgraph = layerGraph(layers);

batchSize = 64;
iterationsPerEpoch = floor(numel(imdsTrain.Files) / batchSize);

options = trainingOptions("adam", ...
    InitialLearnRate=1.5e-3, ...               
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=18, ...                 
    LearnRateDropFactor=0.3, ...                % dropout moins brutal
    L2Regularization=1e-4, ...
    MiniBatchSize=batchSize, ...
    MaxEpochs=45, ...
    Shuffle="every-epoch", ...
    ValidationData=augimdsValidation, ...
    ValidationFrequency=iterationsPerEpoch, ...
    ObjectiveMetric="accuracy", ...
    OutputNetwork="best-validation", ...
    ExecutionEnvironment="gpu", ...
    DispatchInBackground=false, ...             
    Verbose=true, ...
    Plots="training-progress", ...
    Metrics="accuracy");

[net, info] = trainnet(augimdsTrain, dlnetwork(lgraph), "crossentropy", options);

save("ReseauCNNfromScratch_v3.mat", "net", "info");

% 5. Bilan complet (F1 Macro + Frugalité directe)
totalParams = sum(cellfun(@numel, net.Learnables.Value));
R = totalParams / 1e6;
penalite = 0.3 / (1 + (1 / R));

scoresVal = minibatchpredict(net, augimdsValidation);
classNames = categories(imdsVal.Labels);
predLabels = scores2label(scoresVal, classNames);
trueLabels = imdsVal.Labels(:);
predLabels = predLabels(:);

valAccuracy = mean(predLabels == trueLabels) * 100;


figure("Name", "Matrice de confusion CNN 5 couches v3", "NumberTitle", "off");
confusionchart(trueLabels, predLabels, ...
    "RowSummary", "row-normalized", ...
    "ColumnSummary", "column-normalized");
title(sprintf("CNN 5 Couches v3 (%.3f M params) - Précision : %.2f %% ", R, valAccuracy));