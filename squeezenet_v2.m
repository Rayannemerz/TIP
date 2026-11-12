
if ~exist("imdsTrain", "var")
    load("food11_datastores.mat");
end 

%chargement des données
inputSize = [227 227 3];

%Chargement de SqueezeNet
netPretrained = squeezenet;
lgraph = layerGraph(netPretrained);

%Augmentation des données
augmenter = imageDataAugmenter( ...
    RandXReflection=true, ...
    RandXTranslation=[-15 15], ...
    RandYTranslation=[-15 15]);

augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    DataAugmentation=augmenter);

augVal  = augmentedImageDatastore(inputSize(1:2), imdsVal);
augTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

%Modification de la couche de classification
numClasses = 11;

newConvLayer = convolution2dLayer([1 1], numClasses, ...
    Name="new_conv", ...
    WeightLearnRateFactor=10, ...
    BiasLearnRateFactor=10);

lgraph = replaceLayer(lgraph, "conv10", newConvLayer);

%suppr reLu
lgraph = replaceLayer(lgraph, "relu_conv10", ...
    identityLayer(Name="no_relu"));

%Suppr classe final car on a deja un softmax
lgraph = removeLayers(lgraph, "ClassificationLayer_predictions");

%plus d'epoch , moins de batch-size pour contrer sous-apprentissage
batchSize = 64;
numTrainImages = numel(imdsTrain.Files);
maxEpochs = 35;

iterationsPerEpoch = max(1, ceil(numTrainImages / batchSize));

options = trainingOptions("sgdm", ...
    InitialLearnRate=5e-4, ... 
    Momentum=0.9, ... 
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=12, ... %moins violent
    LearnRateDropFactor=0.2, ...
    ExecutionEnvironment="gpu", ...
    MaxEpochs=maxEpochs, ...
    Shuffle="every-epoch", ...
    MiniBatchSize=batchSize, ...
    ValidationData=augVal, ...
    ValidationFrequency=iterationsPerEpoch, ... %réduire le temps de calcul
    DispatchInBackground=true, ...
    Verbose=false, ...
    Plots="training-progress", ...
    Metrics="accuracy");


net = trainnet(augTrain, dlnetwork(lgraph), ...
    "crossentropy", options);

save("squeezenet_trained.mat", "net");

valAccuracy = testnet(net, augVal, "accuracy");

fprintf("--> Précision exacte sur la validation : %.2f %%\n", ...
    valAccuracy);

scoresVal = minibatchpredict(net, augVal);

%Transformation des dimensions
scoresMatrix = squeeze(scoresVal)'; %squeeze pour éliminer le 1*1, ' pour transposer

classNames = categories(imdsVal.Labels);

predLabels = scores2label(scoresMatrix, classNames);

predLabels = predLabels(:);
trueLabels = imdsVal.Labels(:);


figure("Name", "Matrice de confusion SqueezeNet", ...
    "NumberTitle", "off");

confusionchart(trueLabels, predLabels, ...
    "RowSummary", "row-normalized", ...
    "ColumnSummary", "column-normalized");

title(sprintf("Matrice de Confusion - SqueezeNet (Précision : %.2f %%)", ...
    valAccuracy));
