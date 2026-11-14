clearvars -except imdsTrain imdsVal imdsTest;
gpuDevice([]);

if ~exist("imdsTrain", "var")
    load("food11_datastores.mat");
end 

inputSize = [227 227 3];
numClasses = 11;

netPretrained = squeezenet;
lgraph = layerGraph(netPretrained);

% gèle le début du réseau conv1 et fire2 jusqu'a fire5
% fire6 à fire9 dégelés
layers = lgraph.Layers;
for i = 1:numel(layers)
    layer = layers(i);
    name = string(layer.Name);
    isEarlyLayer = startsWith(name, ["conv1", "fire2", "fire3", "fire4", "fire5"]);
    
    if isEarlyLayer
        hasChanged = false;
        if isprop(layer, "WeightLearnRateFactor")
            layer.WeightLearnRateFactor = 0;
            hasChanged = true;
        end
        if isprop(layer, "BiasLearnRateFactor")
            layer.BiasLearnRateFactor = 0;
            hasChanged = true;
        end
        if hasChanged
            lgraph = replaceLayer(lgraph, layer.Name, layer);
        end
    end
end

%remplacement de conv10, suppression de relu_conv10
newConvLayer = convolution2dLayer([1 1], numClasses, ...
    Name="new_conv", ...
    WeightLearnRateFactor=5, ... % Apprend 5 fois plus vite que fire6-fire9
    BiasLearnRateFactor=5);

lgraph = replaceLayer(lgraph, "conv10", newConvLayer);
lgraph = replaceLayer(lgraph, "relu_conv10", identityLayer(Name="no_relu"));
lgraph = replaceLayer(lgraph, "pool10", globalAveragePooling2dLayer(Name="pool10"));
lgraph = removeLayers(lgraph, "ClassificationLayer_predictions");

net = dlnetwork(lgraph);

%Data Augmentation enrichie
augmenter = imageDataAugmenter( ...
    RandXReflection=true, ... %miroir
    RandRotation=[-12 12], ... %rotation
    RandScale=[0.9 1.1], ... %échelle
    RandXTranslation=[-15 15], ... %décalages
    RandYTranslation=[-15 15]);

augTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, DataAugmentation=augmenter);
augVal   = augmentedImageDatastore(inputSize(1:2), imdsVal);

% training options
batchSize = 64;
iterationsPerEpoch = floor(numel(imdsTrain.Files) / batchSize);

options = trainingOptions("adam", ...
    InitialLearnRate=3e-4, ...                  % Doux pour les couches fire6-9 (1.5e-3 effectif sur new_conv)
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=10, ...
    LearnRateDropFactor=0.25, ...
    L2Regularization=2e-4, ...
    ExecutionEnvironment="gpu", ...
    MaxEpochs=28, ...
    Shuffle="every-epoch", ...
    MiniBatchSize=batchSize, ...
    ValidationData=augVal, ...
    ValidationFrequency=iterationsPerEpoch, ...
    ObjectiveMetric="accuracy", ...
    OutputNetwork="best-validation", ...
    DispatchInBackground=false, ...
    Verbose=true, ...
    Plots="training-progress", ...
    Metrics="accuracy");

%Entraînement
[net, info] = trainnet(augTrain, net, "crossentropy", options);

save("squeezenet_v5_finetuned.mat", "net", "info");

%Bilan (Précision, F1 Macro et Frugalité)
totalParams = sum(cellfun(@numel, net.Learnables.Value));
R = totalParams / 1e6;
penalite = 0.3 / (1 + (1 / R));

scoresVal = minibatchpredict(net, augVal);
scoresMatrix = squeeze(scoresVal)';
classNames = categories(imdsVal.Labels);

predLabels = scores2label(scoresMatrix, classNames);
trueLabels = imdsVal.Labels(:);
predLabels = predLabels(:);

valAccuracy = mean(predLabels == trueLabels) * 100;

C = confusionmat(trueLabels, predLabels);
TP = diag(C);
FP = sum(C, 1)' - TP;
FN = sum(C, 2) - TP;
F1_classes = 2 * TP ./ (2 * TP + FP + FN);
f1Macro = mean(F1_classes);
scoreFrugalite = f1Macro - penalite;

fprintf("\n======= SQUEEZENET v5  =======\n");
fprintf("Paramètres totaux     : %d (%.3f M)\n", totalParams, R);
fprintf("Pénalité de frugalité : -%.4f\n", penalite);
fprintf("Précision Validation  : %.2f %%\n", valAccuracy);
fprintf("Score F1 Macro        : %.4f (%.2f %%)\n", f1Macro, f1Macro*100);
fprintf("SCORE DE FRUGALITÉ    : %.4f\n", scoreFrugalite);
fprintf("=========================================================\n");

figure("Name", "Matrice de confusion SqueezeNet v5", "NumberTitle", "off");
confusionchart(trueLabels, predLabels, ...
    "RowSummary", "row-normalized", ...
    "ColumnSummary", "column-normalized");
title(sprintf("SqueezeNet v5 Semi-Gelé 336x336 (Précision : %.2f %% | Frugalité : %.4f)", valAccuracy, scoreFrugalite));
