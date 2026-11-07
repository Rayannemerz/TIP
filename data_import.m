%%% NOUVEAU : fixe graine àléatoire
rng(0); 

baseDir = fullfile("C:\Users\Lucas\Documents\4TC\PROJET_TIP\food-11", "food-11");

trainFolder = fullfile(baseDir, "train");
testFolder  = fullfile(baseDir, "test");

imdsAllTrain = imageDatastore(trainFolder, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

[imdsTrain, imdsVal] = splitEachLabel(imdsAllTrain, 0.8, "randomized");

imdsTest = imageDatastore(testFolder, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

numImages = numel(imdsTest.Labels);
idx = randperm(numImages,16);
I = imtile(imdsTest,Frames=idx);
figure
imshow(I)
%petit exemple pour load des images

%Sauvegarder les variables
save("food11_datastores.mat", "imdsTrain", "imdsVal", "imdsTest");