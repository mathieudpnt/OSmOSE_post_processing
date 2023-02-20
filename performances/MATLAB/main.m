%Get main_PG parameters
clear;clc

% User inputs
% TZ = '+02:00';
TZ = 'Europe/Paris'; %TimeZone

addpath(genpath(fullfile(fileparts(fileparts(pwd)), 'utilities')))

%info written on Aplose csv file
infoAplose.annotator = "PAMGuard";
infoAplose.annotation = "Whistle and moan detector";
infoAplose.dataset = "APOCADO C2D1_070722 (10_144000)";

GeneralFolderWav = uigetdir2('L:\acoustock\Bioacoustique\DATASETS')

GeneralFolderBinary = uigetdir2('L:\acoustock\Bioacoustique\DATASETS');

% Get files - Automatic

GeneralFolderWavInfo = [];
for i = 1:length(GeneralFolderWav)
    GeneralFolderWavInfo = [GeneralFolderWavInfo; dir(fullfile(GeneralFolderWav{i}, '/**/*.wav'))];
end
subFoldersWav = string(unique(extractfield(GeneralFolderWavInfo, 'folder')'));

% GeneralFolderBinaryInfo = dir(fullfile(GeneralFolderBinary, '/**/*.pgdf'));
GeneralFolderBinaryInfo = [];
for i = 1:length(GeneralFolderWav)
    GeneralFolderBinaryInfo = [GeneralFolderBinaryInfo; dir(fullfile(GeneralFolderBinary{i}, '/**/*.pgdf'))];
end
subFoldersBinary = string(unique(extractfield(GeneralFolderBinaryInfo, 'folder')'));


%% Execution of main
%if all the data of a folder is to be analyzed, use the function main_PG
%if only certains dates are to be analyzed in the data folder, create list
%of selected data and use main_PG in a loop
% /!\ input parameters 2 and 3 must be char type, not string type

PG2APLOSE(infoAplose, GeneralFolderWav{1}, GeneralFolderBinary{1}, TZ);



% Manual file selection
% subFoldersWav = subFoldersWav(74:81);
% subFoldersBinary = subFoldersBinary(74:81);
% for i = 1:length(subFoldersWav)
%     main_PG(infoAplose, char(subFoldersWav(i)), char(subFoldersBinary(i)));
%     main_PG(infoAplose, char(subFoldersWav(i)), GeneralFolderBinary);
% end





