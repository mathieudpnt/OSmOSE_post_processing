%Get main_PG parameters
clear;clc

% User inputs
TZ = '+02:00';
% TZ = 'Europe/Paris'; %TimeZone
format_datestr = 'yyMMddHHmmss'; %APOCADO filename format
% format_datestr = 'yyyy-MM-dd_HH-mm-ss'; %CETIROISE filename format

addpath(genpath(fullfile(fileparts(fileparts(pwd)), 'utilities')))

%info written on Aplose csv file
infoAplose.annotator = 'PAMGuard';
infoAplose.annotation = 'Whistle and moan detector';
infoAplose.dataset = 'C7D14_ST7190';


% Get files - Automatic
mode = 'file';
% mode = 'folder';

if isequal(mode, 'file')
%     [GeneralFolderWav0, GeneralFolderWav] = uigetfile('*.wav', 'select waves', 'Multiselect', 'on', 'L:\acoustock\Bioacoustique\DATASETS');
    [GeneralFolderWav0, GeneralFolderWav] = uigetfile('*.wav', 'select waves', 'Multiselect', 'on', 'Y:\Bioacoustique\APOCADO2\Campagne 7\Gwalarn II\7190\wav');
    GeneralFolderWav = {fileparts(GeneralFolderWav)};
    PathWaves=[];
    for i = 1:numel(GeneralFolderWav0)
        PathWaves = [PathWaves; strcat(GeneralFolderWav, '\', GeneralFolderWav0{i})];
    end
    
    
    GeneralFolderWavInfo = [];
    for i = 1:numel(PathWaves)
        GeneralFolderWavInfo = [GeneralFolderWavInfo; dir(PathWaves{i})];
    end
    
elseif isequal(mode, 'folder')
    GeneralFolderWav = uigetdir2('L:\acoustock\Bioacoustique\DATASETS', 'Select wav folders');
    GeneralFolderWavInfo = [];
    for i = 1:numel(GeneralFolderWav)
        GeneralFolderWavInfo = [GeneralFolderWavInfo; dir(fullfile(GeneralFolderWav{i}, '/**/*.wav'))];
    end
end
subFoldersWav = string(unique(extractfield(GeneralFolderWavInfo, 'folder')'));

GeneralFolderBinary = uigetdir2(fileparts(GeneralFolderWav{1}), 'Select binary folders');
% GeneralFolderBinaryInfo = dir(fullfile(GeneralFolderBinary, '/**/*.pgdf'));
GeneralFolderBinaryInfo = [];
for i = 1:numel(GeneralFolderBinary)
    GeneralFolderBinaryInfo = [GeneralFolderBinaryInfo; dir(fullfile(GeneralFolderBinary{i}, '/**/*.pgdf'))];
end
% subFoldersBinary = string(unique(extractfield(GeneralFolderBinaryInfo, 'folder')'));


if numel(GeneralFolderBinaryInfo)~= numel(GeneralFolderWavInfo)
    warning('Number of wav files (%.0f) is different than number of pgdf files (%.0f)', numel(GeneralFolderWavInfo), numel(GeneralFolderBinaryInfo))
end


% Execution of main
%if all the data of a folder is to be analyzed, use the function main_PG
%if only certains dates are to be analyzed in the data folder, create list
%of selected data and use main_PG in a loop
% /!\ input parameters 2 and 3 must be char type, not string type

% PG2APLOSE(infoAplose, GeneralFolderWav{1}, GeneralFolderBinary{2}, format_datestr, TZ);

if numel(GeneralFolderBinaryInfo)== numel(GeneralFolderWavInfo)
    for i=1:numel(GeneralFolderBinary)
        PG2APLOSE(infoAplose, GeneralFolderWav{1}, GeneralFolderBinary{i}, format_datestr, TZ);
    end
else
    warning('Number of wav files is different than number of pgdf files')
    msgbox('Number of wav files is different than number of pgdf files')
end


% Manual file selection
% subFoldersWav = subFoldersWav(74:81);
% subFoldersBinary = subFoldersBinary(74:81);
% for i = 1:length(subFoldersWav)
%     main_PG(infoAplose, char(subFoldersWav(i)), char(subFoldersBinary(i)));
%     main_PG(infoAplose, char(subFoldersWav(i)), GeneralFolderBinary);
% end





