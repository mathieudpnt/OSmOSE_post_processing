clear;clc
cd('U:\Documents_U\Git\post_processing_detections\')
addpath('utilities')
addpath('utilities\pgmatlab')
addpath('performances\MATLAB')
addpath(genpath(fullfile(fileparts(fileparts(pwd)), 'utilities')))

% user inputs
info_deploy.annotator = 'PAMGuard';
info_deploy.annotation = 'Whistle and moan detector';
info_deploy.dataset = 'C7D12_ST7190';
info_deploy.timezone = '+02:00';
info_deploy.dt_format = 'yyMMddHHmmss'; % APOCADO filename format
% info_deploy.dt_format = 'yyyy-MM-dd_HH-mm-ss'; % CETIROISE filename format

% get wav files
%%% mode 'folder': the wav files are located on different folders
%%% mode 'file': the wav files are located on in the same folder
mode = 'file';
% mode = 'folder';

base_folder = 'L:\acoustock\Bioacoustique\DATASETS';

if isequal(mode, 'file')
    msg = sprintf('%s - select waves', info_deploy.dataset);
    [wav_file, folder_wav] = uigetfile('*.wav', msg, 'Multiselect', 'on', base_folder);
    wav_path = fullfile(folder_wav,  wav_file);
    wav_info = cellfun(@dir, wav_path);
    folder_wav = {fileparts(folder_wav)};

elseif isequal(mode, 'folder')
    msg = sprintf('%s - select wav folders', info_deploy.dataset);
    folder_wav = uigetdir2(base_folder, msg);
    wav_info = cellfun(@dir, fullfile(folder_wav, '**/*.wav'));
end

binary_folder = uigetdir2(fileparts(folder_wav{1}), sprintf('%s - select binary folder', info_deploy.dataset));
binary_info = cellfun(@dir, fullfile(binary_folder, '/**/*.pgdf'), 'UniformOutput', false);
binary_info = vertcat(binary_info{:});

% read and export data from binary files to APLOSE csv / Raven txt files
% PG2APLOSE(info_deploy, folder_wav{1}, binary_folder{2});
if numel(binary_info)== numel(wav_info)
    for i=1:numel(binary_folder)
        PG2APLOSE(info_deploy, folder_wav{1}, binary_folder{i});
    end
else
    error('Number of wav files (%.0f) is different than number of pgdf files (%.0f)', numel(wav_info), numel(binary_info))
end