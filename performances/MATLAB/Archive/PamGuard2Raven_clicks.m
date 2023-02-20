%% Fonction qui permet de transformer les résultats de détection du 
% détecteur de clics de PAMGuard sauvegardés dans des fichiers binaires 
% en Selection Table de Raven

% Chemin où se trouve les codes MATLAB pour lire les databases Pamguard
addpath(genpath('C:\Users\torterma\Documents\Projets_Osmose\Sciences\1_PerformanceEvaluation\Benchmark\PAMGuard\Script\PamguardMatlab_20210616'));

% Datenum de la date de début du 1er fichier 
datenum_1stF = datenum(2019,03,02,14,05,39);
% Durée des fichiers audio en secondes
duration_files = 60*10;
% Fréquence d'échantillonnage
Fs = 48000;
% Nombre d'échantillons par fichier
nb_samples_files = duration_files*Fs;

% Load data PAMGuard
folder_data_PG = 'C:\Users\torterma\Documents\Projets_Osmose\Sciences\1_PerformanceEvaluation\Benchmark\PAMGuard\Results\WhistleMoanDet\1_DefaultSettings\20190302';
type_data = 'Click_Detector_Click_Detector_Clicks_*.pgdf';
data = loadPamguardBinaryFolder(folder_data_PG, type_data);

%%

% datenum_files : variable avec les dates des détections en MATLAB
datenum_det={data(1:end).date};
datenum_det = cell2mat(datenum_det);
% duration_det : variable contenant les durée de chaque detection en
% secondes
duration_det = {data(1:end).sampleDuration};
duration_det = cell2mat(duration_det)/Fs;
% Nombre de secondes entre le début de la liste de fichiers et le début de chaque detection 
Beg_sec = (datenum_det-datenum_1stF)*24*60*60;
% Nombre de secondes entre le début de la liste de fichiers et la fin de chaque detection 
End_sec = Beg_sec + duration_det;
% Fréquences limites de chaque détection
freqs={data(1:end).freqLimits};
freqs = cell2mat(freqs);
Low_freq = freqs(1:2:end);
High_freq = freqs(2:2:end);

%% Generate Raven selection Table with appropriate format
L = length(data);
Selection = [1:L]';
View = ones(L,1);
Channel = ones(L,1);

C = [Selection, View, Channel, Beg_sec', End_sec', Low_freq', High_freq']';

file_name = ['test_clicks.txt']
selec_table = fopen(file_name, 'wt');     % create a text file with the same name than the manual selction table + SRD at the end
fprintf(selec_table,'%s\t%s\t%s\t%s\t%s\t%s\t%s\n', 'Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)');
fprintf(selec_table,'%.0f\t%.0f\t%.0f\t%.9f\t%.9f\t%.1f\t%.1f\n',C);
fclose('all');
