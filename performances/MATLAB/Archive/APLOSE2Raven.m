%% Fonction qui permet de transformer les annotations APLOSE Selection Table de Raven
clear;clc

%Select the wav folder
folder_data_wav = uigetdir('','Select folder contening wav files');
if folder_data_wav == 0
    clc; disp("Select folder contening wav files - Error");
    return
end

%List of all .wav dates
wavList = dir(fullfile(folder_data_wav, '*.wav'));
wavNames = '';
wavDates = "";
for i = 1:length(wavList)
    wavNames(i,:) = (wavList(i).name);
    wavDates(i,:) = (wavNames(i,end-15:end-4));
end

wavDates_formated = datetime(wavDates, 'InputFormat', 'yyMMddHHmmss', 'Format', 'yyyy MM dd - HH mm ss');
[FirstDate, posMin] = min(wavDates_formated);

%% Load data APLOSE

[Ap_data, Ap_datapath] = uigetfile(strcat(folder_data_wav,'/*.csv'),'Select Aplose annotations');
if Ap_data == 0
    clc; disp("Select Aplose annotations - Error");
    return
end
Ap_Annotation = importAploseSelectionTable(strcat(Ap_datapath,Ap_data));

msg='Select The annotion type to analyse';
opts=[unique(Ap_Annotation.annotation )];
selection_type_data=menu(msg,opts);
if selection_type_data == 0
    clc; disp("Select The annotion type to analyse - Error");
    return
end
type_selected = opts(selection_type_data);
counter = find(Ap_Annotation.annotation ~= type_selected);
Ap_Annotation(counter,:)=[];

%%
data = Ap_Annotation;

% datenum_files : variable avec les dates des detections en MATLAB
datenum0 = char(strrep(data.start_datetime,'T',' '));
datenum_det = datenum(datenum0(:,1:end-6));

% duration_det : variable contenant les durees de chaque detection en secondes
duration_det = double(data.end_time) - double(data.start_time);

% Nombre de secondes entre le debut de la liste de fichiers et le debut de chaque detection 
Beg_sec = (datenum_det-datenum(FirstDate))*24*60*60;


% Nombre de secondes entre le debut de la liste de fichiers et la fin de chaque detection 
End_sec = Beg_sec + duration_det;

% Frequences limites de chaque detection
Low_freq = double(data.start_frequency);
High_freq = double(data.end_frequency);

% Generate Raven selection Table with appropriate format
L = height(data);
Selection = [1:L]';
View = ones(L,1);
Channel = ones(L,1);

C = [Selection, View, Channel, Beg_sec, End_sec, Low_freq, High_freq]';

file_name = [strcat(folder_data_wav,'\', wavNames(1,1:end-4), ' - APLOSE2Raven Selection Table.txt')];
selec_table = fopen(file_name, 'wt');     % create a text file with the same name than the manual selection table + SRD at the end
fprintf(selec_table,'%s\t%s\t%s\t%s\t%s\t%s\t%s\n', 'Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)');
fprintf(selec_table,'%.0f\t%.0f\t%.0f\t%.9f\t%.9f\t%.1f\t%.1f\n',C);
fclose('all');
clc; disp("Done !");