%% Fonction qui permet de transformer les résultats de détection de PAMGuard
% dans les fichiers binaires en output APLOSE
clear;clc
DATASET_NAME = "APOCADO C2D1 - ST 336363566";
% % DATASET_NAME = str2double(inputdlg("Dataset name ?"));
Label= "whistle and moan";
% % Label = str2double(inputdlg("Label ?"));
Annotator = "mdupon";
% % Annotator = str2double(inputdlg("Annotator ?"));

%Selection of the folder including the PAMGuard functions
addpath(genpath(uigetdir('','Select folder contening PAMGuard MATLAB functions')));

%Select the wav folder
folder_data_wav = uigetdir('','Select folder contening wav files');

%List of all .wav dates
%These lines below are to be adapted to the names of your wav files
wavList = dir(fullfile(folder_data_wav, '*.wav'));
wavNames = "";
wavDates = "";
for i = 1:length(wavList)
    wavNames(i,:) = string(wavList(i).name);   
    splitDates = split(wavNames(i,:),".");
%     wavDates(i,:) = splitDates(3) + splitDates(4);
    wavDates(i,:) = splitDates(2);

end

wavDates_formated = datetime(wavDates, 'InputFormat', 'yyMMddHHmmss', 'Format', 'yyyy MM dd - HH mm ss');

%First measurement
[FirstDate, posMin] = min(wavDates_formated);
wavinfo = audioinfo(strcat(folder_data_wav,"\",string(wavNames(posMin,:))));

% Sampling frequency
Fs = wavinfo.SampleRate;


% Load data PAMGuard
folder_data_PG = uigetdir(folder_data_wav,'Select folder contening PAMGuard binary results');

%List of all .pgdf dates
PG_List = dir(fullfile(folder_data_PG, '*.pgdf'));
PG_Names_temp = '';
PG_Names = "";
PG_Dates = "";
for i = 1:length(PG_List)
    PG_Names_temp = (PG_List(i).name);
    PG_Dates(i,:) = (PG_Names_temp(1,end-19:end-5));
    PG_Names(i,1) = string(PG_Names_temp);
end

PG_Dates_formated = datetime(PG_Dates, 'InputFormat', 'yyyyMMdd_HHmmss', 'Format', 'yyyy MM dd - HH mm ss');
[FirstDate, posMin] = min(PG_Dates_formated);
%datenum corresponding to the beginning of the measurements
datenum_1stF = datenum(FirstDate);



%The lines below allow the user to choose the detector to analyse
FirstDate_f = string(datetime(FirstDate, 'InputFormat','yyyy MM dd - HH mm ss' , 'Format','yyyyMMdd_HHmmss'));
PG_Names_choice = contains(PG_Names, FirstDate_f);
k=find(PG_Names_choice==1);
detectorNames = "";
detectorChar = '';
detectorNames2 = "";
for i =1:length(k)
    detectorNames(i,1) = PG_Names(k(i));
    detectorChar = convertStringsToChars(detectorNames(i));
    detectorNames2(i,1) = string(detectorChar(1,1:end-21));
end

msg='Select The detector to analyse';
opts=[detectorNames2];
selection_type_data=menu(msg,opts);

type_data = opts(selection_type_data);

% Formating the data for APLOSE

data = loadPamguardBinaryFolder(folder_data_PG, convertStringsToChars(strcat(type_data,"*.pgdf")) );

% datestr(cell2mat({data.date}')) %% j'obtiens des date antérieures au début du fichier wav ????????????????????????????



% datenum_files : variable avec les dates des detections en MATLAB
datenum_det = cell2mat({data.date})';

% duration_det : variable contenant les durees de chaque detection en secondes
duration_det = (cell2mat({data.sampleDuration})/Fs)';

% Nombre de secondes entre le debut de la liste de fichiers et le debut de chaque detection 
Beg_sec = (datenum_det-datenum_1stF)*24*60*60;

% Nombre de secondes entre le debut de la liste de fichiers et la fin de chaque detection 
End_sec = Beg_sec + duration_det;

% Liste des fichiers audio dans lesquels sont les détections + liste des début et fin 
% des détection par rapport au début du fichier : infos pas
% dispo avec PAMGuard (ou alors il faut cocher des paramètres dans
% PAMGuard, à voir) donc on ne remplit pas cette info, mais on la faire
% apparaitre quand même dans le csv de résultat pour qu'il ait la même
% forme qu'un csv de résultat APLOSE
L = length(data);
list_dataset(1:L,1) = DATASET_NAME;
list_files(1:L,1)= [0];
list_start_time(1:L,1)= [0];
list_end_time(1:L,1)= [0];

% Fréquences limites de chaque détection
freqs=cell2mat({data.freqLimits}');
list_start_frequency = freqs(:,1);
list_end_frequency = freqs(:,2);
list_annotation(1:L,1) = Label;
list_annotators(1:L,1) = Annotator;

timezone = '+00:00';
datenum_det_end = (datenum_det + (duration_det/(24*60*60)) ) ;
% date début de détection
list_start_datetime=string(datestr(datenum_det, ['yyyy-mm-ddTHH:MM:SS.FFF' timezone]));
% date fin de détection
list_end_datetime=string(datestr(datenum_det_end, ['yyyy-mm-ddTHH:MM:SS.FFF' timezone]));

C = [list_dataset, list_files,list_start_time, list_end_time, list_start_frequency, list_end_frequency,list_annotation,list_annotators, list_start_datetime, list_end_datetime];

%%Generate APLOSE csv
file_name = [strcat(folder_data_wav,"\","PG_detections.csv")]
selec_table = fopen(file_name, 'wt');     % create a text file with the same name than the manual selction table + SRD at the end
    
fprintf(selec_table,'%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n', 'dataset','filename','start_time','end_time','start_frequency','end_frequency','annotation','annotator','start_datetime','end_datetime');
fprintf(selec_table,'%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n',C');
fclose('all');

clc
fprintf('Done !\n');