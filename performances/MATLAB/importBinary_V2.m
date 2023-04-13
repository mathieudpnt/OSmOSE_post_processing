function [data_table_Raven, data_table_Aplose] = importBinary_V2(WavFolderInfo, folder_data_PG, TZ,dt_deployments, infoAp)
%Fonction qui permet d'extraire et de convertir les résultats de détection de PAMGuard binaires

% Sampling frequency
Fs = WavFolderInfo.wavinfo(1).SampleRate;


%List of all .pgdf dates
PG_List = dir(fullfile(folder_data_PG, '/**/*.pgdf'));
PG_Names_temp = extractfield(PG_List,'name')';
for i = 1:length(PG_Names_temp)
    PG_char_temp = char(PG_Names_temp(i));
    PG_Dates(i,:) = PG_char_temp(end-19:end-5);
end
% PG_Dates = PG_Names_temp(:,end-19:end-5);
PG_Names = string(PG_Names_temp);
PG_Dates_formated = datetime(PG_Dates, 'InputFormat', 'yyyyMMdd_HHmmss', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZ');
PG_Dates_formated.TimeZone = TZ;
[FirstDate, posMin] = min(PG_Dates_formated);
datenum_1stF = datenum(FirstDate);



%---------------------------------------------------------------------------------The lines below allow the user to choose the detector to analyse
% FirstDate_f = string(datetime(FirstDate, 'InputFormat','yyyy-MM-dd''T''HH:mm:ss.SSSZ' , 'Format','yyyyMMdd_HHmmss'));
% PG_Names_choice = contains(PG_Names, FirstDate_f);
% k=find(PG_Names_choice==1);

% detectorNames = "";clc
% detectorChar = '';
% detectorNames2 = "";
% for i =1:length(k)
%     detectorNames(i,1) = PG_Names(k(i));
%     detectorChar = convertStringsToChars(detectorNames(i));
%     detectorNames2(i,1) = string(detectorChar(1,1:end-21));
% end


% msg='Select The detector to analyse';
% opts=[detectorNames2];
% selection_type_data=menu(msg,opts);
% 
% if selection_type_data ~= 0
%     type_data = opts(selection_type_data);
% else
%     clc; disp("selection_type_data - Error");
%     return
% end

% Load the data
% data = loadPamguardBinaryFolder(folder_data_PG, convertStringsToChars(strcat(type_data,"*.pgdf")),5);
%----------------------------------------------------------------------------------------------------------------------------------------------

data = loadPamguardBinaryFolder(folder_data_PG, '*.pgdf',1); %if the detector name is not the same throughout the folder

%Here we delete the detections that are not within the deployement's datetimes
test = ([cell2mat({data.date})'] < datenum(dt_deployments(1)) ) | ([cell2mat({data.date})'] > datenum(dt_deployments(2)));
data(test) = [];

datenum_det = cell2mat({data(1:end).date})'; %datetime of the detections, MATLAB format
duration_det = cell2mat({data(1:end).sampleDuration})'/Fs; %durations of detections
 
datetime_begin = datetime(datenum_det, 'ConvertFrom','datenum','Format','yyyy-MM-dd''T''HH:mm:ss.SSSZ', 'TimeZone', TZ);
datetime_end = datetime_begin + seconds(duration_det);


%From datetimes to strings with the Aplose format
for i = 1:length(datetime_begin)
    datetime_begin_str(i,:) = insertBefore(char(datetime_begin(i)), length(char(datetime_begin(i)))-1, ":" );
    datetime_end_str(i,:) = insertBefore(char(datetime_end(i)), length(char(datetime_end(i)))-1, ":" );
end

Beg_sec = seconds(datetime_begin-FirstDate); %number of seconds between the first datetime and beginning of each detection
End_sec = Beg_sec + duration_det; %number of seconds between the first datetime and end of each detection


freqs=cell2mat({data(1:end).freqLimits}'); %frequency min and max of each detection
Low_freq = freqs(:,1);
High_freq = freqs(:,2);


% Generate Raven selection Table with appropriate format
L = length(data);
Selection = [1:L]';
View = ones(L,1);
Channel = ones(L,1);

%index of the corresponding wav file for each detection
idx_wav=[];
for i=1:length(datetime_begin)
    if isempty(max(find(datetime_begin(i) < WavFolderInfo.wavDates_formated ==0)))
        idx_wav(i,1) = 1;
    else
        idx_wav(i,1) = max(find(datetime_begin(i) < WavFolderInfo.wavDates_formated ==0));
    end
end

%-----------------------------------------Adjustment of the timestamps test
%For double checkk on Raven

% for i = 1:length(WavFolderInfo.wavDates_formated)
%     idx_adjust(i,1) = min(find(WavFolderInfo.wavDates_formated(idx_wav) == WavFolderInfo.wavDates_formated(i)));
% end



durations = extractfield(WavFolderInfo.wavinfo, 'Duration')';
adjust = 0;
for i = 1:length(WavFolderInfo.wavDates_formated)-1
    adjust = [adjust; seconds(WavFolderInfo.wavDates_formated(i) - WavFolderInfo.wavDates_formated(i+1)) + durations(i)];
end
cumsum_adjust = cumsum(adjust);


for i =1:length(datenum_det)
    datetime_begin_adjusted(i,1) = datetime_begin(i) + seconds(cumsum_adjust(idx_wav(i)));   
end
datetime_end_adjusted = datetime_begin_adjusted + seconds(duration_det);

% Nombre de secondes entre le debut de la liste de fichiers et le debut de chaque detection 
Beg_sec_adjusted = seconds(datetime_begin_adjusted-FirstDate);

% Nombre de secondes entre le debut de la liste de fichiers et la fin de chaque detection 
End_sec_adjusted = Beg_sec_adjusted + duration_det;
%-------------------------------------------------------------------------

data_table_Raven = [Selection, View, Channel, Beg_sec_adjusted, End_sec_adjusted, Low_freq, High_freq]';
data_table_Raven = sortrows(data_table_Raven', 4)';

% Generate Aplose table
annotator = repmat(infoAp.annotator, [L,1]);
annotation = repmat(infoAp.annotation, [L,1]);
dataset = repmat(infoAp.dataset, [L,1]);
filename = WavFolderInfo.wavNames(idx_wav);

data_table_Aplose = [ array2table([dataset, filename, Beg_sec, End_sec, Low_freq, High_freq, annotation, annotator],...
    'VariableNames',{'dataset', 'filename', 'start_time','end_time','start_frequency','end_frequency', 'annotation', 'annotator'})...
    , table(string(datetime_begin_str), string(datetime_end_str), 'VariableNames',{'start_datetime','end_datetime'})];

data_table_Aplose = sortrows(data_table_Aplose, 'start_datetime');




clc;fprintf('Importation of %s detections done\n', num2str(length(data)))

