function [data_table, data_path, data_name] = importBinary(folder_data_wav, WavFolderInfo, folder_data_PG, index_exclude, time_vector)
%Fonction qui permet d'extraire et de convertir les résultats de détection de PAMGuard binaires

% Sampling frequency
Fs = WavFolderInfo.wavinfo(1).SampleRate;


%List of all .pgdf dates
PG_List = dir(fullfile(folder_data_PG, '*.pgdf'));
% PG_Names_temp = cell2mat(extractfield(PG_List,'name')');
PG_Names_temp = extractfield(PG_List,'name')';
for i = 1:length(PG_Names_temp)
    PG_char_temp = char(PG_Names_temp(i));
    PG_Dates(i,:) = PG_char_temp(end-19:end-5);
end
% PG_Dates = PG_Names_temp(:,end-19:end-5);
PG_Names = string(PG_Names_temp);
PG_Dates_formated = datetime(PG_Dates, 'InputFormat', 'yyyyMMdd_HHmmss', 'Format', 'yyyy MM dd - HH mm ss');
[FirstDate, posMin] = min(PG_Dates_formated);
datenum_1stF = datenum(FirstDate);



%The lines below allow the user to choose the detector to analyse
FirstDate_f = string(datetime(FirstDate, 'InputFormat','yyyy MM dd - HH mm ss' , 'Format','yyyyMMdd_HHmmss'));
PG_Names_choice = contains(PG_Names, FirstDate_f);
k=find(PG_Names_choice==1);

detectorNames = "";clc
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

if selection_type_data ~= 0
    type_data = opts(selection_type_data);
else
    clc; disp("selection_type_data - Error");
    return
end

% Load the data
data = loadPamguardBinaryFolder(folder_data_PG, convertStringsToChars(strcat(type_data,"*.pgdf")));

% datenum_files : variable avec les dates des detections en MATLAB
datenum_det={data(1:end).date};
datenum_det = cell2mat(datenum_det);


WavFolderInfo.wavDates_formated;
data_datetime = datetime(datenum_det','ConvertFrom','datenum');

idx_wav=[];
for i=1:length(data)
    idx_wav(i,1) = max(find(data_datetime(i) < WavFolderInfo.wavDates_formated ==0));
end
% idx_wav = floor(cell2mat({data(1:end).UID}' )/1000000);


filename_formated = WavFolderInfo.wavDates_formated(idx_wav);

%Adjustment of the timestamps test
for i =1:length(datenum_det)
    idx = find(filename_formated(i) == WavFolderInfo.wavDates_formated);
    if idx ~= 1 && ~isempty(index_exclude)
        adjust = seconds(time_vector(index_exclude(idx-1)+1) - WavFolderInfo.wavDates_formated(idx)) /3600/24;
        %index_exclude : indexes of last bins before new wav, then
        %index_exclude(i)+1 : indexes of first timebin of a wav i+1

        datenum_det(1,i) = datenum_det(1,i) + adjust;
    end
end



% duration_det : variable contenant les durees de chaque detection en secondes
duration_det = {data(1:end).sampleDuration}';
duration_det = cell2mat(duration_det)/Fs;

datetime_begin = datetime(datenum_det','ConvertFrom','datenum','Format','yyyy MM dd - HH mm ss');
datetime_end = datetime_begin + seconds(duration_det);



% Nombre de secondes entre le debut de la liste de fichiers et le debut de chaque detection 
Beg_sec = seconds(datetime_begin-FirstDate);

% Nombre de secondes entre le debut de la liste de fichiers et la fin de chaque detection 
End_sec = Beg_sec + duration_det;

% Frequences limites de chaque detection
freqs={data(1:end).freqLimits}';
freqs = cell2mat(freqs);
Low_freq = freqs(1:2:end)';
High_freq = freqs(2:2:end)';


data_table = [ array2table([Beg_sec, End_sec, Low_freq, High_freq],...
    'VariableNames',{'Begin_time','End_time','Low_Freq','High_Freq'})...
    , table(datetime_begin, datetime_end), array2table([End_sec-Beg_sec],'VariableNames',{'Duration (s)'}')];


clc




end

