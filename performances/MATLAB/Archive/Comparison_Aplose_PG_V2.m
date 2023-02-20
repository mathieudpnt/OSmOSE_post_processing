%% This script import binary files info and export it as an Aplose csv & Raven selection table
clear;clc
start = now;

%Info for the Aplose table to be created
infoAplose.annotator = "PAMGuard";
infoAplose.annotation = "Whistle and moan detector";
infoAplose.dataset = "CETIROISE_POINT_B (10_128000)";

%TimeZone
TZ = 'Europe/Paris';


%Time vector resolution
% time_bin = str2double(inputdlg("time bin ? (s)"));
time_bin = 10; %Same size than Aplose annotations

%Add path with matlab functions from PG website
addpath(genpath('L:\acoustock\Bioacoustique\DATASETS\APOCADO\Code_MATLAB'));

%wav folder
% folder_data_wav = uigetdir('','Select folder contening wav files');
folder_data_wav =  'L:\acoustock\Bioacoustique\DATASETS\CETIROISE\DATA\B_Sud Fosse Ouessant\Phase_1\Sylence\2022-07-17';

%Binary folder
% folder_data_PG = uigetdir(folder_data_wav,'Select folder contening PAMGuard binary results');
folder_data_PG = 'L:\acoustock\Bioacoustique\DATASETS\CETIROISE\ANALYSE\PAMGUARD_threshold_7\PHASE_1_POINT_B\Binary\20220717'

%Infos from wav files
WavFolderInfo.wavList = dir(fullfile(folder_data_wav, '/**/*.wav'));
WavFolderInfo.wavNames = string(extractfield(WavFolderInfo.wavList, 'name')');
WavFolderInfo.folder = string(extractfield(WavFolderInfo.wavList, 'folder')');
WavFolderInfo.splitDates = split(WavFolderInfo.wavNames, [".","_"," - "],2);

%%%%%%%%%%%% TO ADAPT ACCORDING TO FILENAME %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% WavFolderInfo.wavDates = WavFolderInfo.splitDates(:,2); %APOCACO
% WavFolderInfo.wavDates_formated = datetime(WavFolderInfo.wavDates, 'InputFormat', 'yyMMddHHmmss', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZ'); %APOCADO
WavFolderInfo.wavDates = strcat(WavFolderInfo.splitDates(:,2),'-',WavFolderInfo.splitDates(:,3)); %CETIROISE
WavFolderInfo.wavDates_formated = datetime(WavFolderInfo.wavDates, 'InputFormat', 'yyyy-MM-dd-HH-mm-ss', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZ');%CETIROISE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
WavFolderInfo.wavDates_formated.TimeZone = TZ;

for i = 1:length(WavFolderInfo.wavList)
    WavFolderInfo.wavinfo(i) = audioinfo(strcat(string(WavFolderInfo.folder(i,:)),"\",string(WavFolderInfo.wavNames(i,:))));
    clc
    disp(['Reading wav files ...', num2str(i),'/',num2str(length(WavFolderInfo.wavList))])
end

n_file_tot = length(WavFolderInfo.wavList);

%inputs used to select the wanted wav files - this is used if all the
%wav files are located in the same folder (APOCADO for instance) but are
%useless if the wanted wav are in the folder (CETIROISE for instance)
if length(unique(WavFolderInfo.folder)) > 1 %==> wav are located in different subfolders
    input1 = datetime(string(inputdlg("Date & Time beginning (dd MM yyyy HH mm ss) :")), 'InputFormat', 'dd MM yyyy HH mm ss', 'Format', 'yyyy MM dd  - HH mm ss', 'TimeZone', TZ);
    input2 = datetime(string(inputdlg("Date & Time end (dd MM yyyy HH mm ss) :")), 'InputFormat', 'dd MM yyyy HH mm ss', 'Format', 'yyyy MM dd  - HH mm ss', 'TimeZone', TZ);

    %File selection
    idx_file{1} = max(find(WavFolderInfo.wavDates_formated < input1));
    if isempty(idx_file{1})
        idx_file_beg = 1;
    else
        [min_val(1,1) min_idx(1,1)]= min(abs(input1-WavFolderInfo.wavDates_formated(idx_file{1}:idx_file{1}+1)));
        if min_idx(1,1) == 1
            idx_file_beg = idx_file{1};
        elseif min_idx(1,1) == 2
            idx_file_beg = idx_file{1}+1;
        end
    end

    idx_file{2} = min(find(WavFolderInfo.wavDates_formated > input2));
    if isempty(idx_file{2})
        idx_file_end = length(WavFolderInfo.wavDates_formated);
    else
        [min_val(1,2) min_idx(1,2)]= min(abs(input1-WavFolderInfo.wavDates_formated(idx_file{2}-1:idx_file{2})));
        if min_idx(1,2) == 1
            idx_file_end = idx_file{2}-1;
        elseif min_idx(1,2) == 2
            idx_file_end = idx_file{2};
        end
    end
    WavFolderInfo.wavList([1:idx_file_beg-1,idx_file_end+1:end])=[];
    WavFolderInfo.wavNames([1:idx_file_beg-1,idx_file_end+1:end])=[];
    WavFolderInfo.splitDates([1:idx_file_beg-1,idx_file_end+1:end])=[];
    WavFolderInfo.wavDates([1:idx_file_beg-1,idx_file_end+1:end])=[];
    WavFolderInfo.wavDates_formated([1:idx_file_beg-1,idx_file_end+1:end])=[];
    WavFolderInfo.wavinfo([1:idx_file_beg-1,idx_file_end+1:end])=[];
end

Firstname = char(WavFolderInfo.wavNames(1));
WavFolderInfo.txt_filename = string(Firstname(1,1:end-4));

clc;disp(strcat("1st wav : ", WavFolderInfo.wavList(1).name))
disp(strcat("last wav : ", WavFolderInfo.wavList(end).name))
disp(strcat(num2str(length(WavFolderInfo.wavList)),"/", num2str(n_file_tot), " files"))


% Export PG data
[PG_Annotation_Raven,  PG_Annotation_Aplose] = importBinary_V2(WavFolderInfo, folder_data_PG, TZ, infoAplose);

% %% Deletion of annotation not within the wanted datetimes
% idx3 = find(PG_dt.beg > input1 == 0);
% if ~isempty(idx3)
%     PG_Annotation_Aplose(idx3,:) = [];
%     PG_Annotation_Raven(:,idx3) = [];
% end
% 
% idx4 = find(PG_dt.end < input2 == 0);
% if ~isempty(idx4)
%     PG_Annotation_Aplose(idx4,:) = [];
%     PG_Annotation_Raven(:,idx4) = [];
% end

if datestr(WavFolderInfo.wavDates_formated(1), 'yymmdd') == datestr(WavFolderInfo.wavDates_formated(end), 'yymmdd')
    PG_title_datestr = datestr(WavFolderInfo.wavDates_formated(1), 'yymmdd');
else
    PG_title_datestr = datestr(WavFolderInfo.wavDates_formated(1), 'yymmdd'),'_', datestr(WavFolderInfo.wavDates_formated(end), 'yymmdd');
end

%Export PG2Aplose
% writetable(PG_Annotation_Aplose, strcat(folder_data_PG,strcat('\PG_rawdata_', PG_title_datestr,'.csv')))
writetable(PG_Annotation_Aplose, strcat('C:\Users\dupontma2\Desktop\testMATLAB\PG_rawdata_', PG_title_datestr,'.csv'))

%Export PG2Raven
% file_name = [strcat(folder_data_PG,'\PG_rawdata_',PG_title_datestr ,'.txt')];
file_name = [strcat('C:\Users\dupontma2\Desktop\testMATLAB\PG_rawdata_',PG_title_datestr ,'.txt')];
selec_table = fopen(file_name, 'wt');     % create a text file with the same name than the manual selection table + SRD at the end
fprintf(selec_table,'%s\t%s\t%s\t%s\t%s\t%s\t%s\n', 'Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)');
fprintf(selec_table,'%.0f\t%.0f\t%.0f\t%.9f\t%.9f\t%.1f\t%.1f\n',PG_Annotation_Raven);
fclose('all');

stop = now;
elapsed_time = (stop-start)*24*3600;

clc
disp(['Raw data files exported in : ', folder_data_PG])
fprintf('\n')
disp(['Elapsed time : ',num2str(elapsed_time,3), ' s'])


