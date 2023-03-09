function [] = PG2APLOSE(infoAp, wavPath, BinaryPath, format_datestr, TZ)
%% This function import binary files info and export it as an Aplose csv & Raven selection table
start = now;

%Info for the Aplose table to be created
infoAplose = infoAp;

%Add path with matlab functions from PG website
% addpath(genpath(fullfile(fileparts(fileparts(pwd)), 'utilities')))

%wav folder
% folder_data_wav = uigetdir('','Select folder contening wav files');
folder_data_wav =  wavPath;

%Binary folder
% folder_data_PG = uigetdir(folder_data_wav,'Select folder contening PAMGuard binary results');
folder_data_PG = BinaryPath;
datetimestr_folder_PG = char(extractfield(dir(fullfile(folder_data_PG, '/**/*.pgdf')), 'name')');
datetime_folder_PG = datetime(datetimestr_folder_PG(:, end-19:end-5), 'InputFormat', 'yyyyMMdd_HHmmss', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZ', 'TimeZone', TZ);

%Infos from wav files
WavFolderInfo.wavList = dir(fullfile(folder_data_wav, '/**/*.wav'));
T = struct2table(WavFolderInfo.wavList);
WavFolderInfo.wavList = table2struct(sortrows(T, 'name')); %Sort the wav files by their names
WavFolderInfo.wavNames = string(extractfield(WavFolderInfo.wavList, 'name')');
WavFolderInfo.folder = string(extractfield(WavFolderInfo.wavList, 'folder')');

% % % % % % % % % % % % WavFolderInfo.splitDates = split(WavFolderInfo.wavNames, [".","_"," - "],2);
% % % % % % % % % % % % %%%%%%%%%%%% TO ADAPT ACCORDING TO FILENAME %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % % % % % % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % % % % % % % % WavFolderInfo.wavDates = WavFolderInfo.splitDates(:,2); %APOCACO
% % % % % % % % % % % % WavFolderInfo.wavDates_formated = datetime(WavFolderInfo.wavDates, 'InputFormat', 'yyMMddHHmmss', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZ', 'TimeZone', TZ); %APOCADO
% % % % % % % % % % % % % WavFolderInfo.wavDates = strcat(WavFolderInfo.splitDates(:,2),'-',WavFolderInfo.splitDates(:,3)); %CETIROISE
% % % % % % % % % % % % % WavFolderInfo.wavDates_formated = datetime(WavFolderInfo.wavDates, 'InputFormat', 'yyyy-MM-dd-HH-mm-ss', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZ', 'TimeZone', TZ);%CETIROISE
% % % % % % % % % % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % % % % % % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
WavFolderInfo.wavDates_formated = convert_datetime(WavFolderInfo.wavNames, format_datestr, TZ);

for i = 1:length(WavFolderInfo.wavList)
    WavFolderInfo.wavinfo(i,1) = audioinfo(strcat(string(WavFolderInfo.folder(i,:)),"\",string(WavFolderInfo.wavNames(i,:))));
    clc
    disp(['Reading wav files ...', num2str(i),'/',num2str(length(WavFolderInfo.wavList))])
end

n_file_tot = length(WavFolderInfo.wavList);

%inputs used to select the wanted wav files - this is used if all the
%wav files are located in the same folder (APOCADO for instance) but are
%useless if the wanted wav are in the folder (CETIROISE for instance)
%----------------------------------------------------------------------------------------
% input1 = datetime(string(inputdlg("Date & Time beginning (dd MM yyyy HH mm ss) :")), 'InputFormat', 'dd MM yyyy HH mm ss', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZ', 'TimeZone', TZ);
% input2 = datetime(string(inputdlg("Date & Time end (dd MM yyyy HH mm ss) :")), 'InputFormat', 'dd MM yyyy HH mm ss', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSZ', 'TimeZone', TZ);
input1 = min(datetime_folder_PG);
input2 = max(datetime_folder_PG);

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
% WavFolderInfo.splitDates([1:idx_file_beg-1,idx_file_end+1:end])=[];
% WavFolderInfo.wavDates([1:idx_file_beg-1,idx_file_end+1:end])=[];
WavFolderInfo.wavDates_formated([1:idx_file_beg-1,idx_file_end+1:end])=[];
WavFolderInfo.wavinfo([1:idx_file_beg-1,idx_file_end+1:end])=[];
%------------------------------------------------------------------------------------

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

if string(datestr(WavFolderInfo.wavDates_formated(1), 'yymmdd')) == string(datestr(WavFolderInfo.wavDates_formated(end), 'yymmdd'))
    PG_title_datestr = datestr(WavFolderInfo.wavDates_formated(1), 'yymmdd');
else
    PG_title_datestr = strcat(datestr(WavFolderInfo.wavDates_formated(1), 'yymmdd'), '_' , datestr(WavFolderInfo.wavDates_formated(end), 'yymmdd'));
end

%Export PG2Aplose
writetable(PG_Annotation_Aplose, strcat(folder_data_PG,strcat('\PG_rawdata_', PG_title_datestr,'.csv')))


%Export PG2Raven
file_name = [strcat(folder_data_PG,'\PG_rawdata_',PG_title_datestr ,'.txt')];
selec_table = fopen(file_name, 'wt');     % create a text file with the same name than the manual selection table + SRD at the end
fprintf(selec_table,'%s\t%s\t%s\t%s\t%s\t%s\t%s\n', 'Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)', 'Low Freq (Hz)', 'High Freq (Hz)');
fprintf(selec_table,'%.0f\t%.0f\t%.0f\t%.9f\t%.9f\t%.1f\t%.1f\n',PG_Annotation_Raven);
fclose('all');

stop = now;
elapsed_time = (stop-start)*24*3600;

clc
disp(['Raw data files exported in : ', folder_data_PG])
fprintf('\n')
disp(['Elapsed time : ',num2str(uint64(elapsed_time)), ' s'])




end

