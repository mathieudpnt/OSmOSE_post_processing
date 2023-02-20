%% This script compares PG detections vs Manual Aplose annotations
%3 vectors are created :
% % % % % %-A time vector is created from the 1st measurement to the end of the last
% % % % % %measurement with a user defined time bin
% % % % % %-An Aplose vector with the timestamps of each annotation
% % % % % %-A PG vector with the timestamps of each detection, the latter is then
% % % % % %formatted so that when one or more detection are present within an
% % % % % %Aplose box, a PG box with the same timestamps is created.
%The formatted PG vector and Aplose vector are then compared to estimate the performances of the PG detector   

% Computation time ~1min for a 24h period

clear;clc

%Add path with matlab functions from PG website
addpath(genpath('U:\Documents\Pamguard\pgmatlab'));
%Add path with matlab functions from PG website
addpath(genpath('L:\acoustock\Bioacoustique\DATASETS\APOCADO\Code_MATLAB'));


%wav folder
folder_data_wav= uigetdir('','Select folder contening wav files');
if folder_data_wav == 0
    clc; disp("Select folder contening wav files - Error");
    return
end

%Infos from wav files
WavFolderInfo.wavList = dir(fullfile(folder_data_wav, '*.wav'));
WavFolderInfo.wavNames = string(extractfield(WavFolderInfo.wavList, 'name')');
WavFolderInfo.splitDates = split(WavFolderInfo.wavNames, [".","_"," - "],2);
% WavFolderInfo.wavDates = WavFolderInfo.splitDates(:,2);
WavFolderInfo.wavDates = strcat(WavFolderInfo.splitDates(:,2),'-',WavFolderInfo.splitDates(:,3));
% WavFolderInfo.wavDates_formated = datetime(WavFolderInfo.wavDates, 'InputFormat', 'yyMMddHHmmss', 'Format', 'yyyy MM dd - HH mm ss');
WavFolderInfo.wavDates_formated = datetime(WavFolderInfo.wavDates, 'InputFormat', 'yyyy-MM-dd-HH-mm-ss', 'Format', 'yyyy MM dd - HH mm ss');
for i = 1:length(WavFolderInfo.wavList)
    WavFolderInfo.wavinfo(i) = audioinfo(strcat(folder_data_wav,"\",string(WavFolderInfo.wavNames(i,:))));
end

Firstname = char(WavFolderInfo.wavNames(1));
WavFolderInfo.txt_filename = string(Firstname(1,1:end-4));

%data folder
folder_data = fileparts(folder_data_wav);

%Binary folder
folder_data_PG = uigetdir(folder_data,'Select folder contening PAMGuard binary results');
if folder_data_PG == 0
    clc; disp("Select folder contening PAMGuard binary results - Error");
    return
end

%If choice = 1, all the wave are analysed
%If choice = 2, the user define a range of study
%TODO : gérer erreurs input
choice = 1;

switch choice
    case 2
    input1 = string(inputdlg("Date & Time beginning (dd MM yyyy HH mm ss) :"));
    input2 = string(inputdlg("Date & Time ending (dd MM yyyy HH mm ss) :"));
end

%Time vector resolution
% time_bin = str2double(inputdlg("time bin ? (s)"));
time_bin = 10; %Same size than Aplose annotations

%%
%Aplose annotation csv file
[Ap_data_name, Ap_datapath] = uigetfile(strcat(fileparts(folder_data_wav),'/*.csv'),'Select Aplose annotations');
if Ap_data_name == 0
    clc; disp("Select Aplose annotations - Error");
    return
end


%% Time vector creation
tic

for i = 1:length(WavFolderInfo.wavList)
    nb_bin_int(i,1) = fix(WavFolderInfo.wavinfo(i).Duration/time_bin); %number of complete aplose time bins per wav file
    last_bins(i,1) = mod(WavFolderInfo.wavinfo(i).Duration,time_bin); %last time bin per wav file
    bins(:,i) = [ones(nb_bin_int(i),1)*time_bin; last_bins(i)];
end
duration_time = bins(:); %All the time bins of the campaign
index_exclude = find(duration_time~=time_bin); %for now, one have to exlude those indexes for Aplose does not include them in the annotation campaign

time_vector_datetime =  [WavFolderInfo.wavDates_formated(1); WavFolderInfo.wavDates_formated(1) + cumsum(seconds(duration_time))];
time_vector = datenum(time_vector_datetime)*24*3600;

elapsed_time.time_vector_creation = toc;

%% Creation of Aplose annotation vector
Ap_Annotation = importAploseSelectionTable(strcat(Ap_datapath,Ap_data_name),WavFolderInfo, time_vector_datetime, index_exclude);

%Selection of the annotator
msg_annotator='Select the annotator';
opts=[unique(Ap_Annotation.annotator );"all"];
selection_annotator=menu(msg_annotator,opts);
annotator_selected = opts(selection_annotator);
tic
if annotator_selected ~= "all"
    counter_annotator = find(Ap_Annotation.annotator ~= annotator_selected);
    Ap_Annotation(counter_annotator,:)=[]; %Deletion of the annotations not correponding to the selected annotator
end

%Selection of the annotation type
msg_annotation='Select the annotion type to analyse';
opts=[unique(Ap_Annotation.annotation )];
selection_type_data=menu(msg_annotation,opts);
type_selected = opts(selection_type_data);
counter_annotation = find(Ap_Annotation.annotation ~= type_selected);
Ap_Annotation(counter_annotation,:)=[]; %Deletion of the annotations not correponding to the type of annotation selected by user

Ap_Annotation = sortrows(Ap_Annotation, 5);

nb_sec_begin_Ap = datenum(Ap_Annotation.start_datetime)*24*3600;
duration_Ap = Ap_Annotation.end_time - Ap_Annotation.start_time;
nb_sec_end_Ap = nb_sec_begin_Ap + duration_Ap;
datenum_Ap = [nb_sec_begin_Ap, nb_sec_end_Ap]; %in second

switch choice
        case 2
    interval_Ap_total = [nb_sec_begin_Ap(1), nb_sec_end_Ap(end)];
    int_t_total = [time_vector(1), time_vector(end)];

    overlap_total_Ap = intersection_vect(interval_Ap_total, int_t_total);
    if overlap_total_Ap ~= 1
        msg = sprintf('Error - No overlap between Aplose annotations (%s // %s)\n\nand user defined period (%s // %s)',...
            datestr(nb_sec_begin_Ap(1)/(3600*24)), datestr(nb_sec_end_Ap(end)/(3600*24)), datestr(time_vector(1)/(3600*24)), datestr(time_vector(end)/(3600*24)) );
        clc; disp(msg);
        return
    elseif overlap_total_Ap == 1
        %on supprime les annotations dont les timestamps sont en dehors de l'intervalle de temps spécifiée par l'utilisateur
        interval_Ap = [nb_sec_begin_Ap, nb_sec_end_Ap];
        for i = 1:length(interval_Ap)
            inter(i,1) = intersection_vect(interval_Ap(i,:), int_t_total); 
        end
        idx = find(inter~=1);
        
        if ~isempty(idx) %if idx is not empty then delete the indexes of datenum__Ap and Ap_Annotation with annotation outside of the time range of interest
            datenum_Ap(idx,:) = [];  
            Ap_Annotation(idx,:)=[];
        end
    end
end

elapsed_time.Ap_vector_creation = toc;
%% Creation of PG annotations vector
tic
PG_Annotation = importBinary(folder_data_wav, WavFolderInfo, folder_data_PG, index_exclude);
if exist('PG_Annotation','var') == 0
    clc; disp("Select PG detections - Error");
    return
end

nb_sec_begin_PG = datenum(PG_Annotation.datetime_begin) *24*3600;
nb_sec_end_PG = datenum(PG_Annotation.datetime_end) *24*3600;
int_PG = [nb_sec_begin_PG, nb_sec_end_PG]; %in second

switch choice
    case 2
        int_PG_total = [int_PG(1,1), int_PG(end,2)];

%         overlap_total_PG = overlaps(int_PG_total, int_t_total);
        overlap_total_PG = intersection_vect(int_PG_total, int_t_total);
        if overlap_total_PG ~= 1
            msg = sprintf('Error - No overlap between PAMGuard detections (%s // %s)\n\nand user defined period (%s // %s)',...
                datestr(int_PG(1,1)/(3600*24)), datestr(int_PG(end,2)/(3600*24)), datestr(time_vector(1)/(3600*24)), datestr(time_vector(end)/(3600*24)) );
            clc; disp(msg);
            return
        elseif overlap_total_PG == 1
            %on supprime les annotations dont les timestamps sont en dehors de l'intervalle de temps spécifiée par l'utilisateur
%             idx2 = find(intersection_vect( int_PG, int_t_total) ~=1); 
            for i = 1:length(int_PG)
                inter2(i,1) = intersection_vect(int_PG(i,:), int_t_total); 
            end
            idx2 = find(inter2~=1);
            if ~isempty(idx2) %if idx2 is not empty then delete the indexes of int_PG and PG_Annotation with annotation outside of the time range of interest
                int_PG(idx2,:) = [];
                PG_Annotation(idx2,:) = [];
            end
        end
end

elapsed_time.PG_vector_creation = toc;

%% Output Aplose
tic

interval_Ap = [nb_sec_begin_Ap, nb_sec_end_Ap]; %Aplose annotations intervals
interval_time = [ time_vector((1:end-1),1), time_vector((2:end),1)]; %Time intervals

output_Ap = [];
for i = 1:length(interval_time)
    for j = 1:length(interval_Ap)
        inter(j,1) = intersection_vect(interval_time(i,:), interval_Ap(j,:))  ;
    end
    idx_overlap = find(inter==1); %indexes of overlapping Ap annotations(j) with timebox(i)
    
    if length(idx_overlap) > 1 %More than 1 overlap
        for k = 1:length(idx_overlap)
            memory_ovrlp(k) = overlap_rate(interval_time(i,:) , interval_Ap(idx_overlap(k),:) )  ;%Pondération car la fenetre Aplose overlap sur plusieurs time bin
        end
        [~, idx_ovlrp_max] = max(memory_ovrlp);
        if max(memory_ovrlp) == 0
            output_Ap(i,1) = 0;
        elseif max(memory_ovrlp) < 0.95 %if a time bin overlap with an aplose window less than 95%, this is an issue
            return
        else
            output_Ap(i,1) = 1;
        end
        
    elseif length(idx_overlap) == 1
        output_Ap(i,1) = 1;
    elseif length(idx_overlap) == 0
        output_Ap(i,1) = 0;
    end
    clc;disp([num2str(i),'/',num2str(length(interval_time))])
end

elapsed_time.output_Ap = toc;

%% Output PG
tic

interval_PG = [nb_sec_begin_PG, nb_sec_end_PG]; %Aplose annotations intervals
interval_time = [ time_vector((1:end-1),1), time_vector((2:end),1)]; %Time intervals

output_PG = [];
for i = 1:length(interval_time)
    for j = 1:length(interval_PG)
        inter(j,1) = intersection_vect(interval_time(i,:), interval_PG(j,:))  ;
    end
    idx_overlap = find(inter==1); %indexes of overlapping PG detections(j) with timebox(i)
    
    if length(idx_overlap) >= 1 %More than 1 overlap
        output_PG(i,1) = 1;
    elseif length(idx_overlap) == 0
        output_PG(i,1) = 0;
    end
    clc;disp([num2str(i),'/',num2str(length(interval_time))])
end


%Conversion from PG detection to Aplose equivalent boxes
start_time = zeros( sum(output_PG),1 );
start_frequency = zeros( sum(output_PG),1 );
end_time = ones( sum(output_PG),1 )*time_bin;
end_frequency = ones( sum(output_PG),1 )*60000;
annotation = repmat(type_selected,[sum(output_PG),1]);

interval_Ap_formatted = interval_time(find(output_PG),:);
for i = 1:length(interval_Ap_formatted)
    start_datetime(i,1) = datetime(interval_Ap_formatted(i,1)/(24*3600), 'ConvertFrom', 'datenum');
    end_datetime(i,1) = datetime(interval_Ap_formatted(i,2)/(24*3600), 'ConvertFrom', 'datenum');
end
PG_Annotation_formatted = table(start_time, end_time, start_frequency, end_frequency, start_datetime, end_datetime, annotation); %export format Aplose des detections PG

elapsed_time.output_PG = toc;
elapsed_time.total_elapsed  = elapsed_time.time_vector_creation + elapsed_time.Ap_vector_creation +elapsed_time.PG_vector_creation +elapsed_time.output_Ap +elapsed_time.output_PG;
%% Results

comparison = "";
for i = 1:length(output_PG)
    if output_PG(i) == 1
        if output_Ap(i) == 1
            comparison(i,1) = "VP";
        elseif output_Ap(i) == 0
            comparison(i,1) = "FP";
        else
            comparison(i,1) = "erreur999";
        end
    elseif output_PG(i) == 0
        if output_Ap(i) == 1
            comparison(i,1) = "FN";
        elseif output_Ap(i) == 0
            comparison(i,1) = "VN";
        else
            comparison(i,1) = "erreur998";
        end
    else
        comparison(i,1) = "erreur997";
    end
end

results.nb_total = length(output_PG);
results.nb_VN = length(find(comparison == "VN"));
results.nb_VP = length(find(comparison == "VP"));
results.nb_FP = length(find(comparison == "FP"));
results.nb_FN = length(find(comparison == "FN"));
results.nb_e = length(find(comparison == "erreur999"))+length(find(comparison == "erreur998"))+length(find(comparison == "erreur997"));

results.precision = results.nb_VP/(results.nb_VP + results.nb_FP);
results.recall = results.nb_VP/(results.nb_VP + results.nb_FN);

clc
% disp(['Precision : ', num2str(results.precision), '; Recall : ', num2str(results.recall)])
disp(results)
disp(elapsed_time)



%% Save results
date_folder = char(datetime(now,'ConvertFrom','datenum','Format','MM-dd_HH-mm-ss'));
folder_result = strcat(folder_data_wav, '\Results\', annotator_selected,'\');
mkdir(folder_result);

D = [results.nb_total, results.nb_VP, results.nb_FP, results.nb_VN, results.nb_FN, results.precision, results.recall  ];

file_name = [strcat(folder_result,'\', WavFolderInfo.txt_filename, ' - results_',date_folder,'.txt')];
selec_table = fopen(file_name, 'wt');
fprintf(selec_table,'%s\n%.0f\n\n%s\n%.0f\n\n%s\n%.0f\n\n%s\n%.0f\n\n%s\n%.0f\n\n%s\n%.4f\n\n%s\n%.4f\n\n%s\n%.0f\n\n', ...
    'nb_total', results.nb_total, 'VP', results.nb_VP, 'FP', results.nb_FP, 'VN', results.nb_VN, 'FN', results.nb_FN,...
    'Precision',results.precision, 'Recall', results.recall, 'Elapsed time', elapsed_time.total_elapsed);

fclose('all');

export_time2Raven(folder_result, WavFolderInfo, time_vector_datetime, time_bin, duration_time) %Time vector as a Raven Table

export_Aplose2Raven(WavFolderInfo, Ap_Annotation, Ap_datapath, Ap_data_name, folder_result)

export_PG2Raven(PG_Annotation, folder_result, WavFolderInfo)

export_Aplose2Raven(WavFolderInfo, PG_Annotation_formatted, Ap_datapath, Ap_data_name, folder_result, ' - PamGuard2Raven formatted Selection Table.txt')

clc;disp("Results and tables saved");