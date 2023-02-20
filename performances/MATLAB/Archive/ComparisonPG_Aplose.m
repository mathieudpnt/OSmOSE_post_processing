%% This script compares PG detections vs Manual Aplose annotations
clear;clc

% Time vector creation
%Here we find the duration of the dataset with the begin time 
%of the first file and the begin time and duration of the last file

folder_data_wav= uigetdir('','Select folder contening wav files');
if folder_data_wav == 0
    clc; disp("Select folder contening wav files - Error");
    return
end

wavList = dir(fullfile(folder_data_wav, '*.wav'));
wavNames = "";
wavDates = "";
for i = 1:length(wavList)
    wavNames(i,:) = (wavList(i).name);
    splitDates = split(wavNames(i,:),".");
    wavDates(i,:) = splitDates(2);
end

wavDates_formated = datetime(wavDates, 'InputFormat', 'yyMMddHHmmss', 'Format', 'yyyy MM dd - HH mm ss');
[FirstDate, posMin] = min(wavDates_formated);
[LastDate, posMax] = max(wavDates_formated);

lastwavinfo = audioinfo(strcat(folder_data_wav,"\",string(wavNames(posMax,:))));

firstwavinfo = audioinfo(strcat(folder_data_wav,"\",string(wavNames(posMin,:))));
% time_bin = str2double(inputdlg("time bin ? (s)")); 
% time_bin = 7199;
time_bin = firstwavinfo.Duration;




%Creation of a datenum time vector from beginning of 1st file to end of last file with time_bin as a
%time step
datenum_begin = datenum(FirstDate)*24*3600;
datenum_end = datenum(LastDate)*24*3600 + lastwavinfo.Duration;

%This is to be implemented into ComparisonPG/Raven!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
time_vector_f1 = [datenum_begin:time_bin:datenum_end]';
if length(time_vector_f1) == length(wavList)+1
    time_vector_f = time_vector_f1;
elseif length(time_vector_f1) == length(wavList)
    %the last bin might not me stricly equal to the time_bin (e.i. 9.9s instead
    %of 10s for example) so we "manually" add the last timebin to the time vector
    %Otherwise, hte time vector would lack the last bin
    total_duration = datenum_end - datenum_begin;
    last_bin = mod(total_duration,time_bin);
    time_vector_f = [time_vector_f1; time_vector_f1(end) + datenum(last_bin)];
end



%% Creation of Aplose annotation vector

[Ap_data, Ap_datapath] = uigetfile(strcat(folder_data_wav,'/*.csv'),'Select Aplose annotations');
if Ap_data == 0
    clc; disp("Select Aplose annotations - Error");
    return
end
Ap_Annotation = importAploseSelectionTable(strcat(Ap_datapath,Ap_data));

msg='Select The annotion type to analyse';
opts=[unique(Ap_Annotation.annotation )];
selection_type_data=menu(msg,opts);
type_selected = opts(selection_type_data);
counter = find(Ap_Annotation.annotation ~= type_selected);
Ap_Annotation(counter,:)=[];

% splitDates = split(Ap_Annotation.filename,"_");
% Annot_Dates = "";
% for i = 1:height(Ap_Annotation)
%     splitDates = strsplit(Ap_Annotation.filename(i),{'_','.'});
%     Annot_Dates(i,1) = splitDates(2);
% end
% % Annot_Dates = splitDates(:,3) + splitDates(:,4);
% Annot_Dates_formated = datetime(Annot_Dates, 'InputFormat', 'yyMMddHHmmss', 'Format', 'yyyy MM dd - HH mm ss');


datenum0 = char(strrep(Ap_Annotation.start_datetime,'T',' '));
datenum_Ap_begin = datenum(datenum0(:,1:end-6))*24*3600;

duration_det = double(Ap_Annotation.end_time) - double(Ap_Annotation.start_time);
datenum_Ap_end = datenum_Ap_begin + duration_det;

datenum_Ap = [datenum_Ap_begin, datenum_Ap_end];



%% Creation of PG annotations vector
[PG_data, PG_datapath] = uigetfile(strcat(folder_data_wav,'/*.csv'),'Select PG detections');
if PG_data == 0
    clc; disp("Select Aplose PG - Error");
    return
end

PG_Annotation = importPG_csvSelectionTable(strcat(PG_datapath,PG_data));
% PG_Annotation_begin = char(string(table2cell(PG_Annotation(:,1))));
PG_Annotation_begin = char(PG_Annotation.start_datetime);
PG_Annotation_end = char(PG_Annotation.end_datetime);
% PG_Annotation_end = char(string(table2cell(PG_Annotation(:,2))));
PG_Annotation_begin2 = "";
PG_Annotation_end2 = "";
for i =1:length(PG_Annotation_begin)
    PG_Annotation_begin2(i,1) = string(strrep(PG_Annotation_begin(i,1:end-6),'T',' '));
    PG_Annotation_end2(i,1) = string(strrep(PG_Annotation_end(i,1:end-6),'T',' '));
end
datenum_PG_begin = datenum(PG_Annotation_begin2)*24*3600;
datenum_PG_end = datenum(PG_Annotation_end2)*24*3600;
datenum_PG = [datenum_PG_begin, datenum_PG_end];


%% Comparison of time vectors / Aplose anotation vector
tic
Ap_output = NaN(length(time_vector_f)-1,1);
k=1;

for i = 1:length(time_vector_f)-1
    counter_exceed=[];
    interval_t = fixed.Interval(time_vector_f(i,1), time_vector_f(i+1,1) );
    overlap_intervals = zeros(1,length(datenum_Ap));

    for j = k:length(datenum_Ap) %on parcours le vecteur R de k à length(R_f) et on regarde s'il y a intersection avec le vecteur temporel
        interval_Annot = fixed.Interval(datenum_Ap(j,1), datenum_Ap(j,2) );
        overlap_intervals(j) = overlaps(interval_Annot, interval_t);
        %Si la fin d'une annotation raven(k) depasse la fin de la timebin(i) et se termine une timebin(i+N),
        %l'indice k doit recommencer à cette valeur pour la timebin suivante (cf l66)
        if interval_Annot.RightEnd > interval_t.RightEnd
            counter_exceed = [counter_exceed;j];
        end
    end
    
    if find( overlap_intervals(k:length(datenum_Ap)) == 1, 1 ) == 1
        Ap_output(i,1) = 1; %output_R(j) = 1 si intersection sinon 0
    elseif sum( overlap_intervals(k:length(datenum_Ap))) == 0
        Ap_output(i,1) = 0; %output_R(j) = 1 si intersection sinon 0
    end
   
    if isempty(counter_exceed)
        k=find(overlap_intervals==1,1,'last')+1;
    elseif isempty(counter_exceed) == 0
        k = min(counter_exceed);
    end
    
    %Si pas d'overlap (i.e. output_R ne contient pas de 1), k est
    %réinitialisé à 1
    if isempty(k) == 1
        k = 1;
    end
    
clc; disp([ num2str(i),'/',num2str(length(time_vector_f)-1) ])

end
clc;toc

%% Comparison of time vectors / PG detection vector - method 1
tic
PG_output = NaN(length(time_vector_f)-1,1);
k=1;

for i = 1:length(time_vector_f)-1
    counter_exceed=[];
    interval_t = fixed.Interval(time_vector_f(i,1), time_vector_f(i+1,1) );

    parfor j = k:length(datenum_PG) %on parcours le vecteur R de k à length(R_f) et on regarde s'il y a intersection avec le vecteur temporel
        interval_Annot = fixed.Interval(datenum_PG(j,1), datenum_PG(j,2) );
        overlap_intervals(j) = overlaps(interval_Annot, interval_t);
        %Si la fin d'une annotation raven(k) depasse la fin de la timebin(i) et se termine une timebin(i+N),
        %l'indice k doit recommencer à cette valeur pour la timebin suivante (cf l66)
        if interval_Annot.RightEnd > interval_t.RightEnd
            counter_exceed = [counter_exceed;j]; 
        end
    end
    
    if find( overlap_intervals(k:length(datenum_PG)) == 1, 1 ) == 1
        PG_output(i,1) = 1; %output_R(j) = 1 si intersection sinon 0
    elseif sum( overlap_intervals(k:length(datenum_PG))) == 0
        PG_output(i,1) = 0; %output_R(j) = 1 si intersection sinon 0
    end
   
    if isempty(counter_exceed)
        k=find(overlap_intervals==1,1,'last')+1;
    elseif isempty(counter_exceed) == 0
        k = min(counter_exceed);
    end
    
    %Si pas d'overlap (i.e. output_R ne contient pas de 1), k est
    %réinitialisé à 1
    if isempty(k) == 1
        k = 1;
    end
    clc; disp([ num2str(i),'/',num2str(length(time_vector_f)-1) ])
end
clc; elapsed_time = toc
%elapsed time ~115s

%% Comparison of time vectors / PG detection vector - method 2
tic
interval_t = fixed.Interval(time_vector_f(1:end-1), time_vector_f(2:end) );

parfor i = 1:length(datenum_PG)
    interval_Annot(i) = fixed.Interval(datenum_PG(i,1),datenum_PG(i,2));
end

parfor i = 1:length(interval_Annot)
    overlap_PG_Ap(:,i) = overlaps(interval_Annot(i), interval_t);
end

PG_output2 = any(overlap_PG_Ap,2);
clc; elapsed_time2 = toc
%elapsed time ~20s

%% Result - method 1
comparison = "";
for i = 1:length(PG_output)
    if PG_output(i) == 1
        if Ap_output(i) == 1
            comparison(i,1) = "VP";
        elseif Ap_output(i) == 0
            comparison(i,1) = "FP";
        else
            comparison(i,1) = "erreur999";
        end
    elseif PG_output(i) == 0
        if Ap_output(i) == 1
            comparison(i,1) = "FN";
        elseif Ap_output(i) == 0
            comparison(i,1) = "VN";
        else
            comparison(i,1) = "erreur998";
        end
    else
        comparison(i,1) = "erreur997";
    end
end

Precision = length(find(comparison == "VP")) / (length(find(comparison == "VP")) + length(find(comparison == "FP")));
Recall = length(find(comparison == "VP")) / (length(find(comparison == "VP")) + length(find(comparison == "FN")));

clc
disp(['Precision : ', num2str(Precision), '; Recall : ', num2str(Recall)])
%% Result - method 2
comparison2 = "";
for i = 1:length(PG_output2)
    if PG_output2(i) == 1
        if Ap_output(i) == 1
            comparison2(i,1) = "VP";
        elseif Ap_output(i) == 0
            comparison2(i,1) = "FP";
        else
            comparison2(i,1) = "erreur999";
        end
    elseif PG_output2(i) == 0
        if Ap_output(i) == 1
            comparison2(i,1) = "FN";
        elseif Ap_output(i) == 0
            comparison2(i,1) = "VN";
        else
            comparison2(i,1) = "erreur998";
        end
    else
        comparison2(i,1) = "erreur997";
    end
end

Precision2 = length(find(comparison2 == "VP")) / (length(find(comparison2 == "VP")) + length(find(comparison2 == "FP")));
Recall2 = length(find(comparison2 == "VP")) / (length(find(comparison2 == "VP")) + length(find(comparison2 == "FN")));

clc
disp(['Precision 2: ', num2str(Precision2), '; Recall 2: ', num2str(Recall2)])



