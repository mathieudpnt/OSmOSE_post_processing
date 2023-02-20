%% This script compares PG detections vs Manual Aplose annotations
%A time vector is created from the 1st measurement to the end of the last
%measurement with a user defined time bin. Each time bin is compared to the
%annotations/detections
clear;clc

folder_data_wav= uigetdir('','Select folder contening wav files');
if folder_data_wav == 0
    clc; disp("Select folder contening wav files - Error");
    return
end

%% Time vector creation

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
time_bin = 60; %Same size than Aplose annotations

choice_time_vector = 1;
switch choice_time_vector
    case 1
        %Creation of a time vector from beginning of 1st file to end of last file with time_bin as a time step
        nb_sec_begin_time = datenum(FirstDate)*24*3600; %in second
        nb_sec_end_time = datenum(LastDate)*24*3600 + lastwavinfo.Duration; %in second
    case 2
        %OR Creation of a datenum time vector with beginning and ending defined by user
        %(used if the user does not want all data but just one deployement for instance)
        %TODO : gérer erreurs input
        input1 = string(inputdlg("Date & Time beginning (dd MM yyyy HH mm ss) :"));
        input2 = string(inputdlg("Date & Time ending (dd MM yyyy HH mm ss) :"));
        nb_sec_begin_time = datenum(datetime(input1, 'InputFormat', 'dd MM yyyy HH mm ss', 'Format', 'yyyy MM dd - HH mm ss'))*24*3600;
        nb_sec_end_time = datenum(datetime(input2, 'InputFormat', 'dd MM yyyy HH mm ss', 'Format', 'yyyy MM dd - HH mm ss'))*24*3600;
end


total_duration = nb_sec_end_time - nb_sec_begin_time;
last_bin = mod(total_duration,time_bin);

if last_bin == 0
    time_vector = [nb_sec_begin_time:time_bin:nb_sec_end_time]';
elseif last_bin ~= 0
    %When creating the time vector, the last bin might not me stricly equal to the time_bin (e.i. 9.9s instead
    %of 10s for example) so we "manually" add the last timebin to the time vector. Otherwise, the time vector would lack the last bin
%     time_vector = [nb_sec_begin_time:time_bin:(nb_sec_end_time-time_bin)]';
    time_vector = [nb_sec_begin_time:time_bin:(nb_sec_end_time)]';
    time_vector = [time_vector; time_vector(end)+last_bin];
end

export_time2Raven(folder_data_wav, time_vector, time_bin, last_bin) %Time vector as a Raven Table - For the sake of control

int_t_total = fixed.Interval(time_vector(1), time_vector(end), '()' );


%% Creation of Aplose annotation vector

[Ap_data_name, Ap_datapath] = uigetfile(strcat(folder_data_wav,'/*.csv'),'Select Aplose annotations');
if Ap_data_name == 0
    clc; disp("Select Aplose annotations - Error");
    return
end
Ap_Annotation = importAploseSelectionTableNEW(strcat(Ap_datapath,Ap_data_name));

msg='Select The annotion type to analyse';
opts=[unique(Ap_Annotation.annotation )];
selection_type_data=menu(msg,opts);
type_selected = opts(selection_type_data);
counter = find(Ap_Annotation.annotation ~= type_selected);
Ap_Annotation(counter,:)=[];


nb_sec_begin_Ap = datenum(Ap_Annotation.start_datetime)*24*3600;
duration_det = Ap_Annotation.end_time - Ap_Annotation.start_time;
nb_sec_end_Ap = nb_sec_begin_Ap + duration_det;

datenum_Ap = [nb_sec_begin_Ap, nb_sec_end_Ap]; %in second


switch choice_time_vector
        case 2
    interval_Ap_total = fixed.Interval(nb_sec_begin_Ap(1), nb_sec_end_Ap(end), '()' );

    overlap_total_Ap = overlaps(interval_Ap_total, int_t_total);
    if overlap_total_Ap ~= 1
        msg = sprintf('Error - No overlap between Aplose annotations (%s // %s)\n\nand user defined period (%s // %s)',...
            datestr(nb_sec_begin_Ap(1)/(3600*24)), datestr(nb_sec_end_Ap(end)/(3600*24)), datestr(time_vector(1)/(3600*24)), datestr(time_vector(end)/(3600*24)) );
        clc; disp(msg);
        return
    elseif overlap_total_Ap == 1
        %on supprime les annotations dont les timestamps sont en dehors de l'intervalle de temps spécifiée par l'utilisateur
        interval_Ap = fixed.Interval(nb_sec_begin_Ap, nb_sec_end_Ap, '()' );
        idx = find( overlaps(interval_Ap, int_t_total)~=1 , 1 ); 
        if ~isempty(idx) %if idx is not empty then delete the indexes of datenum__Ap and Ap_Annotation with annotation outside of the time range of interest
            datenum_Ap(idx,:) = [];  
            Ap_Annotation(idx,:)=[];
        end
    end
end

% Creation of Aplose annotation table in Raven output format
switch choice_time_vector
    case 1 %Nom fichier txt generique
        export_Aplose2Raven(Ap_Annotation, Ap_datapath, Ap_data_name, folder_data_wav)
    case 2 %Nom avec user defined dates
        date1 = string(datetime((nb_sec_begin_time)/(24*3600), 'ConvertFrom', 'datenum', 'Format', 'dd-MM-yy'));
        date2 = string(datetime((nb_sec_end_time)/(24*3600), 'ConvertFrom', 'datenum', 'Format', 'dd-MM-yy'));
        time1 = string(datetime((nb_sec_begin_time)/(24*3600), 'ConvertFrom', 'datenum', 'Format', '_HH_mm_ss'));
        time2 = string(datetime((nb_sec_end_time)/(24*3600), 'ConvertFrom', 'datenum', 'Format', '_HH_mm_ss'));
        string_name1 = strcat(" Aplose2Raven " ,date1,time1," to ",date2,time2, ".txt");

        export_Aplose2Raven(Ap_Annotation, Ap_datapath, Ap_data_name, folder_data_wav, string_name1)
end



%% Creation of PG annotations vector

PG_Annotation = importBinaryNEW(folder_data_wav);

nb_sec_begin_PG = datenum(PG_Annotation.datetime_begin) *24*3600;
nb_sec_end_PG = datenum(PG_Annotation.datetime_end) *24*3600;
datenum_PG = [nb_sec_begin_PG, nb_sec_end_PG]; %in second

switch choice_time_vector
    case 2
        int_PG_total = fixed.Interval(datenum_PG(1,1), datenum_PG(end,2), '()' );

        overlap_total_PG = overlaps(int_PG_total, int_t_total);
        if overlap_total_PG ~= 1
            msg = sprintf('Error - No overlap between PAMGuard detections (%s // %s)\n\nand user defined period (%s // %s)',...
                datestr(datenum_PG(1,1)/(3600*24)), datestr(datenum_PG(end,2)/(3600*24)), datestr(time_vector(1)/(3600*24)), datestr(time_vector(end)/(3600*24)) );
            clc; disp(msg);
            return
        elseif overlap_total_PG == 1
            %on supprime les annotations dont les timestamps sont en dehors de l'intervalle de temps spécifiée par l'utilisateur
            int_PG = fixed.Interval(datenum_PG(:,1), datenum_PG(:,2), '()' );
            idx2 = find(overlaps( int_PG, int_t_total) ~=1); 
    %         tests0 = [string(datestr(datenum_PG(5,1)/(3600*24))), string(datestr(datenum_PG(5,2)/(3600*24)))]
    %         tests1 = [string(datestr(datenum_PG(6,1)/(3600*24))), string(datestr(datenum_PG(6,2)/(3600*24)))]
    %         tests2 = [string(datestr(double(int_t_total.LeftEnd)/(3600*24))), string(datestr(double(int_t_total.RightEnd)/(3600*24)))]

            datenum_PG(idx2,:) = [];
            PG_Annotation(idx2,:) = [];
        end
end

% Creation of PG detection table in Raven output format
switch choice_time_vector
    case 1 %Nom fichier txt generique
        export_PG2Raven(PG_Annotation, folder_data_wav)
    case 2 %Nom avec user defined dates
        date1 = string(datetime((nb_sec_begin_time)/(24*3600), 'ConvertFrom', 'datenum', 'Format', 'dd-MM-yy'));
        date2 = string(datetime((nb_sec_end_time)/(24*3600), 'ConvertFrom', 'datenum', 'Format', 'dd-MM-yy'));
        time1 = string(datetime((nb_sec_begin_time)/(24*3600), 'ConvertFrom', 'datenum', 'Format', '_HH_mm_ss'));
        time2 = string(datetime((nb_sec_end_time)/(24*3600), 'ConvertFrom', 'datenum', 'Format', '_HH_mm_ss'));
        string_name2 = strcat(" PG2Raven " ,date1,time1," to ",date2,time2, ".txt");

        export_PG2Raven(PG_Annotation, folder_data_wav, string_name2)
end


%% Output Aplose
interval_Ap = fixed.Interval(nb_sec_begin_Ap, nb_sec_end_Ap, '()' ); %Aplose annotations intervals
interval_time = fixed.Interval( time_vector((1:end-1),1), time_vector((2:end),1), '()'); %Time intervals

%this loop is used if an Aplose interval overlaps more than one time
%interval, it should not happen. It might happen if the Aplose interval
%overlap on one single value (?) of the following/previous time frame.
%To get rid of this, the overlap rate is calcultated between the aplose
%interval and the overlapping time intervals. If the overlap_rate is below
%a defined treshold (60% here for instance), it is not considered as an
%overlap, the time interval is not kept.
idx_overlap = {};
for i = 1:length(interval_Ap)
rate = [];
idx_overlap(i,1) = {find( overlaps(interval_time, interval_Ap(i)) )};
    if length(idx_overlap{i}) > 1 %More than 1 overlap
        for j = 1:length(idx_overlap{i})
            rate(j) = overlap_rate( interval_time(idx_overlap{i}(j)), interval_Ap(i) );
        end
        for k = 1:length(rate)
            if rate(k) < 0.6
%                 idx_overlap{i}(k) = [];
            end
        end
    end
end
idx_overlap = cell2mat(idx_overlap);

output_Ap = zeros(length(interval_time),1);
output_Ap(idx_overlap) = 1;

%% Output PG
interval_PG = fixed.Interval(nb_sec_begin_PG, nb_sec_end_PG); %Aplose annotations intervals
interval_time = fixed.Interval( time_vector((1:end-1),1), time_vector((2:end),1), '()'); %Time intervals


idx_overlap = {};
for i = 1:length(interval_PG)
rate = [];
idx_overlap(i,1) = {find( overlaps(interval_time, interval_PG(i)) )};
    if length(idx_overlap{i}) > 1 %More than 1 overlap
        for j = 1:length(idx_overlap{i})
            rate(j) = overlap_rate( interval_time(idx_overlap{i}(j)), interval_PG(i) );
        end
        for k = 1:length(rate)
            if rate(k) < 0.6
%                 idx_overlap{i}(k) = [];
            end
        end
    end
end
idx_overlap = unique(cell2mat(idx_overlap));

output_PG = zeros(length(interval_time),1);
output_PG(idx_overlap) = 1;

[interval_time, output_Ap, output_PG]


start_time = zeros( sum(output_PG),1 );
start_frequency = zeros( sum(output_PG),1 );
end_time = ones( sum(output_PG),1 )*time_bin;
end_frequency = ones( sum(output_PG),1 )*60000;
annotation = repmat(type_selected,[sum(output_PG),1]);
interval_Ap_formatted = interval_time(idx_overlap)
for i = 1:length(interval_Ap_formatted)
start_datetime(i,1) = datetime(double(interval_Ap_formatted(i).LeftEnd)/(24*3600), 'ConvertFrom', 'datenum');
end_datetime(i,1) = datetime(double(interval_Ap_formatted(i).RightEnd)/(24*3600), 'ConvertFrom', 'datenum');
end
PG_Annotation_formatted = table(start_time, end_time, start_frequency, end_frequency, start_datetime, end_datetime, annotation) %export format Aplose des detections PG

export_Aplose2Raven(PG_Annotation_formatted, Ap_datapath, Ap_data_name, folder_data_wav, ' - PamGuard2Raven formatted Selection Table.txt')
%
%
%
%
%
%
%
%
%


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

nb_VN = length(find(comparison == "VN"));
nb_VP = length(find(comparison == "VP"));
nb_FP = length(find(comparison == "FP"));
nb_FN = length(find(comparison == "FN"));

Precision = nb_VP/(nb_VP + nb_FP);
Recall = nb_VP/(nb_VP + nb_FN);

clc
disp(['Precision : ', num2str(Precision), '; Recall : ', num2str(Recall)])
%% Comparison of time vectors / Aplose anotation vector
tic
Ap_output = NaN(length(time_vector)-1,1); %N-1 intervals
k=1;

for i = 1:length(time_vector)-1
    counter_exceed=[];
    interval_t = fixed.Interval( time_vector(i,1), time_vector(i+1,1),'()');
    overlap_intervals = zeros(1,length(datenum_Ap));

    for j = k:length(datenum_Ap) %on parcours le vecteur R de k à length(R_f) et on regarde s'il y a intersection avec le vecteur temporel
        interval_Annot = fixed.Interval(datenum_Ap(j,1), datenum_Ap(j,2),'()');
        overlap_intervals(j) = overlaps(interval_Annot, interval_t);
        %Si la fin d'une annotation raven(k) depasse la fin de la timebin(i) et se termine une timebin(i+N),
        %l'indice k doit recommencer à cette valeur pour la timebin suivante (cf l66)
        if interval_Annot.RightEnd > interval_t.RightEnd
            counter_exceed = [counter_exceed;j];
        end
    end
    
%     if find( overlap_intervals(k:length(datenum_Ap)) == 1, 1 ) >= 1
    if any( overlap_intervals(k:length(datenum_Ap)) ) 
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
    
clc; disp([ num2str(i),'/',num2str(length(time_vector)-1) ])

end
clc;toc
%~500s pour 1 déploiement complet
%% Comparison of time vectors / PG detection vector - method 1
tic
PG_output = NaN(length(time_vector)-1,1);
k=1;
interval_Annot=[];

for i = 1:length(time_vector)-1
    counter_exceed=[];
    interval_t = fixed.Interval(time_vector(i,1), time_vector(i+1,1), '()');

    parfor j = k:length(datenum_PG) %on parcours le vecteur datenum_PG de k à length(datenum_PG) et on regarde s'il y a intersection avec le vecteur temporel
        interval_Annot = fixed.Interval(datenum_PG(j,1), datenum_PG(j,2), '()');
%         annot_datetime_test1 = string(datestr(double(datenum_PG(:,1))/(3600*24)))
%         annot_datetime_test2 = string(datestr(double(datenum_PG(:,2))/(3600*24)))
%         annot_test = [annot_datetime_test1, annot_datetime_test2]
%         t_datetime_test = datestr([double(time_vector(i,1)), double(time_vector(i+1,1)) ]/(3600*24));
        overlap_intervals(j) = overlaps(interval_Annot, interval_t); %test intersections interval_Annot avec interval_t(i)
        
        %Si la fin d'une annotation(k) depasse la fin de la timebin(i) et se termine une timebin(i+N),
        %l'indice k doit recommencer à cette valeur pour la timebin suivante (cf l66)
        if interval_Annot.RightEnd > interval_t.RightEnd
            counter_exceed = [counter_exceed;j]; 
        end
    end
    
    if find( overlap_intervals(k:length(datenum_PG)) == 1, 1 ) == 1
        PG_output(i,1) = 1; %PG_output(j) = 1 si intersection sinon 0
    elseif sum( overlap_intervals(k:length(datenum_PG))) == 0
        PG_output(i,1) = 0; %PG_output(j) = 1 si intersection sinon 0
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
    clc; disp([ num2str(i),'/',num2str(length(time_vector)-1) ])
end
clc; elapsed_time = toc
%elapsed time ~9000s pour un deploiement de 15j


%% Comparison of time vectors / PG detection vector - method 2
tic
interval_t = fixed.Interval(time_vector(1:end-1), time_vector(2:end),'()');

for i = 1:length(datenum_PG)
    interval_Annot2(i) = fixed.Interval(datenum_PG(i,1),datenum_PG(i,2),'()');
end

for i = 1:length(interval_Annot2)
    overlap_PG_Ap(:,i) = overlaps(interval_Annot2(i), interval_t);
end

PG_output2 = any(overlap_PG_Ap,2);
clc; elapsed_time2 = toc
%elapsed time ~4000s pour un deploiement de 15j

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
Recall2 = length(find(comparison2 == "VP")) / (length(find(comparison2 == "VP")) + length(find(comparison2 == "FN")) );

clc
disp(['Precision 2: ', num2str(Precision2), '; Recall 2: ', num2str(Recall2)])

















%%
% %si intervalle fermé (= '[]') il y a overlap d'un cadre Aplose avec les
% %cadre du vecteur temps l'encadrant alors qu'il n'y a "overlap" que sur les
% %valeurs limite LeftEnd & RightEnd. Cf exemple ci-dessous
% interval_1 = fixed.Interval(10, 20,'()');
% interval_2 = fixed.Interval(20, 30,'()');
% interval_3 = fixed.Interval(30, 40,'()');
% interval_123 = [test1,test2,test3];
% interval_4 = fixed.Interval(20, 30,'()');
% overlaps(interval_123,interval_4)
