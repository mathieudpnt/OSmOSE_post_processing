function [output] = Box2Timebin(Annotation, time_bin, wav_path)
%Here the dataset is divided into time bins
%If there is an overlap between a detection and a the corresponding time
%bin, then the detection is considered a true positive
%Otherwise the detection is not considered relevant

%Here we find the duration of the dataset with the begin time of the first
%file and the begin time and duration of the last file
% folder_data_wav= uigetdir('','Select folder contening wav files');
folder_data_wav = wav_path;
wavList = dir(fullfile(folder_data_wav, '*.wav'));
wavNames = '';
wavDates = "";
for i = 1:length(wavList)
    wavNames(i,:) = (wavList(i).name);
    wavDates(i,:) = (wavNames(i,end-15:end-4));
end

wavDates_formated = datetime(wavDates, 'InputFormat', 'yyMMddHHmmss', 'Format', 'yyyy MM dd - HH mm ss');
[FirstDate, posMin] = min(wavDates_formated);
[LastDate, posMax] = max(wavDates_formated);

lastwavinfo = audioinfo(strcat(folder_data_wav,"\",string(wavNames(posMax,:))));


%Creation of a datenum time vector from beginning of 1st file to end of last file with time_bin as a
%time step
datenum_begin = datenum(FirstDate)*24*3600;
datenum_end = datenum(LastDate)*24*3600 + lastwavinfo.Duration;

total_duration = datenum_end - datenum_begin;

%the last bin might not me stricly equal to the time_bin (e.i. 9.9s instead
%of 10s for example) so we "manually" add the last timebin to the time vector
last_bin = mod(total_duration,time_bin);

time_vector_f1 = [datenum_begin:time_bin:datenum_end]';
time_vector_f = [time_vector_f1; time_vector_f1(end) + datenum(last_bin)];

%From a relative time to absolute time (datenum)
Annotation_f = (datenum(FirstDate)*24*3600)+ Annotation;

output = NaN(length(time_vector_f)-1,1);
k=1;
interval_Annot = NaN;

for i = 1:length(time_vector_f)-1
    counter_exceed=[];
    interval_t = fixed.Interval(time_vector_f(i,1), time_vector_f(i+1,1) );

    for j = k:length(Annotation_f) %on parcours le vecteur R de k à length(R_f) et on regarde s'il y a intersection avec le vecteur temporel
        interval_Annot = fixed.Interval(Annotation_f(j,1), Annotation_f(j,2) );
        overlap_intervals(j) = overlaps(interval_Annot, interval_t);
        %Si la fin d'une annotation raven(k) depasse la fin de la timebin(i) et se termine une timebin(i+N),
        %l'indice k doit recommencer à cette valeur pour la timebin suivante (cf l66)
        if interval_Annot.RightEnd > interval_t.RightEnd
            counter_exceed = [counter_exceed;j]; 
        end
    end
    
    if find( overlap_intervals(k:length(Annotation_f)) == 1, 1 ) == 1
        output(i,1) = 1; %output_R(j) = 1 si intersection sinon 0
    elseif sum( overlap_intervals(k:length(Annotation_f))) == 0
        output(i,1) = 0; %output_R(j) = 1 si intersection sinon 0
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

end

end

